"""Schritt 1 - System pruefen, Mikrofon auswaehlen, config.json schreiben.

Aufruf:
    python werkzeuge/01_systemcheck.py                 # Geraete zeigen + bestes testen
    python werkzeuge/01_systemcheck.py --liste         # nur auflisten
    python werkzeuge/01_systemcheck.py --geraet 33     # bestimmtes Geraet festlegen
    python werkzeuge/01_systemcheck.py --geraet RODE   # per Namensfragment
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sounddevice as sd

from tastenakustik import audio
from tastenakustik.config import (CONFIG_PFAD, TASTEN, anzeige,
                                  laden_oder_beenden, verzeichnisse_anlegen)

GRUEN, GELB, ROT, GRAU, AUS = "\033[92m", "\033[93m", "\033[91m", "\033[90m", "\033[0m"


def titel(text: str) -> None:
    print(f"\n{text}\n" + "-" * len(text))


def systeminfo() -> None:
    titel("System")
    print(f"  OS          {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python      {platform.python_version()}  -  {sys.executable}")
    # Nicht nur ImportError: Eine kaputte Installation unter Windows meldet
    # sich oft mit OSError (DLL laesst sich nicht laden) - das darf den
    # Systemcheck nicht abbrechen, er soll es ja gerade anzeigen.
    for name in ("numpy", "scipy", "sounddevice", "soundfile", "matplotlib"):
        try:
            mod = __import__(name)
            print(f"  {name:<12}{getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"  {name:<12}{ROT}fehlt{AUS}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<12}{ROT}laesst sich nicht laden: {exc}{AUS}")
    try:
        import torch  # noqa: PLC0415

        cuda = "CUDA " + torch.version.cuda if torch.cuda.is_available() else "nur CPU"
        print(f"  torch       {torch.__version__} ({cuda})  {GRAU}- erst fuers Training noetig{AUS}")
    except ImportError:
        print(f"  torch       {GRAU}nicht installiert - erst fuers Training noetig{AUS}")
    except Exception as exc:  # noqa: BLE001
        print(f"  torch       {GELB}laesst sich nicht laden ({exc}) - erst fuers "
              f"Training noetig{AUS}")


def geraete_zeigen(alle: list[audio.Geraet]) -> None:
    titel(f"Eingangsgeraete ({len(alle)})")
    for g in alle:
        marke = f"{GRUEN}geeignet{AUS}" if g.geeignet else f"{GRAU}ungeeignet{AUS}"
        print(f"  {g.index:>3}  {g.hostapi:<20} {g.latenz_ms:>6.1f} ms  {g.kanaele}ch  "
              f"{g.name[:42]:<42} {marke}")
    print(f"\n  {GRAU}'ungeeignet' = virtuelle Geraete, Loopbacks, Software-Mikrofone "
          f"mit eigener Signalverarbeitung.{AUS}")


def messen(g: audio.Geraet, sekunden: float, samplerate: int, kanaele: int) -> bool:
    titel(f"Testaufnahme: {g.name}")
    print(f"  {sekunden:.0f} s Ruhe aufnehmen - bitte waehrenddessen nicht tippen ...")
    print(f"  {GRAU}Diese Messung dient nur der Pegelpruefung und wird nicht "
          f"gespeichert.{AUS}")
    try:
        x = sd.rec(int(sekunden * samplerate), samplerate=samplerate, channels=kanaele,
                   dtype="float32", device=g.index)
        sd.wait()
    except Exception as exc:  # noqa: BLE001
        print(f"  {ROT}Fehler: {exc}{AUS}")
        return False

    x = x[:, 0]
    rms = audio.rms_dbfs(x)
    peak = audio.peak_dbfs(x)
    dc = float(np.mean(x))
    # Ein sehr sauberes Interface rauscht bei -80 bis -85 dBFS, die Spitze
    # liegt dann um -72. Wirklich stumm ist erst, was darunter bleibt.
    stumm = peak < -90

    print(f"  Rauschboden (RMS)   {rms:7.1f} dBFS")
    print(f"  Spitze              {peak:7.1f} dBFS")
    print(f"  Gleichanteil        {dc:+.5f}")

    if stumm:
        print(f"  {ROT}Kein Signal. Mikrofon stummgeschaltet oder falscher Eingang?{AUS}")
        # Dieselbe Totenstille erzeugen auch ein Noise Gate oder die Windows-
        # Signalverbesserung - der haeufigste Stolperstein. WDM-KS hilft nur
        # gegen Letztere, ein Gate am Interface muss dort abgeschaltet werden.
        # --geraet im Hinweis ist noetig: Ohne config.json bricht 00 sonst ab.
        print(f"  {GRAU}Moeglich sind auch ein Noise Gate am Interface oder eine "
              f"Windows-Signalverbesserung am Aufnahmegeraet (Geraeteeigenschaften -> "
              f"Erweitert). Gegen Letztere hilft dasselbe Geraet ueber WDM-KS statt "
              f"WASAPI (Index aus der Liste oben, --geraet N).{AUS}")
        print(f"  {GRAU}Live pruefen: python werkzeuge/00_signalweg.py "
              f"--geraet {g.index}{AUS}")
        return False
    if rms > -40:
        print(f"  {GELB}Recht lauter Grundpegel. Fuer Transienten ist ein ruhiger "
              f"Raum wichtig - Luefter, Fenster, Klimaanlage pruefen.{AUS}")
    else:
        print(f"  {GRUEN}Rauschboden in Ordnung.{AUS}")
    if abs(dc) > 0.005:
        print(f"  {GELB}Spuerbarer Gleichanteil - der Hochpass in der Analyse faengt das ab.{AUS}")
    return True


def main() -> int:
    # Der Aufruf-Block aus dem Docstring erscheint unter --help als Beispiel.
    p = argparse.ArgumentParser(
        description="Systemcheck und Mikrofonauswahl",
        epilog=__doc__[__doc__.index("Aufruf:"):],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--liste", action="store_true", help="nur Geraete auflisten")
    p.add_argument("--geraet", default=None,
                   help="Index oder Namensfragment, sonst das geeignete mit der "
                        "niedrigsten Latenz")
    p.add_argument("--samplerate", type=int, default=48_000,
                   help="gewuenschte Abtastrate in Hz (Standard 48000); nimmt das "
                        "Geraet sie nicht an, wird eine passende gewaehlt")
    p.add_argument("--sekunden", type=float, default=2.0,
                   help="Dauer der Ruhemessung fuer den Rauschboden in s (Standard 2)")
    args = p.parse_args()

    verzeichnisse_anlegen()
    systeminfo()

    alle = audio.eingaenge()
    if not alle:
        print(f"\n{ROT}Kein Eingangsgeraet gefunden.{AUS}")
        return 1
    geraete_zeigen(alle)

    if args.liste:
        return 0

    g = audio.geraet_finden(args.geraet)
    if g is None:
        print(f"\n{ROT}Geraet {args.geraet!r} nicht gefunden.{AUS}")
        return 1
    if args.geraet is None:
        print(f"\n  Vorschlag (WDM-KS bevorzugt, siehe README): {GRUEN}{g.label}{AUS}")

    sr, kanaele = audio.bestes_format(g, args.samplerate)
    if sr != args.samplerate:
        print(f"  {GELB}{args.samplerate} Hz wird nicht angenommen - benutze {sr} Hz.{AUS}")
    if kanaele != 1:
        print(f"  {GRAU}Geraet liefert {kanaele} Kanaele; ausgewertet wird Kanal 1.{AUS}")

    ok = messen(g, args.sekunden, sr, kanaele)
    if not ok:
        # Ein Eingang, der nichts liefert, soll nicht als neues Mikrofon
        # in der Konfiguration landen.
        print(f"\n  {GELB}Konfiguration nicht geaendert.{AUS}")
        return 1

    cfg = laden_oder_beenden()
    cfg.device = g.index
    cfg.device_name = g.name
    cfg.hostapi = g.hostapi
    cfg.latenz_ms = g.latenz_ms
    cfg.samplerate = sr
    cfg.channels = kanaele
    cfg.speichern()

    titel("Konfiguration")
    print(f"  geschrieben nach  {CONFIG_PFAD}")
    print(f"  Geraet            {g.label}")
    print(f"  Samplerate        {cfg.samplerate} Hz, {cfg.channels} Kanal/Kanaele")
    print(f"  Fenster je Taste  {cfg.pre_roll_ms} ms vor + {cfg.post_roll_ms} ms nach "
          f"= {cfg.fenster_samples} Samples")
    print(f"  Klassen           {' '.join(anzeige(t) for t in TASTEN)}")

    print(f"\n  Naechster Schritt: {GRUEN}python werkzeuge/02_kalibrierung.py{AUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

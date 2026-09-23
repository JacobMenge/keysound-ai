"""Signalweg live pruefen - laeuft, waehrend du am Interface einstellst.

Zeigt jede Sekunde, wie ruhig der Eingang zwischen den Geraeuschen wirklich
ist. Damit laesst sich ein Noise Gate von einem echten Rauschboden
unterscheiden, ohne raten zu muessen:

  Rauschboden eines echten Mikrofons   -55 bis -85 dBFS, durchgehend
  Noise Gate oder Mute                 faellt in der Stille unter -95 dBFS

Ein Gate ist fuer dieses Experiment toedlich: Es verschluckt leise Anschlaege
ganz, schneidet den Einschwingvorgang an und legt jedem Anschlag dieselbe
Ausklingkurve auf.

Aufruf:
    python werkzeuge/00_signalweg.py
    python werkzeuge/00_signalweg.py --geraet 42

Mit Strg+C beenden.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tastenakustik import audio
from tastenakustik.config import Config

GRUEN, GELB, ROT, GRAU, AUS = "\033[92m", "\033[93m", "\033[91m", "\033[90m", "\033[0m"

GATE_GRENZE = -95.0     # darunter kommt kein echtes Mikrofon
BLOCK_MS = 50           # Aufloesung der Ruhemessung


def balken(db: float, tief: float = -120.0, hoch: float = 0.0, breite: int = 32) -> str:
    anteil = (max(db, tief) - tief) / (hoch - tief)
    voll = int(anteil * breite)
    return "#" * voll + "." * (breite - voll)


def main() -> int:
    p = argparse.ArgumentParser(description="Signalweg live pruefen")
    p.add_argument("--geraet", default=None, help="Index oder Namensfragment")
    p.add_argument("--sekunden", type=float, default=1.0, help="Takt der Anzeige")
    args = p.parse_args()

    cfg = Config.laden()
    if args.geraet is not None:
        g = audio.geraet_finden(args.geraet)
        if g is None:
            print(f"Geraet {args.geraet!r} nicht gefunden.")
            return 1
        cfg.device, cfg.device_name = g.index, g.name
        cfg.samplerate, cfg.channels = audio.bestes_format(g, cfg.samplerate)
    if cfg.device is None:
        print("Keine Konfiguration. Bitte zuerst: python werkzeuge/01_systemcheck.py")
        return 1

    print(f"Eingang: {cfg.device_name}  ({cfg.samplerate} Hz)")
    print()
    print("So gehst du vor:")
    print("  1. Erst gar nichts machen - die Zeile zeigt die reine Ruhe.")
    print("  2. Dann ein paar Mal auf die Tastatur tippen.")
    print("  3. Am Interface Noise Gate und Kompressor abschalten und zusehen,")
    print("     wie die Ruhe von unter -95 dBFS auf etwa -60 bis -80 dBFS steigt.")
    print()
    print(f"{'Ruhe':>10} {'laut':>9}   Ruhepegel")
    print("-" * 62)

    ring = audio.Ringpuffer(cfg)
    ring.start()
    w = int(BLOCK_MS / 1000 * cfg.samplerate)
    ruhig_seit = 0
    try:
        while True:
            time.sleep(args.sekunden)
            x = ring.letzte(args.sekunden)
            if x.size < w * 2:
                continue
            bl = x[: x.size // w * w].reshape(-1, w).astype(np.float64)
            pegel = 20 * np.log10(np.maximum(np.sqrt((bl ** 2).mean(axis=1)), 1e-12))
            ruhe = float(np.percentile(pegel, 20))     # was zwischen den Geraeuschen ist
            laut = float(pegel.max())

            if ruhe < GATE_GRENZE:
                farbe, urteil = ROT, "Gate oder Mute aktiv"
            elif ruhe > -45:
                farbe, urteil = GELB, "sehr lauter Raum"
            else:
                farbe, urteil = GRUEN, "sauber"
                ruhig_seit += 1
            if ruhe >= GATE_GRENZE and ruhig_seit >= 3:
                urteil = "sauber - so kann aufgenommen werden"
            if ruhe < GATE_GRENZE:
                ruhig_seit = 0

            print(f"{ruhe:9.1f}  {laut:8.1f}   {farbe}{balken(ruhe)}  {urteil}{AUS}")
    except KeyboardInterrupt:
        print("\nbeendet.")
    finally:
        ring.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

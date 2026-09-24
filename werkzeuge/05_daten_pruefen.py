"""Schritt 5 - eine aufgenommene Sitzung pruefen, bevor trainiert wird.

Beantwortet drei Fragen:

  1. Sind genug Proben je Klasse da, und wie viel wurde verworfen?
  2. Stimmen Pegel, Abstand zum Rauschen und die Lage des Anschlags?
  3. Laeuft eine Dynamikbearbeitung mit (Noise Gate, Kompressor)?

Frage 3 ist die wichtigste. Ein Noise Gate macht den Datensatz unbrauchbar:
Es schluckt leise Anschlaege ganz, schneidet den Einschwingvorgang an und legt
jedem Anschlag dieselbe kuenstliche Ausklingkurve auf. Ein Modell lernt dann
das Gate statt der Taste - und das faellt erst auf, wenn die Trefferquote auf
fremden Daten einbricht.

Aufruf:
    python werkzeuge/05_daten_pruefen.py
    python werkzeuge/05_daten_pruefen.py --sitzung S01_20260922_1410
    python werkzeuge/05_daten_pruefen.py --bild        # Uebersicht nach ausgabe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import soundfile as sf

from tastenakustik import datensatz, features, plots, portrait, storage
from tastenakustik.config import ROH, TASTEN, Config, anzeige, laden_oder_beenden

GRUEN, GELB, ROT, GRAU, AUS = "\033[92m", "\033[93m", "\033[91m", "\033[90m", "\033[0m"

# Ein echtes Mikrofonsignal kommt nicht unter diesen Pegel. Wird er in der
# Stille vor dem Anschlag unterschritten, greift eine Bearbeitung ein.
GATE_VERDACHT_DBFS = -95.0
# Nachhall: aus der Zeit bis -40 dB laesst sich die Nachhallzeit abschaetzen.
# Ein Arbeitsraum liegt bei 0,2 bis 0,5 s. Erst deutlich darueber verschmiert
# der Anschlag so stark, dass die Klassen ineinanderlaufen.
#
# Das Fenster nach dem Anschlag ist nur rund 400 ms lang - mehr als etwa 0,6 s
# laesst sich darin gar nicht messen. Klingt die Mehrheit der Anschlaege im
# Fenster nicht auf -40 dB ab, wird das deshalb ausdruecklich gemeldet statt
# als scheinbar harmloser Wert knapp unter der Warnschwelle.
ABFALL_DB = 40.0
NACHHALL_WARNUNG_S = 0.60


def titel(text: str) -> None:
    print(f"\n{text}\n" + "-" * len(text))


def abklingzeit_ms(x: np.ndarray, sr: int, ab_db: float = ABFALL_DB
                   ) -> tuple[float, bool]:
    """Zeit von der Spitze, bis der Pegel um ab_db gefallen ist.

    Gibt (ms, erreicht) zurueck. erreicht ist False, wenn das Fenster endet,
    bevor der Pegel so weit gefallen ist - dann ist die Zeit eine Untergrenze.
    """
    hoch = np.abs(x)
    i = int(np.argmax(hoch))
    spitze = hoch[i]
    if spitze <= 0:
        return 0.0, True
    schwelle = spitze * (10 ** (-ab_db / 20))
    # gleitendes Maximum ueber 5 ms, damit einzelne Nulldurchgaenge nicht zaehlen
    w = max(int(0.005 * sr), 1)
    rest = hoch[i:]
    if rest.size < w * 2:
        return 0.0, False
    block = rest[: rest.size // w * w].reshape(-1, w).max(axis=1)
    unter = np.flatnonzero(block < schwelle)
    if unter.size:
        return float(unter[0] * w / sr * 1000), True
    return float(rest.size / sr * 1000), False


def main() -> int:
    # Der Aufruf-Block aus dem Docstring erscheint unter --help als Beispiel.
    p = argparse.ArgumentParser(
        description="Aufgenommene Sitzung pruefen",
        epilog=__doc__[__doc__.index("Aufruf:"):],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sitzung", default=None,
                   help="Session-ID, sonst die neueste mit Proben")
    p.add_argument("--bild", action="store_true",
                   help="gemittelte Log-Mel-Bilder je Klasse nach ausgabe/ schreiben")
    args = p.parse_args()
    laden_oder_beenden()   # setzt die gewaehlten Klassen

    alle = storage.sitzungen()
    if not alle:
        print(f"Keine Sitzungen unter {ROH}")
        return 1
    if args.sitzung:
        ordner = ROH / args.sitzung
        if not (ordner / "session.json").exists():
            print(f"Sitzung {args.sitzung} nicht gefunden.")
            return 1
    else:
        # Die neueste Sitzung mit Proben - eine gerade abgebrochene, leere
        # Sitzung sagt nichts ueber die Aufnahmen.
        mit_proben = [o for o in alle if storage.lade_sitzung(o)[1]]
        ordner = mit_proben[-1] if mit_proben else alle[-1]

    kopf, proben = storage.lade_sitzung(ordner)
    cfg = Config(**{k: v for k, v in kopf["aufnahmeparameter"].items()
                    if k in Config.__dataclass_fields__})
    sr = cfg.samplerate

    titel(f"Sitzung {kopf['session_id']}")
    for feld, name in (("tastatur", "Tastatur"), ("mikrofon_position", "Mikrofon"),
                       ("notiz", "Notiz")):
        wert = kopf.get(feld) or f"{GRAU}(leer){AUS}"
        print(f"  {name:<10} {wert}")
    print(f"  {'Geraet':<10} {cfg.device_name}  ({cfg.samplerate} Hz)")
    passt = True
    try:
        datensatz.pruefe_passend(kopf)
        datensatz.pruefe_labels(kopf, proben)
    except ValueError as fehler:
        # Trotzdem pruefen - Pegel und Signalweg sagen auch dann etwas. Nur
        # die Zaehlung je Klasse laeuft ueber die Klassen dieser Sitzung.
        print(f"\n  {GELB}{fehler}{AUS}")
        passt = False
    klassen = list(kopf.get("tasten") or TASTEN)

    # --- 1. Vollstaendigkeit -------------------------------------------
    titel("Proben")
    je_klasse = {t: sum(1 for x in proben if x["label"] == t) for t in klassen}
    for t in klassen:
        print(f"  {anzeige(t):<3} {je_klasse[t]:>4}")
    print(f"  {'gesamt':<3} {len(proben):>4}")
    verworfen = kopf.get("verworfen", {})
    if verworfen:
        gesamt_v = sum(verworfen.values())
        quote = gesamt_v / (gesamt_v + len(proben)) * 100
        farbe = ROT if quote > 50 else (GELB if quote > 20 else GRUEN)
        print(f"\n  verworfen  {farbe}{gesamt_v} ({quote:.0f} % aller Versuche){AUS}")
        for grund, n in sorted(verworfen.items(), key=lambda kv: -kv[1]):
            print(f"    {grund:<14} {n}")

    if not proben:
        print(f"\n  {GELB}Diese Sitzung enthaelt keine Proben - vermutlich direkt "
              f"abgebrochen.{AUS}")
        return 1

    # --- 2. Pegel und Lage ---------------------------------------------
    hol = lambda feld: np.array([x[feld] for x in proben], dtype=float)  # noqa: E731
    peak, snr = hol("peak_dbfs"), hol("snr_db")
    onset_ms = hol("onset_sample") / sr * 1000 - cfg.pre_roll_ms

    titel("Pegel und Lage")
    for name, werte, einheit in (("Spitze", peak, "dBFS"),
                                 ("Abstand", snr, "dB"),
                                 ("Onset", onset_ms, "ms")):
        print(f"  {name:<9} min {werte.min():7.1f}   median {np.median(werte):7.1f}"
              f"   max {werte.max():7.1f}  {einheit}")
    if peak.max() > -3:
        print(f"  {ROT}Einzelne Proben sind nahe an der Vollaussteuerung.{AUS}")
    if np.median(peak) < -30:
        print(f"  {GELB}Sehr leise aufgenommen - mehr Gain waere besser.{AUS}")
    streuung = float(np.percentile(onset_ms, 90) - np.percentile(onset_ms, 10))
    print(f"  Onset-Streuung (10-90 %)  {streuung:.0f} ms")

    # --- 3. Dynamikbearbeitung -----------------------------------------
    titel("Signalweg")
    vorlauf, abkling, erreicht = [], [], []
    for x in proben:
        welle, _ = sf.read(ordner / x["datei"], dtype="float32")
        vor = welle[: int(0.10 * sr)].astype(np.float64)
        vorlauf.append(20 * np.log10(max(float(np.sqrt((vor ** 2).mean())), 1e-12)))
        ms, ok = abklingzeit_ms(welle, sr)
        abkling.append(ms)
        erreicht.append(ok)
    vorlauf, abkling = np.array(vorlauf), np.array(abkling)

    nachhall = float(np.median(abkling)) * 60.0 / ABFALL_DB / 1000.0
    # Klingt die Mehrheit im Fenster nicht weit genug ab, ist die Zahl nur
    # eine Untergrenze.
    untergrenze = bool(np.mean(erreicht) < 0.5)
    print(f"  Pegel vor dem Anschlag   median {np.median(vorlauf):7.1f} dBFS")
    print(f"  Abklingen auf -{ABFALL_DB:.0f} dB     median {np.median(abkling):7.0f} ms"
          f"   (Nachhall {'mindestens' if untergrenze else 'rund'} {nachhall:.2f} s)")

    # Entscheidend ist allein die Stille zwischen den Anschlaegen. Die
    # Abklingzeit haengt am Raum und schwankt auch bei identischem Aufbau -
    # sie taugt nicht als Beweis fuer eine Bearbeitung.
    gate = np.median(vorlauf) < GATE_VERDACHT_DBFS
    print()
    if gate:
        print(f"  {ROT}BEFUND: Es laeuft eine Dynamikbearbeitung mit.{AUS}")
        print(f"    Die Stille vor dem Anschlag liegt bei "
              f"{np.median(vorlauf):.0f} dBFS. Ein echtes Mikrofon kommt")
        print(f"    nicht unter etwa {GATE_VERDACHT_DBFS:.0f} dBFS - das ist ein "
              f"Noise Gate oder Expander.")
        print()
        print(f"  {GELB}Diese Aufnahme taugt nicht zum Trainieren:{AUS}")
        print("    - leise Anschlaege werden ganz verschluckt, der Datensatz")
        print("      enthaelt nur die lauten")
        print("    - der Einschwingvorgang wird angeschnitten, und genau dort")
        print("      steckt die Information ueber die Taste")
        print("    - jeder Anschlag bekommt dieselbe kuenstliche Ausklingkurve")
    else:
        print(f"  {GRUEN}BEFUND: Signalweg ist sauber - kein Gate, kein Kompressor.{AUS}")
        if untergrenze:
            print(f"  {GELB}Die meisten Anschlaege klingen im Aufnahmefenster nicht auf "
                  f"-{ABFALL_DB:.0f} dB ab.{AUS}")
            print(f"  {GELB}Entweder hallt der Raum kraeftig, oder der Rauschboden liegt "
                  f"zu nah an den Anschlaegen.{AUS}")
            print(f"  {GELB}Beides macht die Klassen aehnlicher - naeher ran ans Mikrofon, "
                  f"weiche Oberflaechen helfen.{AUS}")
        elif nachhall > NACHHALL_WARNUNG_S:
            print(f"  {GELB}Der Raum hallt mit rund {nachhall:.2f} s allerdings kraeftig."
                  f" Das verschmiert{AUS}")
            print(f"  {GELB}den Anschlag und macht die Klassen aehnlicher. "
                  f"Weiche Oberflaechen helfen.{AUS}")

    # --- 4. Bild -------------------------------------------------------
    if args.bild and not passt:
        # Das Bild zeichnet die eingestellten Klassen. Fuer eine Sitzung mit
        # anderen Klassen entstuenden leere Felder - und das unter demselben
        # Dateinamen wie das echte Bild aus 06_trennbarkeit.
        print(f"\n  {GRAU}Bild uebersprungen: Die Klassen der Sitzung passen nicht "
              f"zur Einstellung.{AUS}")
    elif args.bild:
        # Derselbe Schnitt wie in 06_trennbarkeit - sonst entstehen unter
        # demselben Dateinamen zwei verschieden geschnittene Bilder.
        n_seg = int(cfg.segment_ms / 1000 * sr)
        vor_n = int(cfg.segment_vor_onset_ms / 1000 * sr)
        mittel = {}
        for t in TASTEN:
            stapel = []
            for x in [q for q in proben if q["label"] == t]:
                welle, _ = sf.read(ordner / x["datei"], dtype="float32")
                s0 = max(0, min(x["onset_sample"] - vor_n, welle.size - n_seg))
                stapel.append(features.log_mel(
                    welle[s0:s0 + n_seg], sr, cfg.mel_nfft,
                    cfg.mel_hop, cfg.mel_baender, cfg.mel_fmin, cfg.mel_fmax))
            if stapel:
                mittel[t] = np.mean(stapel, axis=0)
        fig = plots.klassen_vergleich(mittel, cfg)
        pfad = portrait.exportiere(fig, "04_klassenvergleich", "03_daten")
        print(f"\n  Bild: {pfad}")

    titel("Weiter")
    if gate:
        print("  Erst die Bearbeitung am Interface abschalten, dann einen neuen")
        print("  Probelauf. Diese Sitzung nicht zum Training verwenden.")
    else:
        print(f"  python werkzeuge/03_collector.py --ziel 40   {GRAU}# richtige Sitzung{AUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

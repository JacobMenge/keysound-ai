"""Schritt 7 - das kleine CNN trainieren.

Trainiert auf den Sitzungen mit Rolle "train", geprueft gegen "val". Beides
sind ganze Sitzungen, nie geteilte Proben - siehe datensatz.py.

Am Ende stehen: das Modell, der Verlauf als JSON und die Grafiken mit echten
Zahlen. Kein Ergebnis wird geschoent; steht die Trefferquote bei 30 %, steht
sie bei 30 %.

Die Rechnung selbst liegt in tastenakustik/training.py - hier ist nur die
Bedienung ueber die Kommandozeile. Dasselbe Training laeuft im Studio
(python start.py) mit mitwachsender Kurve.

Aufruf:
    python werkzeuge/07_training.py --rollen           # zeigt, was zugeordnet ist
    python werkzeuge/07_training.py
    python werkzeuge/07_training.py --epochen 80 --segment 250
    python werkzeuge/07_training.py --test             # neuestes Modell gegen "test"

Ohne --segment/--vor gelten segment_ms und segment_vor_onset_ms aus der
config.json. --test nimmt dagegen immer die Werte aus der Modelldatei.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

import numpy as np

from tastenakustik import datensatz, plots, portrait, storage, training
from tastenakustik.config import TASTEN, anzeige, laden_oder_beenden, zufall

GRUEN, GELB, ROT, GRAU, AUS = "\033[92m", "\033[93m", "\033[91m", "\033[90m", "\033[0m"


def naechster_schritt() -> None:
    """Hinweis, wie es weitergeht - ohne Sitzungen ist Zuordnen sinnlos."""
    if not storage.sitzungen():
        print(f"{GRAU}Erst aufnehmen: python werkzeuge/03_collector.py "
              f"(oder python start.py, Schritt 3){AUS}")
    else:
        print(f"{GRAU}Zuordnen mit: python werkzeuge/04_sitzungen.py "
              f"--rolle <ID>=train{AUS}")


def farbe_fuer(quote: float) -> str:
    """Ampel gemessen am Zufall - bei 2 und bei 40 Klassen gleich aussagekraeftig."""
    anteil = (quote - zufall()) / max(1.0 - zufall(), 1e-9)
    return GRUEN if anteil >= 0.5 else GELB if anteil >= 0.15 else ROT


def je_klasse_zeilen(quoten: dict[str, float]) -> None:
    for taste, quote in quoten.items():
        if quote != quote:                      # NaN: keine Probe dieser Klasse
            print(f"  {anzeige(taste)}      -   (keine Probe)")
        else:
            print(f"  {anzeige(taste)}  {quote * 100:5.1f} %")


def main() -> int:
    p = argparse.ArgumentParser(description="Kleines CNN trainieren")
    p.add_argument("--rollen", action="store_true", help="Rollenzuordnung zeigen")
    p.add_argument("--epochen", type=int, default=80, help="Anzahl Epochen (Standard 80)")
    p.add_argument("--segment", type=float, default=None,
                   help="Segmentlaenge in ms (Vorgabe: segment_ms aus config.json)")
    p.add_argument("--vor", type=float, default=None,
                   help="ms vor dem Onset (Vorgabe: segment_vor_onset_ms aus "
                        "config.json)")
    p.add_argument("--lernrate", type=float, default=2e-3, help="Lernrate fuer AdamW")
    p.add_argument("--seed", type=int, default=1,
                   help="Startwert fuer Zufall - gleicher Seed, gleiches Ergebnis")
    p.add_argument("--kein-bild", action="store_true",
                   help="keine Hochformat-Grafiken nach ausgabe/ schreiben")
    p.add_argument("--test", action="store_true",
                   help="nicht trainieren, sondern das neueste Modell auf der "
                        "Testsitzung pruefen")
    args = p.parse_args()
    cfg = laden_oder_beenden()   # setzt die gewaehlten Klassen

    if args.test:
        if training.neuestes_modell() is None:
            print(f"{ROT}Es gibt noch kein trainiertes Modell.{AUS}")
            print(f"{GRAU}Erst trainieren: python werkzeuge/07_training.py{AUS}")
            return 1
        try:
            t = training.teste()
        except (training.DatenFehler, ValueError) as fehler:
            print(f"{ROT}{fehler}{AUS}")
            return 1
        print(f"Modell      {t.modell_pfad.name}")
        print(f"Test        {t.n} Proben aus {t.sitzungen}")
        print(f"Trefferquote {farbe_fuer(t.quote)}{t.quote * 100:.1f} %{AUS}   "
              f"({t.faktor:.1f}-fach ueber Zufall von {zufall() * 100:.1f} %)")
        if t.val_quote is not None:
            print(f"Validation  {t.val_quote * 100:.1f} %   (zum Vergleich)")
        if t.hinweis:
            print(f"{GELB}{t.hinweis}{AUS}")
        print("\nJe Klasse:")
        je_klasse_zeilen(t.je_klasse)
        print(f"\nGespeichert: {t.pfad}")
        if not args.kein_bild:
            # Eigener Name, damit die Test-Matrix nicht mit der Val-Matrix
            # aus dem Training (08_confusion) verwechselt wird.
            fig = plots.konfusionsmatrix(t.konfusion, t.quote)
            print(f"Bild:        "
                  f"{portrait.exportiere(fig, '08b_confusion_test', '04_modell')}")
        return 0

    if args.rollen:
        print("Sitzungen und ihre Rollen:\n")
        print(datensatz.uebersicht())
        print()
        naechster_schritt()
        return 0

    segment = cfg.segment_ms if args.segment is None else args.segment
    vor = cfg.segment_vor_onset_ms if args.vor is None else args.vor
    fenster_ms = cfg.pre_roll_ms + cfg.post_roll_ms
    # Untergrenze in Samples wie beim Schnitt: drei Pooling-Stufen brauchen
    # mindestens 8 Mel-Rahmen, sonst stuerzt das Netz ab.
    n_min = cfg.mel_nfft + 7 * cfg.mel_hop
    min_ms = training.mindest_segment_ms(datensatz.merkmale_von(cfg))
    if args.epochen < 1:
        print(f"{ROT}--epochen muss mindestens 1 sein.{AUS}")
        return 1
    if (int(segment / 1000 * cfg.samplerate) < n_min
            or segment > fenster_ms - vor):
        # Ohne --segment kommt der Wert aus config.json - dann dort ansetzen
        quelle = "--segment" if args.segment is not None else "segment_ms in config.json"
        print(f"{ROT}{quelle} muss zwischen {min_ms:g} und {fenster_ms - vor:.0f} ms "
              f"liegen - kuerzer reicht dem Netz bei {cfg.samplerate} Hz nicht, "
              f"laenger als das aufgenommene Fenster geht nicht.{AUS}")
        return 1

    try:
        train, val = training.pruefe_daten(segment, vor)
    except training.DatenFehler as fehler:
        print(f"{ROT}{fehler}{AUS}\n")
        print(datensatz.uebersicht())
        print()
        naechster_schritt()
        return 1
    except ValueError as fehler:
        # Klassenwechsel, gemischte Abtastraten, zu langes Segment - die
        # Meldung erklaert es selbst, eine Rollenuebersicht hilft da nicht.
        print(f"{ROT}{fehler}{AUS}")
        return 1

    print(f"Training    {len(train):>4} Proben aus {sorted(set(train.sitzungen))}")
    print(f"Validation  {len(val):>4} Proben aus {sorted(set(val.sitzungen))}")
    print(f"Eingang     {train.x.shape[1]} Mel-Baender x {train.x.shape[2]} Zeitschritte"
          f"   ({segment:g} ms ab Onset -{vor:g} ms)")
    print(f"Klassen     {len(TASTEN)}   ({' '.join(anzeige(t) for t in TASTEN)})")
    print(f"Zufall      {zufall() * 100:.1f} %\n")

    ergebnis = None
    for stand in training.trainiere(args.epochen, segment, vor,
                                    args.lernrate, args.seed):
        if isinstance(stand, training.Ergebnis):
            ergebnis = stand
            break
        if stand.epoche % 10 == 0 or stand.epoche == 1:
            print(f"  Epoche {stand.epoche:>3}   Train {stand.train_acc * 100:5.1f} %"
                  f"   Val {stand.val_acc * 100:5.1f} %   Loss {stand.val_loss:.3f}")

    farbe = farbe_fuer(ergebnis.beste_val)
    print(f"\nModell      {ergebnis.parameter:,} Parameter".replace(",", " "))
    print(f"Beste Validation {farbe}{ergebnis.beste_val * 100:.1f} %{AUS} in Epoche "
          f"{ergebnis.beste_epoche}   ({ergebnis.faktor:.1f}-fach ueber Zufall)")
    print(f"Dauer {ergebnis.dauer_s:.0f} s")
    print("\nJe Klasse:")
    je_klasse_zeilen(ergebnis.je_klasse)
    print(f"\nModell:  {ergebnis.modell_pfad}")

    if not args.kein_bild:
        e = np.arange(1, args.epochen + 1)
        v = ergebnis.verlauf
        fig = plots.trainingsverlauf(e, v["train_loss"], v["val_loss"],
                                     v["train_acc"], v["val_acc"])
        print(f"Bild:    {portrait.exportiere(fig, '06_training', '04_modell')}")
        fig = plots.konfusionsmatrix(ergebnis.konfusion, ergebnis.beste_val)
        print(f"Bild:    {portrait.exportiere(fig, '08_confusion', '04_modell')}")

    print(f"\n{GRAU}Das ist die Validation aus einer Sitzung, die am selben Tag")
    print(f"aufgenommen wurde. Die ehrliche Zahl kommt aus einer Testsitzung,")
    print(f"die spaeter und moeglichst an einem anderen Tag entsteht.{AUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

import numpy as np

from tastenakustik import datensatz, plots, portrait, training
from tastenakustik.config import TASTEN, Config, zufall

GRUEN, GELB, ROT, GRAU, AUS = "\033[92m", "\033[93m", "\033[91m", "\033[90m", "\033[0m"


def main() -> int:
    p = argparse.ArgumentParser(description="Kleines CNN trainieren")
    p.add_argument("--rollen", action="store_true", help="Rollenzuordnung zeigen")
    p.add_argument("--epochen", type=int, default=80)
    p.add_argument("--segment", type=float, default=250.0, help="Segmentlaenge in ms")
    p.add_argument("--vor", type=float, default=15.0, help="ms vor dem Onset")
    p.add_argument("--lernrate", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--kein-bild", action="store_true")
    args = p.parse_args()
    Config.laden()   # setzt die gewaehlten Klassen

    if args.rollen:
        print("Sitzungen und ihre Rollen:\n")
        print(datensatz.uebersicht())
        print(f"\n{GRAU}Zuordnen mit: python werkzeuge/04_sitzungen.py "
              f"--rolle <ID>=train{AUS}")
        return 0

    try:
        train, val = training.pruefe_daten(args.segment, args.vor)
    except training.DatenFehler as fehler:
        print(f"{ROT}{fehler}{AUS}\n")
        print(datensatz.uebersicht())
        print(f"\n{GRAU}Zuordnen mit: python werkzeuge/04_sitzungen.py "
              f"--rolle <ID>=train{AUS}")
        return 1

    print(f"Training    {len(train):>4} Proben aus {sorted(set(train.sitzungen))}")
    print(f"Validation  {len(val):>4} Proben aus {sorted(set(val.sitzungen))}")
    print(f"Eingang     {train.x.shape[1]} Mel-Baender x {train.x.shape[2]} Zeitschritte"
          f"   ({args.segment:.0f} ms ab Onset -{args.vor:.0f} ms)")
    print(f"Klassen     {len(TASTEN)}   ({' '.join(t.upper() for t in TASTEN)})")
    print(f"Zufall      {zufall() * 100:.1f} %\n")

    ergebnis = None
    for stand in training.trainiere(args.epochen, args.segment, args.vor,
                                    args.lernrate, args.seed):
        if isinstance(stand, training.Ergebnis):
            ergebnis = stand
            break
        if stand.epoche % 10 == 0 or stand.epoche == 1:
            print(f"  Epoche {stand.epoche:>3}   Train {stand.train_acc * 100:5.1f} %"
                  f"   Val {stand.val_acc * 100:5.1f} %   Loss {stand.val_loss:.3f}")

    farbe = (GRUEN if ergebnis.beste_val > 0.5
             else GELB if ergebnis.beste_val > 2 * zufall() else ROT)
    print(f"\nModell      {ergebnis.parameter:,} Parameter".replace(",", " "))
    print(f"Beste Validation {farbe}{ergebnis.beste_val * 100:.1f} %{AUS} in Epoche "
          f"{ergebnis.beste_epoche}   ({ergebnis.faktor:.1f}-fach ueber Zufall)")
    print(f"Dauer {ergebnis.dauer_s:.0f} s")
    print("\nJe Klasse:")
    for taste, quote in ergebnis.je_klasse.items():
        print(f"  {taste.upper()}  {quote * 100:5.1f} %")
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

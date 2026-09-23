"""Schritt 6 - Sind die Klassen ueberhaupt unterscheidbar?

Diese Pruefung kommt bewusst VOR dem Training. Ein neuronales Netz findet auf
400 Proben fast immer irgendeine Struktur - auch eine, die nur in dieser einen
Aufnahme existiert. Deshalb erst der einfachste denkbare Klassifikator:

  Naechster Mittelwert. Fuer jede Klasse wird das mittlere Log-Mel-Bild
  gebildet, und eine neue Probe bekommt das Label des naechstgelegenen.

Kein Lernen, keine Parameter, nichts zum Ueberanpassen. Was dieser Ansatz
findet, ist wirklich da. Was er nicht findet, kann ein Netz trotzdem finden -
deshalb ist ein schwaches Ergebnis hier kein Ausschlusskriterium, sondern nur
ein Hinweis, wie deutlich das Signal ist.

Zwei Varianten, weil sie verschiedene Fragen beantworten:

  mit Lautstaerke     Wie laut eine Taste ist, gehoert zum Klang. Aber es
                      haengt auch daran, wie fest man drueckt.
  ohne Lautstaerke    Jede Probe wird auf gleiche Energie normiert. Was dann
                      bleibt, ist die Klangfarbe - der robustere Teil.

Aufruf:
    python werkzeuge/06_trennbarkeit.py
    python werkzeuge/06_trennbarkeit.py --train S03_... --test S02_...
    python werkzeuge/06_trennbarkeit.py --bild
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
from tastenakustik.config import ROH, TASTEN, Config, zufall

GRUEN, GELB, ROT, GRAU, AUS = "\033[92m", "\033[93m", "\033[91m", "\033[90m", "\033[0m"



def lade(ordner: Path) -> tuple[Config, list[str], np.ndarray, list[dict]]:
    """Alle Proben einer Sitzung als Log-Mel-Bilder laden."""
    kopf, proben = storage.lade_sitzung(ordner)
    try:
        datensatz.pruefe_passend(kopf)
        datensatz.pruefe_labels(kopf, proben)
    except ValueError as fehler:
        raise SystemExit(f"{ROT}{fehler}{AUS}")
    cfg = Config(**{k: v for k, v in kopf["aufnahmeparameter"].items()
                    if k in Config.__dataclass_fields__})
    sr = cfg.samplerate
    n_seg = int(cfg.segment_ms / 1000 * sr)
    vor = int(cfg.segment_vor_onset_ms / 1000 * sr)

    bilder, labels, behalten = [], [], []
    for p in proben:
        welle, _ = sf.read(ordner / p["datei"], dtype="float32")
        start = max(0, min(p["onset_sample"] - vor, welle.size - n_seg))
        segment = welle[start:start + n_seg]
        if segment.size < n_seg:
            continue
        bilder.append(features.log_mel(segment, sr, cfg.mel_nfft, cfg.mel_hop,
                                       cfg.mel_baender, cfg.mel_fmin, cfg.mel_fmax))
        labels.append(p["label"])
        behalten.append(p)
    return cfg, labels, np.stack(bilder), behalten


def merkmale(bilder: np.ndarray, ohne_lautstaerke: bool) -> np.ndarray:
    """Log-Mel-Bilder zu Vektoren machen."""
    x = bilder.reshape(len(bilder), -1).astype(np.float64)
    if ohne_lautstaerke:
        # Jede Probe auf gleichen Mittelwert und gleiche Streuung bringen -
        # danach zaehlt nur noch die Form, nicht die Lautstaerke.
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)
    return x


def naechster_mittelwert(x_train, y_train, x_test) -> np.ndarray:
    mitten = np.stack([x_train[[i for i, y in enumerate(y_train) if y == t]].mean(axis=0)
                       for t in TASTEN])
    abstand = ((x_test[:, None, :] - mitten[None, :, :]) ** 2).sum(axis=2)
    return np.array([TASTEN[i] for i in abstand.argmin(axis=1)])


def bewerte(y_wahr, y_vorher) -> tuple[float, np.ndarray]:
    treffer = float(np.mean([a == b for a, b in zip(y_wahr, y_vorher)]))
    m = np.zeros((len(TASTEN), len(TASTEN)))
    for a, b in zip(y_wahr, y_vorher):
        m[TASTEN.index(a), TASTEN.index(b)] += 1
    return treffer, m


def main() -> int:
    p = argparse.ArgumentParser(description="Trennbarkeit der Klassen pruefen")
    p.add_argument("--train", default=None, help="Sitzung zum Anlernen, sonst die groesste")
    p.add_argument("--test", default=None, help="Sitzung zum Pruefen, sonst eine andere")
    p.add_argument("--bild", action="store_true", help="Grafiken nach ausgabe")
    args = p.parse_args()
    Config.laden()   # setzt die gewaehlten Klassen

    alle = storage.sitzungen()
    if len(alle) < 1:
        print("Keine Sitzungen gefunden.")
        return 1

    groessen = {o.name: len(storage.lade_sitzung(o)[1]) for o in alle}
    train_name = args.train or max(groessen, key=groessen.get)
    rest = [n for n in groessen if n != train_name and groessen[n] >= len(TASTEN)]
    test_name = args.test or (rest[-1] if rest else None)

    cfg, y_train, b_train, _ = lade(ROH / train_name)
    print(f"Anlernen an {train_name}  ({len(y_train)} Proben)")
    if test_name:
        _, y_test, b_test, _ = lade(ROH / test_name)
        print(f"Pruefen an   {test_name}  ({len(y_test)} Proben)")
        print(f"{GRAU}Beide Sitzungen stammen vom selben Tag mit identischem Aufbau -")
        print(f"das Ergebnis ist deshalb optimistisch. Der ehrliche Test kommt aus")
        print(f"einer spaeter aufgenommenen Sitzung.{AUS}")
    else:
        print(f"{GELB}Nur eine Sitzung vorhanden - geprueft wird gegen zurueckgehaltene")
        print(f"Proben derselben Sitzung. Das ist noch optimistischer.{AUS}")
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y_train))
        schnitt = int(len(idx) * 0.75)
        b_test = b_train[idx[schnitt:]]
        y_test = [y_train[i] for i in idx[schnitt:]]
        b_train = b_train[idx[:schnitt]]
        y_train = [y_train[i] for i in idx[:schnitt]]

    print(f"\nLog-Mel je Probe: {b_train.shape[1]} Zeitschritte x "
          f"{b_train.shape[2]} Baender")
    print(f"Zufallsniveau:    {zufall() * 100:.1f} %")

    ergebnisse = {}
    for ohne, name in ((False, "mit Lautstaerke"), (True, "ohne Lautstaerke")):
        x_tr = merkmale(b_train, ohne)
        x_te = merkmale(b_test, ohne)
        y_vor = naechster_mittelwert(x_tr, y_train, x_te)
        quote, matrix = bewerte(y_test, y_vor)
        ergebnisse[name] = (quote, matrix)
        farbe = GRUEN if quote > 0.4 else (GELB if quote > 0.2 else ROT)
        print(f"\n{name}")
        print(f"  Trefferquote  {farbe}{quote * 100:5.1f} %{AUS}   "
              f"({quote / zufall():.1f}-fach ueber Zufall)")
        je_klasse = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
        beste = int(np.argmax(je_klasse))
        schlechteste = int(np.argmin(je_klasse))
        print(f"  am besten     {TASTEN[beste].upper()}  {je_klasse[beste] * 100:.0f} %")
        print(f"  am schwersten {TASTEN[schlechteste].upper()}  "
              f"{je_klasse[schlechteste] * 100:.0f} %")

    beste_quote = max(q for q, _ in ergebnisse.values())
    print("\n" + "-" * 60)
    if beste_quote > 0.5:
        print(f"{GRUEN}Die Klassen sind deutlich unterscheidbar. Ein kleines CNN")
        print(f"sollte hier klar besser werden als dieser einfache Ansatz.{AUS}")
    elif beste_quote > 0.25:
        print(f"{GELB}Es steckt Struktur in den Daten, aber sie ist nicht eindeutig.")
        print(f"Ein CNN ist einen Versuch wert - mit ehrlicher Erwartung.{AUS}")
    else:
        print(f"{ROT}Kaum Struktur zu finden. Bevor ein Netz trainiert wird, lieber")
        print(f"Aufnahme pruefen: Mikrofonposition, Anschlagsstaerke, Raumhall.{AUS}")

    if args.bild:
        mittel = {t: b_train[[i for i, y in enumerate(y_train) if y == t]].mean(axis=0)
                  for t in TASTEN}
        fig = plots.klassen_vergleich(mittel, cfg)
        print(f"\n  {portrait.exportiere(fig, '04_klassenvergleich', '03_daten')}")
        quote, matrix = ergebnisse["ohne Lautstaerke"]
        fig = plots.konfusionsmatrix(matrix, quote)
        print(f"  {portrait.exportiere(fig, '08_confusion_einfach', '03_daten')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

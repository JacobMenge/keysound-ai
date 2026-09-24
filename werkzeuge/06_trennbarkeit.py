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
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import soundfile as sf

from tastenakustik import datensatz, features, plots, portrait, storage
from tastenakustik.config import ROH, TASTEN, Config, anzeige, zufall, laden_oder_beenden

GRUEN, GELB, ROT, GRAU, AUS = "\033[92m", "\033[93m", "\033[91m", "\033[90m", "\033[0m"


def ueber_zufall(quote: float) -> float:
    """Wie weit die Quote vom Zufall zur Perfektion gekommen ist (0 bis 1).

    Feste Schwellen wie "ueber 50 %" taugen nicht, wenn die Klassenzahl frei
    ist: Bei zwei Klassen ist das Raten, bei vierzig ein starkes Signal.
    """
    return (quote - zufall()) / max(1.0 - zufall(), 1e-9)


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
    if not bilder:
        raise SystemExit(f"{ROT}Sitzung {ordner.name} enthaelt keine verwertbaren "
                         f"Proben.{AUS}")
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
    # Nur Klassen, die beim Anlernen vorkommen: Der Mittelwert einer leeren
    # Klasse waere NaN - und argmin wuerde dann jede Probe ihr zuschlagen.
    vorhanden = [t for t in TASTEN if t in set(y_train)]
    mitten = np.stack([x_train[[i for i, y in enumerate(y_train) if y == t]].mean(axis=0)
                       for t in vorhanden])
    # Klasse fuer Klasse statt als ein Block Proben x Klassen x Merkmale: Der
    # Block braucht bei vierzig Klassen mehrere GB, so bleibt es bei einer
    # Matrix Proben x Merkmale. Das Ergebnis ist bitgleich.
    abstand = np.stack([((x_test - m) ** 2).sum(axis=1) for m in mitten], axis=1)
    return np.array([vorhanden[i] for i in abstand.argmin(axis=1)])


def bewerte(y_wahr, y_vorher) -> tuple[float, np.ndarray]:
    treffer = float(np.mean([a == b for a, b in zip(y_wahr, y_vorher)]))
    m = np.zeros((len(TASTEN), len(TASTEN)))
    for a, b in zip(y_wahr, y_vorher):
        m[TASTEN.index(a), TASTEN.index(b)] += 1
    return treffer, m


def schnitt(kopf: dict) -> tuple:
    """Was die Form der Log-Mel-Bilder einer Sitzung bestimmt.

    Nur Sitzungen mit gleichem Schnitt lassen sich vergleichen - nach einem
    Mikrofonwechsel von 48 auf 44,1 kHz sind die Bilder verschieden lang.
    """
    a = kopf.get("aufnahmeparameter", {})
    # Fehlt ein Wert in einer aelteren Sitzung, gilt wie in lade() der
    # Standardwert - sonst wuerde sie trotz gleichem Schnitt uebergangen.
    felder = Config.__dataclass_fields__
    return tuple(a.get(k, felder[k].default)
                 for k in ("samplerate", "segment_ms", "segment_vor_onset_ms",
                           "mel_nfft", "mel_hop", "mel_baender"))


def abstand_der_aufnahmen(kopf_tr: dict, kopf_te: dict) -> str:
    """Wie weit Anlern- und Pruefsitzung auseinanderliegen - ohne zu raten."""
    try:
        d1 = date.fromisoformat(str(kopf_tr.get("gestartet", ""))[:10])
        d2 = date.fromisoformat(str(kopf_te.get("gestartet", ""))[:10])
    except ValueError:
        return "Die Pruefsitzung ist eine andere Aufnahme."
    aufbau_gleich = all(kopf_tr.get(f) == kopf_te.get(f)
                        for f in ("tastatur", "mikrofon_position"))
    if d1 == d2:
        return ("Beide Sitzungen stammen vom selben Tag"
                + (" mit identischem Aufbau" if aufbau_gleich else "")
                + " -\ndas Ergebnis ist deshalb optimistisch. Der ehrliche Test kommt"
                " aus\neiner spaeter aufgenommenen Sitzung.")
    tage = (d2 - d1).days
    text = (f"Die Pruefsitzung wurde {abs(tage)} Tag{'e' if abs(tage) != 1 else ''} "
            f"{'nach' if tage > 0 else 'vor'} der Anlernsitzung aufgenommen.")
    if not aufbau_gleich:
        text += "\nTastatur oder Mikrofonposition sind laut Sitzung verschieden."
    return text


def main() -> int:
    # Der Aufruf-Block aus dem Docstring erscheint unter --help als Beispiel.
    p = argparse.ArgumentParser(
        description="Trennbarkeit der Klassen pruefen",
        epilog=__doc__[__doc__.index("Aufruf:"):],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default=None,
                   help="Sitzung zum Anlernen, sonst die groesste mit Rolle train "
                        "(ersatzweise offen)")
    p.add_argument("--test", default=None,
                   help="Sitzung zum Pruefen, sonst eine mit Rolle val (ersatzweise "
                        "offen); die Testsitzung nur, wenn hier ausdruecklich genannt")
    p.add_argument("--bild", action="store_true",
                   help="Klassenvergleich und Konfusionsmatrix nach ausgabe/ schreiben")
    args = p.parse_args()
    laden_oder_beenden()   # setzt die gewaehlten Klassen

    alle = storage.sitzungen()
    if len(alle) < 1:
        print("Keine Sitzungen gefunden.")
        return 1

    for name in (args.train, args.test):
        if name and not (ROH / name / "session.json").exists():
            print(f"{ROT}Sitzung {name} nicht gefunden.{AUS}")
            return 1

    # Fuer die automatische Wahl kommen nur Sitzungen mit Proben und den
    # eingestellten Klassen in Frage. Abgebrochene, leere Sitzungen fallen
    # still heraus, fremde Klassen werden genannt.
    info: dict[str, tuple[dict, int]] = {}
    fremde: list[str] = []
    for o in alle:
        kopf, proben = storage.lade_sitzung(o)
        if not proben:
            continue
        try:
            datensatz.pruefe_passend(kopf)
            datensatz.pruefe_labels(kopf, proben)
        except ValueError:
            fremde.append(o.name)
            continue
        info[o.name] = (kopf, len(proben))
    if fremde and not (args.train and args.test):
        print(f"{GRAU}Uebergangen (andere Klassen): {', '.join(fremde)}{AUS}")
    if not info and not args.train:
        print("Es gibt noch keine Sitzung mit Proben"
              + (" und den eingestellten Klassen." if fremde else "."))
        return 1

    def waehle(rollen: tuple[str, ...], ausser: set, mindestens: int = 1,
               zu: tuple | None = None, groesste: bool = False) -> str | None:
        # Rollen der Reihe nach: erst die passende Rolle, dann "offen". Die
        # Testsitzung steht nie in dieser Liste - sie soll unberuehrt bleiben.
        for rolle in rollen:
            namen = [n for n, (k, g) in info.items()
                     if k.get("rolle", "offen") == rolle and n not in ausser
                     and g >= mindestens and (zu is None or schnitt(k) == zu)]
            if namen:
                return max(namen, key=lambda n: info[n][1]) if groesste else namen[-1]
        return None

    train_name = args.train or waehle(("train", "offen"), {args.test}, groesste=True)
    if train_name is None:
        if args.test and set(info) <= {args.test}:
            print(f"{ROT}Zum Anlernen braucht es eine andere Sitzung als "
                  f"{args.test}.{AUS}")
        else:
            print(f"{ROT}Keine Sitzung zum Anlernen gefunden (Rolle train oder offen).")
            print(f"Mit --train ausdruecklich angeben.{AUS}")
        return 1
    kopf_tr = storage.lade_sitzung(ROH / train_name)[0]
    test_name = args.test
    if not test_name:
        test_name = waehle(("val", "offen"), {train_name}, len(TASTEN), schnitt(kopf_tr))
        andere_rate = [n for n, (k, g) in info.items()
                       if n != train_name and k.get("rolle", "offen") in ("val", "offen")
                       and g >= len(TASTEN) and schnitt(k) != schnitt(kopf_tr)]
        if andere_rate and test_name is None:
            print(f"{GRAU}Uebergangen (andere Abtastrate oder anderer Schnitt): "
                  f"{', '.join(andere_rate)}{AUS}")
    if test_name == train_name:
        print(f"{ROT}Anlernen und Pruefen an derselben Sitzung ergibt keine "
              f"Aussage.{AUS}")
        return 1

    cfg, y_train, b_train, _ = lade(ROH / train_name)
    print(f"Anlernen an {train_name}  ({len(y_train)} Proben)")
    if kopf_tr.get("rolle") == "test":
        # Automatisch wird die Testsitzung nie gewaehlt - nur wer sie selbst
        # nennt, landet hier.
        print(f"{GELB}{train_name} ist die Testsitzung. Wer an ihr anlernt, hat die "
              f"unberuehrte\nEndpruefung verbraucht.{AUS}")
    if test_name:
        _, y_test, b_test, _ = lade(ROH / test_name)
        kopf_te = storage.lade_sitzung(ROH / test_name)[0]
        print(f"Pruefen an   {test_name}  ({len(y_test)} Proben)")
        if b_train.shape[1:] != b_test.shape[1:]:
            # Sonst endet der Vergleich in einem Broadcast-Fehler tief in numpy.
            print(f"{ROT}Die Sitzungen passen nicht zusammen (Spektrogramme "
                  f"{'x'.join(map(str, b_train.shape[1:]))} und "
                  f"{'x'.join(map(str, b_test.shape[1:]))}).")
            print(f"Meist wurde zwischendurch ein Mikrofon mit anderer Abtastrate "
                  f"gewaehlt.{AUS}")
            return 1
        if kopf_te.get("rolle") == "test":
            print(f"{GELB}{test_name} ist die Testsitzung. Wer hier schon hinschaut, "
                  f"hat die unberuehrte\nEndpruefung verbraucht.{AUS}")
        print(f"{GRAU}{abstand_der_aufnahmen(kopf_tr, kopf_te)}{AUS}")
    else:
        # Je Klasse ein Viertel zurueckhalten, mindestens eine Probe. Ein
        # freier Zufallsschnitt liesse Klassen ohne Anlernprobe zurueck, die
        # nie getroffen werden koennen - das Urteil waere dann falsch.
        zu_wenig = [t for t in TASTEN if y_train.count(t) < 2]
        if zu_wenig:
            print(f"{ROT}Zu wenige Proben fuer eine Trennbarkeitspruefung: mindestens "
                  f"2 je Klasse noetig,\nfehlt bei: "
                  f"{' '.join(anzeige(t) for t in zu_wenig)}{AUS}")
            return 1
        print(f"{GELB}Keine passende zweite Sitzung - geprueft wird gegen zurueckgehaltene")
        print(f"Proben derselben Sitzung. Das ist noch optimistischer.{AUS}")
        rng = np.random.default_rng(0)
        idx_train, idx_test = [], []
        for t in TASTEN:
            idx = rng.permutation([i for i, y in enumerate(y_train) if y == t])
            n_test = max(1, len(idx) // 4)
            idx_test += idx[:n_test].tolist()
            idx_train += idx[n_test:].tolist()
        b_test = b_train[idx_test]
        y_test = [y_train[i] for i in idx_test]
        b_train = b_train[idx_train]
        y_train = [y_train[i] for i in idx_train]

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
        anteil = ueber_zufall(quote)
        farbe = GRUEN if anteil >= 0.5 else (GELB if anteil >= 0.15 else ROT)
        print(f"\n{name}")
        print(f"  Trefferquote  {farbe}{quote * 100:5.1f} %{AUS}   "
              f"({quote / zufall():.1f}-fach ueber Zufall)")
        # Nur Klassen, die in der Pruefung ueberhaupt vorkamen - eine Klasse
        # ohne Probe ist nicht "am schwersten", sie ist gar nicht gemessen.
        zeilen = matrix.sum(axis=1)
        gemessen = [i for i in range(len(TASTEN)) if zeilen[i] > 0]
        je_klasse = {i: matrix[i, i] / zeilen[i] for i in gemessen}
        beste = max(je_klasse, key=je_klasse.get)
        schlechteste = min(je_klasse, key=je_klasse.get)
        print(f"  am besten     {anzeige(TASTEN[beste])}  {je_klasse[beste] * 100:.0f} %")
        print(f"  am schwersten {anzeige(TASTEN[schlechteste])}  "
              f"{je_klasse[schlechteste] * 100:.0f} %")
        if len(gemessen) < len(TASTEN):
            print(f"  {GRAU}ohne Pruefprobe: "
                  f"{' '.join(anzeige(TASTEN[i]) for i in range(len(TASTEN)) if i not in je_klasse)}"
                  f"{AUS}")

    beste_anteil = max(ueber_zufall(q) for q, _ in ergebnisse.values())
    print("\n" + "-" * 60)
    if beste_anteil >= 0.5:
        print(f"{GRUEN}Die Klassen sind deutlich unterscheidbar. Ein kleines CNN")
        print(f"sollte hier klar besser werden als dieser einfache Ansatz.{AUS}")
    elif beste_anteil >= 0.15:
        print(f"{GELB}Es steckt Struktur in den Daten, aber sie ist nicht eindeutig.")
        print(f"Ein CNN ist einen Versuch wert - mit ehrlicher Erwartung.{AUS}")
    else:
        print(f"{ROT}Kaum Struktur zu finden. Bevor ein Netz trainiert wird, lieber")
        print(f"Aufnahme pruefen: Mikrofonposition, Anschlagsstaerke, Raumhall.{AUS}")

    if args.bild:
        mittel = {t: b_train[[i for i, y in enumerate(y_train) if y == t]].mean(axis=0)
                  for t in TASTEN if t in set(y_train)}
        fig = plots.klassen_vergleich(mittel, cfg)
        print(f"\n  {portrait.exportiere(fig, '04_klassenvergleich', '03_daten')}")
        quote, matrix = ergebnisse["ohne Lautstaerke"]
        fig = plots.konfusionsmatrix(matrix, quote)
        print(f"  {portrait.exportiere(fig, '08_confusion_einfach', '03_daten')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

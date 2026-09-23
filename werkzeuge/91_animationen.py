"""Animierte Video-Grafiken im Hochformat rendern oder live zeigen.

Solange keine Messdaten vorliegen, laufen die Szenen mit Beispieldaten und
tragen den Hinweis "BEISPIELDATEN". Dieselben Szenen zeichnen spaeter die
echten Ergebnisse.

Aufruf:
    python werkzeuge/91_animationen.py --liste
    python werkzeuge/91_animationen.py                      # alle als MP4
    python werkzeuge/91_animationen.py --nur pipeline
    python werkzeuge/91_animationen.py --nur pipeline --zeigen   # Live-Fenster
    python werkzeuge/91_animationen.py --frames             # zusaetzlich PNG-Folge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

import numpy as np

import json

from tastenakustik import (animation, beispiel, features, onset, plots,
                           storage, theme)
from tastenakustik.animation import Szene, phase, weich
from tastenakustik.config import (DATEN, Config, TASTEN, AUSGABE,
                                  setze_klassen, zufall)

rng = np.random.default_rng(20260922)
CFG = Config.laden()


# --- Echte Messdaten -------------------------------------------------------
def lade_ergebnisse() -> dict | None:
    """Trainingsverlauf und Confusion Matrix des letzten Laufs."""
    dateien = sorted((DATEN / "modelle").glob("verlauf_*.json"))
    if not dateien:
        return None
    return json.loads(dateien[-1].read_text(encoding="utf-8"))


def lade_modell():
    """Das zuletzt trainierte Netz, falls vorhanden."""
    dateien = sorted((DATEN / "modelle").glob("cnn_*.pt"))
    if not dateien:
        return None, None
    import torch
    from tastenakustik import modell
    stand = torch.load(dateien[-1], map_location="cpu", weights_only=False)
    if stand.get("klassen"):
        setze_klassen(stand["klassen"])
    netz = modell.KleinesCNN(len(TASTEN))
    netz.load_state_dict(stand["state_dict"])
    netz.eval()
    return netz, stand


def lade_proben(rolle: str = "val") -> list[tuple[str, np.ndarray, int, Config]]:
    """Aufgenommene Fenster einer Rolle, mit Onset und Aufnahmeparametern."""
    import soundfile as sf
    gefunden = []
    for ordner in storage.sitzungen():
        kopf, proben = storage.lade_sitzung(ordner)
        if kopf.get("rolle") != rolle:
            continue
        cfg = Config(**{k: v for k, v in kopf["aufnahmeparameter"].items()
                        if k in Config.__dataclass_fields__})
        for eintrag in proben:
            welle, _ = sf.read(ordner / eintrag["datei"], dtype="float32")
            gefunden.append((eintrag["label"], welle, eintrag["onset_sample"], cfg))
    return gefunden


def lade_test() -> dict | None:
    """Das Ergebnis auf der ungesehenen Testsitzung, falls ausgewertet."""
    pfad = DATEN / "modelle" / "test_ergebnis.json"
    return json.loads(pfad.read_text(encoding="utf-8")) if pfad.exists() else None


ERGEBNIS = lade_ergebnisse()
TEST = lade_test()
PROBEN = lade_proben("test") or lade_proben("val") or lade_proben("train")

# Zeichenfolge fuer die Blindtest-Szene: die Sperrfolge, falls eine gesetzt
# ist - sonst eine kurze Folge aus den vorhandenen Klassen.
SEQUENZ = (CFG.sperrfolge if CFG.sperrfolge and
           all(c in TASTEN for c in CFG.sperrfolge)
           else beispiel.beispiel_sequenz(8))


def _echte_wahrscheinlichkeiten(taste: str, nummer: int | None = None):
    """Eine echte Vorhersage des trainierten Netzes holen.

    Mit `nummer` wird deterministisch ausgewaehlt statt gewuerfelt. Fuer die
    Sequenz ist das wichtig: Bei 88 % je Zeichen ist ein fehlerfreier
    Dreizehnerlauf zu rund 19 % zu erwarten - wer so lange wuerfelt, bis er
    einen erwischt, zeigt nicht das Modell, sondern seine Geduld.
    """
    netz, stand = lade_modell()
    if netz is None or not PROBEN:
        return None, None, None
    import torch
    from tastenakustik import datensatz as ds
    passende = [p for p in PROBEN if p[0] == taste] or PROBEN
    i = (nummer % len(passende)) if nummer is not None else int(rng.integers(len(passende)))
    label, welle, onset, cfg = passende[i]
    n = int(stand["segment_ms"] / 1000 * cfg.samplerate)
    vor = int(stand["vor_ms"] / 1000 * cfg.samplerate)
    start = max(0, min(onset - vor, welle.size - n))
    segment = welle[start:start + n]
    mel = features.log_mel(segment, cfg.samplerate, cfg.mel_nfft, cfg.mel_hop,
                           cfg.mel_baender, cfg.mel_fmin, cfg.mel_fmax).T
    x = ds.normiere(mel[None].astype(np.float32))
    with torch.no_grad():
        p = torch.softmax(netz(torch.from_numpy(x).unsqueeze(1))[0], 0).numpy()
    return label, {t: float(v) for t, v in zip(TASTEN, p)}, mel


def _fenster(taste: str) -> tuple[np.ndarray, int, bool]:
    """Echtes aufgenommenes Fenster, sonst ein synthetisches."""
    passende = [p for p in PROBEN if p[0] == taste]
    if passende:
        _l, welle, onset, _c = passende[rng.integers(len(passende))]
        return welle, onset, True
    x, i = beispiel.fenster(CFG, taste, rng=rng)
    return x, i, False


def _wahrscheinlichkeiten(ziel: str, sicherheit: float) -> dict[str, float]:
    return beispiel.wahrscheinlichkeiten(ziel, sicherheit, rng)


def _stempeln(fig, mit: bool):
    if mit:
        plots.vorschau_stempel(fig)
    return fig


# --- Szenen ---------------------------------------------------------------
def szene_pipeline(stempel: bool) -> Szene:
    """Die sechs Schritte blenden nacheinander auf."""
    n = len(plots.PIPELINE_SCHRITTE)

    def bild(t: float):
        # Pro Schritt ein gleich langer Abschnitt, jeweils weich eingeblendet.
        fortschritt = t * n
        bis = min(int(fortschritt), n - 1)
        anteil = weich(fortschritt - bis)
        return _stempeln(plots.pipeline(CFG, bis_schritt=bis, anteil=anteil), stempel)

    return Szene("05_pipeline", 7.0, bild, "Pipeline baut sich Schritt fuer Schritt auf")


def szene_modell(stempel: bool) -> Szene:
    """Bild rein, Netz rechnet, eine Zahl je Klasse raus."""
    label, p, mel = _echte_wahrscheinlichkeiten(TASTEN[0])
    echt = p is not None
    if not echt:
        p, mel = _wahrscheinlichkeiten(TASTEN[0], 0.90), None

    def bild(t: float):
        return _stempeln(
            plots.modell_erklaerung(p, mel_bild=mel, cfg=CFG if mel is not None else None,
                                    anteil=weich(t)), stempel and not echt)

    quelle = "echte Vorhersage" if echt else "Beispieldaten"
    return Szene("05b_modell", 6.0, bild, f"Was das Netz macht ({quelle})")


def szene_training(stempel: bool) -> Szene:
    """Die Kurven zeichnen sich Epoche fuer Epoche."""
    echt = ERGEBNIS is not None
    if echt:
        v = ERGEBNIS["verlauf"]
        train_loss, val_loss = np.array(v["train_loss"]), np.array(v["val_loss"])
        train_acc, val_acc = np.array(v["train_acc"]), np.array(v["val_acc"])
    else:
        e0 = np.arange(1, 31)
        train_loss = 2.08 * np.exp(-e0 / 7.5) + 0.17 + rng.normal(0, 0.02, 30)
        val_loss = 2.08 * np.exp(-e0 / 8.5) + 0.46 + rng.normal(0, 0.05, 30)
        train_acc = np.clip(1 - 0.86 * np.exp(-e0 / 7.0) + rng.normal(0, .008, 30), 0, 1)
        val_acc = np.clip(1 - 0.94 * np.exp(-e0 / 8.5) + rng.normal(0, .02, 30), 0, 1)
    e = np.arange(1, len(train_loss) + 1)

    def bild(t: float):
        return _stempeln(
            plots.trainingsverlauf(e, train_loss, val_loss, train_acc, val_acc,
                                   anteil=max(t, 1 / len(e))), stempel and not echt)

    quelle = "echter Verlauf" if echt else "Beispielverlauf"
    return Szene("06_training", 7.0, bild, f"Trainingsverlauf zeichnet sich ({quelle})")


def szene_confusion(stempel: bool) -> Szene:
    """Die Matrix fuellt sich Zeile fuer Zeile."""
    # Der Test ist die ehrliche Zahl - er hat Vorrang vor der Validierung.
    echt = TEST is not None or (ERGEBNIS is not None and "konfusion" in ERGEBNIS)
    if TEST is not None:
        matrix = np.array(TEST["konfusion"], dtype=float)
    elif echt:
        matrix = np.array(ERGEBNIS["konfusion"], dtype=float)
    else:
        n = len(TASTEN)
        matrix = np.zeros((n, n))
        for i in range(n):
            richtig = rng.integers(28, 38)
            matrix[i, i] = richtig
            for _ in range(40 - richtig):
                matrix[i, rng.choice([j for j in range(n) if j != i])] += 1
    genauigkeit = float(np.trace(matrix) / max(matrix.sum(), 1))

    def bild(t: float):
        return _stempeln(plots.konfusionsmatrix(matrix, genauigkeit, anteil=weich(t)),
                         stempel and not echt)

    quelle = ("Testsitzung" if TEST is not None
              else ("Validierung" if echt else "Beispieldaten"))
    return Szene("08_confusion", 5.0, bild, f"Confusion Matrix baut sich auf ({quelle})")


def szene_vorhersage(stempel: bool) -> Szene:
    """Balken laufen hoch, dann kommt das Ergebnis."""
    label, p, _mel = _echte_wahrscheinlichkeiten(TASTEN[0])
    echt = p is not None
    if not echt:
        label, p = TASTEN[0], _wahrscheinlichkeiten(TASTEN[0], 0.91)

    def bild(t: float):
        return _stempeln(plots.vorhersage(label, p, anteil=weich(t)),
                         stempel and not echt)

    quelle = "echte Vorhersage" if echt else "Beispieldaten"
    return Szene("09_vorhersage", 4.5, bild, f"Eine Vorhersage entsteht ({quelle})")


def szene_vergleich(stempel: bool) -> Szene:
    """Alle gemessenen Trefferquoten nebeneinander - die ehrlichste Grafik.

    Aufgenommen wird nur, was tatsaechlich gemessen wurde. Wer weitere Stufen
    zeigen will - etwa eine Trennbarkeit ohne Lernen aus 06_trennbarkeit.py
    oder einen Durchgang mit fluessigem Tippen -, traegt sie hier ein. Nichts
    davon wird geraten.
    """
    echt = TEST is not None
    val = TEST["val_quote"] if echt else (
        ERGEBNIS["beste_val"] if ERGEBNIS else 0.99)
    test = TEST["test_quote"] if echt else 0.88
    stufen = [
        (f"Zufall bei {len(TASTEN)} Klassen", zufall(), theme.TEXT_SCHWACH),
        ("Validierung, gleicher Tag", val, theme.AKZENT2),
    ]
    if echt:
        stufen.append(("Test, ungesehene Sitzung", test, theme.OK))

    def bild(t: float):
        hero = ("Test auf ungesehenen Daten", test) if echt else None
        return _stempeln(
            plots.ergebnis_vergleich(stufen, hero=hero, anteil=weich(t)),
            stempel and not echt)

    quelle = "gemessen" if echt else "Beispieldaten"
    return Szene("11_vergleich", 6.0, bild, f"Alle Trefferquoten im Vergleich ({quelle})")


def szene_sequenz(stempel: bool) -> Szene:
    """Der Blindtest: jedes Zeichen wird einzeln erkannt und angehaengt."""
    zeichen = list(SEQUENZ)
    daten, echt = [], True
    for nr, c in enumerate(zeichen):
        _label, p, _mel = _echte_wahrscheinlichkeiten(c, nummer=nr)
        if p is None:
            echt = False
            p = _wahrscheinlichkeiten(c, float(rng.uniform(0.72, 0.95)))
        daten.append((c, p))

    def bild(t: float):
        i = min(int(t * len(zeichen)), len(zeichen) - 1)
        lokal = weich((t * len(zeichen)) - i)
        ziel, p = daten[i]
        # Die Zeichenfolge zeigt, was das Modell sagt - nicht, was getippt
        # wurde. Fehler bleiben sichtbar.
        gesagt = [max(q, key=q.get) for _z, q in daten[:i + 1]]
        bisher = "".join(gesagt) if lokal > 0.66 else "".join(gesagt[:-1])
        return _stempeln(
            plots.vorhersage(ziel, p, sequenz=bisher,
                             titel=f"Blindtest  {i + 1}/{len(zeichen)}",
                             anteil=lokal), stempel and not echt)

    treffer = sum(1 for z, q in daten if max(q, key=q.get) == z)
    quelle = (f"echte Modellausgaben, {treffer}/{len(zeichen)} richtig"
              if echt else "Beispieldaten")
    return Szene("10_sequenz", 13.0, bild,
                 f"{SEQUENZ} Zeichen fuer Zeichen ({quelle})")


def szene_tastenschlag(stempel: bool) -> Szene:
    """Das Fenster laeuft ein, dann wird der Onset markiert."""
    w, _onset_pos, echt = _fenster(TASTEN[0])
    a = onset.analysiere(w, CFG)

    def bild(t: float):
        sichtbar = int(max(phase(t, 0.0, 0.55), 0.02) * w.size)
        teil = np.zeros_like(w)
        teil[:sichtbar] = w[:sichtbar]
        marke = phase(t, 0.55, 0.25)
        leer = onset.Anschlag(False, 0, 0, a.peak_dbfs, a.rausch_dbfs, a.snr_db,
                              False, False, "ok")
        return _stempeln(
            plots.wellenform_mit_onset(teil, TASTEN[0], a if marke > 0.5 else leer, CFG),
            stempel and not echt)

    quelle = "echte Aufnahme" if echt else "Beispieldaten"
    return Szene("02_tastenschlag", 5.0, bild,
                 f"Anschlag laeuft ein und wird markiert ({quelle})")


SZENEN = {
    "tastenschlag": szene_tastenschlag,
    "pipeline": szene_pipeline,
    "modell": szene_modell,
    "training": szene_training,
    "confusion": szene_confusion,
    "vorhersage": szene_vorhersage,
    "sequenz": szene_sequenz,
    "vergleich": szene_vergleich,
}


def main() -> int:
    p = argparse.ArgumentParser(description="Animierte Video-Grafiken")
    p.add_argument("--liste", action="store_true", help="Szenen auflisten")
    p.add_argument("--nur", default=None, help="nur diese Szene")
    p.add_argument("--zeigen", action="store_true", help="Live-Fenster statt Datei")
    p.add_argument("--frames", action="store_true", help="PNG-Bildfolge behalten")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--echt", action="store_true",
                   help="ohne Hinweis BEISPIELDATEN (nur mit echten Zahlen benutzen)")
    args = p.parse_args()

    if args.liste:
        print(f"{'Szene':<16} {'Dauer':>6}   Inhalt")
        print("-" * 68)
        for name, bauen in SZENEN.items():
            s = bauen(True)
            print(f"{name:<16} {s.dauer:>5.1f}s   {s.beschreibung}")
        return 0

    if args.nur and args.nur not in SZENEN:
        print(f"Unbekannte Szene {args.nur!r}. Bekannt: {', '.join(SZENEN)}")
        return 1

    namen = [args.nur] if args.nur else list(SZENEN)
    stempel = not args.echt

    if args.zeigen:
        if not args.nur:
            print("--zeigen braucht --nur <szene>")
            return 1
        szene = SZENEN[args.nur](stempel)
        print(f"{szene.name}: {szene.dauer:.1f}s in Schleife. Fenster schliessen beendet.")
        animation.zeige(szene, args.fps)
        return 0

    if animation._ffmpeg() is None:
        print("ffmpeg nicht gefunden - es werden nur Bildfolgen geschrieben.")

    for name in namen:
        szene = SZENEN[name](stempel)
        n = szene.frames(args.fps)
        print(f"{szene.name}  {szene.dauer:.1f}s  {n} Bilder ...", flush=True)
        try:
            pfad = animation.ablegen(szene, fps=args.fps, frames=args.frames)
            print(f"   {pfad.relative_to(AUSGABE.parent)}")
        except RuntimeError:
            ordner = AUSGABE / "_animation" / f"{szene.name}_frames"
            animation.rendere_bildfolge(szene, ordner, args.fps)
            print(f"   {ordner.relative_to(AUSGABE.parent)}  ({n} PNG)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

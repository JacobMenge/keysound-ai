"""Animierte Video-Grafiken im Hochformat rendern oder live zeigen.

Solange keine Messdaten vorliegen, laufen die Szenen mit Beispieldaten und
tragen den Hinweis "BEISPIELDATEN". Dieselben Szenen zeichnen spaeter die
echten Ergebnisse. Test- und Verlaufszahlen gelten nur, wenn sie zum neuesten
Modell gehoeren. Laufen die Vorhersage-Szenen mangels Testsitzung auf
Validierungs- oder Trainingsdaten, steht das auf dem Bild - auch mit --echt.

Aufruf:
    python werkzeuge/91_animationen.py --liste
    python werkzeuge/91_animationen.py                      # alle als MP4
    python werkzeuge/91_animationen.py --nur pipeline
    python werkzeuge/91_animationen.py --nur pipeline --zeigen   # Live-Fenster:
                                        # erst rendern, dann Echtzeit
    python werkzeuge/91_animationen.py --frames             # zusaetzlich PNG-Folge
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
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
                                  setze_klassen, zufall,
    laden_oder_beenden,
)

rng = np.random.default_rng(20260922)
CFG = laden_oder_beenden()


# --- Echte Messdaten -------------------------------------------------------
def _neuestes_modell() -> Path | None:
    dateien = sorted((DATEN / "modelle").glob("cnn_*.pt"))
    return dateien[-1] if dateien else None


_MODELL: dict = {}


def lade_modell():
    """Das zuletzt trainierte Netz, falls vorhanden - einmal geladen, dann gemerkt.

    Kennt das Modell andere Klassen als die Config, gelten fuer alle Szenen
    die Klassen des Modells: Seine Vorhersagen, Verlaeufe und Matrizen gehoeren
    zu diesen Klassen. Das passiert einmal beim Start, bevor irgendeine Szene
    gebaut wird - sonst wechselten mitten in der Serie Farben und Zufallsquote.
    """
    if not _MODELL:
        pfad = _neuestes_modell()
        netz = stand = None
        if pfad is not None:
            import torch
            from tastenakustik import modell
            stand = torch.load(pfad, map_location="cpu", weights_only=False)
            klassen = list(stand.get("klassen") or TASTEN)
            if klassen != list(TASTEN):
                print(f"Hinweis: Das Modell {pfad.name} kennt die Klassen "
                      f"{''.join(klassen)!r}, eingestellt sind {''.join(TASTEN)!r}. "
                      "Die Szenen zeigen die Klassen des Modells.")
                setze_klassen(klassen)
            netz = modell.KleinesCNN(len(klassen))
            netz.load_state_dict(stand["state_dict"])
            netz.eval()
        _MODELL.update(netz=netz, stand=stand, pfad=pfad)
    return _MODELL["netz"], _MODELL["stand"]


def _passt_zu_klassen(daten: dict) -> bool:
    """Gehoert eine gespeicherte Auswertung zu den aktuellen Klassen?

    Aeltere Dateien haben keine Klassenliste - dann bleibt nur die Form der
    Matrix als Pruefung.
    """
    if daten.get("klassen") and list(daten["klassen"]) != list(TASTEN):
        return False
    matrix = daten.get("konfusion")
    return matrix is None or np.shape(matrix) == (len(TASTEN), len(TASTEN))


def lade_ergebnisse() -> dict | None:
    """Trainingsverlauf und Confusion Matrix zum neuesten Modell.

    Verlauf und Modell tragen denselben Zeitstempel im Namen. Gibt es ein
    Modell, zaehlt nur sein eigener Verlauf - ein anderer zeigte die Kurven
    eines anderen Netzes. Ohne Modell gilt der neueste Verlauf, aber nur, wenn
    seine Klassen passen.
    """
    ordner = DATEN / "modelle"
    pfad = _MODELL.get("pfad")
    if pfad is not None:
        datei = ordner / f"verlauf_{pfad.stem[4:]}.json"
        if not datei.exists():
            print(f"Hinweis: Zu {pfad.name} gibt es keinen Verlauf - "
                  "Verlauf und Matrix laufen mit Beispieldaten.")
            return None
    else:
        dateien = sorted(ordner.glob("verlauf_*.json"))
        if not dateien:
            return None
        datei = dateien[-1]
    daten = json.loads(datei.read_text(encoding="utf-8"))
    if not _passt_zu_klassen(daten):
        print(f"Hinweis: {datei.name} gehoert zu anderen Klassen - "
              "Verlauf und Matrix laufen mit Beispieldaten.")
        return None
    return daten


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
    """Das Ergebnis auf der ungesehenen Testsitzung - nur zum neuesten Modell.

    test_ergebnis.json wird durch ein neues Training nicht ungueltig. Ohne
    diese Pruefung stuenden die Zahlen eines alten Modells als "gemessen"
    neben Verlauf und Vorhersagen des neuen. Eine Datei ohne Modellnamen
    (aus einer aelteren Version) laesst sich nicht zuordnen und zaehlt
    deshalb nicht - einmal "07_training.py --test" schreibt sie neu.
    """
    pfad = DATEN / "modelle" / "test_ergebnis.json"
    if not pfad.exists():
        return None
    daten = json.loads(pfad.read_text(encoding="utf-8"))
    modell_pfad = _MODELL.get("pfad")
    if (modell_pfad is None or daten.get("modell") != modell_pfad.name
            or not _passt_zu_klassen(daten)):
        print("Hinweis: test_ergebnis.json gehoert nicht erkennbar zum neuesten "
              "Modell - die Testzahlen bleiben weg. Mit "
              "'python werkzeuge/07_training.py --test' neu auswerten.")
        return None
    return daten


def _lade_proben_mit_rolle() -> tuple[list, str | None]:
    """Proben fuer die Vorhersage-Szenen und aus welcher Rolle sie stammen.

    Die Testsitzung hat Vorrang - nur sie ist ein Blindtest. Fehlt sie, laufen
    die Szenen mit Validierungs- oder Trainingsdaten, aber gekennzeichnet.
    """
    for rolle in ("test", "val", "train"):
        proben = lade_proben(rolle)
        if proben:
            return proben, rolle
    return [], None


lade_modell()
ERGEBNIS = lade_ergebnisse()
TEST = lade_test()
PROBEN, ROLLE_PROBEN = _lade_proben_mit_rolle()
ROLLEN_NAME = {"val": "VALIDIERUNGSDATEN", "train": "TRAININGSDATEN"}

# Zeichenfolge fuer die Blindtest-Szene: die Sperrfolge, falls eine gesetzt
# ist - sonst eine kurze Folge aus den vorhandenen Klassen.
SEQUENZ = (CFG.sperrfolge if CFG.sperrfolge and
           all(c in TASTEN for c in CFG.sperrfolge)
           else beispiel.beispiel_sequenz(8))


def _echte_wahrscheinlichkeiten(taste: str, nummer: int | None = None,
                                ersatz: bool = False):
    """Eine echte Vorhersage des trainierten Netzes holen.

    Mit `nummer` wird deterministisch ausgewaehlt statt gewuerfelt. Fuer die
    Sequenz ist das wichtig: Bei 88 % je Zeichen ist ein fehlerfreier
    Dreizehnerlauf zu rund 19 % zu erwarten - wer so lange wuerfelt, bis er
    einen erwischt, zeigt nicht das Modell, sondern seine Geduld.

    Gibt es keine Aufnahme dieser Taste, kommt (None, None, None) zurueck -
    ausser mit `ersatz`, dann darf es eine beliebige Taste sein. Das zurueck-
    gegebene Label ist immer das der tatsaechlich verwendeten Aufnahme.
    """
    netz, stand = lade_modell()
    if netz is None or not PROBEN:
        return None, None, None
    import torch
    from tastenakustik import datensatz as ds
    # Nur Aufnahmen mit der Abtastrate des Trainings - eine andere ergaebe
    # ein anders langes Spektrogramm, und das Netz saehe etwas Fremdes.
    # Aeltere Modelle nennen keine Abtastrate, dann zaehlt jede Aufnahme.
    sr_modell = stand.get("samplerate")
    brauchbar = [p for p in PROBEN if sr_modell is None or p[3].samplerate == sr_modell]
    passende = [p for p in brauchbar if p[0] == taste] or (brauchbar if ersatz else [])
    if not passende:
        return None, None, None
    i = (nummer % len(passende)) if nummer is not None else int(rng.integers(len(passende)))
    label, welle, onset, cfg = passende[i]
    n = int(stand["segment_ms"] / 1000 * cfg.samplerate)
    vor = int(stand["vor_ms"] / 1000 * cfg.samplerate)
    start = max(0, min(onset - vor, welle.size - n))
    segment = welle[start:start + n]
    # Mel-Parameter wie im Training - das Modell nennt sie, aeltere nicht;
    # dann gelten die der Aufnahme, mit denen damals auch trainiert wurde.
    # Die Bandzahl nennen auch die aelteren Modelle schon (mel_baender) - die
    # Form der Eingabe muss zum Netz passen, sie hat deshalb Vorrang.
    m = stand.get("mel") or {}
    baender = m.get("baender", stand.get("mel_baender", cfg.mel_baender))
    mel = features.log_mel(segment, cfg.samplerate, m.get("nfft", cfg.mel_nfft),
                           m.get("hop", cfg.mel_hop), baender,
                           m.get("fmin", cfg.mel_fmin), m.get("fmax", cfg.mel_fmax)).T
    x = ds.normiere(mel[None].astype(np.float32))
    with torch.no_grad():
        p = torch.softmax(netz(torch.from_numpy(x).unsqueeze(1))[0], 0).numpy()
    # Das Netz bekommt (Baender, Zeit), die Grafik erwartet (Zeit, Baender).
    return label, {t: float(v) for t, v in zip(TASTEN, p)}, mel.T


def _fenster(taste: str) -> tuple[np.ndarray, int, bool, Config]:
    """Echtes aufgenommenes Fenster, sonst ein synthetisches.

    Mit dabei sind die Aufnahmeparameter, zu denen das Fenster gehoert -
    Abtastrate und Pre-Roll der Sitzung, nicht die der aktuellen Config.
    """
    passende = [p for p in PROBEN if p[0] == taste]
    if passende:
        _l, welle, onset, cfg = passende[rng.integers(len(passende))]
        return welle, onset, True, cfg
    x, i = beispiel.fenster(CFG, taste, rng=rng)
    return x, i, False, CFG


def _modell_schnitt(cfg: Config) -> Config:
    """cfg mit dem Schnitt des Modells statt dem der Config.

    Die markierte Flaeche und die ms-Angabe sollen zeigen, was das Netz hoert.
    Ohne Modell bleibt es beim Schnitt der Config.
    """
    _netz, stand = lade_modell()
    if stand is None:
        return cfg
    return replace(cfg, segment_ms=stand.get("segment_ms", cfg.segment_ms),
                   segment_vor_onset_ms=stand.get("vor_ms", cfg.segment_vor_onset_ms))


def _wahrscheinlichkeiten(ziel: str, sicherheit: float) -> dict[str, float]:
    return beispiel.wahrscheinlichkeiten(ziel, sicherheit, rng)


def _stempeln(fig, mit: bool):
    if mit:
        plots.vorschau_stempel(fig)
    return fig


def _kein_blindtest(fig):
    """Vorhersagen auf bekannten Daten kennzeichnen - auch mit --echt.

    Die Zahlen sind echte Modellausgaben, aber auf einer Sitzung, die das
    Modell beim Lernen oder bei der Auswahl gesehen hat. Als Blindtest
    betitelt waere das eine falsche Behauptung.
    """
    plots.vorschau_stempel(fig, text=f"KEIN BLINDTEST\n{ROLLEN_NAME[ROLLE_PROBEN]}")
    return fig


def _blindtest_titel() -> str:
    return "Blindtest" if ROLLE_PROBEN in (None, "test") else "Probelauf"


# --- Szenen ---------------------------------------------------------------
def szene_pipeline(stempel: bool) -> Szene:
    """Die sechs Schritte blenden nacheinander auf.

    Ohne Stempel, auch ohne --echt: Die Grafik ist schematisch und zeigt keine
    Messung - wie in 90_layoutvorschau. Die Segmentlaenge kommt aus dem
    Modell, falls eines da ist.
    """
    n = len(plots.PIPELINE_SCHRITTE)
    _netz, stand = lade_modell()
    segment_ms = stand.get("segment_ms") if stand is not None else None

    def bild(t: float):
        # Pro Schritt ein gleich langer Abschnitt, jeweils weich eingeblendet.
        fortschritt = t * n
        bis = min(int(fortschritt), n - 1)
        anteil = weich(fortschritt - bis)
        return plots.pipeline(CFG, bis_schritt=bis, anteil=anteil,
                              segment_ms=segment_ms)

    return Szene("05_pipeline", 7.0, bild, "Pipeline baut sich Schritt fuer Schritt auf")


def szene_modell(stempel: bool) -> Szene:
    """Bild rein, Netz rechnet, eine Zahl je Klasse raus."""
    label, p, mel = _echte_wahrscheinlichkeiten(TASTEN[0], ersatz=True)
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
    label, p, _mel = _echte_wahrscheinlichkeiten(TASTEN[0], ersatz=True)
    echt = p is not None
    if not echt:
        label, p = TASTEN[0], _wahrscheinlichkeiten(TASTEN[0], 0.91)
    bekannt = echt and ROLLE_PROBEN != "test"

    def bild(t: float):
        fig = plots.vorhersage(label, p, titel=_blindtest_titel() if echt else "Blindtest",
                               anteil=weich(t))
        return _kein_blindtest(fig) if bekannt else _stempeln(fig, stempel and not echt)

    quelle = (f"echte Vorhersage, {ROLLE_PROBEN}-Sitzung" if echt else "Beispieldaten")
    return Szene("09_vorhersage", 4.5, bild, f"Eine Vorhersage entsteht ({quelle})")


def szene_vergleich(stempel: bool) -> Szene:
    """Alle gemessenen Trefferquoten nebeneinander - die ehrlichste Grafik.

    Aufgenommen wird nur, was tatsaechlich gemessen wurde. Wer weitere Stufen
    zeigen will - etwa eine Trennbarkeit ohne Lernen aus 06_trennbarkeit.py
    oder einen Durchgang mit fluessigem Tippen -, traegt sie hier ein. Nichts
    davon wird geraten.
    """
    echt = TEST is not None
    val = (TEST.get("val_quote") if echt else None) or (
        ERGEBNIS["beste_val"] if ERGEBNIS else 0.99)
    test = TEST["test_quote"] if echt else 0.88
    # Die Validierung ist eine eigene Sitzung, die nie gelernt wurde - nur
    # vom selben Tag. "gleiche Sitzung" wuerde die Sitzungstrennung verleugnen.
    stufen = [
        (f"Zufall bei {len(TASTEN)} Klassen", zufall(), theme.TEXT_SCHWACH),
        ("Validierung, eigene Sitzung, gleicher Tag", val, theme.AKZENT2),
    ]
    if echt:
        stufen.append(("Test, ungesehene Sitzung", test, theme.OK))

    # Ohne gueltigen Test ist die Validierung aus dem Verlauf trotzdem eine
    # Messung - dann ohne Stempel, nur ohne Test-Balken.
    gemessen = echt or ERGEBNIS is not None

    def bild(t: float):
        hero = ("Test auf ungesehenen Daten", test) if echt else None
        return _stempeln(
            plots.ergebnis_vergleich(stufen, hero=hero, anteil=weich(t)),
            stempel and not gemessen)

    quelle = ("gemessen" if echt else
              "gemessen, ohne gueltiges Testergebnis" if gemessen else "Beispieldaten")
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

    bekannt = echt and ROLLE_PROBEN != "test"
    titel = _blindtest_titel() if echt else "Blindtest"

    def bild(t: float):
        i = min(int(t * len(zeichen)), len(zeichen) - 1)
        # Jedes Zeichen ist nach 60 % seines Abschnitts fertig und bleibt dann
        # stehen - erst bei anteil 1 zeigt vorhersage() das Urteil, auch ein
        # "falsch" mitten in der Folge.
        lokal = weich(min((t * len(zeichen) - i) / 0.6, 1.0))
        ziel, p = daten[i]
        # Die Zeichenfolge zeigt, was das Modell sagt - nicht, was getippt
        # wurde. Fehler bleiben sichtbar.
        gesagt = [max(q, key=q.get) for _z, q in daten[:i + 1]]
        bisher = "".join(gesagt) if lokal > 0.66 else "".join(gesagt[:-1])
        fig = plots.vorhersage(ziel, p, sequenz=bisher,
                               titel=f"{titel}  {i + 1}/{len(zeichen)}",
                               anteil=lokal)
        return _kein_blindtest(fig) if bekannt else _stempeln(fig, stempel and not echt)

    treffer = sum(1 for z, q in daten if max(q, key=q.get) == z)
    quelle = (f"echte Modellausgaben, {ROLLE_PROBEN}-Sitzung, "
              f"{treffer}/{len(zeichen)} richtig" if echt else "Beispieldaten")
    # Mindestens 13 s, bei einer langen Sperrfolge eine Sekunde je Zeichen.
    return Szene("10_sequenz", max(13.0, 1.0 * len(zeichen)), bild,
                 f"{SEQUENZ} Zeichen fuer Zeichen ({quelle})")


def szene_tastenschlag(stempel: bool) -> Szene:
    """Das Fenster laeuft ein, dann wird der Onset markiert."""
    w, _onset_pos, echt, cfg = _fenster(TASTEN[0])
    cfg = _modell_schnitt(cfg)
    a = onset.analysiere(w, cfg)
    # Skala aus dem ganzen Fenster, einmal - aus dem gerade sichtbaren Teil
    # berechnet, fuellte erst das Rauschen die Achse, dann sprang sie.
    grenze = float(np.max(np.abs(w))) * 1.25

    def bild(t: float):
        sichtbar = int(max(phase(t, 0.0, 0.55), 0.02) * w.size)
        teil = np.zeros_like(w)
        teil[:sichtbar] = w[:sichtbar]
        marke = phase(t, 0.55, 0.25)
        leer = onset.Anschlag(False, 0, 0, a.peak_dbfs, a.rausch_dbfs, a.snr_db,
                              False, False, "ok")
        return _stempeln(
            plots.wellenform_mit_onset(teil, TASTEN[0], a if marke > 0.5 else leer, cfg,
                                       y_grenze=grenze),
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


def _fps_wert(text: str) -> int:
    """Bildrate pruefen, bevor eine Szene gebaut wird - 0 oder negativ ergaebe
    still zwei Bilder, einen ffmpeg-Fehler oder eine Division durch null."""
    try:
        wert = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} ist keine ganze Zahl") from None
    if not 1 <= wert <= 120:
        raise argparse.ArgumentTypeError(f"{wert} liegt nicht zwischen 1 und 120")
    return wert


def main() -> int:
    p = argparse.ArgumentParser(description="Animierte Video-Grafiken")
    p.add_argument("--liste", action="store_true", help="Szenen auflisten")
    p.add_argument("--nur", default=None, help="nur diese Szene")
    p.add_argument("--zeigen", action="store_true",
                   help="Live-Fenster statt Datei (rendert erst alle Bilder, "
                        "spielt dann in Echtzeit)")
    p.add_argument("--frames", action="store_true", help="PNG-Bildfolge behalten")
    p.add_argument("--fps", type=_fps_wert, default=30, help="Bildrate, 1 bis 120")
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
        print(f"{szene.name}: {szene.frames(args.fps)} Bilder werden erst gerendert, "
              f"dann laeuft die Szene ({szene.dauer:.1f}s) in Echtzeit in Schleife. "
              "Fenster schliessen beendet.")
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
        except RuntimeError as fehler:
            # Fehlt ffmpeg ganz, stand das schon oben - sonst den Grund zeigen.
            if animation._ffmpeg() is not None:
                print(f"   {fehler}")
            ordner = AUSGABE / "_animation" / f"{szene.name}_frames"
            animation.rendere_bildfolge(szene, ordner, args.fps)
            print(f"   {ordner.relative_to(AUSGABE.parent)}  ({n} PNG)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

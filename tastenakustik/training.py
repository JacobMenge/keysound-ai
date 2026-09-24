"""Der Trainingslauf als Bibliothek - damit ihn beide Oberflaechen teilen.

`trainiere()` ist ein Generator: Er gibt nach jeder Epoche einen Zwischenstand
heraus und am Ende das Ergebnis. So kann das Terminal-Werkzeug Zeilen drucken
und das Studio dieselbe Rechnung als wachsende Kurve zeichnen, ohne dass es
zwei Trainingsroutinen gibt, die auseinanderlaufen.

`teste()` prueft ein fertiges Modell gegen die Testsitzungen - die Zahl, die
am Ende zaehlt.

Was hier bewusst NICHT passiert: Train, Validation und Test werden nirgends
gemischt. Die Aufteilung kommt aus der Rolle ganzer Sitzungen (siehe
datensatz.py) und wird hier nur noch benutzt.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn

from . import datensatz, modell
from .config import MODELLE, TASTEN, Config, anzeige, zufall


def _zufall_fuer(klassen: list[str]) -> float:
    """Zufallsniveau einer bestimmten Klassenliste, sonst der eingestellten."""
    return 1.0 / len(klassen) if klassen else zufall()


@dataclass
class Stand:
    """Zwischenstand nach einer Epoche."""

    epoche: int
    epochen: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    beste_val: float
    beste_epoche: int


@dataclass
class Ergebnis:
    """Was nach dem Training vorliegt."""

    verlauf: dict[str, list[float]]
    konfusion: np.ndarray
    beste_val: float
    beste_epoche: int
    dauer_s: float
    parameter: int
    klassen: list[str]
    n_train: int
    n_val: int
    train_sitzungen: list[str]
    val_sitzungen: list[str]
    segment_ms: float
    vor_ms: float
    modell_pfad: Path | None = None
    verlauf_pfad: Path | None = None
    je_klasse: dict[str, float] = field(default_factory=dict)

    @property
    def faktor(self) -> float:
        """Wie viel besser als blindes Raten."""
        # Gemessen an den Klassen des Modells, nicht an der gerade
        # eingestellten Liste.
        return self.beste_val / _zufall_fuer(self.klassen)


class DatenFehler(RuntimeError):
    """Es fehlt etwas, ohne das kein sinnvolles Training moeglich ist."""


class TrainingAbgebrochen(DatenFehler):
    """Das Training wurde auf Wunsch abgebrochen - es wurde nichts gespeichert."""


def _durchlauf(netz, x, y, verlust_fn, optimierer=None, rng=None, batch=32):
    """Eine Epoche. Mit optimierer wird gelernt, ohne nur gemessen."""
    lernen = optimierer is not None
    netz.train(lernen)
    reihenfolge = (rng.permutation(len(y)) if lernen and rng is not None
                   else np.arange(len(y)))
    summe, richtig = 0.0, 0
    with torch.set_grad_enabled(lernen):
        for i in range(0, len(reihenfolge), batch):
            idx = reihenfolge[i:i + batch]
            xb, yb = x[idx], y[idx]
            if lernen:
                xb = modell.augmentiere(xb, rng)
            ausgabe = netz(xb)
            verlust = verlust_fn(ausgabe, yb)
            if lernen:
                optimierer.zero_grad(set_to_none=True)
                verlust.backward()
                optimierer.step()
            summe += float(verlust.detach()) * len(idx)
            richtig += int((ausgabe.argmax(1) == yb).sum())
    return summe / len(y), richtig / len(y)


def _quoten_je_klasse(matrix: np.ndarray) -> np.ndarray:
    """Trefferquote je Klasse; NaN fuer Klassen ohne eine einzige Probe.

    Eine Klasse, die in der Pruefsitzung gar nicht vorkommt, hat keine Quote -
    als "0 %" wuerde sie faelschlich als schwaechste Klasse auftauchen.
    """
    zeilen = matrix.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(zeilen > 0, matrix.diagonal() / np.maximum(zeilen, 1), np.nan)


def _leer_hinweis(leer: list[str], rolle: str) -> str:
    """Sitzungen, die eine Rolle tragen, aber keine einzige Probe enthalten.

    Sie stehen in der Sitzungsliste mit ihrer Rolle - ohne diesen Hinweis
    klaenge "es gibt keine Sitzung mit der Rolle" wie ein Widerspruch.
    """
    if not leer:
        return ""
    wer = ("Die Sitzung " + leer[0] if len(leer) == 1
           else "Die Sitzungen " + ", ".join(leer))
    return (f" {wer} mit der Rolle '{rolle}' "
            f"{'enthält' if len(leer) == 1 else 'enthalten'} keine Proben: "
            "neu aufnehmen oder in Schritt 3 auf 'offen' stellen.")


def mindest_segment_ms(merkmale: dict) -> float:
    """Kuerzestes Segment, das das Netz noch verarbeiten kann.

    Drei Pooling-Stufen halbieren die Zeitachse dreimal - es braucht also
    mindestens 8 Mel-Rahmen, sonst bleibt eine Breite von 0.
    """
    mel = merkmale["mel"]
    n_min = mel["nfft"] + 7 * mel["hop"]
    return math.ceil(n_min / merkmale["samplerate"] * 1000 * 10) / 10


def pruefe_daten(segment_ms: float | None = None, vor_ms: float | None = None,
                 klassen: list[str] | None = None):
    """Train- und Val-Datensatz laden und auf offensichtliche Luecken pruefen.

    segment_ms/vor_ms ohne Angabe aus der config.json.
    """
    segment_ms, vor_ms = datensatz.schnitt(segment_ms, vor_ms)
    klassen = list(TASTEN) if klassen is None else klassen
    _, train = datensatz.lade("train", segment_ms, vor_ms, klassen)
    _, val = datensatz.lade("val", segment_ms, vor_ms, klassen)
    if len(train) == 0 or len(val) == 0:
        raise DatenFehler(
            "Es fehlen Sitzungen mit der Rolle 'train' oder 'val'. "
            "Jede Rolle braucht mindestens eine ganze Sitzung."
            + (_leer_hinweis(train.leer, "train") if len(train) == 0 else "")
            + (_leer_hinweis(val.leer, "val") if len(val) == 0 else "")
        )
    if datensatz.beschreibe_merkmale(train.merkmale) != \
            datensatz.beschreibe_merkmale(val.merkmale):
        # Sonst wuerde die Validation still auf anders gerechneten Bildern
        # messen - das Netz nimmt jede Bildbreite an und liegt nur oefter
        # daneben.
        raise DatenFehler(
            "Train- und Val-Sitzungen wurden mit verschiedenen Einstellungen "
            f"aufgenommen (train: {datensatz.beschreibe_merkmale(train.merkmale)}; "
            f"val: {datensatz.beschreibe_merkmale(val.merkmale)}). Beide Rollen "
            "mit demselben Eingang aufnehmen.")
    if train.x.shape[2] < 8:
        raise DatenFehler(
            f"Ein Segment von {segment_ms:g} ms ist zu kurz: Bei "
            f"{train.merkmale['samplerate']} Hz braucht das Netz mindestens "
            f"{mindest_segment_ms(train.merkmale):g} ms.")
    fehlend = [t for t, n in train.je_klasse.items() if n == 0]
    if fehlend:
        raise DatenFehler(
            "Ohne Trainingsdaten: " + " ".join(anzeige(t) for t in fehlend)
        )
    return train, val


def trainiere(epochen: int = 80, segment_ms: float | None = None,
              vor_ms: float | None = None, lernrate: float = 2e-3, seed: int = 1,
              speichern: bool = True, abbruch: threading.Event | None = None,
              ) -> Iterator[Stand | Ergebnis]:
    """Trainieren und nach jeder Epoche einen Stand herausgeben.

    Der letzte herausgegebene Wert ist das Ergebnis, alle davor sind Staende.
    segment_ms/vor_ms ohne Angabe aus der config.json. Ist `abbruch` nach
    einer Epoche gesetzt, endet das Training mit TrainingAbgebrochen, und es
    wird nichts gespeichert.

    Die Klassenliste wird am Anfang festgehalten: TASTEN kann sich waehrend
    des Trainings aendern (Schritt 2 im Studio), und ein Modell mit den
    Beschriftungen einer anderen Liste zeigte in der Demo still falsche
    Zeichen.
    """
    klassen = list(TASTEN)
    if epochen < 1:
        raise DatenFehler("Es braucht mindestens eine Epoche.")
    segment_ms, vor_ms = datensatz.schnitt(segment_ms, vor_ms)
    rng = modell.setze_zufall(seed)
    train, val = pruefe_daten(segment_ms, vor_ms, klassen)

    x_tr = torch.from_numpy(train.x).unsqueeze(1)
    y_tr = torch.from_numpy(train.y)
    x_va = torch.from_numpy(val.x).unsqueeze(1)
    y_va = torch.from_numpy(val.y)

    netz = modell.KleinesCNN(len(klassen))
    verlust_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(netz.parameters(), lr=lernrate, weight_decay=1e-3)
    plan = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochen)

    verlauf = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    bestes, beste_epoche, bester_stand = 0.0, 0, None
    t0 = time.perf_counter()

    for epoche in range(1, epochen + 1):
        tl, ta = _durchlauf(netz, x_tr, y_tr, verlust_fn, opt, rng)
        vl, va = _durchlauf(netz, x_va, y_va, verlust_fn)
        plan.step()
        for schluessel, wert in (("train_loss", tl), ("val_loss", vl),
                                 ("train_acc", ta), ("val_acc", va)):
            verlauf[schluessel].append(wert)
        # Die erste Epoche gilt immer als bester Stand: Bleibt die Validation
        # durchweg bei 0, gaebe es sonst keinen, und das Laden danach scheitert.
        if va > bestes or bester_stand is None:
            bestes, beste_epoche = va, epoche
            bester_stand = {k: v.detach().clone()
                            for k, v in netz.state_dict().items()}
        yield Stand(epoche, epochen, tl, vl, ta, va, bestes, beste_epoche)
        if abbruch is not None and abbruch.is_set():
            raise TrainingAbgebrochen(
                f"Training nach Epoche {epoche} abgebrochen - kein Modell "
                "gespeichert.")

    if list(TASTEN) != klassen:
        # Die Sperre in der Oberflaeche sollte das verhindern. Falls nicht:
        # lieber kein Modell als eines, dessen Beschriftungen nicht stimmen.
        raise DatenFehler(
            f"Die Klassen wurden während des Trainings von {''.join(klassen)} "
            f"auf {''.join(TASTEN)} geändert - kein Modell gespeichert.")

    dauer = time.perf_counter() - t0

    # Ausgewertet wird der beste Stand, nicht der letzte - bei so wenig Daten
    # schwankt die Validation von Epoche zu Epoche deutlich.
    netz.load_state_dict(bester_stand)
    netz.eval()
    with torch.no_grad():
        vorhersage = netz(x_va).argmax(1).numpy()
    matrix = np.zeros((len(klassen), len(klassen)))
    for wahr, vorher in zip(val.y, vorhersage):
        matrix[wahr, vorher] += 1
    quoten = _quoten_je_klasse(matrix)

    ergebnis = Ergebnis(
        verlauf=verlauf, konfusion=matrix, beste_val=bestes,
        beste_epoche=beste_epoche, dauer_s=dauer,
        parameter=netz.parameterzahl, klassen=list(klassen),
        n_train=len(train), n_val=len(val),
        train_sitzungen=sorted(set(train.sitzungen)),
        val_sitzungen=sorted(set(val.sitzungen)),
        segment_ms=segment_ms, vor_ms=vor_ms,
        je_klasse={t: float(q) for t, q in zip(klassen, quoten)},
    )

    if speichern:
        MODELLE.mkdir(parents=True, exist_ok=True)
        marke = datetime.now().strftime("%Y%m%d_%H%M%S")
        ergebnis.modell_pfad = MODELLE / f"cnn_{marke}.pt"
        # Abtastrate und Mel-Parameter der Trainingssitzungen gehoeren zum
        # Modell: Demo und Test muessen genau diese Merkmale rechnen, egal
        # was config.json inzwischen sagt.
        torch.save({"state_dict": netz.state_dict(), "klassen": list(klassen),
                    "segment_ms": segment_ms, "vor_ms": vor_ms,
                    "mel_baender": train.x.shape[1],
                    "samplerate": int(train.merkmale["samplerate"]),
                    "mel": dict(train.merkmale["mel"])}, ergebnis.modell_pfad)
        ergebnis.verlauf_pfad = MODELLE / f"verlauf_{marke}.json"
        ergebnis.verlauf_pfad.write_text(json.dumps({
            "verlauf": verlauf, "beste_val": bestes, "beste_epoche": beste_epoche,
            "train_sitzungen": ergebnis.train_sitzungen,
            "val_sitzungen": ergebnis.val_sitzungen,
            "n_train": len(train), "n_val": len(val),
            "parameter": netz.parameterzahl, "segment_ms": segment_ms,
            "klassen": list(klassen), "konfusion": matrix.tolist(),
        }, indent=2), encoding="utf-8")

    yield ergebnis


def neuestes_modell() -> Path | None:
    """Der zuletzt gespeicherte Modellstand, falls einer da ist."""
    if not MODELLE.exists():
        return None
    stände = sorted(MODELLE.glob("cnn_*.pt"))
    return stände[-1] if stände else None


def modell_klassen(pfad: Path) -> list[str] | None:
    """Die Klassen, die ein gespeichertes Modell kennt - None, wenn unlesbar."""
    try:
        stand = torch.load(pfad, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    klassen = stand.get("klassen")
    return list(klassen) if klassen else None


def modell_merkmale(stand: dict, cfg: Config | None = None) -> dict:
    """Abtastrate und Mel-Parameter, mit denen ein Modell trainiert wurde.

    Aeltere Modelldateien bringen sie nicht mit - dann gelten die Werte der
    Config, damit diese Modelle weiter laden.
    """
    cfg = cfg or Config.laden(anwenden=False)
    vorgabe = datensatz.merkmale_von(cfg)
    mel = {**vorgabe["mel"], **(stand.get("mel") or {})}
    if "mel" not in stand and stand.get("mel_baender"):
        mel["baender"] = int(stand["mel_baender"])
    return {"samplerate": int(stand.get("samplerate", vorgabe["samplerate"])),
            "mel": mel}


@dataclass
class TestErgebnis:
    """Wie gut ein fertiges Modell auf einer ungesehenen Sitzung ist."""

    quote: float
    konfidenz: float
    n: int
    konfusion: np.ndarray
    je_klasse: dict[str, float]
    sitzungen: list[str]
    modell_pfad: Path
    val_quote: float | None = None
    pfad: Path | None = None
    klassen: list[str] = field(default_factory=list)
    # Leer, oder ein Satz dazu, dass Sitzungen umgerechnet werden mussten
    hinweis: str = ""

    @property
    def faktor(self) -> float:
        return self.quote / _zufall_fuer(self.klassen)


def _vorhersagen(netz, x: np.ndarray, n_klassen: int) -> np.ndarray:
    """Wahrscheinlichkeiten fuer alle Proben, in handlichen Portionen."""
    teile = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            xb = torch.from_numpy(x[i:i + 256]).unsqueeze(1)
            teile.append(torch.softmax(netz(xb), dim=1).numpy())
    return np.concatenate(teile) if teile else np.zeros((0, n_klassen))


def teste(modell_pfad: Path | None = None, rolle: str = "test",
          speichern: bool = True) -> TestErgebnis:
    """Ein fertiges Modell gegen die Testsitzungen pruefen - die ehrliche Zahl.

    Die Testsitzung hat weder beim Lernen noch bei der Auswahl des besten
    Stands mitgewirkt. Was hier herauskommt, ist das, was das Modell auf
    neuen Aufnahmen kann.

    Segment, Abtastrate und Mel-Parameter kommen aus der Modelldatei, nicht
    aus config.json - geprueft wird mit genau den Merkmalen, die das Modell
    gelernt hat. Sitzungen mit anderer Abtastrate werden dafuer umgerechnet.
    """
    eingestellt = list(TASTEN)
    pfad = modell_pfad or neuestes_modell()
    if pfad is None:
        raise DatenFehler("Es gibt noch kein trainiertes Modell.")
    stand = torch.load(pfad, map_location="cpu", weights_only=False)
    klassen = list(stand.get("klassen") or eingestellt)
    if klassen != eingestellt:
        raise DatenFehler(
            f"Das Modell kennt die Klassen {''.join(klassen)}, eingestellt sind "
            f"{''.join(eingestellt)}. Entweder neu trainieren oder die Klassen in "
            "Schritt 2 zurückstellen.")

    segment_ms, vor_ms = stand.get("segment_ms", 250.0), stand.get("vor_ms", 15.0)
    if "samplerate" in stand:
        merkmale = modell_merkmale(stand)
    else:
        # Aeltere Modelldatei: Gelernt hat sie auf den Train-Sitzungen, also
        # gelten deren Parameter - erst wenn es keine gibt, die der Config.
        merkmale = datensatz.merkmale_der_rolle("train") or modell_merkmale(stand)
        if stand.get("mel_baender"):
            merkmale["mel"]["baender"] = int(stand["mel_baender"])
    _, daten = datensatz.lade(rolle, segment_ms, vor_ms, klassen, merkmale)
    if len(daten) == 0 and daten.leer:
        raise DatenFehler(_leer_hinweis(daten.leer, rolle).strip())
    if len(daten) == 0:
        raise DatenFehler(
            f"Es gibt noch keine Sitzung mit der Rolle '{rolle}'. Nimm eine "
            "weitere Sitzung auf - am besten an einem anderen Tag - und gib "
            f"ihr in Schritt 3 die Rolle '{rolle}'.")

    netz = modell.KleinesCNN(len(klassen))
    netz.load_state_dict(stand["state_dict"])
    netz.eval()

    p = _vorhersagen(netz, daten.x, len(klassen))
    vorhersage = p.argmax(1)
    matrix = np.zeros((len(klassen), len(klassen)))
    for wahr, vorher in zip(daten.y, vorhersage):
        matrix[wahr, vorher] += 1
    quoten = _quoten_je_klasse(matrix)

    val_quote = val_konfidenz = None
    umgerechnet = dict(daten.umgerechnet)
    if rolle != "val":
        _, val = datensatz.lade("val", segment_ms, vor_ms, klassen, merkmale)
        if len(val):
            p_val = _vorhersagen(netz, val.x, len(klassen))
            val_quote = float(np.mean(p_val.argmax(1) == val.y))
            val_konfidenz = float(np.mean(p_val.max(1)))
            umgerechnet.update(val.umgerechnet)

    hinweis = ""
    if umgerechnet:
        hinweis = ("Umgerechnet auf die Abtastrate des Modells "
                   f"({merkmale['samplerate']} Hz): "
                   + ", ".join(f"{s} ({sr} Hz)" for s, sr in sorted(umgerechnet.items()))
                   + ". Die Zahl kann dadurch etwas unter der bei gleicher "
                   "Rate liegen.")

    ergebnis = TestErgebnis(
        quote=float(np.mean(vorhersage == daten.y)),
        konfidenz=float(np.mean(p.max(1))),
        n=len(daten), konfusion=matrix,
        je_klasse={t: float(q) for t, q in zip(klassen, quoten)},
        sitzungen=sorted(set(daten.sitzungen)), modell_pfad=pfad,
        val_quote=val_quote, klassen=list(klassen), hinweis=hinweis,
    )
    if speichern:
        ergebnis.pfad = MODELLE / f"{rolle}_ergebnis.json"
        ergebnis.pfad.write_text(json.dumps({
            f"{rolle}_quote": ergebnis.quote,
            f"{rolle}_confidence": ergebnis.konfidenz,
            f"n_{rolle}": ergebnis.n,
            "konfusion": matrix.tolist(),
            "val_quote": val_quote,
            "val_confidence": val_konfidenz,
            "klassen": klassen,
            "sitzungen": ergebnis.sitzungen,
            "modell": pfad.name,
        }, indent=2), encoding="utf-8")
    return ergebnis

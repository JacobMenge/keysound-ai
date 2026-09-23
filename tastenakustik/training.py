"""Der Trainingslauf als Bibliothek - damit ihn beide Oberflaechen teilen.

`trainiere()` ist ein Generator: Er gibt nach jeder Epoche einen Zwischenstand
heraus und am Ende das Ergebnis. So kann das Terminal-Werkzeug Zeilen drucken
und das Studio dieselbe Rechnung als wachsende Kurve zeichnen, ohne dass es
zwei Trainingsroutinen gibt, die auseinanderlaufen.

Was hier bewusst NICHT passiert: Train, Validation und Test werden nirgends
gemischt. Die Aufteilung kommt aus der Rolle ganzer Sitzungen (siehe
datensatz.py) und wird hier nur noch benutzt.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn

from . import datensatz, modell
from .config import MODELLE, TASTEN, zufall


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
        return self.beste_val / zufall()


class DatenFehler(RuntimeError):
    """Es fehlt etwas, ohne das kein sinnvolles Training moeglich ist."""


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


def pruefe_daten(segment_ms: float = 250.0, vor_ms: float = 15.0):
    """Train- und Val-Datensatz laden und auf offensichtliche Luecken pruefen."""
    _, train = datensatz.lade("train", segment_ms, vor_ms)
    _, val = datensatz.lade("val", segment_ms, vor_ms)
    if len(train) == 0 or len(val) == 0:
        raise DatenFehler(
            "Es fehlen Sitzungen mit der Rolle 'train' oder 'val'. "
            "Jede Rolle braucht mindestens eine ganze Sitzung."
        )
    fehlend = [t for t, n in train.je_klasse.items() if n == 0]
    if fehlend:
        raise DatenFehler(
            "Ohne Trainingsdaten: " + " ".join(t.upper() for t in fehlend)
        )
    return train, val


def trainiere(epochen: int = 80, segment_ms: float = 250.0, vor_ms: float = 15.0,
              lernrate: float = 2e-3, seed: int = 1, speichern: bool = True,
              ) -> Iterator[Stand | Ergebnis]:
    """Trainieren und nach jeder Epoche einen Stand herausgeben.

    Der letzte herausgegebene Wert ist das Ergebnis, alle davor sind Staende.
    """
    rng = modell.setze_zufall(seed)
    train, val = pruefe_daten(segment_ms, vor_ms)

    x_tr = torch.from_numpy(train.x).unsqueeze(1)
    y_tr = torch.from_numpy(train.y)
    x_va = torch.from_numpy(val.x).unsqueeze(1)
    y_va = torch.from_numpy(val.y)

    netz = modell.KleinesCNN()
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
        if va > bestes:
            bestes, beste_epoche = va, epoche
            bester_stand = {k: v.detach().clone()
                            for k, v in netz.state_dict().items()}
        yield Stand(epoche, epochen, tl, vl, ta, va, bestes, beste_epoche)

    dauer = time.perf_counter() - t0

    # Ausgewertet wird der beste Stand, nicht der letzte - bei so wenig Daten
    # schwankt die Validation von Epoche zu Epoche deutlich.
    netz.load_state_dict(bester_stand)
    netz.eval()
    with torch.no_grad():
        vorhersage = netz(x_va).argmax(1).numpy()
    matrix = np.zeros((len(TASTEN), len(TASTEN)))
    for wahr, vorher in zip(val.y, vorhersage):
        matrix[wahr, vorher] += 1
    quoten = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)

    ergebnis = Ergebnis(
        verlauf=verlauf, konfusion=matrix, beste_val=bestes,
        beste_epoche=beste_epoche, dauer_s=dauer,
        parameter=netz.parameterzahl, klassen=list(TASTEN),
        n_train=len(train), n_val=len(val),
        train_sitzungen=sorted(set(train.sitzungen)),
        val_sitzungen=sorted(set(val.sitzungen)),
        segment_ms=segment_ms, vor_ms=vor_ms,
        je_klasse={t: float(q) for t, q in zip(TASTEN, quoten)},
    )

    if speichern:
        MODELLE.mkdir(parents=True, exist_ok=True)
        marke = datetime.now().strftime("%Y%m%d_%H%M%S")
        ergebnis.modell_pfad = MODELLE / f"cnn_{marke}.pt"
        torch.save({"state_dict": netz.state_dict(), "klassen": list(TASTEN),
                    "segment_ms": segment_ms, "vor_ms": vor_ms,
                    "mel_baender": train.x.shape[1]}, ergebnis.modell_pfad)
        ergebnis.verlauf_pfad = MODELLE / f"verlauf_{marke}.json"
        ergebnis.verlauf_pfad.write_text(json.dumps({
            "verlauf": verlauf, "beste_val": bestes, "beste_epoche": beste_epoche,
            "train_sitzungen": ergebnis.train_sitzungen,
            "val_sitzungen": ergebnis.val_sitzungen,
            "n_train": len(train), "n_val": len(val),
            "parameter": netz.parameterzahl, "segment_ms": segment_ms,
            "klassen": list(TASTEN), "konfusion": matrix.tolist(),
        }, indent=2), encoding="utf-8")

    yield ergebnis


def neuestes_modell() -> Path | None:
    """Der zuletzt gespeicherte Modellstand, falls einer da ist."""
    if not MODELLE.exists():
        return None
    stände = sorted(MODELLE.glob("cnn_*.pt"))
    return stände[-1] if stände else None

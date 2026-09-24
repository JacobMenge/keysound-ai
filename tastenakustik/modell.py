"""Das kleine CNN und seine Augmentierung.

Bewusst klein gehalten: ein paar hundert Trainingsproben sind wenig, und ein
grosses Netz wuerde sie schlicht auswendig lernen. Drei Faltungsbloecke, danach
ein Mittelwert ueber die ganze Merkmalskarte (Zeit und Frequenz) und eine
lineare Schicht auf genau so viele Ausgaenge, wie es Klassen gibt.

Die Ausgabemenge ist damit auf die gewaehlten Klassen verdrahtet. Etwas anderes
kann dieses Modell nicht vorhersagen - das ist keine Einstellung, sondern die
Form der letzten Schicht.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import TASTEN


class KleinesCNN(nn.Module):
    def __init__(self, n_klassen: int | None = None, breite: int = 16,
                 dropout: float = 0.3):
        # Nicht len(TASTEN) als Vorgabewert: der wuerde beim Import eingefroren
        # und passte nach einem Klassenwechsel nicht mehr.
        n_klassen = len(TASTEN) if n_klassen is None else n_klassen
        super().__init__()

        def block(ein: int, aus: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ein, aus, 3, padding=1, bias=False),
                nn.BatchNorm2d(aus),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.merkmale = nn.Sequential(
            block(1, breite),
            block(breite, breite * 2),
            block(breite * 2, breite * 4),
        )
        self.kopf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(breite * 4, n_klassen),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kopf(self.merkmale(x))

    @property
    def parameterzahl(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def augmentiere(x: torch.Tensor, rng: np.random.Generator,
                zeit_versatz: int = 6, zeit_maske: int = 10,
                frequenz_maske: int = 8, rauschen: float = 0.08) -> torch.Tensor:
    """Vier Stoerungen, die den Klang nicht veraendern, aber die Lage.

    Der Onset sitzt nie exakt gleich, das Mikrofon steht nie exakt gleich, und
    im Raum ist immer etwas los. Genau das wird hier nachgestellt - damit das
    Netz nicht lernt, dass ein J immer bei Bildspalte 14 anfaengt.
    """
    n, _, mel, frames = x.shape
    x = x.clone()

    # zeitlicher Versatz
    for i in range(n):
        v = int(rng.integers(-zeit_versatz, zeit_versatz + 1))
        if v:
            x[i] = torch.roll(x[i], shifts=v, dims=-1)

    # Zeit- und Frequenzbaender ausblenden (SpecAugment)
    for i in range(n):
        if zeit_maske > 0:
            b = int(rng.integers(0, zeit_maske + 1))
            if b:
                s = int(rng.integers(0, max(frames - b, 1)))
                x[i, :, :, s:s + b] = 0.0
        if frequenz_maske > 0:
            b = int(rng.integers(0, frequenz_maske + 1))
            if b:
                s = int(rng.integers(0, max(mel - b, 1)))
                x[i, :, s:s + b, :] = 0.0

    if rauschen > 0:
        x = x + torch.from_numpy(
            rng.normal(0, rauschen, x.shape).astype(np.float32))
    return x


def setze_zufall(seed: int) -> np.random.Generator:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)

"""Selbstgezeichnete Icons - bewusst keine fremde Icon-Bibliothek.

Gruende: keine Lizenzfrage im Video, gleiche Strichstaerke und Rundung wie der
Rest der Grafiken, beliebig skalierbar, und jedes Icon laesst sich fuer die
Animation schrittweise aufbauen.

Jede Funktion zeichnet in eine Achse mit den Koordinaten 0..1 in beiden
Richtungen. Der Parameter `anteil` (0..1) blendet das Icon auf - damit koennen
dieselben Funktionen Standbild und Animation bedienen.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Circle, FancyBboxPatch

from . import theme

STRICH = 5.0


def leinwand(ax) -> None:
    """Achse zu einer quadratischen Zeichenflaeche ohne alles machen."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)


def _a(anteil: float) -> float:
    return float(np.clip(anteil, 0.0, 1.0))


# --- Einzelne Icons -------------------------------------------------------
def taste(ax, farbe: str = theme.AKZENT, anteil: float = 1.0, lw: float = STRICH) -> None:
    """Tastenkappe mit Schallwellen - der Anschlag."""
    leinwand(ax)
    al = _a(anteil)
    ax.add_patch(FancyBboxPatch(
        (0.10, 0.16), 0.46, 0.46,
        boxstyle="round,pad=0.015,rounding_size=0.10",
        fill=False, edgecolor=farbe, linewidth=lw, alpha=al))
    ax.plot([0.19, 0.47], [0.53, 0.53], color=farbe, lw=lw * 0.6, alpha=al * 0.55)
    for i, r in enumerate((0.20, 0.32, 0.44)):
        sichtbar = _a(anteil * 3 - i)
        if sichtbar <= 0:
            continue
        ax.add_patch(Arc((0.58, 0.56), r * 2, r * 2, theta1=-30, theta2=60,
                         color=farbe, lw=lw * 0.85, alpha=sichtbar * (1 - 0.18 * i)))


def mikrofon(ax, farbe: str = theme.AKZENT, anteil: float = 1.0, lw: float = STRICH) -> None:
    """Mikrofon mit Buegel und Fuss."""
    leinwand(ax)
    al = _a(anteil)
    ax.add_patch(FancyBboxPatch(
        (0.38, 0.44), 0.24, 0.36,
        boxstyle="round,pad=0,rounding_size=0.12",
        fill=False, edgecolor=farbe, linewidth=lw, alpha=al))
    ax.add_patch(Arc((0.50, 0.46), 0.52, 0.48, theta1=200, theta2=340,
                     color=farbe, lw=lw, alpha=al))
    ax.plot([0.50, 0.50], [0.22, 0.12], color=farbe, lw=lw, alpha=al)
    ax.plot([0.34, 0.66], [0.12, 0.12], color=farbe, lw=lw, alpha=al,
            solid_capstyle="round")


def welle(ax, farbe: str = theme.AKZENT, anteil: float = 1.0, lw: float = STRICH) -> None:
    """Ein Anschlag als Wellenform - Impuls mit Abklingen."""
    leinwand(ax)
    al = _a(anteil)
    t = np.linspace(0, 1, 600)
    huelle = np.where(t < 0.26, 0.02, np.exp(-(t - 0.26) / 0.10))
    y = 0.5 + 0.40 * huelle * np.sin(2 * np.pi * 17 * (t - 0.26))
    n = max(int(len(t) * al), 2)
    ax.plot([0, 1], [0.5, 0.5], color=farbe, lw=1.4, alpha=al * 0.35)
    ax.plot(t[:n], y[:n], color=farbe, lw=lw * 0.62, solid_joinstyle="round")


def mel(ax, farbe: str = theme.AKZENT, anteil: float = 1.0, lw: float = STRICH) -> None:
    """Log-Mel als Raster - einfarbig, damit das Bild nicht bunt wird."""
    leinwand(ax)
    al = _a(anteil)
    rng = np.random.default_rng(11)
    zeilen, spalten = 7, 10
    werte = 0.10 + rng.random((zeilen, spalten)) * 0.28
    werte[:, 3] = np.linspace(1.00, 0.55, zeilen)
    werte[:, 4] = np.linspace(0.92, 0.45, zeilen)
    werte[:, 5] = np.linspace(0.70, 0.30, zeilen)
    sichtbar = int(np.ceil(spalten * al))
    if sichtbar < 1:
        return
    karte = LinearSegmentedColormap.from_list("mono", ["#0F2233", farbe, "#DFF6FF"])
    ax.imshow(werte[:, :sichtbar], cmap=karte, vmin=0, vmax=1, origin="lower",
              extent=(0.06, 0.06 + 0.88 * sichtbar / spalten, 0.14, 0.86),
              aspect="auto", interpolation="nearest")
    ax.add_patch(FancyBboxPatch(
        (0.06, 0.14), 0.88, 0.72, boxstyle="round,pad=0,rounding_size=0.04",
        fill=False, edgecolor=farbe, linewidth=lw * 0.5, alpha=al * 0.6))


def netz(ax, farbe: str = theme.AKZENT, anteil: float = 1.0, lw: float = STRICH) -> None:
    """Kleines neuronales Netz: drei Schichten, Kanten, Knoten."""
    leinwand(ax)
    al = _a(anteil)
    schichten = (4, 5, 3)
    x_pos = np.linspace(0.16, 0.84, len(schichten))
    knoten = [
        [(x, 0.5 + (i - (n - 1) / 2) * (0.72 / max(n, 2))) for i in range(n)]
        for x, n in zip(x_pos, schichten)
    ]
    kanten_al = _a(anteil * 1.6) * 0.35
    for links, rechts in zip(knoten, knoten[1:]):
        for x1, y1 in links:
            for x2, y2 in rechts:
                ax.plot([x1, x2], [y1, y2], color=farbe, lw=lw * 0.22, alpha=kanten_al)
    for i, schicht in enumerate(knoten):
        sichtbar = _a(anteil * 3 - i)
        for x, y in schicht:
            ax.add_patch(Circle((x, y), 0.055, facecolor=theme.BG, edgecolor=farbe,
                                linewidth=lw * 0.7, alpha=sichtbar, zorder=3))


def balken(ax, farben: list[str], werte: list[float] | None = None,
           anteil: float = 1.0) -> None:
    """Ausgabe: eine Wahrscheinlichkeit je Klasse. Hier duerfen die Klassenfarben rein."""
    leinwand(ax)
    n = len(farben)
    if werte is None:
        # Ein Gewinner, der Rest klein und ungleich - so sieht eine echte
        # Softmax-Ausgabe aus. Laenge folgt der Klassenzahl. Das Muster
        # wiederholt sich ab zwoelf Klassen, der Gewinner darf es nicht:
        # deshalb erst ohne ihn aufbauen und ihn dann genau einmal setzen.
        muster = (0.15, 0.1, 0.92, 0.12, 0.2, 0.1, 0.3, 0.08, 0.18, 0.11, 0.25)
        werte = [muster[i % len(muster)] for i in range(n)]
        werte = [0.14 if w >= 0.9 else w for w in werte]
        if werte:
            werte[min(2, n - 1)] = 0.92
    breite = 0.86 / max(n, 1)
    # Die Balken wachsen nacheinander. Der Versatz je Balken schrumpft bei
    # vielen Klassen, damit bei anteil 1 alle voll stehen - bis zehn Klassen
    # bleibt das Timing wie bisher.
    stufe = min(0.1, 1.0 / max(n, 1))
    for i, (farbe, wert) in enumerate(zip(farben, werte)):
        hoch = 0.70 * wert * _a(anteil * 2 - i * stufe)
        if hoch <= 0.001:
            continue
        ax.add_patch(FancyBboxPatch(
            (0.07 + i * breite + breite * 0.15, 0.14),
            breite * 0.7, hoch,
            boxstyle="round,pad=0,rounding_size=0.02",
            facecolor=farbe, edgecolor="none"))
    ax.plot([0.05, 0.95], [0.13, 0.13], color=theme.GRID, lw=2.0)


def lautsprecher(ax, farbe: str = theme.AKZENT, anteil: float = 1.0,
                 lw: float = STRICH) -> None:
    """Schallquelle - fuer das Call-Experiment."""
    leinwand(ax)
    al = _a(anteil)
    ax.plot([0.16, 0.16, 0.30, 0.46, 0.46, 0.30, 0.16],
            [0.40, 0.60, 0.60, 0.80, 0.20, 0.40, 0.40],
            color=farbe, lw=lw, alpha=al, solid_joinstyle="round")
    for i, r in enumerate((0.16, 0.28)):
        sichtbar = _a(anteil * 2 - i)
        ax.add_patch(Arc((0.52, 0.50), r * 2, r * 2, theta1=-55, theta2=55,
                         color=farbe, lw=lw * 0.85, alpha=sichtbar))


SCHRITTE = {
    "taste": taste,
    "mikrofon": mikrofon,
    "welle": welle,
    "mel": mel,
    "netz": netz,
    "balken": balken,
    "lautsprecher": lautsprecher,
}

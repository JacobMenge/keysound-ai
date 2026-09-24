"""Einheitlicher Look fuer alle Ansichten - abgestimmt auf Bildschirmaufnahme.

Dunkler Hintergrund, kraeftige Linien, grosse Schrift: Alles, was hier
gezeichnet wird, soll auch auf einem Handy noch lesbar sein.
"""

from __future__ import annotations

import colorsys
from datetime import datetime
from pathlib import Path

import matplotlib as mpl

from .config import AUSGABE, TASTEN

# --- Palette --------------------------------------------------------------
BG = "#0B0E14"          # Seitenhintergrund
PANEL = "#131A24"       # Plotflaeche
GRID = "#243346"
TEXT = "#E6EDF3"
TEXT_SCHWACH = "#8B98A9"

AKZENT = "#4CC9F0"      # Signal / Hauptlinie
AKZENT2 = "#F72585"     # Marker, Onset
OK = "#4ADE80"
WARN = "#FBBF24"
FEHLER = "#F87171"
REC = "#FF3B30"

# Bis acht Klassen von Hand gesetzt: gut unterscheidbar und auch bei
# Rot-Gruen-Schwaeche noch trennbar.
BASIS_FARBEN = ("#4CC9F0", "#4ADE80", "#FBBF24", "#F472B6",
                "#A78BFA", "#FB923C", "#2DD4BF", "#F87171")

# Eine feste Farbe je Klasse - ueber alle Grafiken hinweg gleich, damit eine
# Klasse immer dieselbe Farbe hat. Wird von config.setze_klassen() neu gefuellt.
TASTEN_FARBEN: dict[str, str] = {}

SCHRIFTEN = ["Segoe UI", "Inter", "DejaVu Sans", "sans-serif"]

SKALEN = {"normal": 1.0, "gross": 1.35, "hochformat": 2.0}


def _palette(n: int) -> list[str]:
    """n gut unterscheidbare Farben.

    Ab neun Klassen wird der Farbkreis gleichmaessig abgeschritten. Damit
    benachbarte Klassen nicht verschwimmen, wechseln Helligkeit und Saettigung
    zusaetzlich im Dreiertakt - bei 26 Buchstaben reicht der Farbton allein
    nicht mehr aus.
    """
    if n <= len(BASIS_FARBEN):
        return list(BASIS_FARBEN[:n])
    farben = []
    for i in range(n):
        h = (i * 0.61803398875) % 1.0          # goldener Winkel: streut gut
        s = (0.55, 0.75, 0.95)[i % 3]
        v = (0.98, 0.86, 0.92)[i % 3]
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        farben.append(f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}")
    return farben


def palette(n: int) -> list[str]:
    """n Klassenfarben - auch fuer eine Liste, die noch nicht gesetzt ist."""
    return _palette(n)


def farben_aktualisieren() -> None:
    """Klassenfarben an die aktuelle Klassenliste anpassen (in place)."""
    TASTEN_FARBEN.clear()
    TASTEN_FARBEN.update(zip(TASTEN, _palette(len(TASTEN))))


def farbe(taste: str) -> str:
    """Farbe einer Klasse - faellt auf den Akzentton zurueck."""
    return TASTEN_FARBEN.get(taste, AKZENT)


farben_aktualisieren()


def anwenden(groesse: str = "normal") -> None:
    """Globalen Matplotlib-Stil setzen.

    'hochformat' ist fuer die 1080-x-1920-Leinwand gedacht: dort muss Schrift
    auch auf einem Handy in einem kurzen Clip noch lesbar sein.
    """
    skala = SKALEN.get(groesse, 1.0)
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.titleweight": "bold",
            "axes.titlesize": 13 * skala,
            "axes.labelsize": 11 * skala,
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.45,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SCHWACH,
            "ytick.color": TEXT_SCHWACH,
            "xtick.labelsize": 10 * skala,
            "ytick.labelsize": 10 * skala,
            "text.color": TEXT,
            # Als Liste, damit Matplotlib je Zeichen auf DejaVu Sans
            # zurueckfaellt: Segoe UI hat kein Leertasten-Zeichen, und ohne
            # Rueckfall stuende in jeder Kachel ein Ersatzkasten.
            "font.family": ["sans-serif", "DejaVu Sans"],
            "font.sans-serif": SCHRIFTEN,
            "font.size": 11 * skala,
            "legend.facecolor": PANEL,
            "legend.edgecolor": GRID,
            "legend.fontsize": 10 * skala,
            "lines.linewidth": 1.4,
            "lines.solid_capstyle": "round",
            "figure.autolayout": False,
            "savefig.dpi": 200,
            "figure.dpi": 110,
        }
    )


def speichere_asset(fig, name: str, unterordner: str = "", dpi: int = 200) -> Path:
    """Grafik hochaufloesend in ausgabe/ ablegen (mit Zeitstempel)."""
    ziel = AUSGABE / unterordner if unterordner else AUSGABE
    ziel.mkdir(parents=True, exist_ok=True)
    pfad = ziel / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.png"
    fig.savefig(pfad, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    return pfad


def quoten_farbe(quote: float) -> str:
    """Ampelfarbe fuer eine Trefferquote - gemessen am Zufall, nicht absolut.

    Bei zwei Klassen sind 55 % kaum mehr als Raten, bei vierzig Klassen sind
    30 % schon ein deutliches Signal. Deshalb zaehlt, wie weit die Quote auf
    dem Weg vom Zufall zur Perfektion ist.
    """
    zufall = 1.0 / max(len(TASTEN), 1)
    anteil = (quote - zufall) / max(1.0 - zufall, 1e-9)
    if anteil >= 0.5:
        return OK
    if anteil >= 0.15:
        return WARN
    return FEHLER


def pegel_farbe(db: float) -> str:
    """Ampelfarbe fuer einen Spitzenpegel in dBFS."""
    if db >= -3.0:
        return FEHLER
    if db >= -9.0:
        return WARN
    if db >= -30.0:
        return OK
    return TEXT_SCHWACH

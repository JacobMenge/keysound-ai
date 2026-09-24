"""Hochformat-Rahmen 9:16 - gemeinsame Grundlage aller Ansichten.

Alles in diesem Projekt wird fuer 1080 x 1920 gebaut: die Live-Fenster, die
Standbilder und spaeter die Trainings- und Ergebnisgrafiken. Damit passt jedes
Bild ohne Nachschneiden in Shorts, Reels und TikTok.

Zwei Dinge macht dieses Modul:

1. Fenster. Auf einem Hochformat-Monitor laeuft ein Fenster in echten
   1080 x 1920 Pixeln - ohne Hochrechnen, also ohne Schaerfeverlust. Passt das
   nicht, wird das groesste 9:16-Fenster gewaehlt, das auf den Schirm geht, und
   ein Skalierungsfaktor mitgeliefert, mit dem Schriften und Abstaende
   mitwachsen.

2. Grafiken. figur() liefert eine Matplotlib-Flaeche in exakt 1080 x 1920, und
   exportiere() schreibt sie in genau dieser Groesse heraus - ohne
   bbox_inches="tight", das die Masse sonst wieder veraendern wuerde.
"""

from __future__ import annotations

import ctypes
import sys
from datetime import datetime
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from . import theme
from .config import AUSGABE

# wintypes gibt es nur unter Windows sicher - auf anderen Systemen wird die
# Bildschirmgroesse ueber Tk abgefragt (siehe arbeitsflaechen()).
try:
    from ctypes import wintypes
except (ImportError, ValueError):  # pragma: no cover - nur ausserhalb Windows
    wintypes = None

# --- Leinwand -------------------------------------------------------------
BREITE, HOEHE = 1080, 1920
SEITENVERHAELTNIS = BREITE / HOEHE
DPI = 100

# Bereiche, die die Plattformen mit eigener Bedienoberflaeche ueberdecken
# (Statusleiste oben, Bildunterschrift und Fortschritt unten, Knopfleiste
# rechts). Alles Wichtige gehoert in das Kernband dazwischen - mit grosszuegig
# Luft an allen vier Seiten, damit nichts am Rand klebt.
SICHER_OBEN = 240
SICHER_UNTEN = 440
SICHER_LINKS = 140
SICHER_RECHTS = 160

# Schriftgroessen in Punkt bei 100 dpi, also mal 1,39 fuer Pixel.
# Untergrenze ist bewusst hoch: Was im Video auf einem Handy nicht lesbar ist,
# gehoert gar nicht erst aufs Bild.
S_TITEL = 54      # 75 px
S_HERO = 92       # 128 px - die eine Zahl, um die es geht
S_LABEL = 32      # 44 px - benennt, was man sieht
S_ZAHL = 34       # 47 px
S_TICK = 24       # 33 px - kleinste erlaubte Groesse
S_KACHEL = 34


def y(px: float) -> float:
    """Figurkoordinate aus einer Pixelangabe von oben gemessen."""
    return 1.0 - px / HOEHE


def x(px: float) -> float:
    """Figurkoordinate aus einer Pixelangabe von links gemessen."""
    return px / BREITE


INHALT_LINKS = SICHER_LINKS
INHALT_RECHTS = BREITE - SICHER_RECHTS
INHALT_BREITE = INHALT_RECHTS - INHALT_LINKS   # 780 px

# Achsen mit eigener y-Beschriftung ruecken ein, damit die Zahlen innerhalb
# des Kernbands bleiben statt in den Rand zu laufen.
ACHSE_LINKS = INHALT_LINKS + 96
ACHSE_BREITE = INHALT_RECHTS - ACHSE_LINKS


def rechteck(oben: float, hoehe: float, links: float = INHALT_LINKS,
             breite: float = INHALT_BREITE) -> list[float]:
    """Pixelangaben von oben links in ein Matplotlib-Rechteck umrechnen."""
    return [links / BREITE, 1 - (oben + hoehe) / HOEHE, breite / BREITE, hoehe / HOEHE]


def achse(fig: Figure, oben: float, hoehe: float,
          links: float = INHALT_LINKS, breite: float = INHALT_BREITE):
    """Achse pixelgenau setzen, gemessen von oben links.

    Auf einer festen Leinwand ist das verlaesslicher als ein Gridspec: jede
    Zeile bekommt genau den Platz, den ihre Beschriftung braucht.
    """
    return fig.add_axes(rechteck(oben, hoehe, links, breite))


def abschnitt(fig: Figure, text: str, oben: float, links: float = INHALT_LINKS,
              fontsize: int = S_LABEL, farbe: str | None = None):
    """Kurze Ueberschrift ueber einer Achse - benennt, was zu sehen ist."""
    return fig.text(x(links), y(oben), text, fontsize=fontsize, fontweight="bold",
                    color=farbe or theme.TEXT, va="center", ha="left")


def hero(fig: Figure, text: str, oben: float, farbe: str | None = None,
         fontsize: int = S_HERO, links: float | None = None):
    """Die eine grosse Zahl, um die es in der Grafik geht."""
    return fig.text(x(links if links is not None else INHALT_LINKS), y(oben), text,
                    fontsize=fontsize, fontweight="bold",
                    color=farbe or theme.TEXT, va="center", ha="left")


def achse_aufraeumen(ax, x_ticks=None, y_ticks=None, rahmen: bool = False) -> None:
    """Achsen auf das Noetigste reduzieren.

    Voreinstellung ist: keine Beschriftung. Was erklaert wird, muss nicht
    beschriftet sein - und jede weggelassene Zahl macht die uebrigen lesbarer.
    """
    ax.tick_params(labelsize=S_TICK, length=6, width=1.4, pad=10)
    ax.set_xticks(x_ticks if x_ticks is not None else [])
    ax.set_yticks(y_ticks if y_ticks is not None else [])
    for seite, sichtbar in (("top", False), ("right", False),
                            ("left", rahmen or y_ticks is not None),
                            ("bottom", rahmen or x_ticks is not None)):
        ax.spines[seite].set_visible(sichtbar)
        if sichtbar:
            ax.spines[seite].set_color(theme.GRID)
            ax.spines[seite].set_linewidth(1.4)


# --- Grafiken -------------------------------------------------------------
def figur() -> Figure:
    """Leere Flaeche in exakt 1080 x 1920 Pixeln."""
    theme.anwenden("hochformat")
    return Figure(figsize=(BREITE / DPI, HOEHE / DPI), dpi=DPI, facecolor=theme.BG)


def kopf(fig: Figure, titel: str):
    """Nur der Titel. Kein Untertitel - was erklaert wird, gehoert nicht aufs Bild."""
    return fig.text(x(INHALT_LINKS), y(306), titel, fontsize=S_TITEL,
                    fontweight="bold", color=theme.TEXT, va="center", ha="left")


def fuss(fig: Figure, text: str = ""):
    """Kurzer Hinweis im leeren Bereich unten. Sparsam benutzen."""
    if not text:
        return None
    return fig.text(x(INHALT_LINKS), y(1600), text, fontsize=S_TICK,
                    color=theme.TEXT_SCHWACH, va="top", ha="left", linespacing=1.8)


def sicherheitszonen(fig: Figure) -> None:
    """Die von Shorts, Reels und TikTok ueberdeckten Raender einblenden."""
    zonen = (
        (0, 1 - SICHER_OBEN / HOEHE, 1, SICHER_OBEN / HOEHE, "Statusleiste"),
        (0, 0, 1, SICHER_UNTEN / HOEHE, "Bildunterschrift / Fortschritt"),
        (1 - SICHER_RECHTS / BREITE, SICHER_UNTEN / HOEHE,
         SICHER_RECHTS / BREITE, 1 - (SICHER_OBEN + SICHER_UNTEN) / HOEHE, "Knöpfe"),
    )
    for zx, zy, b, h, name in zonen:
        fig.add_artist(
            Rectangle((zx, zy), b, h, transform=fig.transFigure,
                      facecolor=theme.FEHLER, alpha=0.16, zorder=50,
                      edgecolor=theme.FEHLER, linewidth=1.0)
        )
        fig.text(zx + b / 2, zy + h / 2, name, color=theme.FEHLER, fontsize=15,
                 ha="center", va="center", zorder=51, alpha=0.9)


def exportiere(fig: Figure, name: str, unterordner: str = "") -> Path:
    """Als PNG in exakt 1080 x 1920 ablegen."""
    ziel = AUSGABE / unterordner if unterordner else AUSGABE
    ziel.mkdir(parents=True, exist_ok=True)
    pfad = ziel / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.png"
    # Ein verkleinertes Live-Fenster rundet die Figur auf ganze Bildschirm-
    # pixel - dann kaemen ein, zwei Pixel zu wenig heraus. Deshalb fuer das
    # Speichern die Sollgroesse setzen und danach die alte zurueckgeben.
    alt = fig.get_size_inches().copy()
    fig.set_size_inches(BREITE / DPI, HOEHE / DPI, forward=False)
    try:
        # Kein bbox_inches="tight": das wuerde die Masse wieder veraendern.
        fig.savefig(pfad, dpi=DPI, facecolor=fig.get_facecolor())
    finally:
        fig.set_size_inches(alt, forward=False)
    return pfad


# --- Fenster --------------------------------------------------------------
if wintypes is not None:
    class _MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
        ]

# Was ausserhalb von Windows fuer Menueleiste, Dock oder Panel abgezogen wird.
# Tk kennt dort nur die ganze Bildschirmgroesse, nicht die Arbeitsflaeche.
# macOS: Menueleiste und Dock zusammen oft um 100 px, Linux: ein Panel.
RAND_OHNE_ARBEITSFLAECHE = 100 if sys.platform == "darwin" else 60


def dpi_bewusst() -> None:
    """Windows-Skalierung abschalten, damit ein Pixel ein Pixel bleibt.

    Ohne das wuerde bei 125 % Skalierung ein 1080 Pixel breites Fenster
    tatsaechlich 1350 Pixel belegen - und die Aufnahme waere nicht 9:16.
    """
    for aufruf in (
        lambda: ctypes.windll.shcore.SetProcessDpiAwareness(1),
        lambda: ctypes.windll.user32.SetProcessDPIAware(),
    ):
        try:
            aufruf()
            return
        except Exception:  # noqa: BLE001, PERF203
            continue


def arbeitsflaechen() -> list[tuple[int, int, int, int]]:
    """Nutzbare Flaeche je Monitor als (x, y, breite, hoehe).

    Unter Windows exakt je Monitor. Laesst sich das nicht abfragen (macOS,
    Linux), gilt der Bildschirm, den Tk kennt, abzueglich eines Randes -
    sonst wuerde jedes Fenster in voller 1080 x 1920 geoeffnet und liefe auf
    einem Querformat-Bildschirm unten heraus.
    """
    gefunden = _arbeitsflaechen_windows()
    return gefunden or _bildschirm_tk()


def _bildschirm_tk() -> list[tuple[int, int, int, int]]:
    """Bildschirmgroesse ueber Tk - Rueckfall ausserhalb von Windows.

    Gibt es schon ein Tk-Fenster, wird es benutzt, sonst kurz ein
    unsichtbares angelegt (wie in bedienung.pixelfaktor).
    """
    import tkinter as tk

    wurzel = getattr(tk, "_default_root", None)
    eigene = wurzel is None
    try:
        if eigene:
            wurzel = tk.Tk()
            wurzel.withdraw()
        b, h = int(wurzel.winfo_screenwidth()), int(wurzel.winfo_screenheight())
    except Exception:  # noqa: BLE001 - ohne Fenstersystem bleibt nichts
        return []
    finally:
        if eigene and wurzel is not None:
            try:
                wurzel.destroy()
            except Exception:  # noqa: BLE001, S110
                pass
    if b <= 0 or h <= RAND_OHNE_ARBEITSFLAECHE:
        return []
    return [(0, 0, b, h - RAND_OHNE_ARBEITSFLAECHE)]


def _arbeitsflaechen_windows() -> list[tuple[int, int, int, int]]:
    """Arbeitsflaeche je Monitor ueber die Windows-API, sonst leer."""
    gefunden: list[tuple[int, int, int, int]] = []
    if wintypes is None or not hasattr(ctypes, "windll"):
        return gefunden
    try:
        user32 = ctypes.windll.user32
        rueckruf = ctypes.WINFUNCTYPE(
            ctypes.c_int, wintypes.HANDLE, wintypes.HDC,
            ctypes.POINTER(wintypes.RECT), wintypes.LPARAM,
        )

        def sammle(hmonitor, _hdc, _rect, _daten):  # noqa: ANN001
            info = _MONITORINFO()
            info.cbSize = ctypes.sizeof(_MONITORINFO)
            if user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
                w = info.rcWork
                gefunden.append((w.left, w.top, w.right - w.left, w.bottom - w.top))
            return 1

        user32.EnumDisplayMonitors(0, 0, rueckruf(sammle), 0)
    except Exception:  # noqa: BLE001
        return []
    return gefunden


def beste_flaeche(rand: int = 24) -> tuple[int, int, int, int] | None:
    """Arbeitsflaeche des Monitors, auf den ein 9:16-Fenster am besten passt.

    Bevorzugt wird ein Monitor, auf dem echte 1080 x 1920 Pixel gehen, danach
    einer im Hochformat. None, wenn sich die Monitore nicht abfragen lassen.
    """
    flaechen = arbeitsflaechen()
    if not flaechen:
        return None

    def bewertung(f: tuple[int, int, int, int]) -> tuple[int, int]:
        _, _, b, h = f
        passt_nativ = b >= BREITE and h >= HOEHE + rand
        hochformat = h > b
        return (not passt_nativ, not hochformat)

    return sorted(flaechen, key=bewertung)[0]


def fensterplatz(rand: int = 24) -> tuple[int, int, int, int, float]:
    """Groesse und Position eines 9:16-Fensters.

    Gibt (breite, hoehe, x, y, skala) zurueck. skala ist 1.0, wenn echte
    1080 x 1920 Pixel moeglich sind, sonst der Verkleinerungsfaktor, mit dem
    Schriften und Abstaende mitwachsen muessen.
    """
    flaeche = beste_flaeche(rand)
    if flaeche is None:
        return BREITE, HOEHE, 40, 20, 1.0

    x0, y0, b, h = flaeche
    if b >= BREITE and h >= HOEHE + rand:
        breite, hoehe, skala = BREITE, HOEHE, 1.0
    else:
        hoehe = h - rand
        breite = int(round(hoehe * SEITENVERHAELTNIS))
        if breite > b - rand:
            breite = b - rand
            hoehe = int(round(breite / SEITENVERHAELTNIS))
        skala = hoehe / HOEHE
    x = x0 + (b - breite) // 2
    y = y0 + max((h - hoehe) // 2, 0)
    return breite, hoehe, x, y, skala


def fenster_einrichten(fenster, rand: int = 24) -> float:
    """Ein tkinter-Fenster auf 9:16 setzen und positionieren. Gibt skala zurueck."""
    breite, hoehe, x, y, skala = fensterplatz(rand)
    fenster.geometry(f"{breite}x{hoehe}+{x}+{y}")
    fenster.minsize(int(BREITE * 0.4), int(HOEHE * 0.4))
    return skala

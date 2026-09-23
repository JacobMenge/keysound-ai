"""Animierte Fassungen der Video-Grafiken - 1080 x 1920, MP4 oder Bildfolge.

Eine Szene ist eine Funktion, die zu einem Zeitpunkt t (0..1) eine fertige
Figure liefert. Dieselben Zeichenfunktionen wie fuer die Standbilder, nur mit
einem Fortschrittswert - so kann nichts auseinanderlaufen.

Ausgabe:
  * MP4 ueber ffmpeg, H.264, 1080 x 1920 - direkt in den Schnitt ziehbar
  * PNG-Bildfolge, falls im Schnittprogramm lieber Einzelbilder liegen
  * Live-Fenster zum Mitschneiden, wenn es in Echtzeit laufen soll

Die Bewegung ist bewusst ruhig: Schritte blenden nacheinander auf, Kurven
zeichnen sich, Balken laufen hoch. Nichts fliegt, nichts blinkt - es soll
erklaeren, nicht ablenken.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib.figure import Figure

from . import portrait, theme
from .config import AUSGABE

FPS = 30


# --- Zeitverlaeufe --------------------------------------------------------
def weich(t: float) -> float:
    """Sanft an- und abschwellen statt linear - sieht im Video ruhiger aus."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def phase(t: float, start: float, dauer: float) -> float:
    """Anteil eines Abschnitts, der zum Zeitpunkt t schon gelaufen ist."""
    if dauer <= 0:
        return 1.0 if t >= start else 0.0
    return float(np.clip((t - start) / dauer, 0.0, 1.0))


def halten(t: float, ende: float = 0.88) -> float:
    """Bewegung vor Schluss beenden, damit das letzte Bild kurz stehen bleibt."""
    return float(np.clip(t / ende, 0.0, 1.0))


@dataclass
class Szene:
    """Eine animierte Grafik."""

    name: str
    dauer: float                       # Sekunden
    bild: Callable[[float], Figure]    # t in 0..1 -> fertige Figure
    beschreibung: str = ""
    # Bei einer einzelnen Szene soll das fertige Bild am Schluss kurz stehen
    # bleiben - halten() staucht die Bewegung dafuer auf die ersten 88 %.
    # Eine Montage aus mehreren Akten braucht das nicht: dort wuerde es die
    # Aktgrenzen verschieben und am Ende eine Standzeit einbauen.
    haltend: bool = True

    def frames(self, fps: int = FPS) -> int:
        return max(int(round(self.dauer * fps)), 2)

    def zeitpunkt(self, i: int, n: int) -> float:
        """t fuer das i-te von n Bildern."""
        roh = i / (n - 1) if n > 1 else 1.0
        return halten(roh) if self.haltend else float(np.clip(roh, 0.0, 1.0))


# --- Ausgabe --------------------------------------------------------------
def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def rendere_bildfolge(szene: Szene, ordner: Path, fps: int = FPS) -> int:
    """Alle Einzelbilder als PNG schreiben. Gibt die Anzahl zurueck."""
    ordner.mkdir(parents=True, exist_ok=True)
    n = szene.frames(fps)
    for i in range(n):
        fig = szene.bild(szene.zeitpunkt(i, n))
        fig.savefig(ordner / f"{i:05d}.png", dpi=portrait.DPI,
                    facecolor=fig.get_facecolor())
        fig.clear()
        del fig
    return n


def rendere_mp4(szene: Szene, pfad: Path, fps: int = FPS,
                bildfolge_behalten: Path | None = None) -> Path:
    """MP4 in 1080 x 1920 schreiben. Braucht ffmpeg im PATH."""
    exe = _ffmpeg()
    if exe is None:
        raise RuntimeError("ffmpeg nicht gefunden - bitte in den PATH legen.")
    pfad.parent.mkdir(parents=True, exist_ok=True)

    ziel = bildfolge_behalten
    tmp = None
    if ziel is None:
        tmp = tempfile.TemporaryDirectory()
        ziel = Path(tmp.name)
    try:
        rendere_bildfolge(szene, ziel, fps)
        befehl = [
            exe, "-y", "-loglevel", "error",
            "-framerate", str(fps), "-i", str(ziel / "%05d.png"),
            "-c:v", "libx264", "-preset", "slow", "-crf", "16",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(pfad),
        ]
        subprocess.run(befehl, check=True)
    finally:
        if tmp is not None:
            tmp.cleanup()
    return pfad


def zeige(szene: Szene, fps: int = FPS, schleife: bool = True) -> None:
    """Live-Fenster zum Mitschneiden - laeuft in Echtzeit."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    portrait.dpi_bewusst()
    n = szene.frames(fps)
    fig = plt.figure(figsize=(portrait.BREITE / portrait.DPI,
                              portrait.HOEHE / portrait.DPI),
                     dpi=portrait.DPI, facecolor=theme.BG)
    fig.canvas.manager.set_window_title(f"{szene.name} - Tastenakustik")
    try:
        breite, hoehe, px, py, _ = portrait.fensterplatz(rand=90)
        fig.canvas.manager.window.wm_geometry(f"+{px}+{py}")
    except Exception:  # noqa: BLE001
        pass

    # Jedes Bild wird einmal gerendert und dann als fertiges Pixelbild
    # angezeigt. Beim ersten Durchlauf baut sich der Zwischenspeicher auf,
    # danach laeuft die Schleife in Echtzeit - gut zum Mitschneiden.
    from matplotlib.animation import FuncAnimation

    zustand = {"i": 0}
    bild_cache: dict[int, np.ndarray] = {}

    def schritt_bild(_frame):  # noqa: ANN001
        i = zustand["i"]
        zustand["i"] = (i + 1) % n if schleife else min(i + 1, n - 1)
        if i not in bild_cache:
            t = szene.zeitpunkt(i, n)
            vorlage = szene.bild(t)
            vorlage.canvas.draw()
            bild_cache[i] = np.asarray(vorlage.canvas.buffer_rgba()).copy()
            vorlage.clear()
        fig.clear()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(bild_cache[i])
        ax.axis("off")
        return ()

    ani = FuncAnimation(fig, schritt_bild, interval=1000 / fps,
                        blit=False, cache_frame_data=False)
    fig._ani = ani  # Referenz halten
    plt.show()


def ablegen(szene: Szene, unterordner: str = "_animation", fps: int = FPS,
            frames: bool = False) -> Path:
    """Szene als MP4 in ausgabe ablegen, optional mit Bildfolge."""
    ziel = AUSGABE / unterordner
    ziel.mkdir(parents=True, exist_ok=True)
    folge = ziel / f"{szene.name}_frames" if frames else None
    return rendere_mp4(szene, ziel / f"{szene.name}.mp4", fps, folge)

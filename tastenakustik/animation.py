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
    """Alle Einzelbilder als PNG schreiben. Gibt die Anzahl zurueck.

    Nummerndateien eines frueheren Laufs werden vorher geloescht: Hatte der
    mehr Bilder, blieben die hoeheren Nummern sonst liegen, und Schnitt-
    programm wie ffmpeg haengten sie an. Andere Dateien im Ordner bleiben.
    """
    ordner.mkdir(parents=True, exist_ok=True)
    # Nur fuenf Ziffern wie "00042.png" - "?????" traefe auch eigene Dateien
    # wie "cover.png" oder "titel.png".
    for alt in ordner.glob("[0-9][0-9][0-9][0-9][0-9].png"):
        alt.unlink()
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
    """MP4 in 1080 x 1920 schreiben. Braucht ffmpeg im PATH.

    Scheitert ffmpeg - etwa ein Build ohne libx264 -, kommt ein RuntimeError
    mit dem Grund zurueck, wie wenn ffmpeg ganz fehlt. Der Aufrufer faellt
    dann auf die Bildfolge zurueck. Eine halb geschriebene MP4 wird entfernt.
    """
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
        n = rendere_bildfolge(szene, ziel, fps)
        # -frames:v begrenzt auf genau die gerenderten Bilder, auch wenn im
        # Ordner noch fremde Nummerndateien liegen sollten.
        befehl = [
            exe, "-y", "-loglevel", "error",
            "-framerate", str(fps), "-i", str(ziel / "%05d.png"),
            "-frames:v", str(n),
            "-c:v", "libx264", "-preset", "slow", "-crf", "16",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(pfad),
        ]
        try:
            subprocess.run(befehl, check=True, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        except subprocess.CalledProcessError as fehler:
            pfad.unlink(missing_ok=True)
            grund = (fehler.stderr or "").strip()[-300:] or f"Exit-Code {fehler.returncode}"
            raise RuntimeError(f"ffmpeg fehlgeschlagen: {grund}") from fehler
    finally:
        if tmp is not None:
            tmp.cleanup()
    return pfad


def zeige(szene: Szene, fps: int = FPS, schleife: bool = True) -> None:
    """Live-Fenster zum Mitschneiden - erst rendern, dann in Echtzeit.

    Matplotlib schafft ein Bild in 1080 x 1920 nicht in 33 ms, weder beim
    Neuzeichnen der Szene noch als imshow eines fertigen Pixelbilds. Deshalb
    wird jedes Bild vorab einmal gerendert (das Fenster zeigt so lange den
    Fortschritt) und in Fenstergroesse als PNG im Speicher abgelegt - roh
    waeren es 8 MB je Bild, bei einer langen Szene Gigabytes. Beim Abspielen
    wird nur noch ein fertiges Tk-Bild getauscht. Die Bildnummer folgt der Uhr:
    Kommt ein Bild zu spaet, wird es uebersprungen, damit die Szene ihre echte
    Dauer behaelt.

    Die Fenstergroesse kommt wie bei den Live-Fenstern aus LiveMasse: echte
    1080 x 1920 Pixel, wenn sie auf einen Monitor passen, sonst verkleinert.
    """
    import io
    import time
    import tkinter as tk

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image, ImageTk

    from . import bedienung

    portrait.dpi_bewusst()
    masse = bedienung.LiveMasse()
    breite = round(portrait.BREITE * masse.skala)
    hoehe = round(portrait.HOEHE * masse.skala)
    n = szene.frames(fps)

    fenster = tk.Tk()
    fenster.title(f"{szene.name} - Tastenakustik")
    fenster.configure(background=theme.BG)
    fenster.resizable(False, False)
    # Ein leeres Bild in voller Groesse legt die Fenstergroesse fest, bevor
    # das erste gerenderte Bild da ist. Der Fortschritt steht darueber.
    leer = tk.PhotoImage(width=breite, height=hoehe)
    flaeche = tk.Label(fenster, image=leer, compound="center", bg=theme.BG,
                       fg=theme.TEXT_SCHWACH, font=("Segoe UI", 18),
                       bd=0, highlightthickness=0, padx=0, pady=0)
    flaeche.image = leer
    flaeche.pack()
    masse.platzieren(fenster)

    bilder: list[bytes] = []
    zustand = {"zuletzt": -1}

    def rendere(i: int = 0) -> None:
        if i >= n:
            abspielen(time.perf_counter())
            return
        flaeche.configure(text=f"rendere Bild {i + 1}/{n}")
        vorlage = szene.bild(szene.zeitpunkt(i, n))
        leinwand = FigureCanvasAgg(vorlage)
        leinwand.draw()
        bild = Image.fromarray(np.asarray(leinwand.buffer_rgba())[..., :3])
        if bild.size != (breite, hoehe):
            bild = bild.resize((breite, hoehe), Image.LANCZOS)
        puffer = io.BytesIO()
        bild.save(puffer, format="PNG", compress_level=3)
        bilder.append(puffer.getvalue())
        # Die Vorlage sofort freigeben - sonst haelt jede Szene ihre Figuren.
        vorlage.clear()
        del vorlage, leinwand
        fenster.after(1, rendere, i + 1)

    def abspielen(start: float) -> None:
        schritt = int((time.perf_counter() - start) * fps)
        i = schritt % n if schleife else min(schritt, n - 1)
        if i != zustand["zuletzt"]:
            bild = ImageTk.PhotoImage(Image.open(io.BytesIO(bilder[i])))
            flaeche.configure(image=bild, text="")
            flaeche.image = bild   # Referenz halten, sonst raeumt Tk es weg
            zustand["zuletzt"] = i
        if not schleife and schritt >= n - 1:
            return
        naechstes = start + (schritt + 1) / fps
        warten = max(int((naechstes - time.perf_counter()) * 1000), 1)
        fenster.after(warten, abspielen, start)

    fenster.after(1, rendere)
    fenster.mainloop()


def ablegen(szene: Szene, unterordner: str = "_animation", fps: int = FPS,
            frames: bool = False) -> Path:
    """Szene als MP4 in ausgabe ablegen, optional mit Bildfolge."""
    ziel = AUSGABE / unterordner
    ziel.mkdir(parents=True, exist_ok=True)
    folge = ziel / f"{szene.name}_frames" if frames else None
    return rendere_mp4(szene, ziel / f"{szene.name}.mp4", fps, folge)

"""Kuenstliche Anschlaege - fuer Layout-Vorschau, Animationen und Selbsttest.

Hier wird nichts gemessen. Diese Signale sehen einem Tastenanschlag nur
aehnlich genug, um Grafiken zu bauen, solange noch keine echten Aufnahmen
vorliegen. Jede Grafik, die damit entsteht, traegt den Stempel
"BEISPIELDATEN" - sonst waere sie eine Behauptung ohne Messung.

Die Resonanzen leiten sich aus dem Platz der Klasse in der Liste ab, damit das
auch fuer ein ganzes Alphabet funktioniert und nicht nur fuer acht Zeichen.
"""

from __future__ import annotations

import numpy as np

from .config import Config, TASTEN


def profil(taste: str) -> list[tuple[float, float, float]]:
    """Zwei gedaempfte Resonanzen (Frequenz, Amplitude, Abklingzeit).

    Der Abstand zwischen benachbarten Klassen bleibt gleich gross, egal wie
    viele es gibt - die erste und die letzte Klasse liegen immer bei rund
    1,9 und 3,6 kHz.
    """
    n = max(len(TASTEN), 2)
    i = TASTEN.index(taste) if taste in TASTEN else 0
    anteil = i / (n - 1)
    grund = 1900 + anteil * 1700
    ober = 4700 + ((i * 7) % n) / n * 3100
    return [(grund, 0.9 - anteil * 0.3, 0.016 - anteil * 0.008),
            (ober, 0.3 + anteil * 0.3, 0.007 - anteil * 0.003)]


def anschlag_signal(sr: int, taste: str, amp: float = 0.32,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Ein einzelner kuenstlicher Anschlag: Knacken plus zwei Resonanzen."""
    rng = rng if rng is not None else np.random.default_rng(0)
    n = int(0.09 * sr)
    t = np.arange(n) / sr
    x = rng.normal(0, 1, n) * np.exp(-t / 0.004)
    for f, a, tau in profil(taste):
        x += a * np.sin(2 * np.pi * f * t) * np.exp(-t / tau)
    x *= np.exp(-t / 0.02)
    return (x / (np.max(np.abs(x)) + 1e-9) * amp).astype(np.float32)


def fenster(cfg: Config, taste: str, versatz_ms: float = 7.0,
            rng: np.random.Generator | None = None) -> tuple[np.ndarray, int]:
    """Ein Kontextfenster wie aus dem Collector. Gibt (welle, onset) zurueck."""
    rng = rng if rng is not None else np.random.default_rng(0)
    n = cfg.fenster_samples
    x = rng.normal(0, 0.0015, n).astype(np.float32)
    p = anschlag_signal(cfg.samplerate, taste, rng=rng)
    i = int((cfg.pre_roll_ms + versatz_ms) / 1000 * cfg.samplerate)
    x[i:i + p.size] += p[: max(n - i, 0)]
    return x, i


def wahrscheinlichkeiten(ziel: str, sicherheit: float,
                         rng: np.random.Generator | None = None
                         ) -> dict[str, float]:
    """Eine plausible Softmax-Ausgabe: ein Gewinner, der Rest ungleich verteilt."""
    rng = rng if rng is not None else np.random.default_rng(0)
    rest = rng.dirichlet(np.full(len(TASTEN), 0.3))
    if ziel in TASTEN:
        rest[TASTEN.index(ziel)] = 0.0
    rest = rest / max(rest.sum(), 1e-9) * (1 - sicherheit)
    p = {t: float(v) for t, v in zip(TASTEN, rest)}
    if ziel in p:
        p[ziel] = sicherheit
    return p


def beispiel_sequenz(laenge: int = 8) -> str:
    """Eine kurze Zeichenfolge aus den aktuellen Klassen - nur zur Anzeige."""
    if not TASTEN:
        return ""
    return "".join(TASTEN[(i * 3) % len(TASTEN)] for i in range(laenge))

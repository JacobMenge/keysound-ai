"""Spektraldarstellung: Kurzzeit-Spektrum und Log-Mel.

Live-Ansicht, Video-Grafiken und ein spaeteres Modell rechnen ueber dieselben
Funktionen, damit im Video nicht eine andere Darstellung zu sehen ist als die,
auf der trainiert wurde.
"""

from __future__ import annotations

import numpy as np


def rahmen(x: np.ndarray, nfft: int, hop: int) -> np.ndarray:
    """Signal in ueberlappende, gefensterte Rahmen zerlegen."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < nfft:
        x = np.pad(x, (0, nfft - x.size))
    n = max(1 + (x.size - nfft) // hop, 1)
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    return x[idx] * np.hanning(nfft)


def stft_leistung(x: np.ndarray, nfft: int = 512, hop: int = 128) -> np.ndarray:
    """Leistungsspektrum, Form (frames, bins)."""
    return np.abs(np.fft.rfft(rahmen(x, nfft, hop), axis=1)) ** 2


def stft_db(x: np.ndarray, nfft: int = 512, hop: int = 128,
            boden_db: float = -110.0) -> np.ndarray:
    """Kurzzeit-Spektrum in dB, Form (frames, bins).

    Bewusst mit numpy statt scipy.signal.spectrogram: wenige Zeilen, stabile
    API und exakt dieselbe Rechnung ueberall.
    """
    db = 10.0 * np.log10(np.maximum(stft_leistung(x, nfft, hop), 1e-20))
    return np.maximum(db, boden_db)


# --- Mel ------------------------------------------------------------------
def hz_zu_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_zu_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, nfft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Dreiecksfilter auf der Mel-Skala, Form (n_mels, bins)."""
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    kanten = mel_zu_hz(np.linspace(hz_zu_mel(fmin), hz_zu_mel(fmax), n_mels + 2))
    bank = np.zeros((n_mels, f.size))
    for i in range(n_mels):
        links, mitte, rechts = kanten[i], kanten[i + 1], kanten[i + 2]
        auf = (f - links) / max(mitte - links, 1e-9)
        ab = (rechts - f) / max(rechts - mitte, 1e-9)
        bank[i] = np.clip(np.minimum(auf, ab), 0.0, None)
    # Flaechennormierung, damit breite Filter oben nicht dominieren
    bank *= (2.0 / np.maximum(kanten[2:] - kanten[:-2], 1e-9))[:, None]
    return bank


def log_mel(x: np.ndarray, sr: int, nfft: int = 512, hop: int = 64,
            n_mels: int = 64, fmin: float = 100.0, fmax: float | None = None,
            boden_db: float = -100.0) -> np.ndarray:
    """Log-Mel-Spektrogramm in dB, Form (frames, n_mels)."""
    fmax = fmax if fmax is not None else sr / 2
    leistung = stft_leistung(x, nfft, hop)
    mel = leistung @ mel_filterbank(sr, nfft, n_mels, fmin, fmax).T
    db = 10.0 * np.log10(np.maximum(mel, 1e-20))
    return np.maximum(db, db.max() + boden_db if db.size else boden_db)


def min_max_huellkurve(x: np.ndarray, spalten: int) -> tuple[np.ndarray, np.ndarray]:
    """Wellenform auf Bildschirmbreite eindampfen, ohne dass man es sieht.

    Ein 4-Sekunden-Fenster hat bei 48 kHz 192 000 Punkte, die Achse ist aber
    nur rund 900 Pixel breit. Matplotlib rastert trotzdem jeden Punkt - das
    kostet den Grossteil der Zeit je Bild. Stattdessen wird je Bildspalte nur
    das Minimum und das Maximum gezeichnet. Das Ergebnis sieht identisch aus,
    weil genau diese beiden Werte die Spalte ohnehin ausfuellen.

    Gibt (sample_index, wert) zurueck - der Index laesst sich wie zuvor auf
    die Zeitachse abbilden.
    """
    n = x.size
    if n <= spalten * 2 or spalten < 2:
        return np.arange(n, dtype=np.float64), x
    pro = n // spalten
    block = x[: pro * spalten].reshape(spalten, pro)
    idx = np.repeat((np.arange(spalten) + 0.5) * pro, 2)
    werte = np.empty(spalten * 2, dtype=np.float64)
    werte[0::2] = block.min(axis=1)
    werte[1::2] = block.max(axis=1)
    return idx, werte


def mel_mittenfrequenzen(sr: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Mittenfrequenzen der Mel-Baender - fuer die Achsenbeschriftung."""
    return mel_zu_hz(np.linspace(hz_zu_mel(fmin), hz_zu_mel(fmax), n_mels + 2))[1:-1]


def frequenzen(nfft: int, sr: int) -> np.ndarray:
    return np.fft.rfftfreq(nfft, 1.0 / sr)


def bis_frequenz(db: np.ndarray, sr: int, nfft: int, max_hz: int) -> tuple[np.ndarray, float]:
    """Spektrum auf max_hz beschneiden. Gibt (beschnitten, tatsaechliches_max) zurueck."""
    f = frequenzen(nfft, sr)
    n = int(np.searchsorted(f, max_hz)) + 1
    n = min(n, db.shape[1])
    return db[:, :n], float(f[n - 1])

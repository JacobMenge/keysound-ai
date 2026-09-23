"""Segmentierung: wo genau im Kontextfenster sitzt der Anschlag?

Der Zeitstempel des Tastatur-Events ist nur auf wenige Millisekunden genau
(Event-Dispatch + Geraetelatenz). Deshalb wird grosszuegig gefenstert und der
tatsaechliche Anschlag anschliessend akustisch nachgemessen. Das liefert
gleichzeitig die Qualitaetskennzahlen, mit denen schlechte Proben aussortiert
werden - und die Markierung fuer die Video-Ansicht "markierter Tastenschlag".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from .config import Config


@dataclass
class Anschlag:
    """Ergebnis der Analyse eines Kontextfensters."""

    gefunden: bool
    onset_sample: int          # Beginn des Anschlags im Fenster
    peak_sample: int           # lautester Punkt
    peak_dbfs: float
    rausch_dbfs: float
    snr_db: float
    clipping: bool
    mehrfach: bool             # mehr als ein deutlich getrennter Anschlag
    grund: str                 # "ok" oder Ablehnungsgrund

    @property
    def brauchbar(self) -> bool:
        return self.gefunden and self.grund == "ok"

    def as_dict(self) -> dict:
        return asdict(self)


def hochpass(x: np.ndarray, sr: int, fc: int) -> np.ndarray:
    """Raumdroehnen, Lueftergrundrauschen und Netzbrumm wegnehmen."""
    if x.size < 32:
        return x.astype(np.float64)
    fc = min(fc, int(sr * 0.45))
    sos = butter(4, fc / (sr / 2.0), btype="highpass", output="sos")
    return sosfiltfilt(sos, x.astype(np.float64))


def huellkurve(x: np.ndarray, sr: int, fenster_ms: float, hop_ms: float
               ) -> tuple[np.ndarray, int, int]:
    """Kurzzeit-Effektivwert als Huellkurve. Gibt (env, fenster, hop) zurueck."""
    w = max(int(round(fenster_ms / 1000.0 * sr)), 4)
    h = max(int(round(hop_ms / 1000.0 * sr)), 1)
    if x.size < w:
        return np.zeros(1), w, h
    kum = np.concatenate(([0.0], np.cumsum(np.square(x, dtype=np.float64))))
    start = np.arange(0, x.size - w + 1, h)
    env = np.sqrt(np.maximum((kum[start + w] - kum[start]) / w, 0.0))
    return env, w, h


def analysiere(fenster: np.ndarray, cfg: Config) -> Anschlag:
    """Kontextfenster auf genau einen sauberen Anschlag pruefen."""
    leer = Anschlag(False, 0, 0, -120.0, -120.0, 0.0, False, False, "leer")
    if fenster is None or fenster.size < int(0.02 * cfg.samplerate):
        return leer

    sr = cfg.samplerate
    clipping = bool(np.max(np.abs(fenster)) >= cfg.clip_schwelle)

    gefiltert = hochpass(fenster, sr, cfg.hochpass_hz)
    env, w, h = huellkurve(gefiltert, sr, cfg.huellkurve_fenster_ms, cfg.huellkurve_hop_ms)
    if env.size < 4:
        return leer

    env_db = 20.0 * np.log10(np.maximum(env, 1e-9))
    rausch_db = float(np.percentile(env_db, 20))
    i_peak = int(np.argmax(env_db))
    peak_db = float(env_db[i_peak])
    snr = peak_db - rausch_db

    def sample_von(i: int) -> int:
        return int(min(max(i * h + w // 2, 0), fenster.size - 1))

    # Absoluter Spitzenpegel im Originalsignal (fuer Aussteuerungs-Feedback).
    peak_abs_dbfs = float(20.0 * np.log10(max(float(np.max(np.abs(fenster))), 1e-9)))

    if snr < cfg.min_snr_db:
        return Anschlag(False, 0, sample_von(i_peak), peak_abs_dbfs, rausch_db,
                        snr, clipping, False, "zu_leise")
    if peak_abs_dbfs < cfg.min_peak_dbfs:
        return Anschlag(False, 0, sample_von(i_peak), peak_abs_dbfs, rausch_db,
                        snr, clipping, False, "zu_leise")

    # Onset: vom Maximum rueckwaerts bis knapp ueber den Rauschboden.
    schwelle_back = rausch_db + cfg.onset_rueckverfolgung_db
    i_on = i_peak
    while i_on > 0 and env_db[i_on - 1] > schwelle_back:
        i_on -= 1

    # Weitere, deutlich getrennte Anschlaege im selben Fenster?
    min_abstand = max(int(round(cfg.mehrfach_abstand_ms / 1000.0 * sr / h)), 1)
    spitzen, _ = find_peaks(env_db, height=rausch_db + cfg.onset_schwelle_db,
                            distance=min_abstand)
    mehrfach = bool(np.sum(env_db[spitzen] > peak_db - 6.0) > 1) if spitzen.size else False

    grund = "ok"
    if clipping:
        grund = "uebersteuert"
    elif mehrfach:
        grund = "mehrfach"

    return Anschlag(
        gefunden=True,
        onset_sample=sample_von(i_on),
        peak_sample=sample_von(i_peak),
        peak_dbfs=peak_abs_dbfs,
        rausch_dbfs=rausch_db,
        snr_db=snr,
        clipping=clipping,
        mehrfach=mehrfach,
        grund=grund,
    )


def finde_transienten(x: np.ndarray, cfg: Config, min_abstand_ms: float = 150.0
                      ) -> tuple[np.ndarray, float]:
    """Alle Transienten in einem laengeren Ausschnitt - fuer die Live-Ansicht.

    Gibt (sample_indizes, rauschboden_db) zurueck. Rein zur Darstellung; die
    Bewertung einzelner Proben macht analysiere().
    """
    sr = cfg.samplerate
    if x.size < int(0.05 * sr):
        return np.zeros(0, dtype=int), -120.0
    gefiltert = hochpass(x, sr, cfg.hochpass_hz)
    env, w, h = huellkurve(gefiltert, sr, cfg.huellkurve_fenster_ms, cfg.huellkurve_hop_ms)
    if env.size < 4:
        return np.zeros(0, dtype=int), -120.0
    env_db = 20.0 * np.log10(np.maximum(env, 1e-9))
    rausch_db = float(np.percentile(env_db, 20))
    abstand = max(int(round(min_abstand_ms / 1000.0 * sr / h)), 1)
    spitzen, _ = find_peaks(env_db, height=rausch_db + cfg.onset_schwelle_db, distance=abstand)
    return (spitzen * h + w // 2).astype(int), rausch_db


def onset_zu_spitze(x: np.ndarray, cfg: Config, spitze: int,
                    rueckblick_ms: float = 40.0, abfall_db: float = 40.0,
                    korrektur_ms: float = 12.0) -> int:
    """Von einem erkannten Maximum lokal zum Beginn des Anschlags zurueckgehen.

    analysiere() sucht den lautesten Punkt im ganzen Fenster und verfolgt von
    dort zurueck. Das geht schief, sobald mehrere Anschlaege im Fenster liegen
    - dann landet man beim falschen. Hier wird nur ein kurzes Stueck vor der
    uebergebenen Spitze betrachtet, und die Schwelle haengt an der Spitze
    selbst statt am Rauschboden. Damit stoert auch das Ausklingen des
    vorherigen Anschlags nicht.
    """
    sr = cfg.samplerate
    von = max(0, spitze - int(rueckblick_ms / 1000 * sr))
    bis = min(x.size, spitze + int(0.005 * sr))
    if bis - von < 64:
        return int(spitze)
    gefiltert = hochpass(x[von:bis], sr, cfg.hochpass_hz)
    env, w, h = huellkurve(gefiltert, sr, cfg.huellkurve_fenster_ms,
                           cfg.huellkurve_hop_ms)
    if env.size < 3:
        return int(spitze)
    env_db = 20.0 * np.log10(np.maximum(env, 1e-9))
    i_peak = int(np.argmax(env_db))
    schwelle = env_db[i_peak] - abfall_db
    i = i_peak
    while i > 0 and env_db[i - 1] > schwelle:
        i -= 1
    # analysiere() geht bis dicht an den Rauschboden zurueck und landet damit
    # rund 12 ms frueher. Dieser feste Ausgleich bringt beide Verfahren auf
    # denselben Punkt - gemessen an 120 aufgenommenen Anschlaegen betraegt der
    # Restversatz danach unter 4 ms.
    roh = von + i * h + w // 2 - int(korrektur_ms / 1000.0 * sr)
    return int(min(max(roh, 0), x.size - 1))


def segment_grenzen(anschlag: Anschlag, laenge: int, cfg: Config) -> tuple[int, int]:
    """Vorschlag fuer den spaeteren ML-Schnitt: feste Laenge ab kurz vor dem Onset.

    Wird beim Aufnehmen nur als Metadatum notiert - geschnitten wird erst in der
    Feature-Pipeline, damit wir die Fensterlaenge spaeter aendern koennen, ohne
    neu aufzunehmen.
    """
    sr = cfg.samplerate
    n = int(round(cfg.segment_ms / 1000.0 * sr))
    vor = int(round(cfg.segment_vor_onset_ms / 1000.0 * sr))
    start = max(0, min(anschlag.onset_sample - vor, laenge - n))
    return start, start + n


GRUND_TEXT = {
    "ok": "gespeichert",
    "leer": "kein Audio im Puffer",
    "zu_leise": "zu leise, zu wenig Abstand zum Rauschen",
    "uebersteuert": "übersteuert - Gain runter",
    "mehrfach": "mehrere Anschläge im Fenster",
    "kein_fenster": "Audiofenster nicht verfügbar",
}

"""Datensatz: von den aufgenommenen Sitzungen zu Log-Mel-Bildern.

Die Aufteilung laeuft ueber die Rolle der Sitzung, nie ueber einzelne Proben.
Alles andere waere Selbstbetrug: Proben derselben Aufnahme teilen Raum,
Mikrofonposition und Tagesform, und ein Modell erkennt das wieder. Die
Trefferquote saehe dann gut aus und wuerde auf neuen Daten einbrechen.

Zwei Entscheidungen, die hier fallen:

  Schnitt        Das Segment wird ab dem akustisch gemessenen Onset
                 geschnitten, nicht ab dem Tastatur-Ereignis. Dessen
                 Zeitstempel schwankt um mehrere Millisekunden.
  Normierung     Jede Probe wird auf eigenen Mittelwert und eigene Streuung
                 normiert. Damit faellt die Lautstaerke heraus - und mit ihr
                 die Frage, wie fest jemand gerade gedrueckt hat. Uebrig
                 bleibt die Klangfarbe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf

from . import features, storage
from .config import ROH, TASTEN, Config


@dataclass
class Datensatz:
    x: np.ndarray            # (n, mel, frames), float32
    y: np.ndarray            # (n,) int64, Index in TASTEN
    sitzungen: list[str]     # Herkunft je Probe
    rolle: str

    def __len__(self) -> int:
        return len(self.y)

    @property
    def je_klasse(self) -> dict[str, int]:
        return {t: int((self.y == i).sum()) for i, t in enumerate(TASTEN)}


def _segment(welle: np.ndarray, onset: int, cfg: Config,
             segment_ms: float, vor_ms: float) -> np.ndarray | None:
    n = int(segment_ms / 1000 * cfg.samplerate)
    vor = int(vor_ms / 1000 * cfg.samplerate)
    start = max(0, min(onset - vor, welle.size - n))
    stueck = welle[start:start + n]
    return stueck if stueck.size == n else None


def pruefe_passend(kopf: dict) -> None:
    """Passt die Sitzung zur aktuellen Klassenliste?

    Wer die Klassen wechselt, ohne die alten Aufnahmen beiseitezuraeumen, wuerde
    sonst stillschweigend einen Datensatz mischen, in dem einzelne Klassen nur
    aus einer einzigen Sitzung stammen. Das faellt beim Training nicht auf und
    macht jedes Ergebnis wertlos - deshalb hier ein klarer Abbruch.
    """
    alt = kopf.get("tasten")
    if alt and list(alt) != list(TASTEN):
        raise ValueError(
            f"Sitzung {kopf.get('session_id', '?')} wurde mit den Klassen "
            f"{''.join(alt)} aufgenommen, eingestellt sind {''.join(TASTEN)}. "
            "Entweder die Klassen zurueckstellen oder die alten Sitzungen aus "
            "daten/roh/ wegraeumen."
        )


def pruefe_labels(kopf: dict, proben: list[dict]) -> None:
    """Stecken in der Sitzung Zeichen, die gar keine eingestellte Klasse sind?

    Aeltere Sitzungen haben keine Klassenliste im Kopf - dann faellt ein
    Wechsel der Klassen erst an den Proben selbst auf. Auch das mit einer
    klaren Meldung statt eines Tracebacks tief im Training.
    """
    fremd = sorted({p["label"] for p in proben} - set(TASTEN))
    if fremd:
        raise ValueError(
            f"Sitzung {kopf.get('session_id', '?')} enthaelt die Zeichen "
            f"{''.join(fremd)}, eingestellt sind {''.join(TASTEN)}. "
            "Entweder die Klassen in Schritt 2 passend einstellen oder die "
            "alten Sitzungen aus daten/roh/ wegraeumen."
        )


def lade_sitzung(ordner, segment_ms: float, vor_ms: float
                 ) -> tuple[Config, np.ndarray, np.ndarray]:
    kopf, proben = storage.lade_sitzung(ordner)
    pruefe_passend(kopf)
    pruefe_labels(kopf, proben)
    cfg = Config(**{k: v for k, v in kopf["aufnahmeparameter"].items()
                    if k in Config.__dataclass_fields__})
    bilder, labels = [], []
    for p in proben:
        welle, _ = sf.read(ordner / p["datei"], dtype="float32")
        stueck = _segment(welle, p["onset_sample"], cfg, segment_ms, vor_ms)
        if stueck is None:
            continue
        mel = features.log_mel(stueck, cfg.samplerate, cfg.mel_nfft, cfg.mel_hop,
                               cfg.mel_baender, cfg.mel_fmin, cfg.mel_fmax)
        bilder.append(mel.T.astype(np.float32))      # (mel, frames)
        labels.append(TASTEN.index(p["label"]))
    if not bilder:
        return cfg, np.zeros((0, 1, 1), np.float32), np.zeros((0,), np.int64)
    return cfg, np.stack(bilder), np.array(labels, dtype=np.int64)


def normiere(x: np.ndarray) -> np.ndarray:
    """Je Probe auf Mittelwert 0 und Streuung 1 - nimmt die Lautstaerke heraus."""
    m = x.mean(axis=(1, 2), keepdims=True)
    s = x.std(axis=(1, 2), keepdims=True)
    return ((x - m) / (s + 1e-6)).astype(np.float32)


def lade(rolle: str, segment_ms: float = 250.0, vor_ms: float = 15.0
         ) -> tuple[Config | None, Datensatz]:
    """Alle Sitzungen einer Rolle zu einem Datensatz zusammenfassen."""
    teile_x, teile_y, herkunft = [], [], []
    cfg = None
    for ordner in storage.sitzungen():
        kopf, _ = storage.lade_sitzung(ordner)
        if kopf.get("rolle") != rolle:
            continue
        cfg, x, y = lade_sitzung(ordner, segment_ms, vor_ms)
        if len(y) == 0:
            continue
        teile_x.append(x)
        teile_y.append(y)
        herkunft += [kopf["session_id"]] * len(y)
    if not teile_x:
        return cfg, Datensatz(np.zeros((0, 1, 1), np.float32),
                              np.zeros((0,), np.int64), [], rolle)
    formen = sorted({t.shape[1:] for t in teile_x})
    if len(formen) > 1:
        # Passiert, wenn zwischen zwei Sitzungen das Mikrofon mit einer
        # anderen Abtastrate gewechselt wurde: Die Spektrogramme sind dann
        # unterschiedlich lang und lassen sich nicht stapeln.
        raise ValueError(
            f"Die Sitzungen mit der Rolle '{rolle}' passen nicht zusammen "
            f"(Spektrogramme {' und '.join('x'.join(map(str, f)) for f in formen)}). "
            "Meist wurde zwischendurch ein Mikrofon mit anderer Abtastrate "
            "gewählt. Alle Sitzungen einer Rolle mit demselben Eingang aufnehmen.")
    x = normiere(np.concatenate(teile_x))
    y = np.concatenate(teile_y)
    return cfg, Datensatz(x, y, herkunft, rolle)


def uebersicht() -> str:
    """Welche Sitzung hat welche Rolle - zum Nachschauen vor dem Training."""
    zeilen = []
    for ordner in storage.sitzungen():
        kopf, proben = storage.lade_sitzung(ordner)
        zeilen.append(f"  {kopf['session_id']:<22} {kopf.get('rolle', 'offen'):<7} "
                      f"{len(proben):>4} Proben   {kopf.get('notiz', '')[:28]}")
    return "\n".join(zeilen) if zeilen else f"  keine Sitzungen unter {ROH}"

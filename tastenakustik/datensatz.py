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

from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from . import features, storage
from .config import ROH, TASTEN, Config


@dataclass
class Datensatz:
    x: np.ndarray            # (n, mel, frames), float32
    y: np.ndarray            # (n,) int64, Index in klassen
    sitzungen: list[str]     # Herkunft je Probe
    rolle: str
    # Die Klassenliste zum Zeitpunkt des Ladens - y zeigt in genau diese
    # Liste, auch wenn TASTEN sich danach aendert.
    klassen: list[str] = field(default_factory=list)
    # Abtastrate und Mel-Parameter, mit denen die Bilder gerechnet wurden
    merkmale: dict | None = None
    # Sitzungen dieser Rolle ohne eine einzige Probe
    leer: list[str] = field(default_factory=list)
    # Sitzungen, die auf eine andere Abtastrate umgerechnet wurden: ID -> Rate
    umgerechnet: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def je_klasse(self) -> dict[str, int]:
        klassen = self.klassen or TASTEN
        return {t: int((self.y == i).sum()) for i, t in enumerate(klassen)}


def schnitt(segment_ms: float | None = None, vor_ms: float | None = None
            ) -> tuple[float, float]:
    """Segmentlaenge und Vorlauf - ohne Angabe aus der config.json.

    Eine Quelle fuer den Schnitt: Was dort als "Schnitt fuer das Modell"
    steht, ist auch das, womit trainiert wird. Ein fertiges Modell traegt
    seine eigenen Werte in der Modelldatei und wird mit diesen geprueft.
    """
    if segment_ms is None or vor_ms is None:
        cfg = Config.laden(anwenden=False)
        segment_ms = cfg.segment_ms if segment_ms is None else segment_ms
        vor_ms = cfg.segment_vor_onset_ms if vor_ms is None else vor_ms
    return float(segment_ms), float(vor_ms)


def merkmale_von(cfg: Config) -> dict:
    """Alles, was Form und Inhalt eines Mel-Bilds bestimmt.

    Gleiche Bildform heisst nicht gleiche Merkmale: Andere fmin/fmax ergeben
    dieselbe Form mit anderem Inhalt. Verglichen wird deshalb dieser ganze
    Satz, nicht nur die Form. Im selben Aufbau steht er in der Modelldatei.
    """
    fmax = None if cfg.mel_fmax is None else float(cfg.mel_fmax)
    return {"samplerate": int(cfg.samplerate),
            "mel": {"nfft": int(cfg.mel_nfft), "hop": int(cfg.mel_hop),
                    "baender": int(cfg.mel_baender),
                    "fmin": float(cfg.mel_fmin), "fmax": fmax}}


def beschreibe_merkmale(m: dict) -> str:
    """Kurzform fuer Meldungen, z. B. '48000 Hz, Mel 512/64/64, 100-16000 Hz'."""
    mel = m["mel"]
    fmax = "Nyquist" if mel["fmax"] is None else f"{mel['fmax']:g}"
    return (f"{m['samplerate']} Hz, Mel {mel['nfft']}/{mel['hop']}/"
            f"{mel['baender']}, {mel['fmin']:g}-{fmax} Hz")


def merkmale_der_rolle(rolle: str) -> dict | None:
    """Merkmalsparameter der ersten Sitzung einer Rolle, die Proben hat.

    Liest nur die Sitzungskoepfe, kein Audio - fuer aeltere Modelldateien,
    die ihre Parameter noch nicht selbst mitbringen.
    """
    for ordner in storage.sitzungen():
        kopf, proben = storage.lade_sitzung(ordner)
        if kopf.get("rolle") == rolle and proben:
            return merkmale_von(_sitzungs_cfg(kopf))
    return None


def _sitzungs_cfg(kopf: dict) -> Config:
    return Config(**{k: v for k, v in kopf["aufnahmeparameter"].items()
                     if k in Config.__dataclass_fields__})


def _segment(welle: np.ndarray, onset: int, samplerate: int,
             segment_ms: float, vor_ms: float) -> np.ndarray | None:
    n = int(segment_ms / 1000 * samplerate)
    vor = int(vor_ms / 1000 * samplerate)
    start = max(0, min(onset - vor, welle.size - n))
    stueck = welle[start:start + n]
    return stueck if stueck.size == n else None


def pruefe_passend(kopf: dict, klassen: list[str] | None = None) -> None:
    """Passt die Sitzung zur aktuellen Klassenliste?

    Wer die Klassen wechselt, ohne die alten Aufnahmen beiseitezuraeumen, wuerde
    sonst stillschweigend einen Datensatz mischen, in dem einzelne Klassen nur
    aus einer einzigen Sitzung stammen. Das faellt beim Training nicht auf und
    macht jedes Ergebnis wertlos - deshalb hier ein klarer Abbruch.
    """
    klassen = list(TASTEN) if klassen is None else klassen
    alt = kopf.get("tasten")
    if alt and list(alt) != list(klassen):
        raise ValueError(
            f"Sitzung {kopf.get('session_id', '?')} wurde mit den Klassen "
            f"{''.join(alt)} aufgenommen, eingestellt sind {''.join(klassen)}. "
            "Entweder die Klassen zurueckstellen oder die alten Sitzungen aus "
            "daten/roh/ wegraeumen."
        )


def pruefe_labels(kopf: dict, proben: list[dict],
                  klassen: list[str] | None = None) -> None:
    """Stecken in der Sitzung Zeichen, die gar keine eingestellte Klasse sind?

    Aeltere Sitzungen haben keine Klassenliste im Kopf - dann faellt ein
    Wechsel der Klassen erst an den Proben selbst auf. Auch das mit einer
    klaren Meldung statt eines Tracebacks tief im Training.
    """
    klassen = list(TASTEN) if klassen is None else klassen
    fremd = sorted({p["label"] for p in proben} - set(klassen))
    if fremd:
        raise ValueError(
            f"Sitzung {kopf.get('session_id', '?')} enthaelt die Zeichen "
            f"{''.join(fremd)}, eingestellt sind {''.join(klassen)}. "
            "Entweder die Klassen in Schritt 2 passend einstellen oder die "
            "alten Sitzungen aus daten/roh/ wegraeumen."
        )


def lade_sitzung(ordner, segment_ms: float, vor_ms: float,
                 klassen: list[str] | None = None, merkmale: dict | None = None,
                 ) -> tuple[Config, np.ndarray, np.ndarray]:
    """Eine Sitzung zu Mel-Bildern.

    merkmale: Abtastrate und Mel-Parameter, mit denen gerechnet wird - etwa
    die eines fertigen Modells. Ohne Angabe gelten die der Sitzung selbst.
    Weicht die Rate ab, wird das Audio vorher umgerechnet.
    """
    klassen = list(TASTEN) if klassen is None else klassen
    kopf, proben = storage.lade_sitzung(ordner)
    pruefe_passend(kopf, klassen)
    pruefe_labels(kopf, proben, klassen)
    cfg = _sitzungs_cfg(kopf)
    ziel = merkmale or merkmale_von(cfg)
    sr, mel_p = ziel["samplerate"], ziel["mel"]
    bilder, labels = [], []
    for p in proben:
        welle, _ = sf.read(ordner / p["datei"], dtype="float32")
        onset = p["onset_sample"]
        if sr != cfg.samplerate:
            welle = features.umrechnen(welle, cfg.samplerate, sr)
            onset = int(round(onset * sr / cfg.samplerate))
        stueck = _segment(welle, onset, sr, segment_ms, vor_ms)
        if stueck is None:
            continue
        mel = features.log_mel(stueck, sr, mel_p["nfft"], mel_p["hop"],
                               mel_p["baender"], mel_p["fmin"], mel_p["fmax"])
        bilder.append(mel.T.astype(np.float32))      # (mel, frames)
        labels.append(klassen.index(p["label"]))
    if proben and not bilder:
        # Alle Fenster einer Sitzung sind gleich lang - faellt jede Probe
        # heraus, ist das Segment laenger als das aufgenommene Fenster.
        fenster_ms = cfg.pre_roll_ms + cfg.post_roll_ms
        raise ValueError(
            f"Sitzung {kopf.get('session_id', '?')}: Ein Segment von "
            f"{segment_ms:g} ms passt nicht in die aufgenommenen Fenster von "
            f"{fenster_ms} ms. Ein kürzeres Segment wählen (segment_ms in "
            "config.json).")
    if not bilder:
        return cfg, np.zeros((0, 1, 1), np.float32), np.zeros((0,), np.int64)
    return cfg, np.stack(bilder), np.array(labels, dtype=np.int64)


def normiere(x: np.ndarray) -> np.ndarray:
    """Je Probe auf Mittelwert 0 und Streuung 1 - nimmt die Lautstaerke heraus."""
    m = x.mean(axis=(1, 2), keepdims=True)
    s = x.std(axis=(1, 2), keepdims=True)
    return ((x - m) / (s + 1e-6)).astype(np.float32)


def lade(rolle: str, segment_ms: float | None = None, vor_ms: float | None = None,
         klassen: list[str] | None = None, merkmale: dict | None = None,
         ) -> tuple[Config | None, Datensatz]:
    """Alle Sitzungen einer Rolle zu einem Datensatz zusammenfassen.

    segment_ms/vor_ms ohne Angabe aus der config.json. merkmale siehe
    lade_sitzung - ohne Angabe muessen alle Sitzungen der Rolle mit denselben
    Parametern aufgenommen sein.
    """
    segment_ms, vor_ms = schnitt(segment_ms, vor_ms)
    klassen = list(TASTEN) if klassen is None else list(klassen)
    # Eine Explorer-Kopie traegt dieselbe session_id wie das Original. Laege
    # die Kopie in einer anderen Rolle, stuende dieselbe Aufnahme zugleich in
    # Training und Test - genau das Leakage, das die Sitzungstrennung verhindert.
    doppelt = storage.doppelte_sitzungen()
    if doppelt:
        raise ValueError(
            "Dieselbe Sitzung liegt in mehreren Ordnern: "
            + "; ".join(f"{sid} ({', '.join(o)})" for sid, o in doppelt.items())
            + f". Erst die Kopien aus {storage.ROH} entfernen.")
    teile_x, teile_y, herkunft = [], [], []
    je_sitzung: dict[str, dict] = {}
    leer: list[str] = []
    umgerechnet: dict[str, int] = {}
    cfg = None
    for ordner in storage.sitzungen():
        kopf, _ = storage.lade_sitzung(ordner)
        if kopf.get("rolle") != rolle:
            continue
        cfg, x, y = lade_sitzung(ordner, segment_ms, vor_ms, klassen, merkmale)
        if len(y) == 0:
            leer.append(kopf["session_id"])
            continue
        eigene = merkmale_von(cfg)
        je_sitzung[kopf["session_id"]] = merkmale or eigene
        if merkmale and eigene["samplerate"] != merkmale["samplerate"]:
            umgerechnet[kopf["session_id"]] = eigene["samplerate"]
        teile_x.append(x)
        teile_y.append(y)
        herkunft += [kopf["session_id"]] * len(y)
    if not teile_x:
        return cfg, Datensatz(np.zeros((0, 1, 1), np.float32),
                              np.zeros((0,), np.int64), [], rolle, klassen,
                              merkmale, leer, umgerechnet)
    verschieden = {beschreibe_merkmale(m) for m in je_sitzung.values()}
    if len(verschieden) > 1:
        # Passiert, wenn zwischen zwei Sitzungen das Mikrofon mit einer
        # anderen Abtastrate gewechselt wurde. Verglichen werden die
        # Parameter, nicht nur die Bildform - andere fmin/fmax ergeben
        # dieselbe Form mit anderem Inhalt.
        liste = "; ".join(f"{s}: {beschreibe_merkmale(m)}"
                          for s, m in je_sitzung.items())
        raise ValueError(
            f"Die Sitzungen mit der Rolle '{rolle}' passen nicht zusammen "
            f"({liste}). Meist wurde zwischendurch ein Mikrofon mit anderer "
            "Abtastrate gewählt. Alle Sitzungen einer Rolle mit demselben "
            "Eingang aufnehmen.")
    x = normiere(np.concatenate(teile_x))
    y = np.concatenate(teile_y)
    return cfg, Datensatz(x, y, herkunft, rolle, klassen,
                          next(iter(je_sitzung.values())), leer, umgerechnet)


def uebersicht() -> str:
    """Welche Sitzung hat welche Rolle - zum Nachschauen vor dem Training."""
    zeilen = []
    for ordner in storage.sitzungen():
        kopf, proben = storage.lade_sitzung(ordner)
        zeilen.append(f"  {kopf['session_id']:<22} {kopf.get('rolle', 'offen'):<7} "
                      f"{len(proben):>4} Proben   {kopf.get('notiz', '')[:28]}")
    return "\n".join(zeilen) if zeilen else f"  keine Sitzungen unter {ROH}"

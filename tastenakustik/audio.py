"""Audio-Eingang: Geraetesuche und ein Ringpuffer fuer den Recording-Modus.

Der Ringpuffer haelt die letzten Sekunden Mikrofonsignal im Arbeitsspeicher.
Er laeuft ausschliesslich zwischen start() und stop(), also nur solange der
Recording-Modus aktiv ist, und schreibt von sich aus nichts auf die Platte.
Auf die Platte kommt nur das kurze Fenster um einen bestaetigten, vorher
angekuendigten Tastendruck (siehe storage.py).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from .config import Config

# Hostapis nach Eignung fuer dieses Experiment. ASIO und WDM-KS reichen das
# Signal am Effektpaket der Windows-Audio-Engine vorbei durch. WASAPI kann
# leise Transienten glaetten (siehe README) - Jacobs WASAPI-Sitzung verlor
# drei Viertel der Anschlaege, die WDM-KS-Sitzungen keinen.
HOSTAPI_RANG = {
    "ASIO": 0,
    "Windows WDM-KS": 0,
    "Windows WASAPI": 1,
    "Windows DirectSound": 2,
    "MME": 3,
}

# Geraete, die fuer eine akustische Messung nichts taugen: virtuelle Router,
# Loopbacks, Software-Mikrofone mit eigener Signalverarbeitung.
UNGEEIGNET = (
    "steam streaming",
    "stereomix",
    "stereo mix",
    "soundmapper",
    "soundaufnahmetreiber",
    "nvidia broadcast",
    "virtual",
    "midi",
)


@dataclass
class Geraet:
    index: int
    name: str
    hostapi: str
    kanaele: int
    samplerate: int
    latenz_ms: float
    geeignet: bool

    @property
    def label(self) -> str:
        return f"[{self.index}] {self.name} ({self.hostapi}, {self.latenz_ms:.0f} ms)"


def eingaenge() -> list[Geraet]:
    """Alle Eingangsgeraete, sortiert: geeignet zuerst, dann nach Latenz."""
    hostapis = [h["name"] for h in sd.query_hostapis()]
    gefunden: list[Geraet] = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        name = str(d["name"]).strip()
        api = hostapis[d["hostapi"]]
        klein = name.lower()
        gefunden.append(
            Geraet(
                index=i,
                name=name,
                hostapi=api,
                kanaele=int(d["max_input_channels"]),
                samplerate=int(d["default_samplerate"]),
                latenz_ms=float(d["default_low_input_latency"]) * 1000.0,
                geeignet=not any(u in klein for u in UNGEEIGNET),
            )
        )
    gefunden.sort(key=lambda g: (not g.geeignet, HOSTAPI_RANG.get(g.hostapi, 9), g.latenz_ms))
    return gefunden


def geraet_finden(suche: str | int | None) -> Geraet | None:
    """Geraet per Index oder per Namensfragment suchen."""
    alle = eingaenge()
    if not alle:
        return None
    if suche is None or suche == "":
        return alle[0]
    text = str(suche)
    if text.lstrip("-").isdigit():
        idx = int(text)
        return next((g for g in alle if g.index == idx), None)
    klein = text.lower()
    return next((g for g in alle if klein in g.name.lower()), None)


def neu_einlesen() -> None:
    """PortAudio neu starten, damit umgesteckte Geraete auftauchen.

    PortAudio liest die Geraeteliste nur einmal beim Start. Ein Prozess, der
    lange laeuft (das Studio), sieht ein neu angestecktes Headset sonst nie.
    Nur aufrufen, solange in diesem Prozess kein Stream offen ist.
    """
    sd._terminate()
    sd._initialize()


def geraet_aufloesen(cfg: Config) -> int | None:
    """Den aktuellen PortAudio-Index des gespeicherten Mikrofons finden.

    Die Indizes verschieben sich, sobald ein Audiogeraet dazukommt oder
    wegfaellt (USB-Headset, Webcam, Bluetooth). Massgeblich ist deshalb der
    gespeicherte Name samt Host-API; der Index ist nur die Vorauswahl.
    Ohne gespeicherten Namen (alte oder handgemachte Konfiguration) bleibt
    es beim Index. Ist das Mikrofon nicht mehr da oder nicht eindeutig,
    bricht das mit einer klaren Meldung ab, statt still ein anderes Geraet
    aufzunehmen.
    """
    name = (cfg.device_name or "").strip()
    if not name:
        return cfg.device
    try:
        alle = eingaenge()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Audio-Geraete nicht lesbar: {exc}") from exc

    def passt(g: Geraet) -> bool:
        return g.name == name and (not cfg.hostapi or g.hostapi == cfg.hostapi)

    if any(g.index == cfg.device and passt(g) for g in alle):
        return cfg.device
    treffer = [g for g in alle if passt(g)]
    if not treffer:
        raise RuntimeError(
            f"Mikrofon {name!r} ({cfg.hostapi or 'Host-API unbekannt'}) ist nicht "
            "mehr da - angesteckt? Sonst in Schritt 1 neu waehlen.")
    if cfg.hostapi and len(treffer) > 1:
        raise RuntimeError(
            f"Mikrofon {name!r} ({cfg.hostapi}) gibt es {len(treffer)}-mal - "
            "in Schritt 1 das richtige neu waehlen.")
    # Ohne gespeicherte Host-API: die beste nach HOSTAPI_RANG, wie eingaenge()
    # sie ohnehin zuerst listet.
    g = treffer[0]
    if g.index != cfg.device:
        print(f"Hinweis: {g.name} ({g.hostapi}) steht jetzt unter Index {g.index} "
              f"statt {cfg.device}.")
    return g.index


def bestes_format(geraet: Geraet, wunsch_sr: int) -> tuple[int, int]:
    """Funktionierende (samplerate, kanaele) fuer dieses Geraet ermitteln.

    Viele Interfaces melden sich unter WASAPI zweikanalig und lehnen einen
    Mono-Stream ab. Wir nehmen dann zwei Kanaele entgegen und verwenden
    spaeter Kanal 1 - das Ergebnis ist identisch, nur der Weg dorthin robuster.
    """
    kandidaten_sr = [wunsch_sr, geraet.samplerate, 48_000, 44_100]
    kandidaten_ch = [1, geraet.kanaele] if geraet.kanaele > 1 else [1]
    for sr in dict.fromkeys(kandidaten_sr):
        for ch in dict.fromkeys(kandidaten_ch):
            try:
                sd.check_input_settings(device=geraet.index, samplerate=sr,
                                        channels=ch, dtype="float32")
            except Exception:  # noqa: BLE001, PERF203
                continue
            return sr, ch
    return wunsch_sr, 1


def rms_dbfs(x: np.ndarray) -> float:
    """Effektivpegel in dBFS."""
    if x.size == 0:
        return -120.0
    wert = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return float(20.0 * np.log10(max(wert, 1e-9)))


def peak_dbfs(x: np.ndarray) -> float:
    """Spitzenpegel in dBFS."""
    if x.size == 0:
        return -120.0
    return float(20.0 * np.log10(max(float(np.max(np.abs(x))), 1e-9)))


class Ringpuffer:
    """Kontinuierlicher Mitlauf-Puffer im RAM waehrend des Recording-Modus."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.N = max(int(cfg.ring_sekunden * cfg.samplerate), cfg.blocksize * 8)
        self._buf = np.zeros(self.N, dtype=np.float32)
        self._pos = 0        # naechste Schreibposition im Ring
        self._total = 0      # insgesamt geschriebene Samples (monoton steigend)
        self._t_anker = 0.0  # perf_counter beim letzten Callback
        self._n_anker = 0    # Stand von _total zu diesem Zeitpunkt
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        # Aussetzer (Overflow) seit start() - als Zaehler, damit ein langer
        # Lauf keinen Speicher frisst. Der Collector vermerkt je Probe, wie
        # viele in ihre Zeit fielen.
        self.stoerungen_n = 0
        self.letzte_stoerung = ""
        self.latenz_s = cfg.latenz_ms / 1000.0

    # -- Lebenszyklus ---------------------------------------------------
    @property
    def laeuft(self) -> bool:
        return self._stream is not None and self._stream.active

    def start(self) -> None:
        if self.laeuft:
            return
        with self._lock:
            self._buf[:] = 0.0
            self._pos = 0
            self._total = 0
        self.stoerungen_n = 0
        self.letzte_stoerung = ""
        # Der Index kann seit der Wahl in Schritt 1 gewandert sein. Nur im
        # Speicher korrigieren - wer die Konfiguration speichert, entscheidet
        # das Studio, sonst schreiben Studio und Werkzeug gegeneinander.
        self.cfg.device = geraet_aufloesen(self.cfg)
        letzter_fehler: Exception | None = None
        for kanaele in dict.fromkeys([self.cfg.channels, 2]):
            stream = None
            try:
                stream = sd.InputStream(
                    samplerate=self.cfg.samplerate,
                    blocksize=self.cfg.blocksize,
                    device=self.cfg.device,
                    channels=kanaele,
                    dtype="float32",
                    latency="low",
                    callback=self._callback,
                )
                stream.start()
                self._stream = stream
                break
            except Exception as exc:  # noqa: BLE001, PERF203
                letzter_fehler = exc
                # Ein geoeffneter, aber nicht gestarteter Stream haelt das
                # Geraet fest - unter WDM-KS exklusiv, dann scheitert auch
                # der Versuch mit zwei Kanaelen und jeder weitere Start.
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:  # noqa: BLE001
                        pass
        if self._stream is None:
            raise RuntimeError(
                f"Audioeingang {self.cfg.device} ({self.cfg.device_name}) liess sich "
                f"nicht oeffnen: {letzter_fehler}"
            )
        self.latenz_s = float(self._stream.latency)

    def stop(self) -> None:
        """Stream anhalten und schliessen - wirft nie.

        Auf einem abgezogenen Geraet kann schon stream.stop() eine
        PortAudioError werfen. Wer gerade pausieren oder aufraeumen will,
        soll daran nicht mitten im Tk-Callback scheitern.
        """
        if self._stream is not None:
            stream, self._stream = self._stream, None
            try:
                stream.stop()
            except Exception:  # noqa: BLE001
                pass
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "Ringpuffer":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    # -- Callback (Audio-Thread) ----------------------------------------
    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            self.stoerungen_n += 1
            self.letzte_stoerung = str(status)
        x = indata[:, 0] if indata.ndim > 1 else indata
        n = x.shape[0]
        with self._lock:
            p, N = self._pos, self.N
            ende = p + n
            if ende <= N:
                self._buf[p:ende] = x
            else:
                k = N - p
                self._buf[p:] = x[:k]
                self._buf[: ende - N] = x[k:]
            self._pos = ende % N
            self._total += n
            self._n_anker = self._total
            self._t_anker = time.perf_counter()

    # -- Lesen ----------------------------------------------------------
    def _schneiden(self, von_abs: int, bis_abs: int, pos: int, total: int) -> np.ndarray:
        """Absoluten Sample-Bereich aus dem Ring holen (mit Umbruch).

        Bewusst mit Scheiben statt mit einer Index-Tabelle: Fuer ein
        Vier-Sekunden-Fenster muesste man sonst 192 000 Indizes erzeugen und
        modulo rechnen. Als Speicherblock kopiert ist dasselbe rund zehnmal
        schneller - und das faellt in der Live-Ansicht bei jedem Bild an.
        """
        laenge = bis_abs - von_abs
        if laenge <= 0:
            return np.zeros(0, dtype=np.float32)
        start = (pos - total + von_abs) % self.N
        if start + laenge <= self.N:
            return self._buf[start:start + laenge].copy()
        kopf = self.N - start
        return np.concatenate([self._buf[start:], self._buf[: laenge - kopf]])

    @property
    def gesamt(self) -> int:
        """Bisher insgesamt aufgenommene Samples - monoton steigend."""
        with self._lock:
            return self._total

    def letzte(self, sekunden: float) -> np.ndarray:
        """Die letzten n Sekunden - fuer Live-Ansichten."""
        with self._lock:
            total, pos = self._total, self._pos
            n = min(int(sekunden * self.cfg.samplerate), self.N, total)
            if n <= 0:
                return np.zeros(0, dtype=np.float32)
            return self._schneiden(total - n, total, pos, total)

    def fenster_um(self, t_perf: float, vor_s: float, nach_s: float) -> np.ndarray | None:
        """Fenster um einen Zeitpunkt der perf_counter-Uhr.

        None, wenn der Nachlauf noch nicht aufgenommen ist oder der Bereich
        bereits aus dem Ring gelaufen ist.
        """
        sr = self.cfg.samplerate
        with self._lock:
            total, pos = self._total, self._pos
            t_anker, n_anker = self._t_anker, self._n_anker
            if total == 0:
                return None
            # Das Sample mit Index n_anker wurde rund latenz_s vor t_anker erfasst.
            n_ereignis = n_anker + (t_perf - t_anker) * sr - self.latenz_s * sr
            von = int(round(n_ereignis - vor_s * sr))
            bis = int(round(n_ereignis + nach_s * sr))
            if bis > total or von < max(0, total - self.N) or von >= bis:
                return None
            return self._schneiden(von, bis, pos, total)

    def fenster_absolut(self, von_abs: int, laenge: int) -> np.ndarray | None:
        """Segment ueber absolute Sample-Indizes holen.

        Fuer die Live-Demo: Dort kommt die Position aus dem Audiosignal selbst,
        nicht aus einer Uhr - es gibt gar kein Tastatur-Ereignis, auf das man
        sich beziehen koennte.
        """
        with self._lock:
            total, pos = self._total, self._pos
            bis = von_abs + laenge
            if laenge <= 0 or bis > total or von_abs < max(0, total - self.N):
                return None
            return self._schneiden(von_abs, bis, pos, total)

    def fenster_um_taste(self, t_perf: float) -> np.ndarray | None:
        """Das konfigurierte Kontextfenster (pre_roll + post_roll)."""
        return self.fenster_um(
            t_perf, self.cfg.pre_roll_ms / 1000.0, self.cfg.post_roll_ms / 1000.0
        )

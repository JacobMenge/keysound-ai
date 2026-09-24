"""Zentrale Konfiguration - Klassen, Aufnahmeparameter, Verzeichnisse.

Die Klassenliste ist frei waehlbar: acht Tasten, das ganze Alphabet, Ziffern,
Satzzeichen. Sie steht in der config.json und wird beim Start geladen.

Sie ist gleichzeitig die inhaltliche Grenze des Programms. Der Collector nimmt
nichts ausserhalb dieser Liste entgegen, und ein trainiertes Modell kann per
Konstruktion nichts anderes ausgeben - die Ausgabeschicht hat genau so viele
Neuronen wie es Klassen gibt.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# ---------------------------------------------------------------------------
# Verzeichnisse
# ---------------------------------------------------------------------------
PROJEKT = Path(__file__).resolve().parents[1]
DATEN = PROJEKT / "daten"
ROH = DATEN / "roh"
MODELLE = DATEN / "modelle"
AUSGABE = PROJEKT / "ausgabe"
CONFIG_PFAD = PROJEKT / "config.json"

# ---------------------------------------------------------------------------
# Die Klassen
#
# TASTEN ist bewusst eine Liste, die an Ort und Stelle veraendert wird: Andere
# Module halten dieselbe Liste in der Hand, und ein `setze_klassen()` wirkt
# damit ueberall sofort - ohne dass jedes Modul neu geladen werden muesste.
# ---------------------------------------------------------------------------
STANDARD_KLASSEN = "asdfjkl."

TASTEN: list[str] = list(STANDARD_KLASSEN)

# Zeichen, die sich nicht von selbst erklaeren oder nicht als Dateiname taugen.
NAMEN = {
    ".": ("Punkt", "punkt"), ",": ("Komma", "komma"), ";": ("Semikolon", "semikolon"),
    ":": ("Doppelpunkt", "doppelpunkt"), "-": ("Bindestrich", "bindestrich"),
    "_": ("Unterstrich", "unterstrich"), " ": ("Leertaste", "leertaste"),
    "/": ("Schraegstrich", "schraegstrich"), "?": ("Fragezeichen", "fragezeichen"),
    "!": ("Ausrufezeichen", "ausrufezeichen"), "'": ("Apostroph", "apostroph"),
    "+": ("Plus", "plus"), "#": ("Raute", "raute"), "<": ("Kleiner", "kleiner"),
    "ä": ("A-Umlaut", "ae"), "ö": ("O-Umlaut", "oe"), "ü": ("U-Umlaut", "ue"),
    "ß": ("Scharfes S", "sz"),
}


# Erlaubte Proben je Taste - Studio und Collector pruefen gegen dieselben
# Grenzen. Unter 5 taugt eine Sitzung nicht zum Lernen, ueber 200 wird die
# Promptfolge unhandlich lang.
ZIEL_MIN, ZIEL_MAX = 5, 200


class KlassenFehler(ValueError):
    """Eine Klassenliste, mit der sich nicht arbeiten laesst."""


class ConfigFehler(ValueError):
    """config.json ist nicht lesbar oder enthaelt ungueltige Werte."""


def schreibe_atomar(pfad: Path, text: str, versuche: int = 10) -> None:
    """Datei so ersetzen, dass Leser nie eine halbe oder leere Datei sehen.

    Erst in eine Nachbardatei schreiben, dann per os.replace austauschen.
    Unter Windows schlaegt os.replace fehl, solange ein anderer Prozess die
    Zieldatei gerade offen hat (etwa das Studio beim Einlesen) - deshalb
    einige kurze Wiederholungen. Klappt es gar nicht, wird direkt
    geschrieben: Lieber kurz nicht atomar als die Daten verlieren.
    """
    # Je Prozess eine eigene Nachbardatei: Schreiben Collector und Studio
    # zugleich dieselbe session.json, darf keiner die halbe Datei des
    # anderen einsetzen oder an ihrem Verschwinden scheitern.
    tmp = pfad.with_name(f"{pfad.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    for _ in range(versuche):
        try:
            os.replace(tmp, pfad)
            return
        except PermissionError:
            time.sleep(0.05)
    pfad.write_text(text, encoding="utf-8")
    try:
        tmp.unlink()
    except OSError:
        pass


def pruefe_klassen(zeichen) -> list[str]:
    """Klassenliste pruefen und als Liste einzelner Zeichen zurueckgeben."""
    liste = [z.lower() for z in zeichen]
    if len(liste) < 2:
        raise KlassenFehler("Mindestens zwei Klassen werden gebraucht.")
    if any(len(z) != 1 for z in liste):
        raise KlassenFehler("Jede Klasse ist genau ein Zeichen.")
    doppelt = {z for z in liste if liste.count(z) > 1}
    if doppelt:
        raise KlassenFehler(f"Doppelte Zeichen: {sorted(doppelt)}")
    if len(liste) > 40:
        raise KlassenFehler("Mehr als 40 Klassen sind nicht vorgesehen.")
    return liste


def setze_klassen(zeichen) -> list[str]:
    """Klassenliste wechseln - wirkt sofort im ganzen Programm."""
    from . import theme

    neu = pruefe_klassen(zeichen)
    TASTEN[:] = neu
    theme.farben_aktualisieren()
    return list(TASTEN)


def zufall() -> float:
    """Trefferquote von blindem Raten - die Messlatte fuer jedes Ergebnis.

    Bewusst eine Funktion und keine Konstante: Die Klassenliste kann sich zur
    Laufzeit aendern, ein beim Import eingefrorener Wert waere dann falsch.
    """
    return 1.0 / len(TASTEN)


def ist_demo_taste(zeichen: str) -> bool:
    """True nur fuer die festgelegten Klassen."""
    return bool(zeichen) and len(zeichen) == 1 and zeichen.lower() in TASTEN


def anzeige(taste: str) -> str:
    """Grosses Zeichen fuer die Anzeige.

    Die Leertaste bekommt ein sichtbares Zeichen - als Leerzeichen stuende
    sie in jeder Tabelle, Kachel und Matrix als Luecke da.

    Ein Anschlag ist immer genau ein Zeichen. Wird ein Zeichen gross
    geschrieben zu zweien ('ß' -> 'SS'), bleibt es klein - sonst saehe es
    aus wie zweimal S.
    """
    if taste == " ":
        return "␣"
    gross = taste.upper()
    return gross if len(gross) == len(taste) else taste


def beiname(taste: str) -> str:
    """Erklaerender Name unter dem Zeichen, sonst leer."""
    return NAMEN.get(taste, ("", ""))[0]


def datei_token(taste: str) -> str:
    """Dateinamentauglicher Name der Klasse."""
    if taste in NAMEN:
        return NAMEN[taste][1]
    return taste if taste.isalnum() else f"zeichen{ord(taste)}"


def pruefe_keine_sperrfolge(folge: list[str], sperrfolge: str) -> None:
    """Sicherstellen, dass eine Promptfolge eine Zielsequenz nicht enthaelt.

    Wer das Programm mit einem spaeteren Blindtest auf ein bestimmtes Wort
    benutzt, will dieses Wort nicht schon im Training gehabt haben. Trainiert
    werden einzelne, zufaellig angeordnete Anschlaege - nicht das Wort.
    """
    if sperrfolge and sperrfolge.lower() in "".join(folge):
        raise KlassenFehler(
            f"Die Promptfolge enthaelt die gesperrte Sequenz {sperrfolge!r}."
        )


@dataclass
class Config:
    """Aufnahme- und Analyseparameter. Wird als config.json abgelegt."""

    # --- Klassen ----------------------------------------------------------
    klassen: str = STANDARD_KLASSEN
    sperrfolge: str = ""          # darf in keiner Aufnahmefolge vorkommen

    # --- Audio-Eingang ----------------------------------------------------
    samplerate: int = 48_000
    channels: int = 1
    blocksize: int = 256
    device: int | None = None
    device_name: str = ""
    hostapi: str = ""
    latenz_ms: float = 10.0

    # --- Kontextfenster um einen Tastendruck ------------------------------
    ring_sekunden: float = 6.0
    pre_roll_ms: int = 200
    post_roll_ms: int = 400

    # --- Schnitt fuer das Modell -----------------------------------------
    segment_ms: int = 250
    segment_vor_onset_ms: int = 15

    # --- Onset-Erkennung --------------------------------------------------
    hochpass_hz: int = 1_200
    huellkurve_fenster_ms: float = 2.0
    huellkurve_hop_ms: float = 0.5
    onset_schwelle_db: float = 12.0
    onset_rueckverfolgung_db: float = 6.0
    mehrfach_abstand_ms: float = 200.0

    # --- Qualitaetsgrenzen ------------------------------------------------
    clip_schwelle: float = 0.98
    min_snr_db: float = 12.0
    min_peak_dbfs: float = -45.0

    # --- Log-Mel ----------------------------------------------------------
    mel_nfft: int = 512
    mel_hop: int = 64
    mel_baender: int = 64
    mel_fmin: float = 100.0
    mel_fmax: float = 16_000.0

    # --- Ablauf -----------------------------------------------------------
    ziel_pro_taste: int = 40
    pause_nach_probe_ms: int = 550

    # --- Darstellung ------------------------------------------------------
    spektro_max_hz: int = 16_000

    # ------------------------------------------------------------------
    @property
    def fenster_samples(self) -> int:
        return int(round((self.pre_roll_ms + self.post_roll_ms) / 1000 * self.samplerate))

    def anwenden(self) -> None:
        """Klassen dieser Konfiguration global setzen."""
        setze_klassen(self.klassen)

    def speichern(self, pfad: Path | None = None) -> Path:
        pfad = pfad or CONFIG_PFAD
        pfad.parent.mkdir(parents=True, exist_ok=True)
        schreibe_atomar(pfad, json.dumps(asdict(self), indent=2, ensure_ascii=False))
        return pfad

    def _typen_pruefen(self) -> None:
        """Jeden Wert gegen den Typ seines Standardwerts pruefen.

        Ein Tippfehler von Hand ("samplerate": "48000") fiele sonst erst tief
        im Audio- oder Trainingscode auf, mit einer Meldung ohne Bezug zur
        config.json. Ganzzahlige Kommazahlen (250.0) werden still zu int.
        """
        for feld in fields(self):
            wert, standard = getattr(self, feld.name), feld.default
            if isinstance(wert, bool) and not isinstance(standard, bool):
                ok = False
            elif standard is None or isinstance(standard, int):
                if isinstance(wert, float) and wert.is_integer():
                    wert = int(wert)
                    setattr(self, feld.name, wert)
                ok = isinstance(wert, int) or (standard is None and wert is None)
            elif isinstance(standard, float):
                ok = isinstance(wert, (int, float))
            elif isinstance(standard, str):
                ok = isinstance(wert, str)
            else:
                ok = True
            if not ok:
                if isinstance(standard, str):
                    erwartet = "Text"
                elif isinstance(standard, float):
                    erwartet = "eine Zahl"
                else:
                    erwartet = "eine ganze Zahl"
                raise ValueError(f"{feld.name!r} hat den Wert {wert!r} - erwartet "
                                 f"{erwartet}")

    @classmethod
    def laden(cls, pfad: Path | None = None, anwenden: bool = True) -> "Config":
        """config.json lesen - bei Fehlern ein ConfigFehler mit Pfad und Zeile.

        Bewusst kein stiller Rueckfall auf Standardwerte: Der naechste
        speichern()-Aufruf wuerde sonst Jacobs echte Konfiguration
        (Mikrofon, Klassen) mit den Standardwerten ueberschreiben.
        """
        pfad = pfad or CONFIG_PFAD
        ausweg = ("Die Datei korrigieren - oder loeschen, dann gelten die "
                  "Standardwerte und Mikrofon und Klassen werden neu gewaehlt.")
        if pfad.exists():
            try:
                daten = json.loads(pfad.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ConfigFehler(
                    f"{pfad} ist kein gueltiges JSON (Zeile {exc.lineno}, Spalte "
                    f"{exc.colno}: {exc.msg}).\n{ausweg}") from exc
            except (OSError, UnicodeDecodeError) as exc:
                raise ConfigFehler(f"{pfad} laesst sich nicht lesen: {exc}") from exc
            if not isinstance(daten, dict):
                raise ConfigFehler(f"{pfad} enthaelt kein JSON-Objekt.\n{ausweg}")
            gueltig = set(cls.__dataclass_fields__)
            try:
                cfg = cls(**{k: v for k, v in daten.items() if k in gueltig})
                cfg._typen_pruefen()
                pruefe_klassen(cfg.klassen)
            except (TypeError, ValueError) as exc:
                raise ConfigFehler(f"{pfad}: {exc}\n{ausweg}") from exc
        else:
            cfg = cls()
        if anwenden:
            cfg.anwenden()
        return cfg


def laden_oder_beenden(anwenden: bool = True) -> Config:
    """Config.laden fuer die Werkzeuge: bei Fehlern Meldung statt Traceback.

    Die Meldung geht auf stderr - aus dem Studio gestartet liest es sie mit
    und zeigt sie an. Der Exit-Code 1 sagt dem Studio, dass es nicht ging.
    """
    try:
        return Config.laden(anwenden=anwenden)
    except ConfigFehler as exc:
        print(f"Konfiguration fehlerhaft: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


def verzeichnisse_anlegen() -> None:
    for p in (DATEN, ROH, MODELLE, AUSGABE):
        p.mkdir(parents=True, exist_ok=True)

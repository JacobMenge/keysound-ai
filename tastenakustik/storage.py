"""Ablage der Proben: eine Sitzung = ein Ordner.

Datenaufteilung ohne Leakage: Train / Validation / Test werden spaeter nach
ganzen Sitzungen vergeben, nie nach einzelnen Proben. Deshalb bekommt jede
Sitzung eine eigene ID, einen Zeitstempel und ein Feld "rolle", das bis zur
Entscheidung auf "offen" steht. Die Rolle steht nur in session.json - nicht
in den Probenzeilen, sonst gaebe es zwei Stellen, die sich widersprechen.

Identitaet einer Sitzung ist ihr Ordnername. Normalerweise ist er gleich der
session_id im Kopf; nach einer Kopie im Explorer oder einem Umbenennen nicht
mehr - uebersicht() und doppelte_sitzungen() machen das sichtbar.
"""

from __future__ import annotations

import json
import re
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import Config, ROH, TASTEN, datei_token, schreibe_atomar

ROLLEN = ("offen", "train", "val", "test")

# Felder, die man im Kopf nachtraeglich aendern kann (Studio, Werkzeug 04,
# von Hand) - eine laufende Sitzung darf sie beim Abschliessen nicht
# zuruecksetzen.
NACHTRAEGLICH = ("rolle", "notiz", "tastatur", "mikrofon_position")

_NUMMER = re.compile(r"S(\d+)_")


def _jetzt() -> datetime:
    return datetime.now().astimezone()


def _nummer(ordner: Path) -> int:
    treffer = _NUMMER.match(ordner.name)
    return int(treffer.group(1)) if treffer else 0


def sitzungen() -> list[Path]:
    """Alle vorhandenen Sitzungsordner, chronologisch.

    Sortiert nach dem Zeitstempel im Namen (S07_20260922_1410 -> 20260922_1410),
    nicht nach der Nummer: Bei Altbestaenden kann dieselbe Nummer doppelt
    vorkommen, und "die neueste Sitzung" soll wirklich die zuletzt begonnene
    sein.
    """
    if not ROH.exists():
        return []
    return sorted(
        (p for p in ROH.iterdir() if p.is_dir() and (p / "session.json").exists()),
        key=lambda p: (p.name.split("_", 1)[-1], _nummer(p), p.name),
    )


class Sitzung:
    """Schreibt die Proben einer Aufnahmesitzung."""

    def __init__(self, cfg: Config, notiz: str = "", rolle: str = "offen",
                 tastatur: str = "", mikrofon_position: str = ""):
        if rolle not in ROLLEN:
            raise ValueError(f"rolle muss aus {ROLLEN} sein")
        self.cfg = cfg
        self.start = _jetzt()
        # Hoechste vorhandene Nummer + 1, nicht Anzahl + 1: Nach dem
        # Wegraeumen alter Sitzungen kaeme sonst eine Nummer doppelt vor.
        # Gezaehlt werden alle S..-Ordner, auch solche ohne session.json.
        # Den Ordner exklusiv anlegen - eine vorhandene Sitzung wird nie
        # still weiterbeschrieben.
        ROH.mkdir(parents=True, exist_ok=True)
        self.index = max((_nummer(p) for p in ROH.iterdir() if p.is_dir()), default=0) + 1
        while True:
            self.id = f"S{self.index:02d}_{self.start:%Y%m%d_%H%M}"
            self.ordner = ROH / self.id
            try:
                self.ordner.mkdir(exist_ok=False)
                break
            except FileExistsError:
                self.index += 1
        self.proben_datei = self.ordner / "proben.jsonl"
        self.zaehler: dict[str, int] = {t: 0 for t in TASTEN}
        self.verworfen: dict[str, int] = {}

        self.kopf = {
            "session_id": self.id,
            "index": self.index,
            "rolle": rolle,
            "notiz": notiz,
            "tastatur": tastatur,
            "mikrofon_position": mikrofon_position,
            "gestartet": self.start.isoformat(timespec="seconds"),
            "beendet": None,
            "tasten": list(TASTEN),
            "aufnahmeparameter": asdict(cfg),
        }
        self._kopf_schreiben()

    def _kopf_schreiben(self) -> None:
        _kopf_speichern(self.ordner, self.kopf)

    # ------------------------------------------------------------------
    def speichere(self, fenster: np.ndarray, label: str, anschlag, extra: dict | None = None) -> Path:
        """Eine akzeptierte Probe als 32-Bit-Float-WAV plus Metadatenzeile."""
        if label not in self.zaehler:
            raise ValueError(f"Label {label!r} gehoert nicht zur festen Tastenliste")

        nummer = self.zaehler[label] + 1
        # Der Punkt taugt nicht als Dateinamensanfang, deshalb ein Token.
        name = f"{datei_token(label)}_{nummer:03d}.wav"
        pfad = self.ordner / name
        sf.write(pfad, fenster.astype(np.float32), self.cfg.samplerate, subtype="FLOAT")

        zeile = {
            "datei": name,
            "label": label,
            "nummer": nummer,
            "session_id": self.id,
            "zeitstempel": _jetzt().isoformat(timespec="milliseconds"),
            "samplerate": self.cfg.samplerate,
            "laenge_samples": int(fenster.size),
            "pre_roll_ms": self.cfg.pre_roll_ms,
            "post_roll_ms": self.cfg.post_roll_ms,
            "geraet": self.cfg.device_name,
            "hostapi": self.cfg.hostapi,
            **anschlag.as_dict(),
            **(extra or {}),
        }
        with self.proben_datei.open("a", encoding="utf-8") as f:
            f.write(json.dumps(zeile, ensure_ascii=False) + "\n")

        self.zaehler[label] = nummer
        return pfad

    def notiere_verwurf(self, grund: str) -> None:
        self.verworfen[grund] = self.verworfen.get(grund, 0) + 1

    # ------------------------------------------------------------------
    @property
    def gesamt(self) -> int:
        return sum(self.zaehler.values())

    def fehlend(self, ziel: int) -> dict[str, int]:
        return {t: max(0, ziel - n) for t, n in self.zaehler.items()}

    def abschliessen(self) -> None:
        # Waehrend der Aufnahme kann im Studio schon eine Rolle vergeben
        # worden sein - die steht auf der Platte, nicht in self.kopf.
        try:
            auf_platte = lade_kopf(self.ordner)
        except (OSError, ValueError):
            auf_platte = {}
        for feld in NACHTRAEGLICH:
            if feld in auf_platte:
                self.kopf[feld] = auf_platte[feld]
        self.kopf["beendet"] = _jetzt().isoformat(timespec="seconds")
        self.kopf["proben_je_taste"] = dict(self.zaehler)
        self.kopf["proben_gesamt"] = self.gesamt
        self.kopf["verworfen"] = dict(self.verworfen)
        self._kopf_schreiben()


def _kopf_speichern(ordner: Path, kopf: dict) -> None:
    """session.json atomar ersetzen - das Studio liest sie jederzeit mit."""
    schreibe_atomar(ordner / "session.json",
                    json.dumps(kopf, indent=2, ensure_ascii=False))


def lade_kopf(ordner: Path) -> dict:
    """Nur session.json einer Sitzung - ValueError mit Pfad, wenn unlesbar.

    Unter Windows ist die Datei fuer einen Augenblick gesperrt, waehrend ein
    anderer Prozess sie per os.replace austauscht (PermissionError). Das ist
    kein Schaden - deshalb kurz wiederholen, statt die Sitzung als unlesbar
    zu melden oder beim Abschliessen die Rolle von der Platte zu verpassen.
    """
    pfad = ordner / "session.json"
    for versuch in range(10):
        try:
            kopf = json.loads(pfad.read_text(encoding="utf-8"))
            break
        except FileNotFoundError:
            raise
        except PermissionError as exc:
            if versuch == 9:
                raise ValueError(f"{pfad} ist nicht lesbar: {exc}") from exc
            time.sleep(0.02)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{pfad} ist nicht lesbar: {exc}") from exc
    if not isinstance(kopf, dict):
        raise ValueError(f"{pfad} enthaelt kein JSON-Objekt.")
    return kopf


def lade_sitzung(ordner: Path) -> tuple[dict, list[dict]]:
    """Kopf und Probenliste einer Sitzung einlesen.

    Ein unlesbarer Kopf ist ein ValueError mit Pfad. Kaputte Probenzeilen -
    meist eine abgeschnittene letzte Zeile nach Absturz oder Abbruch -
    werden mit Datei und Zeilennummer gemeldet und uebersprungen; die
    uebrigen Proben der Sitzung bleiben nutzbar.

    Jede Probe bekommt die Rolle aus dem Kopf. Aeltere proben.jsonl tragen
    noch ein eigenes Feld "rolle", das nie nachgefuehrt wurde.
    """
    kopf = lade_kopf(ordner)
    proben_datei = ordner / "proben.jsonl"
    proben: list[dict] = []
    if proben_datei.exists():
        text = proben_datei.read_text(encoding="utf-8", errors="replace")
        for nr, zeile in enumerate(text.splitlines(), start=1):
            if not zeile.strip():
                continue
            try:
                probe = json.loads(zeile)
            except json.JSONDecodeError:
                probe = None
            if not isinstance(probe, dict) or "label" not in probe or "datei" not in probe:
                warnings.warn(f"{proben_datei}, Zeile {nr}: nicht lesbar - uebersprungen",
                              stacklevel=2)
                continue
            probe["rolle"] = kopf.get("rolle", "offen")
            proben.append(probe)
    return kopf, proben


def doppelte_sitzungen() -> dict[str, list[str]]:
    """session_ids, die in mehr als einem Ordner stehen -> diese Ordner.

    Typisch nach einer Kopie im Explorer ("S04_... - Kopie"). Beide Ordner
    enthalten dieselben Aufnahmen; landen sie in verschiedenen Rollen, steht
    dieselbe Aufnahme in val und test.
    """
    ordner_je_id: dict[str, list[str]] = {}
    for ordner in sitzungen():
        try:
            sid = lade_kopf(ordner).get("session_id", ordner.name)
        except (OSError, ValueError):
            continue
        ordner_je_id.setdefault(sid, []).append(ordner.name)
    return {sid: namen for sid, namen in ordner_je_id.items() if len(namen) > 1}


def uebersicht() -> list[dict]:
    """Kurzueberblick ueber alle aufgenommenen Sitzungen.

    "ordner" ist die Identitaet fuer rolle_setzen. "hinweis" ist leer oder
    nennt, warum mit der Sitzung etwas nicht stimmt (Kopie, umbenannt).
    Sitzungen mit unlesbarem Kopf fehlen hier - mit einer Warnung, statt
    Studio und Werkzeuge am Start scheitern zu lassen.
    """
    zeilen = []
    doppelt = doppelte_sitzungen()
    for ordner in sitzungen():
        try:
            kopf, proben = lade_sitzung(ordner)
        except (OSError, ValueError) as exc:
            warnings.warn(f"Sitzung {ordner.name} uebersprungen: {exc}", stacklevel=2)
            continue
        je_taste: dict[str, int] = {t: 0 for t in TASTEN}
        for p in proben:
            je_taste[p["label"]] = je_taste.get(p["label"], 0) + 1
        sid = kopf.get("session_id", ordner.name)
        hinweis = ""
        if sid in doppelt:
            hinweis = (f"dieselbe Sitzung liegt in {len(doppelt[sid])} Ordnern "
                       f"({', '.join(doppelt[sid])}) - Kopie entfernen")
        elif sid != ordner.name:
            hinweis = f"Ordner heisst anders als die Sitzung ({sid})"
        zeilen.append(
            {
                "session_id": sid,
                "ordner": ordner.name,
                "rolle": kopf.get("rolle", "offen"),
                "gestartet": kopf.get("gestartet", ""),
                "beendet": kopf.get("beendet"),
                "notiz": kopf.get("notiz", ""),
                "gesamt": len(proben),
                "je_taste": je_taste,
                "hinweis": hinweis,
            }
        )
    return zeilen


def rolle_setzen(ordner_name: str, rolle: str) -> Path:
    """Rolle einer Sitzung nachtraeglich vergeben (train / val / test).

    Erwartet den Ordnernamen. Fuer aeltere Aufrufer, die die session_id
    uebergeben, wird ein umbenannter Ordner ueber die ID im Kopf gefunden.
    Liegt dieselbe Sitzung in mehreren Ordnern, wird abgelehnt: Sonst liesse
    sich dieselbe Aufnahme zugleich als val und als test vergeben.
    """
    if rolle not in ROLLEN:
        raise ValueError(f"rolle muss aus {ROLLEN} sein")
    ordner = ROH / ordner_name
    if not (ordner / "session.json").exists():
        passend = [o for o in sitzungen() if _session_id(o) == ordner_name]
        if len(passend) != 1:
            raise FileNotFoundError(f"Keine Sitzung {ordner_name!r} unter {ROH}")
        ordner = passend[0]
    kopf = lade_kopf(ordner)
    doppelt = doppelte_sitzungen().get(kopf.get("session_id", ordner.name))
    if doppelt:
        raise ValueError(
            f"Sitzung {kopf.get('session_id')} liegt doppelt vor "
            f"({', '.join(doppelt)}) - erst die Kopie aus {ROH} entfernen.")
    kopf["rolle"] = rolle
    _kopf_speichern(ordner, kopf)
    return ordner


def _session_id(ordner: Path) -> str | None:
    try:
        return lade_kopf(ordner).get("session_id")
    except (OSError, ValueError):
        return None

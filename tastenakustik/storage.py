"""Ablage der Proben: eine Sitzung = ein Ordner.

Datenaufteilung ohne Leakage: Train / Validation / Test werden spaeter nach
ganzen Sitzungen vergeben, nie nach einzelnen Proben. Deshalb bekommt jede
Sitzung eine eigene ID, einen Zeitstempel und ein Feld "rolle", das bis zur
Entscheidung auf "offen" steht.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import Config, ROH, TASTEN, datei_token

ROLLEN = ("offen", "train", "val", "test")


def _jetzt() -> datetime:
    return datetime.now().astimezone()


def sitzungen() -> list[Path]:
    """Alle vorhandenen Sitzungsordner, chronologisch nach ID."""
    if not ROH.exists():
        return []
    return sorted(p for p in ROH.iterdir() if p.is_dir() and (p / "session.json").exists())


class Sitzung:
    """Schreibt die Proben einer Aufnahmesitzung."""

    def __init__(self, cfg: Config, notiz: str = "", rolle: str = "offen",
                 tastatur: str = "", mikrofon_position: str = ""):
        if rolle not in ROLLEN:
            raise ValueError(f"rolle muss aus {ROLLEN} sein")
        self.cfg = cfg
        self.start = _jetzt()
        self.index = len(sitzungen()) + 1
        self.id = f"S{self.index:02d}_{self.start:%Y%m%d_%H%M}"
        self.ordner = ROH / self.id
        self.ordner.mkdir(parents=True, exist_ok=True)
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
        (self.ordner / "session.json").write_text(
            json.dumps(self.kopf, indent=2, ensure_ascii=False), encoding="utf-8"
        )

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
            "rolle": self.kopf["rolle"],
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
        self.kopf["beendet"] = _jetzt().isoformat(timespec="seconds")
        self.kopf["proben_je_taste"] = dict(self.zaehler)
        self.kopf["proben_gesamt"] = self.gesamt
        self.kopf["verworfen"] = dict(self.verworfen)
        self._kopf_schreiben()


def lade_sitzung(ordner: Path) -> tuple[dict, list[dict]]:
    """Kopf und Probenliste einer Sitzung einlesen."""
    kopf = json.loads((ordner / "session.json").read_text(encoding="utf-8"))
    proben_datei = ordner / "proben.jsonl"
    proben: list[dict] = []
    if proben_datei.exists():
        for zeile in proben_datei.read_text(encoding="utf-8").splitlines():
            if zeile.strip():
                proben.append(json.loads(zeile))
    return kopf, proben


def uebersicht() -> list[dict]:
    """Kurzueberblick ueber alle aufgenommenen Sitzungen."""
    zeilen = []
    for ordner in sitzungen():
        kopf, proben = lade_sitzung(ordner)
        je_taste: dict[str, int] = {t: 0 for t in TASTEN}
        for p in proben:
            je_taste[p["label"]] = je_taste.get(p["label"], 0) + 1
        zeilen.append(
            {
                "session_id": kopf["session_id"],
                "rolle": kopf.get("rolle", "offen"),
                "gestartet": kopf.get("gestartet", ""),
                "notiz": kopf.get("notiz", ""),
                "gesamt": len(proben),
                "je_taste": je_taste,
            }
        )
    return zeilen


def rolle_setzen(session_id: str, rolle: str) -> Path:
    """Rolle einer Sitzung nachtraeglich vergeben (train / val / test)."""
    if rolle not in ROLLEN:
        raise ValueError(f"rolle muss aus {ROLLEN} sein")
    ordner = ROH / session_id
    kopf, _ = lade_sitzung(ordner)
    kopf["rolle"] = rolle
    (ordner / "session.json").write_text(
        json.dumps(kopf, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return ordner

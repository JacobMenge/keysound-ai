"""Schritt 4 - Sitzungen ansehen und Rollen vergeben (Train / Validation / Test).

Die Aufteilung erfolgt bewusst nach ganzen Sitzungen, nicht nach einzelnen
Proben. Waeren Proben derselben Aufnahme auf Train und Test verteilt, wuerde
das Modell Raum, Mikrofonposition und Tagesform wiedererkennen statt der Taste
- und die Genauigkeit waere geschoent. Der Test soll aus einer spaeter
aufgenommenen Sitzung stammen.

Aufruf:
    python werkzeuge/04_sitzungen.py
    python werkzeuge/04_sitzungen.py --rolle S01_20260922_1430=train
    python werkzeuge/04_sitzungen.py --rolle S01_...=train --rolle S03_...=test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tastenakustik import storage
from tastenakustik.config import ROH, TASTEN, Config, anzeige

GRUEN, GELB, GRAU, AUS = "\033[92m", "\033[93m", "\033[90m", "\033[0m"
ROLLEN_FARBE = {"train": GRUEN, "val": GELB, "test": "\033[96m", "offen": GRAU}


def tabelle() -> None:
    zeilen = storage.uebersicht()
    if not zeilen:
        print(f"Noch keine Sitzungen unter {ROH}")
        return

    kopf = "  ".join(f"{anzeige(t):>3}" for t in TASTEN)
    print(f"\n{'Sitzung':<22} {'Rolle':<7} {'ges':>4}   {kopf}   Notiz")
    print("-" * (22 + 8 + 6 + len(kopf) + 12))
    summe = {t: 0 for t in TASTEN}
    fremde = []
    for z in zeilen:
        farbe = ROLLEN_FARBE.get(z["rolle"], GRAU)
        zahlen = "  ".join(f"{z['je_taste'].get(t, 0):>3}" for t in TASTEN)
        for t in TASTEN:
            summe[t] += z["je_taste"].get(t, 0)
        # Sitzungen mit anderen Klassen fallen sonst nur als Luecke auf.
        andere = any(n and t not in TASTEN for t, n in z["je_taste"].items())
        if andere:
            fremde.append(z["session_id"])
        print(f"{z['session_id']:<22} {farbe}{z['rolle']:<7}{AUS} {z['gesamt']:>4}   "
              f"{zahlen}   {'* ' if andere else ''}{z['notiz'][:28]}")
    print("-" * (22 + 8 + 6 + len(kopf) + 12))
    gesamt = sum(z["gesamt"] for z in zeilen)
    print(f"{'Summe':<22} {'':<7} {gesamt:>4}   "
          + "  ".join(f"{summe[t]:>3}" for t in TASTEN))

    nach_rolle: dict[str, int] = {}
    for z in zeilen:
        nach_rolle[z["rolle"]] = nach_rolle.get(z["rolle"], 0) + z["gesamt"]
    print("\nProben je Rolle:  " + "   ".join(f"{k}: {v}" for k, v in sorted(nach_rolle.items())))

    if fremde:
        print(f"\n{GELB}* mit anderen Klassen aufgenommen: {', '.join(fremde)}.{AUS}")
        print(f"{GRAU}  Diese Sitzungen passen nicht zu {''.join(TASTEN)} - fuer das "
              f"Training aus daten/roh/ wegraeumen oder die Klassen zurueckstellen.{AUS}")

    if nach_rolle.get("offen"):
        print(f"\n{GELB}Noch nicht zugeordnete Sitzungen. Empfehlung: die zuletzt "
              f"aufgenommene Sitzung als Test.{AUS}")
        print(f"{GRAU}  python werkzeuge/04_sitzungen.py --rolle <SESSION_ID>=train{AUS}")


def main() -> int:
    p = argparse.ArgumentParser(description="Sitzungen ansehen und Rollen vergeben")
    p.add_argument("--rolle", action="append", default=[],
                   metavar="SESSION_ID=ROLLE",
                   help="Rolle setzen, z.B. S01_20260922_1430=train")
    args = p.parse_args()
    Config.laden()   # setzt die gewaehlten Klassen

    for eintrag in args.rolle:
        if "=" not in eintrag:
            print(f"Erwartet SESSION_ID=ROLLE, bekommen: {eintrag!r}")
            return 1
        sid, rolle = eintrag.split("=", 1)
        try:
            storage.rolle_setzen(sid.strip(), rolle.strip())
        except (FileNotFoundError, ValueError) as exc:
            print(f"{sid}: {exc}")
            return 1
        print(f"{sid} -> {rolle}")

    tabelle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

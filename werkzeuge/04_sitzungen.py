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
from tastenakustik.config import ROH, TASTEN, anzeige, laden_oder_beenden

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
        # Der Ordnername ist eindeutig, auch wenn eine Kopie dieselbe
        # session_id traegt - und mit ihm funktioniert --rolle.
        hinweis = z.get("hinweis", "")
        print(f"{z.get('ordner', z['session_id']):<22} {farbe}{z['rolle']:<7}{AUS} "
              f"{z['gesamt']:>4}   {zahlen}   {'* ' if andere else ''}"
              f"{(GELB + hinweis + AUS) if hinweis else z['notiz'][:28]}")
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
        print(f"{GRAU}  Diese Sitzungen passen nicht zu "
              f"{''.join(anzeige(t) for t in TASTEN)} - fuer das "
              f"Training aus daten/roh/ wegraeumen oder die Klassen zurueckstellen.{AUS}")

    # Die Empfehlung "zuletzt aufgenommene als Test" nur, solange es noch
    # keine Testsitzung gibt - sonst waere sie eine zweite.
    if nach_rolle.get("offen"):
        if "test" not in nach_rolle:
            print(f"\n{GELB}Noch nicht zugeordnete Sitzungen. Empfehlung: die zuletzt "
                  f"aufgenommene Sitzung als Test.{AUS}")
            print(f"{GRAU}  python werkzeuge/04_sitzungen.py --rolle <SESSION_ID>=test{AUS}")
        else:
            print(f"\n{GRAU}Offene Sitzungen werden beim Training nicht benutzt. Wer eine "
                  f"davon doch verwenden will:{AUS}")
            print(f"{GRAU}  python werkzeuge/04_sitzungen.py --rolle <SESSION_ID>=train{AUS}")


def _session_id(ordner: Path) -> str | None:
    """session_id aus dem Kopf - None, wenn session.json unlesbar ist."""
    try:
        return storage.lade_kopf(ordner).get("session_id")
    except (OSError, ValueError):
        return None


def main() -> int:
    # Der Aufruf-Block aus dem Docstring erscheint unter --help als Beispiel.
    p = argparse.ArgumentParser(
        description="Sitzungen ansehen und Rollen vergeben",
        epilog=__doc__[__doc__.index("Aufruf:"):],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rolle", action="append", default=[],
                   metavar="SESSION_ID=ROLLE",
                   help="Rolle setzen (train, val, test oder offen), z.B. "
                        "S01_20260922_1430=train; mehrfach angebbar")
    args = p.parse_args()
    laden_oder_beenden()   # setzt die gewaehlten Klassen

    # Erst alle Eintraege pruefen, dann schreiben: Scheitert der dritte von
    # drei, sollen die ersten beiden nicht schon still gesetzt sein.
    auftraege: list[tuple[str, str, str]] = []
    doppelt = storage.doppelte_sitzungen() if args.rolle else {}
    for eintrag in args.rolle:
        if "=" not in eintrag:
            print(f"Erwartet SESSION_ID=ROLLE, bekommen: {eintrag!r}")
            return 1
        sid, rolle = (teil.strip() for teil in eintrag.split("=", 1))
        if rolle not in storage.ROLLEN:
            print(f"{sid}: Rolle muss {'/'.join(storage.ROLLEN)} sein, "
                  f"bekommen: {rolle!r}")
            return 1
        # Wie rolle_setzen: Ein umbenannter Ordner wird auch ueber die
        # session_id in seinem Kopf gefunden - die zeigt die Tabelle.
        ordner = ROH / sid if sid else None
        if ordner is not None and not (ordner / "session.json").exists():
            passend = [o for o in storage.sitzungen() if _session_id(o) == sid]
            ordner = passend[0] if passend else None
        if ordner is None:
            vorhanden = ", ".join(o.name for o in storage.sitzungen()) or "keine"
            print(f"Sitzung {sid!r} nicht gefunden. Vorhanden: {vorhanden}")
            return 1
        # Einen defekten Kopf und eine Sitzung in zwei Ordnern lehnt
        # rolle_setzen ab. Beides schon hier pruefen, sonst waeren die
        # Eintraege davor bereits geschrieben.
        try:
            eigene_id = storage.lade_kopf(ordner).get("session_id", ordner.name)
        except (OSError, ValueError) as exc:
            print(f"{sid}: {exc}")
            return 1
        if eigene_id in doppelt:
            print(f"Sitzung {eigene_id} liegt doppelt vor "
                  f"({', '.join(doppelt[eigene_id])}) - erst die Kopie aus {ROH} "
                  f"entfernen.")
            return 1
        auftraege.append((sid, ordner.name, rolle))

    for sid, ordner_name, rolle in auftraege:
        try:
            storage.rolle_setzen(ordner_name, rolle)
        except (OSError, ValueError) as exc:
            # Nur noch fuer Schreibfehler oder eine Datei, die sich seit der
            # Pruefung oben geaendert hat.
            print(f"{sid}: {exc}")
            return 1
        print(f"{sid} -> {rolle}")

    tabelle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

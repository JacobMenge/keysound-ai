"""Tastenakustik - das Studio. Ein Fenster fuer das ganze Experiment.

    python start.py

Fuehrt der Reihe nach durch alles: Mikrofon einrichten, Klassen festlegen,
Daten aufnehmen, Daten pruefen, Modell trainieren, Modell testen. Jeder Schritt
sagt, woran er gerade haengt, und schaltet den naechsten frei, sobald er fertig
ist. Die Vollbild-Werkzeuge (Pegelanzeige, Collector, Live-Demo) gehen aus
diesem Fenster auf, weil sie im Hochformat laufen und zum Filmen gedacht sind.

Was dieses Programm NICHT tut: Es liest keine normalen Texteingaben mit. Der
Collector nimmt nur im ausdruecklich gestarteten Aufnahmemodus auf, nur
angekuendigte einzelne Anschlaege, und nur, solange sein eigenes Fenster im
Vordergrund ist. Die Live-Demo liest ueberhaupt keine Tastatur-Ereignisse -
sie findet Anschlaege allein im Audiosignal.
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox

PROJEKT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJEKT))


def _fehlende_pakete() -> list[str]:
    """Welche Abhaengigkeiten fehlen in genau diesem Python?"""
    import importlib.util

    noetig = {"numpy": "numpy", "scipy": "scipy", "matplotlib": "matplotlib",
              "soundfile": "soundfile", "sounddevice": "sounddevice",
              "torch": "torch", "PIL": "pillow"}
    return [paket for modul, paket in noetig.items()
            if importlib.util.find_spec(modul) is None]


def _abbruch_fehlende_pakete(fehlt: list[str]) -> None:
    """Klar sagen, was fehlt - statt eines nackten ImportError.

    Der haeufigste Stolperstein: Das Programm wird mit einem anderen Python
    gestartet als dem, in dem installiert wurde. Deshalb steht hier, welcher
    Interpreter gerade laeuft.
    """
    text = (
        "Es fehlen Pakete:  " + "  ".join(fehlt) + "\n\n"
        f"Dieser Python laeuft gerade:\n{sys.executable}\n\n"
        "Wahrscheinlich ist das ein anderer als der, in dem du installiert\n"
        "hast. So richtest du eine eigene Umgebung ein - im Ordner\n"
        f"{PROJEKT}:\n\n"
        "    python -m venv .venv\n"
        "    .venv\\Scripts\\activate\n"
        "    pip install -r requirements.txt\n"
        "    python start.py\n"
    )
    print(text, file=sys.stderr)
    try:
        wurzel = tk.Tk()
        wurzel.withdraw()
        messagebox.showerror("Tastenakustik - Pakete fehlen", text)
        wurzel.destroy()
    except Exception:                                  # noqa: BLE001, S110
        # Ohne Fenstersystem bleibt es bei der Ausgabe im Terminal.
        pass
    raise SystemExit(1)


if _fehlt := _fehlende_pakete():
    _abbruch_fehlende_pakete(_fehlt)

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from tastenakustik import audio, storage, theme, training
from tastenakustik.config import (
    Config,
    KlassenFehler,
    TASTEN,
    anzeige,
    pruefe_klassen,
    verzeichnisse_anlegen,
    zufall,
)

WERKZEUGE = PROJEKT / "werkzeuge"
ANSI = re.compile(r"\x1b\[[0-9;]*m")

VORLAGEN = {
    "Grundreihe (8)": "asdfjkl.",
    "Vier Tasten": "asdf",
    "Ziffern (10)": "0123456789",
    "Alphabet (26)": "abcdefghijklmnopqrstuvwxyz",
}

# Wie die Vollbild-Werkzeuge in Meldungen heissen.
WERKZEUG_NAMEN = {
    "02_kalibrierung.py": "Pegelanzeige",
    "03_collector.py": "Collector",
    "08_demo.py": "Live-Demo",
    "09_selbsttest.py": "Selbsttest",
}

ZIEL_MIN, ZIEL_MAX = 5, 200


# ---------------------------------------------------------------------------
# Kleine Bausteine
# ---------------------------------------------------------------------------
def rundes_rechteck(c: tk.Canvas, x0, y0, x1, y1, r, **kw):
    punkte = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
              x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return c.create_polygon(punkte, smooth=True, **kw)


class Knopf(tk.Label):
    """Flacher Knopf - tk.Button sieht unter Windows nicht zum Rest passend aus."""

    def __init__(self, eltern, text: str, befehl, fett: bool = False,
                 farbe: str | None = None, **kw):
        self.farbe = farbe or (theme.AKZENT if fett else theme.PANEL)
        self.vordergrund = theme.BG if fett else theme.TEXT
        super().__init__(eltern, text=text, bg=self.farbe, fg=self.vordergrund,
                         padx=18, pady=9, cursor="hand2", **kw)
        self.befehl = befehl
        self.aktiv = True
        self.bind("<Button-1>", self._klick)
        self.bind("<Enter>", lambda _e: self.aktiv and self.configure(
            bg=theme.TEXT if self.farbe == theme.AKZENT else theme.GRID))
        self.bind("<Leave>", lambda _e: self.configure(bg=self.farbe))

    def _klick(self, _ereignis=None) -> None:
        if self.aktiv:
            self.befehl()

    def setze_aktiv(self, aktiv: bool) -> None:
        self.aktiv = aktiv
        self.configure(bg=self.farbe if aktiv else theme.PANEL,
                       fg=self.vordergrund if aktiv else theme.TEXT_SCHWACH,
                       cursor="hand2" if aktiv else "arrow")


# ---------------------------------------------------------------------------
# Das Fenster
# ---------------------------------------------------------------------------
class Studio(tk.Tk):
    SCHRITTE = [
        ("Überblick", "_ueberblick"),
        ("1  Mikrofon", "_mikrofon"),
        ("2  Klassen", "_klassen"),
        ("3  Aufnehmen", "_aufnehmen"),
        ("4  Daten prüfen", "_pruefen"),
        ("5  Training", "_training"),
        ("6  Live testen", "_testen"),
    ]

    def __init__(self) -> None:
        super().__init__()
        verzeichnisse_anlegen()
        self.cfg = Config.laden()
        self.aktiv = 0
        # Laufende Werkzeuge je Datei, mit den letzten Zeilen ihrer Ausgabe -
        # die braucht es, um zu sagen, warum eines nicht aufging.
        self.prozesse: dict[str, tuple[subprocess.Popen, deque]] = {}
        self.trainings_queue: queue.Queue = queue.Queue()
        self.trainings_thread: threading.Thread | None = None
        self.ergebnis: training.Ergebnis | None = None
        self.verlauf: dict[str, list[float]] | None = None
        self.test_ergebnis: training.TestErgebnis | None = None
        self.test_laeuft = False

        self.title("Tastenakustik - Studio")
        self.configure(bg=theme.BG)
        self.geometry("1180x840")
        self.minsize(1060, 760)
        self.protocol("WM_DELETE_WINDOW", self._schliessen)

        self._schriften()
        self._aufbau()
        self.zeige(0)
        self.after(700, self._takt)

    # -- Aufbau ---------------------------------------------------------
    def _schriften(self) -> None:
        fam = "Segoe UI" if "Segoe UI" in tkfont.families() else "Arial"
        self.fam = fam
        self.f_titel = tkfont.Font(family=fam, size=22, weight="bold")
        self.f_kopf = tkfont.Font(family=fam, size=15, weight="bold")
        self.f_normal = tkfont.Font(family=fam, size=10)
        self.f_klein = tkfont.Font(family=fam, size=9)
        self.f_nav = tkfont.Font(family=fam, size=11)
        self.f_nav_fett = tkfont.Font(family=fam, size=11, weight="bold")
        self.f_mono = tkfont.Font(family="Consolas", size=10)
        self.f_zahl = tkfont.Font(family=fam, size=34, weight="bold")

    def _aufbau(self) -> None:
        kopf = tk.Frame(self, bg=theme.BG, padx=28, pady=18)
        kopf.pack(fill="x")
        tk.Label(kopf, text="Tastenakustik", font=self.f_titel,
                 bg=theme.BG, fg=theme.TEXT).pack(side="left")
        tk.Label(kopf, text="Klingen einzelne Tasten unterschiedlich?",
                 font=self.f_normal, bg=theme.BG,
                 fg=theme.TEXT_SCHWACH).pack(side="left", padx=(16, 0), pady=(10, 0))
        self.l_kurzstatus = tk.Label(kopf, text="", font=self.f_klein,
                                     bg=theme.BG, fg=theme.TEXT_SCHWACH)
        self.l_kurzstatus.pack(side="right", pady=(10, 0))

        koerper = tk.Frame(self, bg=theme.BG)
        koerper.pack(fill="both", expand=True, padx=28, pady=(0, 20))

        nav = tk.Frame(koerper, bg=theme.BG, width=210)
        nav.pack(side="left", fill="y")
        nav.pack_propagate(False)
        self.nav_eintraege: list[tk.Label] = []
        for i, (beschriftung, _) in enumerate(self.SCHRITTE):
            zeile = tk.Label(nav, text=beschriftung, font=self.f_nav, anchor="w",
                             bg=theme.BG, fg=theme.TEXT_SCHWACH, padx=14, pady=10,
                             cursor="hand2")
            zeile.pack(fill="x", pady=1)
            zeile.bind("<Button-1>", lambda _e, k=i: self.zeige(k))
            self.nav_eintraege.append(zeile)

        self.inhalt = tk.Frame(koerper, bg=theme.PANEL)
        self.inhalt.pack(side="left", fill="both", expand=True, padx=(18, 0))

        fuss = tk.Frame(self, bg=theme.BG, padx=28, pady=0)
        fuss.pack(fill="x", side="bottom")
        self.l_status = tk.Label(fuss, text="", font=self.f_klein, anchor="w",
                                 bg=theme.BG, fg=theme.TEXT_SCHWACH)
        self.l_status.pack(fill="x", pady=(0, 12))

    # -- Zustand --------------------------------------------------------
    def sitzungen_nach_rolle(self) -> dict[str, list[dict]]:
        nach_rolle: dict[str, list[dict]] = {r: [] for r in storage.ROLLEN}
        for zeile in storage.uebersicht():
            nach_rolle.setdefault(zeile["rolle"], []).append(zeile)
        return nach_rolle

    def fertig(self, schritt: int, rollen: dict[str, list[dict]] | None = None,
               modell_da: bool | None = None) -> bool:
        """Ist dieser Schritt erledigt?"""
        if rollen is None:
            rollen = self.sitzungen_nach_rolle()
        if modell_da is None:
            modell_da = training.neuestes_modell() is not None
        if schritt == 1:
            return self.cfg.device is not None
        if schritt == 2:
            return len(TASTEN) >= 2
        if schritt == 3:
            return bool(rollen["train"]) and bool(rollen["val"])
        if schritt == 4:
            return bool(rollen["train"])
        if schritt == 5:
            return modell_da
        if schritt == 6:
            return False
        return True

    def _takt(self) -> None:
        """Regelmaessig nachsehen, ob ein gestartetes Werkzeug fertig ist."""
        beendet = [(datei, p, ausgabe)
                   for datei, (p, ausgabe) in self.prozesse.items()
                   if p.poll() is not None]
        for datei, p, ausgabe in beendet:
            del self.prozesse[datei]
            self._werkzeug_beendet(datei, p.returncode, ausgabe)
        # Nur die Seiten neu aufbauen, die zeigen, was ein Werkzeug erzeugt
        # hat. Andere wuerden dabei halb ausgefuellte Eingaben verlieren.
        if beendet and self.aktiv in (0, 3):
            self.zeige(self.aktiv)
        self._nav_auffrischen()
        self.after(700, self._takt)

    def _nav_auffrischen(self) -> None:
        # Einmal je Takt lesen, nicht einmal je Schritt: Bei vielen Sitzungen
        # sind das sonst dutzende Dateizugriffe pro Sekunde.
        rollen = self.sitzungen_nach_rolle()
        modell_da = training.neuestes_modell() is not None
        for i, zeile in enumerate(self.nav_eintraege):
            ist_fertig = self.fertig(i, rollen, modell_da)
            marke = "✓  " if ist_fertig and i > 0 else "     "
            if i == 0:
                marke = "     "
            zeile.configure(
                text=marke + self.SCHRITTE[i][0],
                font=self.f_nav_fett if i == self.aktiv else self.f_nav,
                bg=theme.PANEL if i == self.aktiv else theme.BG,
                fg=(theme.TEXT if i == self.aktiv
                    else theme.OK if ist_fertig and i > 0 else theme.TEXT_SCHWACH),
            )
        gesamt = sum(z["gesamt"] for liste in rollen.values() for z in liste)
        self.l_kurzstatus.configure(
            text=f"{len(TASTEN)} Klassen     {gesamt} Proben     "
                 f"{'Modell vorhanden' if modell_da else 'noch kein Modell'}")

    def melde(self, text: str, farbe: str = "") -> None:
        self.l_status.configure(text=text, fg=farbe or theme.TEXT_SCHWACH)

    # -- Navigation -----------------------------------------------------
    def zeige(self, schritt: int) -> None:
        self.aktiv = schritt
        for kind in self.inhalt.winfo_children():
            kind.destroy()
        rahmen = tk.Frame(self.inhalt, bg=theme.PANEL, padx=30, pady=26)
        rahmen.pack(fill="both", expand=True)
        getattr(self, self.SCHRITTE[schritt][1])(rahmen)
        self._nav_auffrischen()

    def _kopfzeile(self, eltern, titel: str, unterzeile: str = "") -> None:
        tk.Label(eltern, text=titel, font=self.f_kopf, bg=theme.PANEL,
                 fg=theme.TEXT, anchor="w").pack(fill="x")
        if unterzeile:
            tk.Label(eltern, text=unterzeile, font=self.f_normal, bg=theme.PANEL,
                     fg=theme.TEXT_SCHWACH, anchor="w", justify="left",
                     wraplength=780).pack(fill="x", pady=(6, 0))

    def _absatz(self, eltern, text: str, farbe: str = "") -> tk.Label:
        l = tk.Label(eltern, text=text, font=self.f_normal, bg=theme.PANEL,
                     fg=farbe or theme.TEXT_SCHWACH, anchor="w", justify="left",
                     wraplength=780)
        l.pack(fill="x", pady=(14, 0))
        return l

    def _knopfreihe(self, eltern) -> tk.Frame:
        reihe = tk.Frame(eltern, bg=theme.PANEL)
        reihe.pack(fill="x", pady=(20, 0))
        return reihe

    # -- Werkzeuge starten ----------------------------------------------
    def starte_werkzeug(self, datei: str, *argumente: str) -> bool:
        """Ein Vollbild-Werkzeug in einem eigenen Prozess oeffnen.

        Jedes Werkzeug laeuft hoechstens einmal: Zwei Collector-Fenster
        wuerden zwei Sitzungen gleichzeitig schreiben, zwei Demos sich um das
        Mikrofon streiten.
        """
        name = WERKZEUG_NAMEN.get(datei, datei)
        laufend = self.prozesse.get(datei)
        if laufend is not None and laufend[0].poll() is None:
            self.melde(f"{name} ist schon offen - erst dieses Fenster schließen.",
                       theme.WARN)
            return False
        pfad = WERKZEUGE / datei
        if not pfad.exists():
            messagebox.showerror("Fehlt", f"{pfad} wurde nicht gefunden.")
            return False
        umgebung = {**os.environ, "PYTHONIOENCODING": "utf-8",
                    "PYTHONUNBUFFERED": "1"}
        try:
            p = subprocess.Popen([sys.executable, str(pfad), *argumente],
                                 cwd=str(PROJEKT), env=umgebung,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace")
        except OSError as fehler:
            messagebox.showerror("Start fehlgeschlagen", str(fehler))
            return False
        ausgabe: deque = deque(maxlen=14)

        def mitlesen() -> None:
            # Ausgabe weiterreichen wie bisher - und die letzten Zeilen
            # behalten, falls das Werkzeug mit einem Fehler endet.
            for zeile in p.stdout:
                ausgabe.append(zeile.rstrip())
                if sys.stdout is not None:
                    try:
                        sys.stdout.write(zeile)
                        sys.stdout.flush()
                    except (OSError, ValueError):
                        pass

        threading.Thread(target=mitlesen, daemon=True).start()
        self.prozesse[datei] = (p, ausgabe)
        self.melde(f"{name} läuft - dieses Fenster bleibt offen.", theme.AKZENT)
        return True

    def _werkzeug_beendet(self, datei: str, code: int, ausgabe: deque) -> None:
        """Sagen, wie ein Werkzeug ausgegangen ist - vor allem, wenn es scheiterte."""
        name = WERKZEUG_NAMEN.get(datei, datei)
        if code == 0:
            if datei == "09_selbsttest.py":
                self.melde("Selbsttest bestanden - die Installation läuft.", theme.OK)
            else:
                self.melde(f"{name} geschlossen.")
            return
        zeilen = [ANSI.sub("", z) for z in ausgabe if z.strip()]
        letzte = "\n".join(zeilen[-8:]) or "(keine Ausgabe)"
        self.melde(f"{name} wurde mit einem Fehler beendet.", theme.FEHLER)
        messagebox.showerror(f"{name} - beendet mit Fehler",
                             f"{name} ist nicht sauber gelaufen. Die letzten "
                             f"Meldungen:\n\n{letzte}")

    def _schliessen(self) -> None:
        for p, _ausgabe in self.prozesse.values():
            if p.poll() is None:
                p.terminate()
        self.destroy()

    # ===================================================================
    # 0 - Überblick
    # ===================================================================
    def _ueberblick(self, f) -> None:
        self._kopfzeile(
            f, "Worum es geht",
            "Jede Taste sitzt anders auf der Platte und klingt deshalb etwas "
            "anders. Dieses Programm misst, ob ein kleines neuronales Netz "
            "diesen Unterschied in einem selbst aufgenommenen, kontrollierten "
            "Datensatz wiederfindet - und wie gut."
        )
        self._absatz(
            f,
            "Der Ablauf: Du legst fest, welche Tasten mitspielen. Das Programm "
            "sagt dir eine Taste an, du drückst sie, es speichert nur das kurze "
            "Audiofenster um diesen einen Anschlag. Aus mehreren solchen "
            "Sitzungen entsteht ein Datensatz, darauf lernt ein Netz mit "
            "rund 24 000 Parametern. Getestet wird gegen eine Sitzung, die es "
            "nie gesehen hat.")

        kasten = tk.Frame(f, bg=theme.BG, padx=20, pady=16)
        kasten.pack(fill="x", pady=(20, 0))
        tk.Label(kasten, text="Was dieses Programm nicht ist", font=self.f_normal,
                 bg=theme.BG, fg=theme.WARN, anchor="w").pack(fill="x")
        tk.Label(
            kasten,
            text="Kein Keylogger und kein Decoder für fremde Eingaben. Es wird "
                 "nur im gestarteten Aufnahmemodus aufgenommen, nur die vorher "
                 "angesagte Taste, nur im eigenen Fenster. Das Modell kennt "
                 "ausschließlich einzelne Anschläge aus deinen eigenen "
                 "Aufnahmen und gibt nie etwas anderes aus als die von dir "
                 "gewählten Klassen.",
            font=self.f_klein, bg=theme.BG, fg=theme.TEXT_SCHWACH, anchor="w",
            justify="left", wraplength=760).pack(fill="x", pady=(8, 0))

        # Stand der Dinge
        stand = tk.Frame(f, bg=theme.PANEL)
        stand.pack(fill="x", pady=(24, 0))
        rollen = self.sitzungen_nach_rolle()
        zeilen = [
            ("Mikrofon", self.cfg.device_name or "noch nicht gewählt",
             self.cfg.device is not None),
            ("Klassen", f"{len(TASTEN)}   {' '.join(anzeige(t) for t in TASTEN)}",
             len(TASTEN) >= 2),
            ("Sitzungen", "  ".join(f"{r}: {len(rollen[r])}" for r in storage.ROLLEN),
             bool(rollen["train"]) and bool(rollen["val"])),
            ("Modell", (training.neuestes_modell().name
                        if training.neuestes_modell() else "noch keins"),
             training.neuestes_modell() is not None),
        ]
        for i, (name, wert, ok) in enumerate(zeilen):
            tk.Label(stand, text=name, font=self.f_normal, bg=theme.PANEL,
                     fg=theme.TEXT_SCHWACH, anchor="w", width=12).grid(
                row=i, column=0, sticky="w", pady=3)
            tk.Label(stand, text=wert, font=self.f_mono, bg=theme.PANEL,
                     fg=theme.OK if ok else theme.TEXT_SCHWACH, anchor="w").grid(
                row=i, column=1, sticky="w", pady=3)

        reihe = self._knopfreihe(f)
        Knopf(reihe, "Loslegen", lambda: self.zeige(1), fett=True,
              font=self.f_normal).pack(side="left")
        Knopf(reihe, "Installation prüfen", self._selbsttest,
              font=self.f_normal).pack(side="left", padx=(10, 0))

    def _selbsttest(self) -> None:
        if self.starte_werkzeug("09_selbsttest.py", "--behalten"):
            self.melde("Der Selbsttest läuft im Terminal - er braucht kein "
                       "Mikrofon und fasst deine Aufnahmen nicht an.", theme.AKZENT)

    # ===================================================================
    # 1 - Mikrofon
    # ===================================================================
    def _mikrofon(self, f) -> None:
        self._kopfzeile(
            f, "Mikrofon wählen",
            "Ein Tastenanschlag ist kurz und leise. Gut geeignet ist ein "
            "Mikrofon nah an der Tastatur, ohne Rauschunterdrückung und ohne "
            "Noise Gate - beide schlucken genau den Anteil, um den es hier geht."
        )

        try:
            self.geraete = audio.eingaenge()
        except Exception as fehler:                        # noqa: BLE001
            self._absatz(f, f"Audio-Geräte nicht lesbar: {fehler}", theme.FEHLER)
            return
        if not self.geraete:
            self._absatz(f, "Kein Eingangsgerät gefunden.", theme.FEHLER)
            return

        rahmen = tk.Frame(f, bg=theme.PANEL)
        rahmen.pack(fill="both", expand=True, pady=(18, 0))
        leiste = tk.Scrollbar(rahmen, orient="vertical")
        self.liste = tk.Listbox(
            rahmen, font=self.f_mono, bg=theme.BG, fg=theme.TEXT,
            selectbackground=theme.AKZENT, selectforeground=theme.BG,
            highlightthickness=0, borderwidth=0, activestyle="none",
            yscrollcommand=leiste.set, height=11)
        leiste.configure(command=self.liste.yview)
        leiste.pack(side="right", fill="y")
        self.liste.pack(side="left", fill="both", expand=True)

        for i, g in enumerate(self.geraete):
            marke = "  " if g.geeignet else "! "
            self.liste.insert("end", f"{marke}{g.label}")
            if not g.geeignet:
                self.liste.itemconfigure(i, foreground=theme.TEXT_SCHWACH)
            if g.index == self.cfg.device:
                self.liste.selection_set(i)
                self.liste.see(i)
        if not self.liste.curselection():
            self.liste.selection_set(0)

        self.l_messung = tk.Label(f, text="", font=self.f_mono, bg=theme.PANEL,
                                  fg=theme.TEXT_SCHWACH, anchor="w", justify="left")
        self.l_messung.pack(fill="x", pady=(14, 0))

        reihe = self._knopfreihe(f)
        self.k_messen = Knopf(reihe, "2 Sekunden mithören", self._pegel_messen,
                              font=self.f_normal)
        self.k_messen.pack(side="left")
        Knopf(reihe, "Übernehmen", self._geraet_uebernehmen, fett=True,
              font=self.f_normal).pack(side="left", padx=(10, 0))
        Knopf(reihe, "Pegel im Vollbild", lambda: self.starte_werkzeug(
            "02_kalibrierung.py"), font=self.f_normal).pack(side="left", padx=(10, 0))

    def _gewaehltes_geraet(self):
        auswahl = self.liste.curselection()
        return self.geraete[auswahl[0]] if auswahl else None

    def _pegel_messen(self) -> None:
        g = self._gewaehltes_geraet()
        if g is None:
            return
        self.k_messen.setze_aktiv(False)
        self.l_messung.configure(text="läuft - bitte jetzt ein paar Mal tippen …",
                                 fg=theme.AKZENT)
        ergebnis: queue.Queue = queue.Queue()

        def messen() -> None:
            import numpy as np
            try:
                sr, ch = audio.bestes_format(g, self.cfg.samplerate)
                probe = Config(**{**self.cfg.__dict__, "device": g.index,
                                  "device_name": g.name, "hostapi": g.hostapi,
                                  "samplerate": sr, "channels": ch})
                ring = audio.Ringpuffer(probe)
                ring.start()
                threading.Event().wait(2.0)
                x = ring.letzte(2.0)
                ring.stop()
                # Rauschboden als 20. Perzentil der Kurzzeit-Energie: robust
                # gegen die Anschlaege, die waehrend der Messung passieren.
                fenster = max(int(0.01 * sr), 16)
                teile = x[: (x.size // fenster) * fenster].reshape(-1, fenster)
                pegel = 20 * np.log10(np.maximum(
                    np.sqrt(np.mean(np.square(teile, dtype=np.float64), axis=1)), 1e-9))
                ergebnis.put(("ok", audio.peak_dbfs(x),
                              float(np.percentile(pegel, 20)), sr, ch))
            except Exception as fehler:                    # noqa: BLE001
                ergebnis.put(("fehler", str(fehler), 0, 0, 0))

        threading.Thread(target=messen, daemon=True).start()

        def nachsehen() -> None:
            try:
                art, a, b, sr, ch = ergebnis.get_nowait()
            except queue.Empty:
                self.after(120, nachsehen)
                return
            self.k_messen.setze_aktiv(True)
            if art == "fehler":
                self.l_messung.configure(text=f"Öffnen fehlgeschlagen: {a}",
                                         fg=theme.FEHLER)
                return
            abstand = a - b
            if abstand >= 25:
                urteil, farbe = "gut - Anschläge heben sich klar ab", theme.OK
            elif abstand >= 15:
                urteil, farbe = "brauchbar - näher ran wäre besser", theme.WARN
            else:
                urteil, farbe = ("zu wenig Abstand zum Rauschen - näher ran, "
                                 "Gain hoch, Lüfter aus"), theme.FEHLER
            self.l_messung.configure(
                text=f"Spitze {a:6.1f} dBFS     Rauschboden {b:6.1f} dBFS     "
                     f"Abstand {abstand:4.1f} dB     {sr} Hz / {ch} Kanal\n{urteil}",
                fg=farbe)

        self.after(120, nachsehen)

    def _geraet_uebernehmen(self) -> None:
        g = self._gewaehltes_geraet()
        if g is None:
            return
        sr, ch = audio.bestes_format(g, self.cfg.samplerate)
        self.cfg.device, self.cfg.device_name = g.index, g.name
        self.cfg.hostapi, self.cfg.samplerate, self.cfg.channels = g.hostapi, sr, ch
        self.cfg.latenz_ms = g.latenz_ms
        self.cfg.speichern()
        self.melde(f"Gespeichert: {g.name} ({g.hostapi}, {sr} Hz).", theme.OK)
        self.zeige(2)

    # ===================================================================
    # 2 - Klassen
    # ===================================================================
    def _klassen(self, f) -> None:
        self._kopfzeile(
            f, "Klassen festlegen",
            "Welche Tasten soll das Modell unterscheiden? Zwei bis vierzig "
            "Zeichen. Je mehr Klassen, desto schwerer die Aufgabe - und desto "
            "mehr Aufnahmen braucht es. Acht Klassen mit je 40 Proben sind ein "
            "guter Anfang, das ganze Alphabet ist ein Abend Arbeit."
        )

        vorlagen = tk.Frame(f, bg=theme.PANEL)
        vorlagen.pack(fill="x", pady=(18, 0))
        tk.Label(vorlagen, text="Vorlage", font=self.f_klein, bg=theme.PANEL,
                 fg=theme.TEXT_SCHWACH).pack(side="left", padx=(0, 10))
        for name, zeichen in VORLAGEN.items():
            Knopf(vorlagen, name, lambda z=zeichen: self._vorlage(z),
                  font=self.f_klein).pack(side="left", padx=(0, 8))

        raster = tk.Frame(f, bg=theme.PANEL)
        raster.pack(fill="x", pady=(18, 0))

        tk.Label(raster, text="Zeichen", font=self.f_normal, bg=theme.PANEL,
                 fg=theme.TEXT_SCHWACH, anchor="w", width=16).grid(
            row=0, column=0, sticky="w", pady=6)
        self.e_klassen = tk.Entry(raster, font=self.f_mono, bg=theme.BG,
                                  fg=theme.TEXT, insertbackground=theme.TEXT,
                                  relief="flat", width=46)
        self.e_klassen.insert(0, "".join(TASTEN))
        self.e_klassen.grid(row=0, column=1, sticky="w", ipady=6, ipadx=8)
        self.e_klassen.bind("<KeyRelease>", lambda _e: self._klassen_vorschau())

        tk.Label(raster, text="Proben je Klasse", font=self.f_normal,
                 bg=theme.PANEL, fg=theme.TEXT_SCHWACH, anchor="w", width=16).grid(
            row=1, column=0, sticky="w", pady=6)
        self.e_ziel = tk.Spinbox(raster, from_=ZIEL_MIN, to=ZIEL_MAX,
                                 font=self.f_mono, bg=theme.BG, fg=theme.TEXT,
                                 width=6, relief="flat",
                                 buttonbackground=theme.PANEL,
                                 insertbackground=theme.TEXT,
                                 command=self._klassen_vorschau)
        self.e_ziel.delete(0, "end")
        self.e_ziel.insert(0, str(self.cfg.ziel_pro_taste))
        self.e_ziel.grid(row=1, column=1, sticky="w", ipady=5)
        self.e_ziel.bind("<KeyRelease>", lambda _e: self._klassen_vorschau())

        tk.Label(raster, text="Sperrfolge", font=self.f_normal, bg=theme.PANEL,
                 fg=theme.TEXT_SCHWACH, anchor="w", width=16).grid(
            row=2, column=0, sticky="w", pady=6)
        self.e_sperre = tk.Entry(raster, font=self.f_mono, bg=theme.BG,
                                 fg=theme.TEXT, insertbackground=theme.TEXT,
                                 relief="flat", width=24)
        self.e_sperre.insert(0, self.cfg.sperrfolge)
        self.e_sperre.grid(row=2, column=1, sticky="w", ipady=6, ipadx=8)
        tk.Label(raster, text="optional: ein Wort aus deinen Klassen, das du später "
                              "blind testen willst. Es kommt dann in keiner "
                              "Aufnahmefolge vor.",
                 font=self.f_klein, bg=theme.PANEL, fg=theme.TEXT_SCHWACH,
                 anchor="w").grid(row=3, column=1, sticky="w")

        self.vorschau = tk.Canvas(f, bg=theme.PANEL, height=120,
                                  highlightthickness=0)
        self.vorschau.pack(fill="x", pady=(20, 0))
        self.l_klassen_hinweis = tk.Label(f, text="", font=self.f_normal,
                                          bg=theme.PANEL, fg=theme.TEXT_SCHWACH,
                                          anchor="w", justify="left", wraplength=780)
        self.l_klassen_hinweis.pack(fill="x", pady=(8, 0))

        reihe = self._knopfreihe(f)
        Knopf(reihe, "Speichern", self._klassen_speichern, fett=True,
              font=self.f_normal).pack(side="left")
        self._klassen_vorschau()

    def _vorlage(self, zeichen: str) -> None:
        self.e_klassen.delete(0, "end")
        self.e_klassen.insert(0, zeichen)
        self._klassen_vorschau()

    def _klassen_vorschau(self) -> None:
        text = self.e_klassen.get()
        self.vorschau.delete("all")
        try:
            liste = pruefe_klassen(text)
        except KlassenFehler as fehler:
            self.l_klassen_hinweis.configure(text=str(fehler), fg=theme.FEHLER)
            return

        farben = theme.palette(len(liste))
        kachel, luecke = 34, 6
        pro_zeile = max(int((self.vorschau.winfo_width() or 760) // (kachel + luecke)), 1)
        for i, (zeichen, farbe) in enumerate(zip(liste, farben)):
            zeile, spalte = divmod(i, pro_zeile)
            x0 = spalte * (kachel + luecke)
            y0 = zeile * (kachel + luecke) + 4
            rundes_rechteck(self.vorschau, x0, y0, x0 + kachel, y0 + kachel, 7,
                            fill=farbe, outline="")
            self.vorschau.create_text(x0 + kachel / 2, y0 + kachel / 2,
                                      text=anzeige(zeichen), fill=theme.BG,
                                      font=(self.fam, 13, "bold"))

        try:
            ziel = int(self.e_ziel.get())
        except (ValueError, tk.TclError):
            ziel = self.cfg.ziel_pro_taste
        minuten = len(liste) * ziel * 2.4 / 60
        self.l_klassen_hinweis.configure(
            text=f"{len(liste)} Klassen - blindes Raten trifft "
                 f"{100 / len(liste):.1f} %. Eine volle Sitzung sind "
                 f"{len(liste) * ziel} Anschläge, ungefähr {minuten:.0f} Minuten.",
            fg=theme.TEXT_SCHWACH)

    def _klassen_speichern(self) -> None:
        try:
            liste = pruefe_klassen(self.e_klassen.get())
        except KlassenFehler as fehler:
            messagebox.showerror("Geht so nicht", str(fehler))
            return
        try:
            ziel = int(self.e_ziel.get())
        except ValueError:
            ziel = 0
        if not ZIEL_MIN <= ziel <= ZIEL_MAX:
            messagebox.showerror(
                "Geht so nicht",
                f"Proben je Klasse: eine ganze Zahl von {ZIEL_MIN} bis {ZIEL_MAX}.")
            return
        sperre = self.e_sperre.get().strip().lower()
        fremd = sorted({c for c in sperre if c not in liste})
        if fremd:
            messagebox.showerror(
                "Geht so nicht",
                "Die Sperrfolge enthält Zeichen, die keine Klasse sind: "
                f"{' '.join(fremd)}\n\nSie soll ein Wort sein, das du später "
                "blind testest - das geht nur mit Zeichen, die das Modell kennt.")
            return
        if len(sperre) == 1:
            messagebox.showerror(
                "Geht so nicht",
                "Die Sperrfolge braucht mindestens zwei Zeichen. Ein einzelnes "
                "Zeichen ließe sich nicht aufnehmen, ohne es zu verwenden.")
            return

        vorhanden = storage.uebersicht()
        if vorhanden and list(TASTEN) != liste:
            weiter = messagebox.askokcancel(
                "Es liegen schon Aufnahmen vor",
                f"In daten/roh/ liegen {len(vorhanden)} Sitzungen mit den Klassen "
                f"{''.join(TASTEN)}.\n\nMit einem anderen Zeichensatz lassen sie "
                f"sich nicht weiterverwenden - das Training bricht dann ab, statt "
                f"stillschweigend zu mischen.\n\nTrotzdem auf {''.join(liste)} "
                f"umstellen?")
            if not weiter:
                return

        if list(TASTEN) != liste:
            # Ergebnisse zu den alten Klassen gelten nicht mehr.
            self.ergebnis = self.verlauf = self.test_ergebnis = None
        self.cfg.klassen = "".join(liste)
        self.cfg.sperrfolge = sperre
        self.cfg.ziel_pro_taste = ziel
        self.cfg.anwenden()
        self.cfg.speichern()
        self.melde(f"{len(liste)} Klassen gespeichert: "
                   f"{' '.join(anzeige(t) for t in TASTEN)}", theme.OK)
        self.zeige(3)

    # ===================================================================
    # 3 - Aufnehmen
    # ===================================================================
    def _aufnehmen(self, f) -> None:
        self._kopfzeile(
            f, "Daten aufnehmen",
            "Der Collector sagt dir eine Taste an, du drückst sie, er speichert "
            "das Audiofenster darum. Immer in derselben Haltung, ruhig und "
            "gleichmäßig - und zwischen zwei Anschlägen kurz warten."
        )
        self._absatz(
            f,
            "Du brauchst mindestens drei Sitzungen: eine oder mehrere zum "
            "Trainieren, eine zum Mitprüfen während des Trainings (val) und eine "
            "zum ehrlichen Testen (test). Die Testsitzung entsteht am besten an "
            "einem anderen Tag - nur dann misst sie, was das Modell wirklich "
            "kann, und nicht bloß, wie gleich der Raum geblieben ist.")

        if self.cfg.device is None:
            self._absatz(f, "Erst in Schritt 1 ein Mikrofon wählen.", theme.WARN)
            return

        reihe = self._knopfreihe(f)
        Knopf(reihe, "Collector öffnen", lambda: self.starte_werkzeug(
            "03_collector.py", "--ziel", str(self.cfg.ziel_pro_taste)),
            fett=True, font=self.f_normal).pack(side="left")
        tk.Label(reihe, text=f"{self.cfg.ziel_pro_taste} Proben je Klasse, "
                             f"{len(TASTEN) * self.cfg.ziel_pro_taste} insgesamt",
                 font=self.f_klein, bg=theme.PANEL,
                 fg=theme.TEXT_SCHWACH).pack(side="left", padx=(14, 0))

        tk.Label(f, text="Aufgenommene Sitzungen", font=self.f_normal,
                 bg=theme.PANEL, fg=theme.TEXT, anchor="w").pack(
            fill="x", pady=(26, 8))

        zeilen = storage.uebersicht()
        if not zeilen:
            self._absatz(f, "Noch keine. Oben den Collector öffnen.",
                         theme.TEXT_SCHWACH)
            return

        tabelle = tk.Frame(f, bg=theme.PANEL)
        tabelle.pack(fill="x")
        self.rollen_wahl: dict[str, tk.StringVar] = {}
        for i, zeile in enumerate(zeilen):
            tk.Label(tabelle, text=zeile["session_id"], font=self.f_mono,
                     bg=theme.PANEL, fg=theme.TEXT, anchor="w").grid(
                row=i, column=0, sticky="w", pady=2, padx=(0, 18))
            tk.Label(tabelle, text=f"{zeile['gesamt']:>4} Proben", font=self.f_mono,
                     bg=theme.PANEL, fg=theme.TEXT_SCHWACH, anchor="w").grid(
                row=i, column=1, sticky="w", padx=(0, 18))
            var = tk.StringVar(value=zeile["rolle"])
            self.rollen_wahl[zeile["session_id"]] = var
            wahl = tk.OptionMenu(tabelle, var, *storage.ROLLEN)
            wahl.configure(bg=theme.BG, fg=theme.TEXT, font=self.f_klein,
                           relief="flat", highlightthickness=0, width=6,
                           activebackground=theme.GRID, activeforeground=theme.TEXT)
            wahl["menu"].configure(bg=theme.BG, fg=theme.TEXT, font=self.f_klein)
            wahl.grid(row=i, column=2, sticky="w", padx=(0, 18))
            tk.Label(tabelle, text=zeile["notiz"][:40], font=self.f_klein,
                     bg=theme.PANEL, fg=theme.TEXT_SCHWACH, anchor="w").grid(
                row=i, column=3, sticky="w")

        reihe2 = self._knopfreihe(f)
        Knopf(reihe2, "Rollen speichern", self._rollen_speichern, fett=True,
              font=self.f_normal).pack(side="left")

        rollen = self.sitzungen_nach_rolle()
        fehlt = [r for r in ("train", "val", "test") if not rollen[r]]
        if fehlt:
            self._absatz(f, "Es fehlt noch eine Sitzung für: " + ", ".join(fehlt),
                         theme.WARN)

    def _rollen_speichern(self) -> None:
        for sitzung_id, var in self.rollen_wahl.items():
            storage.rolle_setzen(sitzung_id, var.get())
        self.melde("Rollen gespeichert.", theme.OK)
        self.zeige(3)

    # ===================================================================
    # 4 - Daten prüfen
    # ===================================================================
    def _pruefen(self, f) -> None:
        self._kopfzeile(
            f, "Daten prüfen",
            "Bevor trainiert wird: Sind die Aufnahmen brauchbar? Geprüft werden "
            "Pegel, Abstand zum Rauschen, Übersteuerung und ob eine "
            "Rauschunterdrückung dazwischengefunkt hat."
        )
        if not storage.sitzungen():
            self._absatz(f, "Noch keine Aufnahmen da.", theme.WARN)
            return

        reihe = self._knopfreihe(f)
        Knopf(reihe, "Prüfen", self._pruefung_starten, fett=True,
              font=self.f_normal).pack(side="left")
        Knopf(reihe, "Trennbarkeit messen", self._trennbarkeit,
              font=self.f_normal).pack(side="left", padx=(10, 0))

        self.ausgabe = tk.Text(f, font=self.f_mono, bg=theme.BG, fg=theme.TEXT,
                               relief="flat", height=16, wrap="none",
                               insertbackground=theme.TEXT)
        self.ausgabe.pack(fill="both", expand=True, pady=(18, 0))
        self.ausgabe.insert("1.0", "Noch nichts geprüft.")
        self.ausgabe.configure(state="disabled")

    def _werkzeug_ausgabe(self, datei: str, *argumente: str) -> None:
        """Ein Kommandozeilen-Werkzeug laufen lassen und die Ausgabe zeigen."""
        self.ausgabe.configure(state="normal")
        self.ausgabe.delete("1.0", "end")
        self.ausgabe.insert("1.0", "läuft …")
        self.ausgabe.configure(state="disabled")
        self.update_idletasks()

        ergebnis: queue.Queue = queue.Queue()

        def laufen() -> None:
            try:
                # UTF-8 erzwingen: Sonst schreibt das Werkzeug in die Pipe mit
                # der Windows-Codepage, und Umlaute kommen hier kaputt an.
                p = subprocess.run(
                    [sys.executable, str(WERKZEUGE / datei), *argumente],
                    cwd=str(PROJEKT), capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=600,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                ergebnis.put(ANSI.sub("", (p.stdout or "") + (p.stderr or "")))
            except Exception as fehler:                    # noqa: BLE001
                ergebnis.put(f"{type(fehler).__name__}: {fehler}")

        threading.Thread(target=laufen, daemon=True).start()

        def nachsehen() -> None:
            try:
                text = ergebnis.get_nowait()
            except queue.Empty:
                self.after(150, nachsehen)
                return
            self.ausgabe.configure(state="normal")
            self.ausgabe.delete("1.0", "end")
            self.ausgabe.insert("1.0", text.strip() or "(keine Ausgabe)")
            self.ausgabe.configure(state="disabled")

        self.after(150, nachsehen)

    def _pruefung_starten(self) -> None:
        self._werkzeug_ausgabe("05_daten_pruefen.py")

    def _trennbarkeit(self) -> None:
        self._werkzeug_ausgabe("06_trennbarkeit.py")

    # ===================================================================
    # 5 - Training
    # ===================================================================
    def _training(self, f) -> None:
        self._kopfzeile(
            f, "Modell trainieren",
            "Ein kleines Faltungsnetz lernt aus den Log-Mel-Bildern der "
            "Trainingssitzungen. Geprüft wird nach jeder Epoche gegen die "
            "Validierungssitzung - eine andere Aufnahme, nie dieselben Proben."
        )

        kopf = tk.Frame(f, bg=theme.PANEL)
        kopf.pack(fill="x", pady=(16, 0))
        tk.Label(kopf, text="Epochen", font=self.f_klein, bg=theme.PANEL,
                 fg=theme.TEXT_SCHWACH).pack(side="left")
        self.e_epochen = tk.Spinbox(kopf, from_=10, to=400, increment=10,
                                    font=self.f_mono, bg=theme.BG, fg=theme.TEXT,
                                    width=5, relief="flat",
                                    buttonbackground=theme.PANEL,
                                    insertbackground=theme.TEXT)
        self.e_epochen.delete(0, "end")
        self.e_epochen.insert(0, "80")
        self.e_epochen.pack(side="left", padx=(8, 20), ipady=3)
        self.k_training = Knopf(kopf, "Training starten", self._training_starten,
                                fett=True, font=self.f_normal)
        self.k_training.pack(side="left")
        self.l_training = tk.Label(kopf, text="", font=self.f_klein, bg=theme.PANEL,
                                   fg=theme.TEXT_SCHWACH)
        self.l_training.pack(side="left", padx=(16, 0))

        self.fig_training = Figure(figsize=(8.2, 3.3), dpi=100, facecolor=theme.PANEL)
        self.ax_verlust = self.fig_training.add_subplot(1, 2, 1)
        self.ax_quote = self.fig_training.add_subplot(1, 2, 2)
        self._training_achsen()
        self.fig_training.tight_layout(pad=1.6)
        self.canvas_training = FigureCanvasTkAgg(self.fig_training, master=f)
        self.canvas_training.get_tk_widget().pack(fill="both", expand=True,
                                                  pady=(14, 0))

        self.l_ergebnis = tk.Label(f, text="", font=self.f_normal, bg=theme.PANEL,
                                   fg=theme.TEXT_SCHWACH, anchor="w",
                                   justify="left", wraplength=800)
        self.l_ergebnis.pack(fill="x", pady=(10, 0))

        # Die Reihe steht von Anfang an, damit sie beim Zurueckkommen auf
        # diesen Schritt nicht fehlt - sie ist nur so lange stumm, bis ein
        # Ergebnis vorliegt.
        reihe = self._knopfreihe(f)
        self.k_grafiken = Knopf(reihe, "Grafiken im Hochformat speichern",
                                self._grafiken_exportieren, font=self.f_normal)
        self.k_grafiken.pack(side="left")
        self.k_test = Knopf(reihe, "Auf Testsitzung prüfen", self._test_starten,
                            font=self.f_normal)
        self.k_test.pack(side="left", padx=(10, 0))
        self.k_weiter = Knopf(reihe, "Weiter zum Live-Test", lambda: self.zeige(6),
                              fett=True, font=self.f_normal)
        self.k_weiter.pack(side="left", padx=(10, 0))
        for k in (self.k_grafiken, self.k_weiter):
            k.setze_aktiv(self.ergebnis is not None)

        self.l_test = tk.Label(f, text="", font=self.f_normal, bg=theme.PANEL,
                               fg=theme.TEXT_SCHWACH, anchor="w",
                               justify="left", wraplength=800)
        self.l_test.pack(fill="x", pady=(12, 0))

        laeuft = (self.trainings_thread is not None
                  and self.trainings_thread.is_alive())
        if self.verlauf and self.verlauf["epoche"]:
            # Beim Zurueckkommen auf diesen Schritt die Kurven wieder zeigen -
            # das Panel wird jedes Mal neu gebaut.
            self._kurven_zeichnen()
        if laeuft:
            self.k_training.setze_aktiv(False)
            self.l_training.configure(text="läuft …")
        elif self.ergebnis is not None:
            self._training_ergebnis_zeigen(self.ergebnis)
        else:
            try:
                train, val = training.pruefe_daten()
                self.l_training.configure(
                    text=f"{len(train)} Trainings- und {len(val)} "
                         f"Validierungsproben bereit")
            except training.DatenFehler as fehler:
                self.l_training.configure(text=str(fehler), fg=theme.WARN)
                self.k_training.setze_aktiv(False)
            except ValueError as fehler:
                self.l_training.configure(text=str(fehler), fg=theme.FEHLER)
                self.k_training.setze_aktiv(False)
        self._test_anzeige()

    def _training_achsen(self) -> None:
        for ax, titel in ((self.ax_verlust, "Fehler"), (self.ax_quote, "Trefferquote")):
            ax.clear()
            ax.set_facecolor(theme.BG)
            ax.set_title(titel, color=theme.TEXT, fontsize=10, fontweight="bold")
            ax.tick_params(colors=theme.TEXT_SCHWACH, labelsize=8)
            ax.grid(color=theme.GRID, alpha=0.4, linewidth=0.7)
            for seite in ("top", "right"):
                ax.spines[seite].set_visible(False)
            for seite in ("bottom", "left"):
                ax.spines[seite].set_color(theme.GRID)
        self.ax_quote.set_ylim(0, 100)
        self.ax_quote.axhline(zufall() * 100, color=theme.TEXT_SCHWACH,
                              ls="--", lw=1.4)

    def _training_starten(self) -> None:
        if self.trainings_thread is not None and self.trainings_thread.is_alive():
            return
        try:
            epochen = int(self.e_epochen.get())
        except (ValueError, tk.TclError):
            return
        self.k_training.setze_aktiv(False)
        self.l_ergebnis.configure(text="")
        self.verlauf = {"epoche": [], "train_loss": [], "val_loss": [],
                        "train_acc": [], "val_acc": []}
        self._training_achsen()
        self.canvas_training.draw()

        def laufen() -> None:
            try:
                for stand in training.trainiere(epochen=epochen):
                    self.trainings_queue.put(stand)
            except Exception as fehler:                    # noqa: BLE001
                self.trainings_queue.put(fehler)

        self.trainings_thread = threading.Thread(target=laufen, daemon=True)
        self.trainings_thread.start()
        self.after(120, self._training_takt)

    def _sichtbar(self) -> bool:
        """Steht der Trainingsschritt noch auf dem Schirm?

        Das Training laeuft in einem eigenen Thread weiter, auch wenn jemand
        zwischendurch auf einen anderen Schritt klickt. Dann sind die Widgets
        dieses Panels zerstoert - eingesammelt wird trotzdem, gezeichnet nicht.
        """
        return self.aktiv == 5 and bool(self.l_training.winfo_exists())

    def _training_takt(self) -> None:
        neu = False
        while True:
            try:
                nachricht = self.trainings_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(nachricht, Exception):
                if self._sichtbar():
                    self.k_training.setze_aktiv(True)
                    self.l_training.configure(text=str(nachricht), fg=theme.FEHLER)
                else:
                    self.melde(f"Training abgebrochen: {nachricht}", theme.FEHLER)
                return
            if isinstance(nachricht, training.Ergebnis):
                self.ergebnis = nachricht
                self.test_ergebnis = None      # gehoerte zum vorigen Modell
                if self._sichtbar():
                    self.k_training.setze_aktiv(True)
                    self._training_ergebnis_zeigen(nachricht)
                else:
                    self.melde(f"Training fertig: beste Validation "
                               f"{nachricht.beste_val * 100:.1f} %", theme.OK)
                self._nav_auffrischen()
                return
            v = self.verlauf
            v["epoche"].append(nachricht.epoche)
            v["train_loss"].append(nachricht.train_loss)
            v["val_loss"].append(nachricht.val_loss)
            v["train_acc"].append(nachricht.train_acc * 100)
            v["val_acc"].append(nachricht.val_acc * 100)
            if self._sichtbar():
                self.l_training.configure(
                    text=f"Epoche {nachricht.epoche} von {nachricht.epochen}     "
                         f"Validation {nachricht.val_acc * 100:.1f} %     "
                         f"beste {nachricht.beste_val * 100:.1f} %",
                    fg=theme.TEXT_SCHWACH)
            neu = True

        if neu and self._sichtbar():
            self._kurven_zeichnen()

        self.after(120, self._training_takt)

    def _kurven_zeichnen(self) -> None:
        v = self.verlauf
        self._training_achsen()
        self.ax_verlust.plot(v["epoche"], v["train_loss"], color=theme.AKZENT,
                             lw=1.6, label="Training")
        self.ax_verlust.plot(v["epoche"], v["val_loss"], color=theme.AKZENT2,
                             lw=1.6, label="Validation")
        self.ax_verlust.legend(fontsize=8, facecolor=theme.BG,
                               edgecolor=theme.GRID, labelcolor=theme.TEXT)
        self.ax_quote.plot(v["epoche"], v["train_acc"], color=theme.AKZENT, lw=1.6)
        self.ax_quote.plot(v["epoche"], v["val_acc"], color=theme.AKZENT2, lw=1.6)
        self.ax_quote.set_ylim(0, 100)
        self.fig_training.tight_layout(pad=1.6)
        self.canvas_training.draw()

    def _training_ergebnis_zeigen(self, e: training.Ergebnis) -> None:
        farbe = theme.quoten_farbe(e.beste_val)
        # Klassen ohne Validierungsprobe haben keine Quote - sie tauchen hier
        # nicht als "0 %" auf.
        schwach = sorted(((t, q) for t, q in e.je_klasse.items() if q == q),
                         key=lambda p: p[1])[:3]
        self.l_training.configure(
            text=f"fertig in {e.dauer_s:.0f} s", fg=theme.TEXT_SCHWACH)
        self.l_ergebnis.configure(
            text=f"Beste Validation {e.beste_val * 100:.1f} %  "
                 f"(Epoche {e.beste_epoche}, {e.faktor:.1f}-fach über dem "
                 f"Zufall von {zufall() * 100:.1f} %)   ·   {e.parameter:,} "
                 f"Parameter   ·   {e.n_train} Trainings- und {e.n_val} "
                 f"Validierungsproben\n"
                 f"Am schwächsten: "
                 + "   ".join(f"{anzeige(t)} {q * 100:.0f} %" for t, q in schwach)
                 + f"\nGespeichert: {e.modell_pfad.name if e.modell_pfad else '-'}"
                 + "\nDas ist noch nicht die ehrliche Zahl - die kommt aus einer "
                   "Testsitzung, die das Modell nie gesehen hat.",
            fg=farbe)
        for k in (self.k_grafiken, self.k_weiter):
            k.setze_aktiv(True)
        self._test_anzeige()

    def _grafiken_exportieren(self) -> None:
        if self.ergebnis is None:
            return
        import numpy as np
        from tastenakustik import plots, portrait

        e = self.ergebnis
        pfade = []
        theme.anwenden("hochformat")
        try:
            epochen = np.arange(1, len(e.verlauf["val_acc"]) + 1)
            fig = plots.trainingsverlauf(epochen, e.verlauf["train_loss"],
                                         e.verlauf["val_loss"], e.verlauf["train_acc"],
                                         e.verlauf["val_acc"])
            pfade.append(portrait.exportiere(fig, "06_training", "04_modell"))
            fig = plots.konfusionsmatrix(e.konfusion, e.beste_val)
            pfade.append(portrait.exportiere(fig, "08_confusion", "04_modell"))
        except Exception as fehler:                        # noqa: BLE001
            self.melde(f"Grafiken nicht gespeichert: {fehler}", theme.FEHLER)
            return
        finally:
            # Sonst bleibt der grosse Hochformat-Stil fuer die Live-Kurven stehen.
            theme.anwenden("normal")
        self.melde(f"Gespeichert in {pfade[0].parent}", theme.OK)

    # -- Testsitzung ----------------------------------------------------
    def _test_anzeige(self) -> None:
        """Knopf und Zeile fuer die Testsitzung auf den aktuellen Stand bringen."""
        l_test = getattr(self, "l_test", None)
        if self.aktiv != 5 or l_test is None or not l_test.winfo_exists():
            return
        modell_da = training.neuestes_modell() is not None
        test_da = bool(self.sitzungen_nach_rolle()["test"])
        trainiert = (self.trainings_thread is not None
                     and self.trainings_thread.is_alive())
        self.k_test.setze_aktiv(modell_da and test_da and not trainiert
                                and not self.test_laeuft)
        t = self.test_ergebnis
        if self.test_laeuft:
            self.l_test.configure(text="Testsitzung wird ausgewertet …",
                                  fg=theme.AKZENT)
        elif t is not None:
            vergleich = (f"   ·   Validation zum Vergleich {t.val_quote * 100:.1f} %"
                         if t.val_quote is not None else "")
            self.l_test.configure(
                text=f"Testsitzung: {t.quote * 100:.1f} %  ({t.faktor:.1f}-fach "
                     f"über dem Zufall, {t.n} Proben aus "
                     f"{', '.join(t.sitzungen)}){vergleich}\n"
                     f"Das ist die ehrliche Zahl. Gespeichert in "
                     f"{t.pfad.name if t.pfad else '-'}.",
                fg=theme.quoten_farbe(t.quote))
        elif modell_da and not test_da:
            self.l_test.configure(
                text="Für die ehrliche Zahl fehlt noch eine Sitzung mit der Rolle "
                     "„test“ - am besten an einem anderen Tag aufgenommen.",
                fg=theme.TEXT_SCHWACH)
        else:
            self.l_test.configure(text="", fg=theme.TEXT_SCHWACH)

    def _test_starten(self) -> None:
        if self.test_laeuft:
            return
        self.test_laeuft = True
        self._test_anzeige()
        ergebnis: queue.Queue = queue.Queue()

        def laufen() -> None:
            try:
                ergebnis.put(training.teste())
            except Exception as fehler:                    # noqa: BLE001
                ergebnis.put(fehler)

        threading.Thread(target=laufen, daemon=True).start()

        def nachsehen() -> None:
            try:
                antwort = ergebnis.get_nowait()
            except queue.Empty:
                self.after(150, nachsehen)
                return
            self.test_laeuft = False
            if isinstance(antwort, Exception):
                self.test_ergebnis = None
                self.melde(f"Test nicht möglich: {antwort}", theme.FEHLER)
            else:
                self.test_ergebnis = antwort
                self.melde(f"Testsitzung: {antwort.quote * 100:.1f} %",
                           theme.quoten_farbe(antwort.quote))
            self._test_anzeige()

        self.after(150, nachsehen)

    # ===================================================================
    # 6 - Live testen
    # ===================================================================
    def _testen(self, f) -> None:
        self._kopfzeile(
            f, "Modell live testen",
            "Die Demo öffnet sich im Hochformat und hört zu. Sie liest keine "
            "Tastatur-Ereignisse - sie findet Anschläge allein im Audiosignal "
            "und ordnet jeden einer Klasse zu. Bedient wird sie deshalb nur mit "
            "der Maus: Knöpfe unter dem Bild, im Bühnenmodus ein Rechtsklick "
            "ins Bild. Die Tastatur bleibt ganz frei zum Testen."
        )

        modell_pfad = training.neuestes_modell()
        if modell_pfad is None:
            self._absatz(f, "Es gibt noch kein trainiertes Modell.", theme.WARN)
            return
        # Die Demo nimmt die Klassen aus der Modelldatei. Nach einem
        # Klassenwechsel ohne neues Training sind das andere als eingestellt.
        self.demo_klassen = training.modell_klassen(modell_pfad) or list(TASTEN)

        self._absatz(
            f,
            "Zwei Dinge, die du wissen solltest, bevor du dich wunderst:\n\n"
            "Einzeln und mit Pause gedrückte Tasten erkennt das Modell deutlich "
            "besser als flüssiges Tippen. Es hat nie gelernt, wie Tippen klingt "
            "- in den Trainingsdaten lag rund eine Sekunde zwischen zwei "
            "Anschlägen.\n\n"
            "Zwischen zwei erkannten Anschlägen liegen mindestens 300 ms. Sonst "
            "zählt das Loslassen einer Taste als eigener Anschlag, und du "
            "bekommst doppelte Buchstaben.")

        tk.Label(f, text=f"Modell: {modell_pfad.name}   Klassen: "
                         f"{' '.join(anzeige(t) for t in self.demo_klassen)}",
                 font=self.f_mono, bg=theme.PANEL, fg=theme.TEXT_SCHWACH,
                 anchor="w").pack(fill="x", pady=(20, 0))
        if self.demo_klassen != list(TASTEN):
            self._absatz(
                f, "Dieses Modell wurde mit anderen Klassen trainiert als "
                   "gerade eingestellt. Die Demo nimmt die Klassen des Modells - "
                   "für die eingestellten Klassen erst neu trainieren.",
                theme.WARN)

        vergleich = tk.Frame(f, bg=theme.PANEL)
        vergleich.pack(fill="x", pady=(16, 0))
        tk.Label(vergleich, text="Vergleichstext (optional)", font=self.f_klein,
                 bg=theme.PANEL, fg=theme.TEXT_SCHWACH).pack(side="left")
        self.e_soll = tk.Entry(vergleich, font=self.f_mono, bg=theme.BG,
                               fg=theme.TEXT, insertbackground=theme.TEXT,
                               relief="flat", width=24)
        self.e_soll.pack(side="left", padx=(10, 0), ipady=5, ipadx=8)
        if self.cfg.sperrfolge and all(c in self.demo_klassen
                                       for c in self.cfg.sperrfolge):
            self.e_soll.insert(0, self.cfg.sperrfolge)
        tk.Label(vergleich, text="nur für die Anzeige - geht nicht in die "
                                 "Vorhersage ein",
                 font=self.f_klein, bg=theme.PANEL,
                 fg=theme.TEXT_SCHWACH).pack(side="left", padx=(12, 0))

        reihe = self._knopfreihe(f)
        Knopf(reihe, "Live-Demo öffnen", self._demo_oeffnen, fett=True,
              font=self.f_normal).pack(side="left")
        Knopf(reihe, "ohne Titel und Knöpfe (zum Filmen)",
              lambda: self._demo_oeffnen(buehne=True),
              font=self.f_normal).pack(side="left", padx=(10, 0))

    def _demo_oeffnen(self, buehne: bool = False) -> None:
        argumente = []
        soll = self.e_soll.get().strip().lower()
        if soll:
            unbekannt = sorted({c for c in soll if c not in self.demo_klassen})
            if unbekannt:
                messagebox.showerror(
                    "Zeichen nicht dabei",
                    "Der Vergleichstext enthält Zeichen, die nicht zu den "
                    f"Klassen gehören: {' '.join(unbekannt)}")
                return
            argumente += ["--soll", soll]
        if buehne:
            argumente.append("--buehne")
        self.starte_werkzeug("08_demo.py", *argumente)


def main() -> int:
    theme.anwenden("normal")
    Studio().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

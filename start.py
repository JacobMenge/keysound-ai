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

import json
import os
import queue
import re
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

PROJEKT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJEKT))

try:
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import messagebox
except ImportError:
    # Manche Linux- und Homebrew-Pythons kommen ohne Tk. Dann gibt es auch
    # kein Fenster fuer die Meldung - also ein klarer Hinweis im Terminal
    # statt eines Tracebacks.
    if sys.platform.startswith("linux"):
        _tk_hinweis = "sudo apt install python3-tk   (oder das Paket deiner Distribution)"
    elif sys.platform == "darwin":
        _tk_hinweis = "brew install python-tk   (bzw. Python von python.org nehmen)"
    else:
        _tk_hinweis = ("Python von python.org neu installieren, mit der Option "
                       "\"tcl/tk and IDLE\"")
    print("Diesem Python fehlt Tk (tkinter) - ohne das geht kein Fenster auf.\n"
          f"Dieser Python laeuft gerade: {sys.executable}\n\n"
          f"So kommt Tk dazu:  {_tk_hinweis}", file=sys.stderr)
    raise SystemExit(1) from None


def _fehlende_pakete() -> list[str]:
    """Welche Abhaengigkeiten fehlen in genau diesem Python?"""
    import importlib.util

    noetig = {"numpy": "numpy", "scipy": "scipy", "matplotlib": "matplotlib",
              "soundfile": "soundfile", "sounddevice": "sounddevice",
              "torch": "torch", "PIL": "pillow"}
    return [paket for modul, paket in noetig.items()
            if importlib.util.find_spec(modul) is None]


def _portaudio_fehler() -> str:
    """Laedt sounddevice wirklich? Leer, wenn ja, sonst die Fehlermeldung.

    Das Paket kann installiert sein, ohne dass PortAudio da ist (Linux) -
    dann scheitert erst der Import, und find_spec allein merkt nichts davon.
    """
    try:
        import sounddevice  # noqa: F401
    except Exception as fehler:                        # noqa: BLE001
        # Unter Windows kommt eine fehlende oder kaputte DLL ebenfalls als
        # OSError - deshalb allgemein abgefangen.
        return f"{type(fehler).__name__}: {fehler}"
    return ""


def _installationshinweis() -> str:
    """Die Befehle, die in genau dieser Lage weiterhelfen.

    Ohne Aktivieren der Umgebung: .venv\\Scripts\\activate scheitert in der
    PowerShell an der Standard-Richtlinie, und pip landet dann im globalen
    Python. Der direkte Aufruf des Umgebungs-Pythons geht in cmd, PowerShell
    und unter Linux/macOS gleich.

    Die Befehle stehen relativ zum Projektordner: Ein Befehl in
    Anfuehrungszeichen ("C:\\...\\python.exe" -m pip) ist in der PowerShell
    ein Syntaxfehler, .venv\\Scripts\\python dagegen geht in cmd und
    PowerShell.
    """
    windows = os.name == "nt"
    venv_python = (PROJEKT / ".venv" / ("Scripts" if windows else "bin")
                   / ("python.exe" if windows else "python"))
    befehl = ".venv\\Scripts\\python" if windows else ".venv/bin/python"
    installieren = f"{befehl} -m pip install -r requirements.txt"
    starten = ".\\start.bat" if windows else f"{befehl} start.py"
    if venv_python.exists():
        # Die Umgebung gibt es schon - es fehlt nur die Installation darin.
        return ("Im Projektordner gibt es schon eine Umgebung (.venv). "
                "Installiere dort hinein - im Ordner\n"
                f"{PROJEKT}:\n\n"
                f"    {installieren}\n\n"
                f"Danach starten mit:\n\n    {starten}\n")
    anlegen = ("py -m venv .venv      (oder: python -m venv .venv)" if windows
               else "python3 -m venv .venv")
    return ("So richtest du eine eigene Umgebung ein - im Ordner\n"
            f"{PROJEKT}:\n\n"
            f"    {anlegen}\n"
            f"    {installieren}\n\n"
            f"Danach starten mit:\n\n    {starten}\n")


def _abbrechen(titel: str, text: str) -> None:
    """Meldung ins Terminal und, wenn moeglich, als Fenster - dann Ende."""
    print(text, file=sys.stderr)
    try:
        wurzel = tk.Tk()
        wurzel.withdraw()
        messagebox.showerror(titel, text)
        wurzel.destroy()
    except Exception:                                  # noqa: BLE001, S110
        # Ohne Fenstersystem bleibt es bei der Ausgabe im Terminal.
        pass
    raise SystemExit(1)


def _abbruch_fehlende_pakete(fehlt: list[str]) -> None:
    """Klar sagen, was fehlt - statt eines nackten ImportError.

    Der haeufigste Stolperstein: Das Programm wird mit einem anderen Python
    gestartet als dem, in dem installiert wurde. Deshalb steht hier, welcher
    Interpreter gerade laeuft.
    """
    _abbrechen(
        "Tastenakustik - Pakete fehlen",
        "Es fehlen Pakete:  " + "  ".join(fehlt) + "\n\n"
        f"Dieser Python laeuft gerade:\n{sys.executable}\n\n"
        "Vielleicht ist das ein anderer als der, in dem du installiert hast.\n"
        + _installationshinweis())


def _abbruch_portaudio(fehler: str) -> None:
    """sounddevice ist da, aber PortAudio nicht - das hilft kein pip install -r."""
    if sys.platform.startswith("linux"):
        abhilfe = "sudo apt install libportaudio2   (oder das Paket deiner Distribution)"
    elif sys.platform == "darwin":
        abhilfe = "brew install portaudio"
    else:
        # In der PowerShell braucht ein Befehl in Anfuehrungszeichen ein &.
        abhilfe = (f"\"{sys.executable}\" -m pip install --force-reinstall "
                   "sounddevice\n\n    (bringt PortAudio unter Windows mit; in "
                   "der PowerShell ein & davor setzen)")
    _abbrechen(
        "Tastenakustik - PortAudio fehlt",
        "sounddevice ist installiert, laedt aber nicht - meist fehlt die "
        f"Audio-Bibliothek PortAudio.\n\n{fehler}\n\n"
        f"Dieser Python laeuft gerade:\n{sys.executable}\n\n"
        f"Abhilfe:\n\n    {abhilfe}\n")


if _fehlt := _fehlende_pakete():
    _abbruch_fehlende_pakete(_fehlt)
if _portaudio := _portaudio_fehler():
    _abbruch_portaudio(_portaudio)

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from tastenakustik import audio, storage, theme, training
from tastenakustik.config import (
    MODELLE,
    ROH,
    Config,
    KlassenFehler,
    ConfigFehler,
    TASTEN,
    ZIEL_MAX,
    ZIEL_MIN,
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
    "05_daten_pruefen.py": "Datenprüfung",
    "06_trennbarkeit.py": "Trennbarkeitsmessung",
}

# Diese Werkzeuge oeffnen das Mikrofon. Unter WDM-KS geht das nur exklusiv -
# zwei gleichzeitig enden mit einem rohen "Device unavailable".
MIKRO_WERKZEUGE = ("02_kalibrierung.py", "03_collector.py", "08_demo.py")

EPOCHEN_MIN, EPOCHEN_MAX = 1, 400


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
        # Ein gesperrter Knopf bleibt auch nach dem Ueberfahren grau.
        self.bind("<Leave>", lambda _e: self.configure(
            bg=self.farbe if self.aktiv else theme.PANEL))

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
        self.trainings_abbruch = threading.Event()
        self.ergebnis: training.Ergebnis | None = None
        self.verlauf: dict[str, list[float]] | None = None
        # Was vom neuesten Modell auf der Platte liegt (verlauf_*.json) - damit
        # Kurven und Grafik-Export auch nach einem Neustart da sind.
        self.gespeichert: dict | None = None
        self.test_ergebnis: training.TestErgebnis | None = None
        self.test_laeuft = False
        self.messung_laeuft = False
        self._modell_cache: dict[tuple[str, float], list[str] | None] = {}

        self.title("Tastenakustik - Studio")
        self.configure(bg=theme.BG)
        # Auf niedrigen Bildschirmen (1080p mit 150 %, 1366x768) waere
        # 840 hoeher als der Platz - der Inhalt scrollt dann statt zu fehlen.
        hoehe = max(min(840, self.winfo_screenheight() - 120), 480)
        self.geometry(f"1180x{hoehe}")
        self.minsize(1060, min(hoehe, 520))
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

        # Die Statuszeile zuerst packen: Wird das Fenster knapp, gibt pack
        # den Platz in Packreihenfolge ab - sie soll nie als erste fehlen.
        fuss = tk.Frame(self, bg=theme.BG, padx=28, pady=0)
        fuss.pack(fill="x", side="bottom")
        self.l_status = tk.Label(fuss, text="", font=self.f_klein, anchor="w",
                                 bg=theme.BG, fg=theme.TEXT_SCHWACH)
        self.l_status.pack(fill="x", pady=(0, 12))

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

        # Der Inhalt liegt in einer scrollbaren Flaeche. Ist genug Platz da,
        # bekommt die Seite die volle Hoehe (Kurven und Listen wachsen mit);
        # ist er knapp, behaelt sie ihre natuerliche Hoehe und scrollt, statt
        # Knoepfe und Zeilen zu quetschen.
        huelle = tk.Frame(koerper, bg=theme.PANEL)
        huelle.pack(side="left", fill="both", expand=True, padx=(18, 0))
        self.leiste = tk.Scrollbar(huelle, orient="vertical")
        self.flaeche = tk.Canvas(huelle, bg=theme.PANEL, highlightthickness=0,
                                 borderwidth=0, yscrollcommand=self.leiste.set)
        self.leiste.configure(command=self.flaeche.yview)
        self.flaeche.pack(side="left", fill="both", expand=True)
        self.inhalt = tk.Frame(self.flaeche, bg=theme.PANEL)
        self._inhalt_fenster = self.flaeche.create_window(0, 0, anchor="nw",
                                                          window=self.inhalt)
        self.flaeche.bind("<Configure>", lambda _e: self._flaeche_anpassen())
        self.bind_all("<MouseWheel>", self._mausrad, add="+")

        self._flaeche_takt()

    def _flaeche_anpassen(self) -> None:
        """Seite und Scrollbereich an Fenster und Inhalt angleichen.

        Die Seite ist mindestens so hoch wie der sichtbare Bereich, damit
        Kurven und Listen wie bisher mitwachsen, und nie niedriger als ihr
        Inhalt - fehlt Platz, erscheint die Leiste.
        """
        breite = self.flaeche.winfo_width()
        sicht = self.flaeche.winfo_height()
        noetig = self.inhalt.winfo_reqheight()
        hoehe = max(noetig, sicht)
        self.flaeche.itemconfigure(self._inhalt_fenster, width=breite, height=hoehe)
        self.flaeche.configure(scrollregion=(0, 0, breite, hoehe))
        zu_hoch = noetig > sicht + 1
        if zu_hoch and not self.leiste.winfo_ismapped():
            self.leiste.pack(side="right", fill="y", before=self.flaeche)
        elif not zu_hoch and self.leiste.winfo_ismapped():
            self.leiste.pack_forget()
            self.flaeche.yview_moveto(0)
        self._flaeche_stand = (breite, sicht, noetig)

    def _flaeche_takt(self) -> None:
        # Texte wie das Trainingsergebnis wachsen nachtraeglich; ein Ereignis
        # dafuer gibt es in Tk nicht. Nachgesehen wird deshalb regelmaessig -
        # angepasst nur, wenn sich etwas geaendert hat.
        stand = (self.flaeche.winfo_width(), self.flaeche.winfo_height(),
                 self.inhalt.winfo_reqheight())
        if stand != getattr(self, "_flaeche_stand", None):
            self._flaeche_anpassen()
        self.after(250, self._flaeche_takt)

    def _mausrad(self, ereignis) -> None:
        """Mausrad ueber der Seite scrollt die Seite - Listen scrollen selbst."""
        w = ereignis.widget
        if not isinstance(w, tk.Misc) or isinstance(w, (tk.Listbox, tk.Text)):
            return
        if not str(w).startswith(str(self.flaeche)):
            return
        if self.inhalt.winfo_reqheight() <= self.flaeche.winfo_height() + 1:
            return
        schritte = -int(ereignis.delta / 120) or (-1 if ereignis.delta > 0 else 1)
        self.flaeche.yview_scroll(schritte * 2, "units")

    # -- Zustand --------------------------------------------------------
    def _sitzung_passt(self, zeile: dict) -> bool:
        """Passt die Sitzung zu den eingestellten Klassen?

        Dieselbe Frage wie datensatz.pruefe_passend/pruefe_labels, nur ohne
        Audio zu laden: Klassenliste im Kopf gleich, sonst (aeltere Sitzungen)
        keine Labels ausserhalb der Klassen.
        """
        if "passt" in zeile:
            return bool(zeile["passt"])
        # Der Ordner, nicht die session_id: Eine umbenannte Kopie liegt sonst
        # unter einem Namen, den es auf der Platte nicht gibt.
        ordner = zeile.get("ordner") or zeile["session_id"]
        try:
            kopf = json.loads((ROH / ordner / "session.json")
                              .read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        alt = kopf.get("tasten")
        if alt:
            return list(alt) == list(TASTEN)
        labels = {t for t, n in zeile.get("je_taste", {}).items() if n}
        return labels <= set(TASTEN)

    def sitzungen_nach_rolle(self) -> dict[str, list[dict]]:
        nach_rolle: dict[str, list[dict]] = {r: [] for r in storage.ROLLEN}
        for zeile in storage.uebersicht():
            zeile["passt"] = self._sitzung_passt(zeile)
            nach_rolle.setdefault(zeile["rolle"], []).append(zeile)
        return nach_rolle

    def _modell_kopf(self, pfad: Path | None) -> tuple[list[str] | None, frozenset]:
        """Klassen und Schluessel einer Modelldatei, gemerkt je Datei und Stand.

        torch.load bei jedem Takt waere zu teuer - die Datei aendert sich aber
        nur, wenn neu trainiert wird.
        """
        if pfad is None:
            return None, frozenset()
        try:
            schluessel = (str(pfad), pfad.stat().st_mtime)
        except OSError:
            return None, frozenset()
        if schluessel not in self._modell_cache:
            try:
                import torch
                stand = torch.load(pfad, map_location="cpu", weights_only=False)
                klassen = list(stand.get("klassen") or []) or None
                self._modell_cache[schluessel] = (klassen, frozenset(stand))
            except Exception:                              # noqa: BLE001
                self._modell_cache[schluessel] = (None, frozenset())
        return self._modell_cache[schluessel]

    def modell_klassen(self, pfad: Path | None) -> list[str] | None:
        """Die Klassen, die ein gespeichertes Modell kennt - None, wenn unlesbar."""
        return self._modell_kopf(pfad)[0]

    def modell_passt(self, pfad: Path | None = None) -> bool:
        """Kennt das neueste Modell genau die eingestellten Klassen?"""
        pfad = pfad or training.neuestes_modell()
        return pfad is not None and self.modell_klassen(pfad) == list(TASTEN)

    def training_laeuft(self) -> bool:
        return self.trainings_thread is not None and self.trainings_thread.is_alive()

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
        # Nach einem Klassenwechsel zaehlen alte Sitzungen nicht mehr: Das
        # Laden bricht an jeder unpassenden Sitzung einer Rolle ab.
        if schritt == 3:
            return (bool(rollen["train"]) and bool(rollen["val"])
                    and all(z["passt"] for z in rollen["train"] + rollen["val"]))
        if schritt == 4:
            return bool(rollen["train"]) and all(z["passt"] for z in rollen["train"])
        if schritt == 5:
            return modell_da
        if schritt == 6:
            return False
        return True

    def _takt(self) -> None:
        """Regelmaessig nachsehen, ob ein gestartetes Werkzeug fertig ist.

        Der naechste Takt wird in jedem Fall angesetzt: Scheitert ein Durchlauf
        (etwa an einer gerade halb geschriebenen Sitzungsdatei), liefe das
        Studio sonst ohne Aktualisierung weiter.
        """
        try:
            self._takt_schritt()
        except Exception as fehler:                       # noqa: BLE001
            self.melde(f"Anzeige nicht aktualisiert: {fehler}", theme.WARN)
        finally:
            self.after(700, self._takt)

    def _takt_schritt(self) -> None:
        beendet = [(datei, p, ausgabe)
                   for datei, (p, ausgabe) in self.prozesse.items()
                   if p.poll() is not None]
        for datei, p, ausgabe in beendet:
            del self.prozesse[datei]
            self._werkzeug_beendet(datei, p.returncode, ausgabe)
        # Nur die Seiten neu aufbauen, die zeigen, was ein Werkzeug erzeugt
        # hat. Die noch nicht gespeicherte Rollenwahl in Schritt 3 wird dabei
        # uebernommen - neue Sitzungen erscheinen trotzdem.
        if beendet and self.aktiv in (0, 3):
            alt = ({k: v.get() for k, v in getattr(self, "rollen_wahl", {}).items()}
                   if self.aktiv == 3 else {})
            position = self.flaeche.yview()[0]
            self.zeige(self.aktiv)
            neu = getattr(self, "rollen_wahl", {}) if self.aktiv == 3 else {}
            for sitzung_id, wert in alt.items():
                if sitzung_id in neu:
                    neu[sitzung_id].set(wert)
            self.update_idletasks()
            self._flaeche_anpassen()
            self.flaeche.yview_moveto(position)
        self._nav_auffrischen()

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
        # Die Demo nutzt auch ein Modell mit anderen Klassen weiter - es
        # zaehlt also als vorhanden, aber nicht als passend.
        if not modell_da:
            modell_text = "noch kein Modell"
        elif self.modell_passt():
            modell_text = "Modell vorhanden"
        else:
            modell_text = "Modell (andere Klassen)"
        self.l_kurzstatus.configure(
            text=f"{len(TASTEN)} Klassen     {gesamt} Proben     {modell_text}")

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
        self.flaeche.yview_moveto(0)
        self.after_idle(self._flaeche_anpassen)
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
        if datei in MIKRO_WERKZEUGE:
            belegt = self._mikrofon_belegt()
            if belegt:
                self.melde(f"{name} braucht das Mikrofon - erst {belegt} beenden.",
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

    def _mikrofon_belegt(self) -> str:
        """Wer hat das Mikrofon gerade offen? Leer, wenn niemand."""
        if self.messung_laeuft:
            return "das Mithören in Schritt 1"
        artikel = {"03_collector.py": "den"}
        for datei in MIKRO_WERKZEUGE:
            laufend = self.prozesse.get(datei)
            if laufend is not None and laufend[0].poll() is None:
                return f"{artikel.get(datei, 'die')} {WERKZEUG_NAMEN[datei]}"
        return ""

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
        """Fenster zu - aber nicht, ohne zu sagen, was dabei mit endet.

        Das Studio verspricht beim Start eines Werkzeugs, offen zu bleiben;
        viele schliessen es danach als erledigt. Eine laufende Aufnahme oder
        ein Training ohne Rueckfrage abzuwuergen, waere hier die Falle.
        """
        aktiv = [WERKZEUG_NAMEN.get(datei, datei)
                 for datei, (p, _ausgabe) in self.prozesse.items()
                 if p.poll() is None]
        if self.training_laeuft():
            aktiv.append("Training")
        if aktiv:
            weiter = messagebox.askokcancel(
                "Studio schließen?",
                f"Noch aktiv: {', '.join(aktiv)}.\n\nSchließen beendet das "
                "mit. Bereits gespeicherte Proben bleiben erhalten, ein "
                "laufendes Training speichert kein Modell.")
            if not weiter:
                return
        self.trainings_abbruch.set()
        for p, _ausgabe in self.prozesse.values():
            if p.poll() is None:
                p.terminate()
        # Wartende Takte abmelden - sonst meldet Tcl sie nach dem Schliessen
        # als "invalid command name".
        for kennung in self.tk.splitlist(self.tk.call("after", "info")):
            self.after_cancel(kennung)
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
        # Ohne --behalten: Jeder Lauf liesse sonst 55-75 MB in %TEMP% liegen.
        # Wer die Testbilder sehen will, ruft 09 von Hand mit --behalten auf.
        if self.starte_werkzeug("09_selbsttest.py"):
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
            self._neu_einlesen_reihe(f)
            return
        if not self.geraete:
            self._absatz(f, "Kein Eingangsgerät gefunden. Mikrofon einstecken und "
                            "„Liste neu einlesen“ drücken.", theme.FEHLER)
            self._neu_einlesen_reihe(f)
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

        # Vorauswahl ueber Name und Schnittstelle, nicht ueber den Index: Nach
        # dem Neueinlesen koennen sich die Indizes verschoben haben.
        gespeichert = (self.cfg.device_name, self.cfg.hostapi)
        for i, g in enumerate(self.geraete):
            marke = "  " if g.geeignet else "! "
            self.liste.insert("end", f"{marke}{g.label}")
            if not g.geeignet:
                self.liste.itemconfigure(i, foreground=theme.TEXT_SCHWACH)
            if (g.name, g.hostapi) == gespeichert and not self.liste.curselection():
                self.liste.selection_set(i)
                self.liste.see(i)
        if not self.liste.curselection():
            for i, g in enumerate(self.geraete):
                if g.index == self.cfg.device:
                    self.liste.selection_set(i)
                    self.liste.see(i)
                    break
        if not self.liste.curselection():
            self.liste.selection_set(0)

        tk.Label(f, text="Gerät fehlt? Nach dem Einstecken „Liste neu einlesen“ "
                         "drücken.",
                 font=self.f_klein, bg=theme.PANEL, fg=theme.TEXT_SCHWACH,
                 anchor="w").pack(fill="x", pady=(6, 0))

        self.l_messung = tk.Label(f, text="", font=self.f_mono, bg=theme.PANEL,
                                  fg=theme.TEXT_SCHWACH, anchor="w", justify="left")
        self.l_messung.pack(fill="x", pady=(14, 0))

        reihe = self._knopfreihe(f)
        self.k_messen = Knopf(reihe, "2 Sekunden mithören", self._pegel_messen,
                              font=self.f_normal)
        self.k_messen.pack(side="left")
        if self.messung_laeuft:
            self.k_messen.setze_aktiv(False)
        Knopf(reihe, "Übernehmen", self._geraet_uebernehmen, fett=True,
              font=self.f_normal).pack(side="left", padx=(10, 0))
        Knopf(reihe, "Pegel im Vollbild", self._pegel_vollbild,
              font=self.f_normal).pack(side="left", padx=(10, 0))
        Knopf(reihe, "Liste neu einlesen", self._geraete_neu_einlesen,
              font=self.f_normal).pack(side="left", padx=(10, 0))

    def _neu_einlesen_reihe(self, f) -> None:
        reihe = self._knopfreihe(f)
        Knopf(reihe, "Liste neu einlesen", self._geraete_neu_einlesen,
              font=self.f_normal).pack(side="left")

    def _geraete_neu_einlesen(self) -> None:
        """PortAudio neu starten, damit ein spaeter eingestecktes Geraet auftaucht.

        Geht nur, solange hier niemand das Mikrofon offen hat - ein laufender
        Stream wuerde dabei abgerissen. Die Vollbild-Werkzeuge sind eigene
        Prozesse und bleiben unberuehrt.
        """
        if self.messung_laeuft:
            self.melde("Erst das Mithören abwarten.", theme.WARN)
            return
        try:
            if hasattr(audio, "neu_einlesen"):
                audio.neu_einlesen()
            else:
                import sounddevice as sd
                sd._terminate()                            # noqa: SLF001
                sd._initialize()                           # noqa: SLF001
        except Exception as fehler:                        # noqa: BLE001
            self.melde(f"Neu einlesen fehlgeschlagen: {fehler}", theme.FEHLER)
            return
        self.zeige(1)
        self.melde(f"Geräteliste neu eingelesen: {len(getattr(self, 'geraete', []))} "
                   "Eingänge.", theme.OK)

    def _gewaehltes_geraet(self):
        auswahl = self.liste.curselection()
        return self.geraete[auswahl[0]] if auswahl else None

    def _pegel_vollbild(self) -> None:
        # Das in der Liste markierte Geraet zeigen, nicht das aus config.json -
        # 02 schreibt die Wahl nicht zurueck, "Übernehmen" bleibt noetig.
        g = self._gewaehltes_geraet()
        argumente = ("--geraet", str(g.index)) if g is not None else ()
        self.starte_werkzeug("02_kalibrierung.py", *argumente)

    def _pegel_messen(self) -> None:
        g = self._gewaehltes_geraet()
        if g is None or self.messung_laeuft:
            return
        belegt = self._mikrofon_belegt()
        if belegt:
            self.melde(f"Das Mikrofon ist belegt - erst {belegt} beenden.", theme.WARN)
            return
        self.messung_laeuft = True
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
                fenster = max(int(0.01 * sr), 16)
                if x.size < 2 * fenster:
                    # Offen, aber stumm - das Perzentil ueber nichts wuerde
                    # sonst als raetselhafter Indexfehler ankommen.
                    ergebnis.put(("stumm", "", 0, sr, ch))
                    return
                # Rauschboden als 20. Perzentil der Kurzzeit-Energie: robust
                # gegen die Anschlaege, die waehrend der Messung passieren.
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
            self.messung_laeuft = False
            if art == "fehler":
                text, farbe = f"Öffnen fehlgeschlagen: {a}", theme.FEHLER
            elif art == "stumm":
                text, farbe = ("Das Gerät liefert kein Signal - anderes Gerät oder "
                               "WASAPI/WDM-KS wählen."), theme.FEHLER
            else:
                abstand = a - b
                if abstand >= 25:
                    urteil, farbe = "gut - Anschläge heben sich klar ab", theme.OK
                elif abstand >= 15:
                    urteil, farbe = "brauchbar - näher ran wäre besser", theme.WARN
                else:
                    urteil, farbe = ("zu wenig Abstand zum Rauschen - näher ran, "
                                     "Gain hoch, Lüfter aus"), theme.FEHLER
                text = (f"Spitze {a:6.1f} dBFS     Rauschboden {b:6.1f} dBFS     "
                        f"Abstand {abstand:4.1f} dB     {sr} Hz / {ch} Kanal\n{urteil}")
            # Wer inzwischen weitergeklickt hat, hat die Widgets dieser Seite
            # nicht mehr - das Ergebnis kommt dann in die Statuszeile.
            if self.aktiv == 1 and self.l_messung.winfo_exists():
                self.k_messen.setze_aktiv(True)
                self.l_messung.configure(text=text, fg=farbe)
            else:
                self.melde(f"Mithören ({g.name}): " + text.replace("\n", "  -  "), farbe)

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
        # Erst nach dem Auslegen ist die echte Breite bekannt - und bei jeder
        # Groessenaenderung passen andere Kacheln in eine Zeile.
        self.vorschau.bind("<Configure>", lambda _e: self._klassen_vorschau())
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
        # Vor dem ersten Auslegen meldet Tk eine Breite von 1, nicht 0.
        breite = self.vorschau.winfo_width()
        pro_zeile = max(int((breite if breite > 50 else 760) // (kachel + luecke)), 1)
        for i, (zeichen, farbe) in enumerate(zip(liste, farben)):
            zeile, spalte = divmod(i, pro_zeile)
            x0 = spalte * (kachel + luecke)
            y0 = zeile * (kachel + luecke) + 4
            rundes_rechteck(self.vorschau, x0, y0, x0 + kachel, y0 + kachel, 7,
                            fill=farbe, outline="")
            self.vorschau.create_text(x0 + kachel / 2, y0 + kachel / 2,
                                      text=anzeige(zeichen), fill=theme.BG,
                                      font=(self.fam, 13, "bold"))
        # Hoehe an die Zeilenzahl anpassen - nur bei Aenderung, sonst loeste
        # jedes configure ein neues <Configure> aus.
        zeilen = -(-len(liste) // pro_zeile)
        hoehe = max(zeilen * (kachel + luecke) + 8, 48)
        if int(self.vorschau.cget("height")) != hoehe:
            self.vorschau.configure(height=hoehe)

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
        # Training und Test lesen die Klassenliste im eigenen Thread. Ein
        # Wechsel mittendrin gaebe ein Modell mit falschen Beschriftungen.
        if self.training_laeuft() or self.test_laeuft:
            was = "das Training" if self.training_laeuft() else "die Testauswertung"
            messagebox.showinfo(
                "Moment",
                f"Gerade läuft {was}. Erst abwarten (oder in Schritt 5 "
                "abbrechen), dann die Klassen ändern.")
            return
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
            self.gespeichert = None
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
            "03_collector.py", "--ziel",
            str(min(max(self.cfg.ziel_pro_taste, ZIEL_MIN), ZIEL_MAX))),
            fett=True, font=self.f_normal).pack(side="left")
        tk.Label(reihe, text=f"{self.cfg.ziel_pro_taste} Proben je Klasse, "
                             f"{len(TASTEN) * self.cfg.ziel_pro_taste} insgesamt",
                 font=self.f_klein, bg=theme.PANEL,
                 fg=theme.TEXT_SCHWACH).pack(side="left", padx=(14, 0))

        zeilen = storage.uebersicht()
        # "Rollen speichern" steht in der Kopfzeile der Tabelle: Bei vielen
        # Sitzungen liegt das Tabellenende sonst unter dem Fensterrand.
        titel = tk.Frame(f, bg=theme.PANEL)
        titel.pack(fill="x", pady=(26, 8))
        tk.Label(titel, text="Aufgenommene Sitzungen (neueste oben)" if zeilen
                 else "Aufgenommene Sitzungen", font=self.f_normal,
                 bg=theme.PANEL, fg=theme.TEXT, anchor="w").pack(side="left")
        if not zeilen:
            self._absatz(f, "Noch keine. Oben den Collector öffnen.",
                         theme.TEXT_SCHWACH)
            return
        Knopf(titel, "Rollen speichern", self._rollen_speichern, fett=True,
              font=self.f_normal).pack(side="left", padx=(18, 0))

        rollen = self.sitzungen_nach_rolle()
        fehlt = [r for r in ("train", "val", "test") if not rollen[r]]
        if fehlt:
            tk.Label(f, text="Es fehlt noch eine Sitzung für: " + ", ".join(fehlt),
                     font=self.f_normal, bg=theme.PANEL, fg=theme.WARN,
                     anchor="w").pack(fill="x", pady=(0, 8))

        tabelle = tk.Frame(f, bg=theme.PANEL)
        tabelle.pack(fill="x")
        self.rollen_wahl: dict[str, tk.StringVar] = {}
        for i, zeile in enumerate(reversed(zeilen)):
            # Der Ordnername ist die Identitaet - bei einer Explorer-Kopie
            # tragen zwei Ordner dieselbe session_id.
            ordner = zeile.get("ordner", zeile["session_id"])
            tk.Label(tabelle, text=ordner, font=self.f_mono,
                     bg=theme.PANEL, fg=theme.TEXT, anchor="w").grid(
                row=i, column=0, sticky="w", pady=2, padx=(0, 18))
            tk.Label(tabelle, text=f"{zeile['gesamt']:>4} Proben", font=self.f_mono,
                     bg=theme.PANEL, fg=theme.TEXT_SCHWACH, anchor="w").grid(
                row=i, column=1, sticky="w", padx=(0, 18))
            var = tk.StringVar(value=zeile["rolle"])
            self.rollen_wahl[ordner] = var
            wahl = tk.OptionMenu(tabelle, var, *storage.ROLLEN)
            wahl.configure(bg=theme.BG, fg=theme.TEXT, font=self.f_klein,
                           relief="flat", highlightthickness=0, width=6,
                           activebackground=theme.GRID, activeforeground=theme.TEXT)
            wahl["menu"].configure(bg=theme.BG, fg=theme.TEXT, font=self.f_klein)
            wahl.grid(row=i, column=2, sticky="w", padx=(0, 18))
            # Nach einem Klassenwechsel sichtbar machen, welche Sitzungen nicht
            # mehr mitspielen koennen.
            passt = self._sitzung_passt(zeile)
            hinweis = zeile.get("hinweis", "")
            notiz = (hinweis if hinweis
                     else zeile["notiz"][:40] if passt
                     else "andere Klassen - " + zeile["notiz"][:30])
            tk.Label(tabelle, text=notiz, font=self.f_klein, bg=theme.PANEL,
                     fg=theme.TEXT_SCHWACH if passt and not hinweis else theme.WARN,
                     anchor="w").grid(row=i, column=3, sticky="w")

    def _rollen_speichern(self) -> None:
        # Jede Sitzung einzeln: Eine abgelehnte (z. B. doppelt vorhandene)
        # Sitzung darf die Rollen der uebrigen nicht mit sich reissen.
        fehler = []
        for ordner, var in self.rollen_wahl.items():
            try:
                storage.rolle_setzen(ordner, var.get())
            except (OSError, ValueError) as grund:
                fehler.append(f"{ordner}: {grund}")
        if fehler:
            messagebox.showerror("Nicht alle Rollen gespeichert", "\n\n".join(fehler))
            self.melde("Rollen teilweise gespeichert.", theme.WARN)
        else:
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
        zeilen = storage.uebersicht()
        if not zeilen:
            self._absatz(f, "Noch keine Aufnahmen da.", theme.WARN)
            return

        # Geprueft wird eine Sitzung - welche, steht sichtbar daneben. Ohne
        # Auswahl nahm 05 still die neueste, und aeltere blieben ungeprueft.
        reihe = self._knopfreihe(f)
        # 05 --sitzung erwartet den Ordnernamen (ROH / name) - er ist auch
        # bei einer Kopie mit derselben session_id eindeutig.
        namen = {}
        for z in reversed(zeilen):
            ordner = z.get("ordner") or z["session_id"]
            namen[f"{ordner}  ({z['rolle']})"] = ordner
        vorwahl = next((n for n, z in zip(namen, reversed(zeilen)) if z["gesamt"]),
                       next(iter(namen)))
        self.pruef_namen = namen
        self.pruef_wahl = tk.StringVar(value=vorwahl)
        wahl = tk.OptionMenu(reihe, self.pruef_wahl, *namen)
        wahl.configure(bg=theme.BG, fg=theme.TEXT, font=self.f_klein,
                       relief="flat", highlightthickness=0,
                       activebackground=theme.GRID, activeforeground=theme.TEXT)
        wahl["menu"].configure(bg=theme.BG, fg=theme.TEXT, font=self.f_klein)
        wahl.pack(side="left", padx=(0, 10), ipady=4)
        Knopf(reihe, "Prüfen", self._pruefung_starten, fett=True,
              font=self.f_normal).pack(side="left")
        Knopf(reihe, "Trennbarkeit messen", self._trennbarkeit,
              font=self.f_normal).pack(side="left", padx=(10, 0))

        self.ausgabe = tk.Text(f, font=self.f_mono, bg=theme.BG, fg=theme.TEXT,
                               relief="flat", height=16, wrap="none",
                               insertbackground=theme.TEXT)
        self.ausgabe.pack(fill="both", expand=True, pady=(18, 0))
        self.ausgabe.insert("1.0", getattr(self, "letzte_pruefung", None)
                            or "Noch nichts geprüft.")
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

        ausgabe = self.ausgabe

        def nachsehen() -> None:
            try:
                text = ergebnis.get_nowait()
            except queue.Empty:
                self.after(150, nachsehen)
                return
            text = text.strip() or "(keine Ausgabe)"
            # Wer inzwischen weitergeklickt hat, hat dieses Textfeld nicht
            # mehr. Das Ergebnis wird dann gemerkt und beim naechsten Aufbau
            # von Schritt 4 gezeigt, statt mit einem TclError zu verschwinden.
            self.letzte_pruefung = text
            ziel = ausgabe if ausgabe.winfo_exists() else None
            if ziel is None and self.aktiv == 4:
                aktuell = getattr(self, "ausgabe", None)
                if aktuell is not None and aktuell.winfo_exists():
                    ziel = aktuell
            if ziel is None:
                self.melde(f"{WERKZEUG_NAMEN.get(datei, datei)} fertig - das "
                           "Ergebnis steht in Schritt 4.", theme.OK)
                return
            ziel.configure(state="normal")
            ziel.delete("1.0", "end")
            ziel.insert("1.0", text)
            ziel.configure(state="disabled")

        self.after(150, nachsehen)

    def _pruefung_starten(self) -> None:
        sitzung = self.pruef_namen.get(self.pruef_wahl.get())
        self._werkzeug_ausgabe("05_daten_pruefen.py",
                               *(("--sitzung", sitzung) if sitzung else ()))

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
        self.e_epochen = tk.Spinbox(kopf, from_=10, to=EPOCHEN_MAX, increment=10,
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
        self.k_abbrechen = Knopf(kopf, "Abbrechen", self._training_abbrechen,
                                 font=self.f_normal)
        self.k_abbrechen.pack(side="left", padx=(10, 0))
        # Umbrechen statt rechts abschneiden: Die Datenpruefung meldet mitunter
        # einen ganzen Satz.
        self.l_training = tk.Label(kopf, text="", font=self.f_klein, bg=theme.PANEL,
                                   fg=theme.TEXT_SCHWACH, justify="left",
                                   anchor="w", wraplength=400)
        self.l_training.pack(side="left", padx=(16, 0))

        # Auf niedrigen Bildschirmen etwas flacher - Jacobs Hochformat-
        # Monitore behalten die gewohnte Hoehe.
        fig_hoehe = 3.3 if self.winfo_screenheight() >= 900 else 2.6
        self.fig_training = Figure(figsize=(8.2, fig_hoehe), dpi=100,
                                   facecolor=theme.PANEL)
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

        self.l_test = tk.Label(f, text="", font=self.f_normal, bg=theme.PANEL,
                               fg=theme.TEXT_SCHWACH, anchor="w",
                               justify="left", wraplength=800)
        self.l_test.pack(fill="x", pady=(12, 0))

        laeuft = self.training_laeuft()
        if not laeuft and self.ergebnis is None:
            # Nach einem Neustart: Kurven und Kennzahlen des neuesten Modells
            # aus seiner verlauf_*.json holen, statt eine leere Seite zu zeigen.
            self._gespeichert_laden()
        if self.verlauf and self.verlauf["epoche"]:
            # Beim Zurueckkommen auf diesen Schritt die Kurven wieder zeigen -
            # das Panel wird jedes Mal neu gebaut.
            self._kurven_zeichnen()
        self.k_abbrechen.setze_aktiv(laeuft and not self.trainings_abbruch.is_set())
        if laeuft:
            self.k_training.setze_aktiv(False)
            self.l_training.configure(
                text="wird nach dieser Epoche abgebrochen …"
                if self.trainings_abbruch.is_set() else "läuft …")
        elif self.ergebnis is not None:
            self._training_ergebnis_zeigen(self.ergebnis)
        else:
            if self.gespeichert is not None:
                self._gespeichert_zeigen()
            self._daten_pruefen_im_hintergrund()
        self._knoepfe_auffrischen()
        self._test_anzeige()

    def _daten_pruefen_im_hintergrund(self) -> None:
        """Die Zeile "N Trainings- und M Validierungsproben bereit" fuellen.

        pruefe_daten liest alle WAVs und rechnet alle Log-Mels - im UI-Thread
        fror das Fenster bei jedem Besuch von Schritt 5 ein bis zwei Sekunden
        ein. Bis die Antwort da ist, bleibt "Training starten" gesperrt.
        """
        self.k_training.setze_aktiv(False)
        self.l_training.configure(text="Daten werden geprüft …",
                                  fg=theme.TEXT_SCHWACH)
        antwort: queue.Queue = queue.Queue()
        label = self.l_training
        klassen = list(TASTEN)

        def pruefen() -> None:
            try:
                train, val = training.pruefe_daten()
                antwort.put(("ok", f"{len(train)} Trainings- und {len(val)} "
                                   "Validierungsproben bereit"))
            except training.DatenFehler as fehler:
                antwort.put(("warn", str(fehler)))
            except ValueError as fehler:
                antwort.put(("fehler", str(fehler)))
            except Exception as fehler:                    # noqa: BLE001
                # Etwa eine von Hand geloeschte WAV-Datei: lieber eine Zeile
                # hier als ein halb aufgebautes Panel.
                antwort.put(("fehler", f"Daten nicht lesbar: {fehler}"))

        threading.Thread(target=pruefen, daemon=True).start()

        def nachsehen() -> None:
            try:
                art, text = antwort.get_nowait()
            except queue.Empty:
                self.after(150, nachsehen)
                return
            # Seite inzwischen neu gebaut, Training gestartet oder Klassen
            # gewechselt: Diese Antwort gilt nicht mehr.
            if (not label.winfo_exists() or self.aktiv != 5
                    or self.training_laeuft() or self.ergebnis is not None
                    or klassen != list(TASTEN)):
                return
            farbe = {"ok": theme.TEXT_SCHWACH, "warn": theme.WARN,
                     "fehler": theme.FEHLER}[art]
            label.configure(text=text, fg=farbe)
            self.k_training.setze_aktiv(art == "ok")

        self.after(150, nachsehen)

    def _knoepfe_auffrischen(self) -> None:
        """Grafik-Export und "Weiter" passend zum Stand sperren oder freigeben."""
        if not self._sichtbar():
            return
        laeuft = self.training_laeuft()
        self.k_grafiken.setze_aktiv(not laeuft and self._grafiken_moeglich())
        # "Weiter" braucht nur ein Modell - auch eines von vor dem Neustart.
        self.k_weiter.setze_aktiv(not laeuft
                                  and training.neuestes_modell() is not None)

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
        if self.training_laeuft():
            return
        if self.test_laeuft:
            self.l_training.configure(text="Erst die Testauswertung abwarten.",
                                      fg=theme.WARN)
            return
        try:
            epochen = int(self.e_epochen.get())
        except (ValueError, tk.TclError):
            epochen = 0
        if not EPOCHEN_MIN <= epochen <= EPOCHEN_MAX:
            self.l_training.configure(
                text=f"Epochen: eine ganze Zahl von {EPOCHEN_MIN} bis {EPOCHEN_MAX}",
                fg=theme.FEHLER)
            return
        self.k_training.setze_aktiv(False)
        self.l_ergebnis.configure(text="")
        self.l_training.configure(text="läuft …", fg=theme.TEXT_SCHWACH)
        self.verlauf = {"epoche": [], "train_loss": [], "val_loss": [],
                        "train_acc": [], "val_acc": []}
        self._training_achsen()
        self.canvas_training.draw()

        # Je Lauf eine eigene Queue und ein eigenes Abbruch-Signal: Staende
        # eines alten Laufs duerfen nie als Ergebnis des neuen ankommen.
        nachrichten: queue.Queue = queue.Queue()
        abbruch = threading.Event()
        self.trainings_queue = nachrichten
        self.trainings_abbruch = abbruch

        def laufen() -> None:
            try:
                for stand in training.trainiere(epochen=epochen, abbruch=abbruch):
                    nachrichten.put(stand)
            except Exception as fehler:                    # noqa: BLE001
                nachrichten.put(fehler)

        self.trainings_thread = threading.Thread(target=laufen, daemon=True)
        self.trainings_thread.start()
        # Erst nach dem Start: is_alive() ist vorher noch False.
        self.k_abbrechen.setze_aktiv(True)
        self._knoepfe_auffrischen()
        self._test_anzeige()
        self.after(120, lambda: self._training_takt(nachrichten))

    def _training_abbrechen(self) -> None:
        """Nach der laufenden Epoche aufhoeren - gespeichert wird dann nichts."""
        if not self.training_laeuft():
            return
        self.trainings_abbruch.set()
        self.k_abbrechen.setze_aktiv(False)
        self.l_training.configure(text="wird nach dieser Epoche abgebrochen …",
                                  fg=theme.WARN)

    def _sichtbar(self) -> bool:
        """Steht der Trainingsschritt noch auf dem Schirm?

        Das Training laeuft in einem eigenen Thread weiter, auch wenn jemand
        zwischendurch auf einen anderen Schritt klickt. Dann sind die Widgets
        dieses Panels zerstoert - eingesammelt wird trotzdem, gezeichnet nicht.
        """
        l_training = getattr(self, "l_training", None)
        return (self.aktiv == 5 and l_training is not None
                and bool(l_training.winfo_exists()))

    def _training_takt(self, nachrichten: queue.Queue) -> None:
        if nachrichten is not self.trainings_queue:
            # Nachrichten eines frueheren Laufs - verwerfen.
            return
        fertig = False
        try:
            fertig = self._training_nachrichten(nachrichten)
            # Erst nach dem Thread fragen, dann nach der Queue: Ist er tot,
            # liegt alles, was er geschickt hat, schon darin. Umgekehrt koennte
            # das Ergebnis genau dazwischen ankommen.
            if (not fertig and not self.training_laeuft()
                    and nachrichten.empty()):
                # Der Thread ist weg, ohne sich abzumelden - nicht ewig warten.
                fertig = True
                self._training_beendet("Training unerwartet beendet.", theme.FEHLER)
        finally:
            # Auch nach einem Fehler beim Zeichnen weiter abholen - sonst
            # bliebe die Abfrage still stehen, waehrend der Thread weiterlaeuft.
            if not fertig:
                self.after(120, lambda: self._training_takt(nachrichten))

    def _training_nachrichten(self, nachrichten: queue.Queue) -> bool:
        """Alles Angekommene abholen. True, sobald der Lauf zu Ende ist."""
        neu = False
        while True:
            try:
                nachricht = nachrichten.get_nowait()
            except queue.Empty:
                break
            if isinstance(nachricht, (Exception, training.Ergebnis)):
                # Das war die letzte Nachricht; der Thread endet gleich danach.
                # Kurz auf ihn warten, sonst saehe training_laeuft() ihn noch
                # und Grafiken, Weiter und Test blieben gesperrt.
                self._thread_abwarten()
            if isinstance(nachricht, Exception):
                if isinstance(nachricht, getattr(training, "TrainingAbgebrochen", ())):
                    self._training_beendet(
                        "abgebrochen, kein Modell gespeichert", theme.WARN,
                        f"Training abgebrochen: {nachricht}")
                else:
                    self._training_beendet(str(nachricht), theme.FEHLER,
                                           f"Training abgebrochen: {nachricht}")
                return True
            if isinstance(nachricht, training.Ergebnis):
                if list(nachricht.klassen) != list(TASTEN):
                    # Sollte die Sperre in Schritt 2 verhindern - falls nicht,
                    # kein Ergebnis mit fremden Beschriftungen uebernehmen.
                    self._training_beendet(
                        "Klassen während des Trainings geändert - Ergebnis "
                        "verworfen.", theme.FEHLER)
                    return True
                self.ergebnis = nachricht
                self.gespeichert = None
                self.test_ergebnis = None      # gehoerte zum vorigen Modell
                if self._sichtbar():
                    self.k_training.setze_aktiv(True)
                    self.k_abbrechen.setze_aktiv(False)
                    self._training_ergebnis_zeigen(nachricht)
                else:
                    self.melde(f"Training fertig: beste Validation "
                               f"{nachricht.beste_val * 100:.1f} %", theme.OK)
                self._nav_auffrischen()
                return True
            v = self.verlauf
            if v is None:
                # Die Anzeige wurde zurueckgesetzt - diesen Stand verwerfen.
                continue
            v["epoche"].append(nachricht.epoche)
            v["train_loss"].append(nachricht.train_loss)
            v["val_loss"].append(nachricht.val_loss)
            v["train_acc"].append(nachricht.train_acc * 100)
            v["val_acc"].append(nachricht.val_acc * 100)
            if self._sichtbar() and not self.trainings_abbruch.is_set():
                self.l_training.configure(
                    text=f"Epoche {nachricht.epoche} von {nachricht.epochen}     "
                         f"Validation {nachricht.val_acc * 100:.1f} %     "
                         f"beste {nachricht.beste_val * 100:.1f} %",
                    fg=theme.TEXT_SCHWACH)
            neu = True

        if neu and self._sichtbar():
            self._kurven_zeichnen()
        return False

    def _training_beendet(self, text: str, farbe: str, meldung: str = "") -> None:
        """Lauf ohne neues Modell zu Ende: Knoepfe zurueck, Grund nennen."""
        # Die halben Kurven gehoeren zu keinem Modell. Beim naechsten Aufbau
        # wieder die des bisherigen Stands zeigen (Ergebnis oder Datei).
        if self.ergebnis is not None:
            self.verlauf = self._anzeige_verlauf(self.ergebnis.verlauf)
        else:
            self.verlauf = self.gespeichert = None
            self._gespeichert_laden()
        if self._sichtbar():
            # Auch auf dem Schirm die halben Kurven durch den bisherigen Stand
            # ersetzen - so, wie ein Neuaufbau der Seite ihn zeigen wuerde.
            if self.verlauf and self.verlauf["epoche"]:
                self._kurven_zeichnen()
            else:
                self._training_achsen()
                self.canvas_training.draw()
            if self.ergebnis is not None:
                self._training_ergebnis_zeigen(self.ergebnis)
            elif self.gespeichert is not None:
                self._gespeichert_zeigen()
            self.k_training.setze_aktiv(True)
            self.k_abbrechen.setze_aktiv(False)
            self.l_training.configure(text=text, fg=farbe)
            self._knoepfe_auffrischen()
            self._test_anzeige()
        else:
            self.melde(meldung or text, farbe)
        self._nav_auffrischen()

    def _thread_abwarten(self) -> None:
        """Dem Trainings-Thread die letzten Augenblicke bis zum Ende lassen."""
        thread = self.trainings_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

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

    @staticmethod
    def _parameter_text(anzahl: int) -> str:
        # Tausender mit Leerzeichen wie in 07, 08 und den Grafiken - geschuetzt,
        # damit der Zeilenumbruch die Zahl nicht zerteilt.
        return f"{anzahl:,}".replace(",", " ")

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
                 f"Zufall von {zufall() * 100:.1f} %)   ·   "
                 f"{self._parameter_text(e.parameter)} "
                 f"Parameter   ·   {e.n_train} Trainings- und {e.n_val} "
                 f"Validierungsproben\n"
                 f"Am schwächsten: "
                 + "   ".join(f"{anzeige(t)} {q * 100:.0f} %" for t, q in schwach)
                 + f"\nGespeichert: {e.modell_pfad.name if e.modell_pfad else '-'}"
                 + "\nDas ist noch nicht die ehrliche Zahl - die kommt aus einer "
                   "Testsitzung, die das Modell nie gesehen hat.",
            fg=farbe)
        self.k_abbrechen.setze_aktiv(False)
        self._knoepfe_auffrischen()
        self._test_anzeige()

    # -- Gespeicherter Stand (nach einem Neustart) ----------------------
    def _gespeichert_laden(self) -> None:
        """verlauf_<marke>.json zum neuesten Modell lesen, wenn es passt.

        Kein Ergebnis-Objekt nachbauen - nur Kurven und die Kennzahlen, die
        wirklich in der Datei stehen. Gilt nur, wenn die Klassen des Modells
        die eingestellten sind.
        """
        pfad = training.neuestes_modell()
        if pfad is None:
            self.gespeichert = None
            return
        if self.gespeichert is not None and self.gespeichert["modell"] == pfad.name:
            return
        self.gespeichert = None
        verlauf_pfad = MODELLE / ("verlauf_" + pfad.stem.removeprefix("cnn_") + ".json")
        try:
            daten = json.loads(verlauf_pfad.read_text(encoding="utf-8"))
            v = daten["verlauf"]
            konfusion = daten["konfusion"]
            # Aeltere Dateien vermerken die Klassen nicht - dann gelten die
            # des Modells mit derselben Marke.
            klassen = daten.get("klassen") or self.modell_klassen(pfad) or []
            if list(klassen) != list(TASTEN) or len(konfusion) != len(TASTEN):
                return
            gespeichert = {
                "modell": pfad.name, "verlauf": v, "konfusion": konfusion,
                "beste_val": float(daten["beste_val"]),
                "beste_epoche": int(daten.get("beste_epoche", 0)),
                "parameter": daten.get("parameter"),
                "n_train": daten.get("n_train"), "n_val": daten.get("n_val"),
            }
            verlauf = self._anzeige_verlauf(v)
        except (OSError, ValueError, KeyError, TypeError):
            return
        self.gespeichert = gespeichert
        self.verlauf = verlauf

    @staticmethod
    def _anzeige_verlauf(v: dict) -> dict[str, list[float]]:
        """Verlauf wie trainiere() ihn speichert -> wie die Live-Kurven ihn zeigen."""
        return {"epoche": list(range(1, len(v["val_acc"]) + 1)),
                "train_loss": list(v["train_loss"]),
                "val_loss": list(v["val_loss"]),
                "train_acc": [a * 100 for a in v["train_acc"]],
                "val_acc": [a * 100 for a in v["val_acc"]]}

    def _gespeichert_zeigen(self) -> None:
        g = self.gespeichert
        teile = [f"Beste Validation {g['beste_val'] * 100:.1f} %  "
                 f"(Epoche {g['beste_epoche']})"]
        if g.get("parameter"):
            teile.append(f"{self._parameter_text(int(g['parameter']))} Parameter")
        if g.get("n_train") is not None and g.get("n_val") is not None:
            teile.append(f"{g['n_train']} Trainings- und {g['n_val']} "
                         "Validierungsproben")
        self.l_ergebnis.configure(
            text=f"Letztes Training: {g['modell']}\n" + "   ·   ".join(teile),
            fg=theme.quoten_farbe(g["beste_val"]))

    def _grafiken_moeglich(self) -> bool:
        test = self._testdaten()
        return (self.ergebnis is not None or self.gespeichert is not None
                or (test is not None and not test["unsicher"]))

    def _testdaten(self) -> dict | None:
        """Testergebnis fuer Anzeige und Export - aus dem Speicher oder der Datei.

        Aus test_ergebnis.json nur, wenn es zum neuesten Modell und zu den
        eingestellten Klassen gehoert. Aeltere Dateien vermerken das Modell
        nicht (und Dateizeiten sind nach einem Kopieren wertlos): Sie werden
        nur zu einem Modell aus derselben Zeit gezeigt, ausdruecklich als
        unsicher, und nicht exportiert.
        """
        t = self.test_ergebnis
        pfad = training.neuestes_modell()
        # Hat 07_training.py inzwischen ein neueres Modell gespeichert, gehoert
        # die Zahl im Speicher nicht mehr dazu.
        if t is not None and t.modell_pfad == pfad:
            return {"quote": t.quote, "val_quote": t.val_quote,
                    "konfusion": t.konfusion, "n": t.n, "sitzungen": t.sitzungen,
                    "klassen": list(getattr(t, "klassen", None) or TASTEN),
                    "pfad": t.pfad, "hinweis": getattr(t, "hinweis", ""),
                    "gespeichert": False, "unsicher": False}
        datei = MODELLE / "test_ergebnis.json"
        if pfad is None or not datei.exists():
            return None
        try:
            daten = json.loads(datei.read_text(encoding="utf-8"))
            modell_name = daten.get("modell")
            if modell_name is not None and modell_name != pfad.name:
                return None
            if modell_name is None and "samplerate" in self._modell_kopf(pfad)[1]:
                # Ein Modell im neuen Format (mit Abtastrate) bekommt beim
                # Testen immer eine Datei mit Modellnamen - eine ohne gehoert
                # also sicher zu einem aelteren Modell.
                return None
            klassen = daten.get("klassen") or self.modell_klassen(pfad) or []
            if (list(klassen) != list(TASTEN)
                    or len(daten["konfusion"]) != len(TASTEN)):
                return None
            return {"quote": float(daten["test_quote"]),
                    "val_quote": daten.get("val_quote"),
                    "konfusion": daten["konfusion"], "n": int(daten["n_test"]),
                    "sitzungen": list(daten.get("sitzungen") or []),
                    "klassen": list(klassen), "pfad": datei,
                    "hinweis": "", "gespeichert": True,
                    "unsicher": modell_name is None}
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _grafiken_exportieren(self) -> None:
        import numpy as np
        from tastenakustik import plots, portrait

        # Validierung: aus diesem Lauf oder aus der Datei des neuesten Modells.
        if self.ergebnis is not None and list(self.ergebnis.klassen) == list(TASTEN):
            e = self.ergebnis
            val = {"verlauf": e.verlauf, "konfusion": e.konfusion,
                   "beste_val": e.beste_val}
        else:
            val = self.gespeichert
        test = self._testdaten()
        if test is not None and (test["klassen"] != list(TASTEN) or test["unsicher"]):
            test = None
        if val is None and test is None:
            return

        pfade = []
        theme.anwenden("hochformat")
        try:
            if val is not None:
                v = val["verlauf"]
                epochen = np.arange(1, len(v["val_acc"]) + 1)
                fig = plots.trainingsverlauf(epochen, v["train_loss"], v["val_loss"],
                                             v["train_acc"], v["val_acc"])
                pfade.append(portrait.exportiere(fig, "06_training", "04_modell"))
                fig = plots.konfusionsmatrix(np.asarray(val["konfusion"]),
                                             val["beste_val"])
                pfade.append(portrait.exportiere(fig, "08_confusion", "04_modell"))
            if test is not None:
                # Eigener Name, damit die Test-Matrix nie mit der aus der
                # Validierung verwechselt wird.
                fig = plots.konfusionsmatrix(np.asarray(test["konfusion"]),
                                             test["quote"])
                pfade.append(portrait.exportiere(fig, "08b_confusion_test",
                                                 "04_modell"))
                # Dieselben Stufen wie die Szene "vergleich" in 91.
                stufen = [(f"Zufall bei {len(TASTEN)} Klassen", zufall(),
                           theme.TEXT_SCHWACH)]
                if test["val_quote"] is not None:
                    stufen.append(("Validierung, gleicher Tag",
                                   float(test["val_quote"]), theme.AKZENT2))
                stufen.append(("Test, ungesehene Sitzung", test["quote"], theme.OK))
                fig = plots.ergebnis_vergleich(
                    stufen, hero=("Test auf ungesehenen Daten", test["quote"]))
                pfade.append(portrait.exportiere(fig, "11_vergleich", "04_modell"))
        except Exception as fehler:                        # noqa: BLE001
            self.melde(f"Grafiken nicht gespeichert: {fehler}", theme.FEHLER)
            return
        finally:
            # Sonst bleibt der grosse Hochformat-Stil fuer die Live-Kurven stehen.
            theme.anwenden("normal")
        self.melde(f"{len(pfade)} Grafiken gespeichert in {pfade[0].parent}", theme.OK)

    # -- Testsitzung ----------------------------------------------------
    def _test_anzeige(self) -> None:
        """Knopf und Zeile fuer die Testsitzung auf den aktuellen Stand bringen."""
        l_test = getattr(self, "l_test", None)
        if self.aktiv != 5 or l_test is None or not l_test.winfo_exists():
            return
        modell = training.neuestes_modell()
        modell_da = modell is not None
        modell_passt = self.modell_passt(modell)
        test_zeilen = self.sitzungen_nach_rolle()["test"]
        test_da = bool(test_zeilen)
        tests_passen = all(z["passt"] for z in test_zeilen)
        trainiert = self.training_laeuft()
        self.k_test.setze_aktiv(modell_da and modell_passt and test_da
                                and tests_passen and not trainiert
                                and not self.test_laeuft)
        self._knoepfe_auffrischen()
        t = None if trainiert or self.test_laeuft else self._testdaten()
        if self.test_laeuft:
            self.l_test.configure(text="Testsitzung wird ausgewertet …",
                                  fg=theme.AKZENT)
        elif trainiert:
            self.l_test.configure(
                text="Training läuft - danach „Auf Testsitzung prüfen“ für die "
                     "ehrliche Zahl des neuen Modells.",
                fg=theme.TEXT_SCHWACH)
        elif t is not None:
            vergleich = (f"   ·   Validation zum Vergleich {t['val_quote'] * 100:.1f} %"
                         if t["val_quote"] is not None else "")
            if t["unsicher"]:
                herkunft = (f"Aus {t['pfad'].name} - ältere Datei ohne Angabe des "
                            "Modells, zur Sicherheit neu prüfen.")
            elif t["gespeichert"]:
                herkunft = f"Zuletzt ausgewertet, aus {t['pfad'].name}."
            else:
                herkunft = f"Gespeichert in {t['pfad'].name if t['pfad'] else '-'}."
            hinweis = f"\n{t['hinweis']}" if t["hinweis"] else ""
            aus = f" aus {', '.join(t['sitzungen'])}" if t["sitzungen"] else ""
            self.l_test.configure(
                text=f"Testsitzung: {t['quote'] * 100:.1f} %  "
                     f"({t['quote'] / zufall():.1f}-fach über dem Zufall, "
                     f"{t['n']} Proben{aus}){vergleich}\n"
                     f"Das ist die ehrliche Zahl. {herkunft}{hinweis}",
                fg=theme.quoten_farbe(t["quote"]))
        elif modell_da and not modell_passt:
            self.l_test.configure(
                text="Das neueste Modell wurde mit anderen Klassen trainiert - "
                     "für die eingestellten Klassen erst neu trainieren.",
                fg=theme.WARN)
        elif test_da and not tests_passen:
            self.l_test.configure(
                text="Die Testsitzung wurde mit anderen Klassen aufgenommen - "
                     "in Schritt 3 eine passende Sitzung auf „test“ stellen.",
                fg=theme.WARN)
        elif modell_da and not test_da:
            self.l_test.configure(
                text="Für die ehrliche Zahl fehlt noch eine Sitzung mit der Rolle "
                     "„test“ - am besten an einem anderen Tag aufgenommen.",
                fg=theme.TEXT_SCHWACH)
        else:
            self.l_test.configure(text="", fg=theme.TEXT_SCHWACH)

    def _test_starten(self) -> None:
        if self.test_laeuft or self.training_laeuft():
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
            elif antwort.modell_pfad != training.neuestes_modell():
                # Inzwischen ist ein neueres Modell entstanden (etwa ueber
                # 07_training.py) - diese Zahl gehoert nicht mehr dazu.
                self.melde("Inzwischen gibt es ein neueres Modell - bitte noch "
                           "einmal auf der Testsitzung prüfen.", theme.WARN)
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
            "- in den Trainingsdaten lagen rund zwei Sekunden zwischen zwei "
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
            # Mit "=" verbunden: Ein Text wie "-ab" ("-" ist eine Klasse)
            # hielte argparse sonst fuer eine Option.
            argumente.append(f"--soll={soll}")
        if buehne:
            argumente.append("--buehne")
        self.starte_werkzeug("08_demo.py", *argumente)


def main() -> int:
    theme.anwenden("normal")
    try:
        studio = Studio()
    except ConfigFehler as fehler:
        # Eine von Hand kaputt editierte config.json soll nicht als roher
        # Traceback enden, sondern sagen, wo es hakt.
        text = (f"config.json laesst sich nicht lesen:\n\n{fehler}\n\n"
                "Fehler darin beheben oder die Datei loeschen - dann startet das "
                "Studio mit den Voreinstellungen.")
        print(text, file=sys.stderr)
        try:
            wurzel = tk.Tk()
            wurzel.withdraw()
            messagebox.showerror("Tastenakustik - config.json", text)
            wurzel.destroy()
        except tk.TclError:
            pass
        return 1
    studio.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

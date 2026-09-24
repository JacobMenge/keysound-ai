"""Schritt 3 - Guided Data Collector fuer die gewaehlten Klassen, Hochformat 9:16.

Ablauf: Das Programm gibt eine Taste vor, du drueckst genau diese Taste, und
nur das kurze Audiofenster um diesen angekuendigten Anschlag wird gespeichert.

Grenzen, die im Code verankert sind:
  * Aufnahme laeuft nur zwischen "Aufnahme starten" und "Stopp".
  * Tastendruecke werden ueber das Fensterereignis dieses Programms gelesen -
    kein globaler Hook. Ist das Fenster nicht im Vordergrund, pausiert die
    Aufnahme automatisch.
  * Zeichen ausserhalb der gewaehlten Klassen werden verworfen, ohne dass sie
    gezaehlt, angezeigt oder protokolliert werden.
  * Gespeichert wird ausschliesslich ein Fenster um einen Anschlag, der zu dem
    gerade angezeigten Prompt passt.
  * Die Promptfolge ist zufaellig. Ist in der Konfiguration eine Sperrfolge
    gesetzt, kommt sie nie zusammenhaengend vor - schon beim Ziehen der Folge
    ausgeschlossen und vor jedem Lauf noch einmal geprueft.

Das Fenster ist 1080 x 1920 gross und legt sich von selbst auf einen
Hochformat-Monitor, falls einer vorhanden ist.

Die Steuertasten (Esc, F7-F9) liegen bewusst ausserhalb jeder moeglichen
Klasse - hier kann nichts mit einer aufzunehmenden Taste kollidieren.

Aufruf:
    python werkzeuge/03_collector.py
    python werkzeuge/03_collector.py --ziel 50
    python werkzeuge/03_collector.py --geraet 39

Tasten im Fenster:
    die gewaehlten Klassen  Probe aufnehmen
    Esc                     pausieren / fortsetzen
    F7                      Buehne: Sitzungskennung und Hinweise ausblenden
    F8                      Sicherheitszonen fuer Shorts / Reels einblenden
    F9                      aktuelle Ansicht nach ausgabe/ speichern
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from tastenakustik import audio, bedienung, features, onset, portrait, storage, theme
from tastenakustik.config import (
    Config,
    KlassenFehler,
    TASTEN,
    anzeige,
    beiname,
    ist_demo_taste,
    pruefe_keine_sperrfolge,
    verzeichnisse_anlegen,
)

NFFT = 512
HOP = 64

# Hoehenbudget in Leinwandpixeln (1920 gesamt), von oben nach unten.
H_KOPF_LEER = 150
H_KOPF = 60
H_PROMPT = 460
H_STATUS = 66
H_FIGUR = 600
H_FORTSCHRITT = 300
H_FUSS = 170


def erzeuge_folge(ziel: int, rng: random.Random, sperrfolge: str = "",
                  versuche: int = 200) -> list[str]:
    """Ausgewogene, gemischte Promptfolge - hoechstens zwei gleiche in Reihe.

    Zufaellige Reihenfolge ist zwingend: Wuerden wir alle A hintereinander
    aufnehmen, waere jede spaetere Trennbarkeit womoeglich nur ein Abbild von
    Drift (Handhaltung, Raum, Pegel) statt vom Tastenklang.

    Wer eine Sperrfolge gesetzt hat, will dieses Wort spaeter blind testen -
    dann darf es hier nicht zusammenhaengend vorkommen. Die Folge wird deshalb
    Zeichen fuer Zeichen gezogen und laesst dabei nichts zu, was eine
    Dreierreihe oder die Sperrfolge vollenden wuerde. Einfach mischen und
    hinterher pruefen reicht nicht: Bei einem kurzen Wort steckt es in einer
    Folge von einigen hundert Zeichen fast immer irgendwo drin.
    """
    sperre = (sperrfolge or "").lower()
    if sperre and any(c not in TASTEN for c in sperre):
        sperre = ""                   # kann ohnehin nie vorkommen
    if len(sperre) == 1:
        raise KlassenFehler(
            f"Die Sperrfolge {sperre!r} ist ein einzelnes Zeichen - das laesst "
            "sich nicht aufnehmen, ohne es zu verwenden.")

    gesamt = ziel * len(TASTEN)
    for _ in range(versuche):
        rest = {t: ziel for t in TASTEN}
        folge: list[str] = []
        while len(folge) < gesamt:
            kandidaten = [t for t, n in rest.items() if n > 0]
            if len(folge) >= 2 and folge[-1] == folge[-2]:
                kandidaten = [t for t in kandidaten if t != folge[-1]]
            if sperre:
                ende = "".join(folge[len(folge) - len(sperre) + 1:])
                if len(ende) == len(sperre) - 1:
                    kandidaten = [t for t in kandidaten if ende + t != sperre]
            if not kandidaten:
                break                 # festgefahren - neu ziehen
            # Nach verbleibender Anzahl gewichtet: ohne Nebenbedingungen ist
            # das genau eine gleichverteilte Mischung.
            wahl = rng.choices(kandidaten, weights=[rest[t] for t in kandidaten])[0]
            folge.append(wahl)
            rest[wahl] -= 1
        if len(folge) == gesamt:
            pruefe_keine_sperrfolge(folge, sperre)
            return folge
    raise KlassenFehler(
        f"Mit den Klassen {''.join(TASTEN)} laesst sich keine Aufnahmefolge "
        f"bilden, in der {sperrfolge!r} nicht vorkommt. Eine laengere "
        "Sperrfolge waehlen oder sie in Schritt 2 leeren.")


def rundes_rechteck(canvas: tk.Canvas, x0, y0, x1, y1, r, **kw):
    """Abgerundetes Rechteck - tkinter kann das nicht von Haus aus."""
    punkte = [
        x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
        x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
    ]
    return canvas.create_polygon(punkte, smooth=True, **kw)


class Collector(tk.Tk):
    def __init__(self, cfg: Config, ziel: int, seed: int | None,
                 buehne: bool = False):
        super().__init__()
        self.buehne = buehne
        self.cfg = cfg
        self.ziel = ziel
        self.rng = random.Random(seed)
        self.ring = audio.Ringpuffer(cfg)
        self.sitzung: storage.Sitzung | None = None
        self.zustand = "start"
        self.warteschlange: list[str] = []
        self.aktuelle_taste = ""
        self.t_taste = 0.0
        self.t_prompt = 0.0
        self.t_start = 0.0
        self.prompt_nr = 0
        self.falsche = 0
        self.letztes_fenster: np.ndarray | None = None
        self.zonen: list[tk.Frame] = []
        self.vor_pause = ""           # Zustand, in dem pausiert wurde
        self.prompt_offen = False     # naechster Prompt kam waehrend der Pause
        self.l_startfehler: tk.Label | None = None

        self.title("Guided Collector - Tastenakustik")
        self.configure(bg=theme.BG)
        self.s = portrait.fenster_einrichten(self)
        # Windows-Skalierung (125 %, 150 %): Matplotlib vergroessert die
        # eingebettete Figur um diesen Faktor, Tk die Schrift in Punkt. Beides
        # wird herausgerechnet, sonst laeuft das Layout unten aus dem Fenster.
        self.pixelfaktor = bedienung.pixelfaktor(self)
        self.protocol("WM_DELETE_WINDOW", self._schliessen)

        self._schriften()
        self._startansicht()
        self._messansicht()
        self.mess.pack_forget()
        self.bind("<KeyPress>", self._taste)

    # -- Hilfen ---------------------------------------------------------
    def px(self, wert: float) -> int:
        """Leinwandpixel in tatsaechliche Fensterpixel umrechnen."""
        return max(int(round(wert * self.s)), 1)

    def pt(self, wert: float) -> int:
        """Schriftgroesse in Punkt fuer die Matplotlib-Figur."""
        return max(int(round(wert * self.s)), 6)

    def schrift(self, wert: float) -> int:
        """Schriftgroesse fuer Tk - in Pixeln, damit sie zum Layout passt.

        Tk rechnet Punkt mit der Windows-Skalierung in Pixel um, das Layout
        hier ist aber in echten Pixeln gebaut. Negative Groessen sind in Tk
        Pixelangaben; bei 100 % Skalierung kommt dasselbe heraus wie vorher.
        """
        return -max(int(round(wert * self.s * 96 / 72)), 8)

    def _schriften(self) -> None:
        fam = "Segoe UI" if "Segoe UI" in tkfont.families() else "Arial"
        self.fam = fam
        self.f_titel = tkfont.Font(family=fam, size=self.schrift(30), weight="bold")
        self.f_normal = tkfont.Font(family=fam, size=self.schrift(14))
        self.f_klein = tkfont.Font(family=fam, size=self.schrift(12))
        self.f_mono = tkfont.Font(family="Consolas", size=self.schrift(13))
        self.f_label = tkfont.Font(family=fam, size=self.schrift(17))
        self.f_status = tkfont.Font(family=fam, size=self.schrift(18), weight="bold")

    # -- Startansicht ---------------------------------------------------
    def _startansicht(self) -> None:
        f = tk.Frame(self, bg=theme.BG, padx=self.px(70), pady=self.px(210))
        f.pack(fill="both", expand=True)
        self.start = f

        tk.Label(f, text="Guided Collector", font=self.f_titel,
                 bg=theme.BG, fg=theme.TEXT).pack(anchor="w")

        # Klassenleiste - bricht um, damit auch ein ganzes Alphabet passt.
        leiste = tk.Frame(f, bg=theme.BG)
        leiste.pack(anchor="w", pady=(self.px(24), self.px(28)))
        pro_zeile = 13 if len(TASTEN) > 8 else 8
        kachel = tkfont.Font(family=self.fam,
                             size=self.schrift(20 if len(TASTEN) <= 8 else 15),
                             weight="bold")
        for i, taste in enumerate(TASTEN):
            zeile, spalte = divmod(i, pro_zeile)
            tk.Label(leiste, text=anzeige(taste), font=kachel,
                     bg=theme.farbe(taste), fg=theme.BG,
                     width=2, pady=self.px(4)).grid(
                row=zeile, column=spalte,
                padx=(0, self.px(10)), pady=(0, self.px(8)))

        hinweis = (
            "Die Aufnahme laeuft nur, solange dieses Fenster im Vordergrund ist,\n"
            "und speichert ausschliesslich das kurze Fenster um den jeweils\n"
            "angekuendigten Anschlag. Alle anderen Tasten werden ignoriert und\n"
            "nicht protokolliert.\n\n"
            "Die Reihenfolge ist zufaellig - trainiert werden einzelne\n"
            "Anschlaege, keine Woerter."
        )
        if self.cfg.sperrfolge:
            hinweis += (f"\n\nDie Folge {self.cfg.sperrfolge!r} kommt in keiner\n"
                        "Aufnahmefolge vor - sie bleibt dem Blindtest vorbehalten.")
        tk.Label(f, text=hinweis, font=self.f_normal, justify="left",
                 bg=theme.PANEL, fg=theme.TEXT_SCHWACH,
                 padx=self.px(22), pady=self.px(18)).pack(anchor="w", fill="x")

        raster = tk.Frame(f, bg=theme.BG)
        raster.pack(anchor="w", pady=(self.px(34), 0), fill="x")
        self.felder: dict[str, tk.Entry] = {}
        for i, (schluessel, beschriftung) in enumerate(
            (("tastatur", "Tastatur (Modell, Schalter)"),
             ("mikrofon_position", "Mikrofonposition"),
             ("notiz", "Notiz zur Sitzung"))
        ):
            tk.Label(raster, text=beschriftung, font=self.f_normal, anchor="w",
                     bg=theme.BG, fg=theme.TEXT_SCHWACH).grid(
                row=i * 2, column=0, sticky="w", pady=(self.px(10), 0))
            e = tk.Entry(raster, font=self.f_normal, bg=theme.PANEL, fg=theme.TEXT,
                         insertbackground=theme.TEXT, relief="flat", highlightthickness=1,
                         highlightbackground=theme.GRID, highlightcolor=theme.AKZENT)
            e.grid(row=i * 2 + 1, column=0, sticky="we", ipady=self.px(7))
            self.felder[schluessel] = e
        raster.grid_columnconfigure(0, weight=1)

        tk.Label(raster, text="Proben je Taste", font=self.f_normal, anchor="w",
                 bg=theme.BG, fg=theme.TEXT_SCHWACH).grid(
            row=6, column=0, sticky="w", pady=(self.px(10), 0))
        self.ziel_var = tk.StringVar(value=str(self.ziel))
        tk.Entry(raster, textvariable=self.ziel_var, font=self.f_normal, width=6,
                 bg=theme.PANEL, fg=theme.TEXT, insertbackground=theme.TEXT,
                 relief="flat", highlightthickness=1,
                 highlightbackground=theme.GRID).grid(
            row=7, column=0, sticky="w", ipady=self.px(7))

        tk.Label(f, text=f"Eingang:  {self.cfg.device_name}\n"
                        f"{self.cfg.hostapi}, {self.cfg.samplerate} Hz",
                 font=self.f_klein, justify="left",
                 bg=theme.BG, fg=theme.TEXT_SCHWACH).pack(anchor="w", pady=(self.px(30), 0))

        tk.Button(f, text="Aufnahme starten", font=self.f_label, bg=theme.OK,
                  fg="#06240F", activebackground=theme.OK, relief="flat",
                  padx=self.px(36), pady=self.px(16), takefocus=0,
                  command=self._starten).pack(anchor="w", pady=(self.px(30), 0))

    # -- Messansicht ----------------------------------------------------
    def _messansicht(self) -> None:
        m = tk.Frame(self, bg=theme.BG)
        m.pack(fill="both", expand=True)
        self.mess = m
        rand = self.px(portrait.INHALT_LINKS)

        tk.Frame(m, bg=theme.BG, height=self.px(H_KOPF_LEER)).pack(fill="x")

        kopf = tk.Frame(m, bg=theme.BG, height=self.px(H_KOPF))
        kopf.pack(fill="x", padx=rand)
        self.l_sitzung = tk.Label(kopf, text="", font=self.f_mono,
                                  bg=theme.BG, fg=theme.TEXT_SCHWACH)
        self.l_sitzung.pack(side="left")
        self.l_rec = tk.Label(kopf, text="●  AUFNAHME", font=self.f_label,
                              bg=theme.BG, fg=theme.REC)
        self.l_rec.pack(side="right")

        # Prompt: grosses Feld in der Klassenfarbe, Zeichen dunkel darin.
        # Ohne Feld waere der Punkt als Klasse auf dem Schirm kaum zu sehen.
        self.prompt = tk.Canvas(m, height=self.px(H_PROMPT), bg=theme.BG,
                                highlightthickness=0, takefocus=0)
        self.prompt.pack(fill="x")
        self.prompt.bind("<Configure>", lambda _e: self._zeichne_prompt())

        self.l_status = tk.Label(m, text="", font=self.f_status, bg=theme.BG,
                                 fg=theme.TEXT_SCHWACH, height=1)
        self.l_status.pack(pady=(self.px(8), 0))

        theme.anwenden("normal")
        breite_px = self.px(portrait.BREITE - 2 * portrait.INHALT_LINKS + 60)
        self.fig = Figure(figsize=(breite_px / 100, self.px(H_FIGUR) / 100),
                          dpi=100 / self.pixelfaktor, facecolor=theme.BG)
        self.ax_w = self.fig.add_subplot(2, 1, 1)
        self.ax_s = self.fig.add_subplot(2, 1, 2)
        self.fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.11, hspace=0.62)
        self.canvas = FigureCanvasTkAgg(self.fig, master=m)
        self.canvas.get_tk_widget().pack(padx=rand - self.px(30))
        self.canvas.get_tk_widget().configure(highlightthickness=0, takefocus=0)
        self._leere_plots()

        self.fortschritt = tk.Canvas(m, height=self.px(H_FORTSCHRITT), bg=theme.BG,
                                     highlightthickness=0, takefocus=0)
        self.fortschritt.pack(fill="x", padx=rand, pady=(self.px(10), 0))
        self.fortschritt.bind("<Configure>", lambda _e: self._zeichne_fortschritt())

        fuss = tk.Frame(m, bg=theme.BG, height=self.px(H_FUSS))
        fuss.pack(fill="x", side="bottom", padx=rand, pady=(0, self.px(40)))
        self.l_summe = tk.Label(fuss, text="", font=self.f_mono, bg=theme.BG, fg=theme.TEXT)
        self.l_summe.pack(anchor="w", pady=(0, self.px(12)))
        knoepfe = tk.Frame(fuss, bg=theme.BG)
        knoepfe.pack(anchor="w")
        self.b_pause = tk.Button(knoepfe, text="Pause (Esc)", font=self.f_normal,
                                 bg=theme.PANEL, fg=theme.TEXT, activebackground=theme.GRID,
                                 relief="flat", padx=self.px(22), pady=self.px(10),
                                 takefocus=0, command=self._pause_umschalten)
        self.b_pause.pack(side="left")
        tk.Button(knoepfe, text="Stopp und speichern", font=self.f_normal, bg=theme.PANEL,
                  fg=theme.FEHLER, activebackground=theme.GRID, relief="flat",
                  padx=self.px(22), pady=self.px(10), takefocus=0,
                  command=self._beenden).pack(side="left", padx=(self.px(14), 0))
        self.l_hinweis = tk.Label(knoepfe, text="F7 Buehne   F8 Zonen   F9 Standbild",
                                  font=self.f_klein, bg=theme.BG, fg=theme.TEXT_SCHWACH)
        self.l_hinweis.pack(side="left", padx=(self.px(22), 0))
        self._setze_buehne()

    def _setze_buehne(self) -> None:
        """Werkzeugansicht oder Buehne.

        Auf der Buehne bleiben Prompt, Wellenform und Fortschritt - also das,
        worum es geht. Sitzungskennung, Tempo und Tastenhinweise sind beim
        Einrichten nuetzlich und beim Filmen nur Beiwerk.
        """
        if self.buehne:
            self.l_sitzung.pack_forget()
            self.l_hinweis.pack_forget()
        else:
            self.l_sitzung.pack(side="left")
            self.l_hinweis.pack(side="left", padx=(self.px(22), 0))
        if self.sitzung is not None:
            self._zeichne_fortschritt()

    def _leere_plots(self) -> None:
        for ax, titel in ((self.ax_w, "Aufgenommenes Fenster"), (self.ax_s, "Spektrogramm")):
            ax.clear()
            ax.set_title(titel, fontsize=self.pt(11))
            ax.set_xticks([])
            ax.set_yticks([])
        self.canvas.draw_idle()

    def _zeichne_prompt(self) -> None:
        c = self.prompt
        c.delete("all")
        breite = c.winfo_width()
        if breite < 100:
            breite = self.px(portrait.BREITE)
        hoehe = self.px(H_PROMPT)
        mitte = breite / 2

        c.create_text(mitte, self.px(44), text="DRÜCKE", fill=theme.TEXT_SCHWACH,
                      font=(self.fam, self.schrift(22)))
        if not self.aktuelle_taste:
            return
        farbe = theme.farbe(self.aktuelle_taste)
        kante = self.px(300)
        oben = self.px(90)
        rundes_rechteck(c, mitte - kante / 2, oben, mitte + kante / 2, oben + kante,
                        self.px(46), fill=farbe, outline="")
        c.create_text(mitte, oben + kante / 2, text=anzeige(self.aktuelle_taste),
                      fill=theme.BG, font=(self.fam, self.schrift(160), "bold"))
        name = beiname(self.aktuelle_taste)
        if name:
            c.create_text(mitte, oben + kante + self.px(36), text=name, fill=farbe,
                          font=(self.fam, self.schrift(20), "bold"))

    # -- Ablauf ---------------------------------------------------------
    def _startfehler(self, text: str) -> None:
        """Grund anzeigen, warum die Aufnahme nicht startet - immer an einer Stelle."""
        if self.l_startfehler is None:
            self.l_startfehler = tk.Label(
                self.start, font=self.f_normal, bg=theme.BG, fg=theme.FEHLER,
                wraplength=self.px(880), justify="left")
            self.l_startfehler.pack(anchor="w", pady=(self.px(18), 0))
        self.l_startfehler.configure(text=text)

    def _starten(self) -> None:
        if self.zustand != "start":
            return
        try:
            self.ziel = max(1, int(self.ziel_var.get()))
        except ValueError:
            self.ziel = 40
        # Erst die Folge, dann das Mikrofon, dann der Sitzungsordner: Scheitert
        # ein Schritt, bleibt kein leerer Sitzungsordner in daten/roh/ liegen.
        try:
            folge = erzeuge_folge(self.ziel, self.rng, self.cfg.sperrfolge)
        except KlassenFehler as exc:
            self._startfehler(str(exc))
            return
        try:
            self.ring.start()
        except RuntimeError as exc:
            self._startfehler(str(exc))
            return

        self.sitzung = storage.Sitzung(
            self.cfg,
            notiz=self.felder["notiz"].get().strip(),
            tastatur=self.felder["tastatur"].get().strip(),
            mikrofon_position=self.felder["mikrofon_position"].get().strip(),
        )
        self.warteschlange = folge
        self.t_start = time.perf_counter()
        self.start.pack_forget()
        self.mess.pack(fill="both", expand=True)
        self.l_sitzung.configure(
            text=f"{self.sitzung.id}   {self.cfg.samplerate} Hz   Ziel {self.ziel}"
        )
        self._zeichne_fortschritt()
        self.focus_force()
        self._naechster_prompt()
        self.after(1500, self._fokus_pruefen)

    def _naechster_prompt(self) -> None:
        if self.zustand in ("pause", "fertig"):
            # Pausiert in der kurzen Wartezeit nach einer Probe: Der naechste
            # Prompt darf die Pause nicht aufheben - er kommt beim Fortsetzen.
            self.prompt_offen = self.zustand == "pause"
            return
        if not self.warteschlange:
            self._fertig()
            return
        self.aktuelle_taste = self.warteschlange.pop(0)
        self.prompt_nr += 1
        self.t_prompt = time.perf_counter()
        self.zustand = "warte"
        self._zeichne_prompt()
        self._rec_anzeige()

    def _taste(self, event) -> None:  # noqa: ANN001
        if event.keysym == "F9":
            self._asset_speichern()
            return
        if event.keysym == "F7":
            self.buehne = not self.buehne
            self._setze_buehne()
            return
        if event.keysym == "F8":
            self._zonen_umschalten()
            return
        if event.keysym == "Escape":
            self._pause_umschalten()
            return
        if self.zustand != "warte":
            return

        zeichen = event.keysym.lower()
        if len(zeichen) != 1:
            zeichen = (event.char or "").lower()
        # Alles ausserhalb der gewaehlten Klassen endet hier: nicht gezaehlt,
        # nicht angezeigt, nicht gespeichert.
        if not ist_demo_taste(zeichen):
            return
        if zeichen != self.aktuelle_taste:
            self.falsche += 1
            self._status(f"{anzeige(zeichen)} statt {anzeige(self.aktuelle_taste)} "
                         f"- nichts gespeichert", theme.WARN)
            return

        self.zustand = "erfasst"
        self.t_taste = time.perf_counter()
        self._status("...", theme.TEXT_SCHWACH)
        self.after(self.cfg.post_roll_ms + 140, self._verarbeite)

    def _verarbeite(self) -> None:
        # Wurde zwischendurch pausiert oder gestoppt, wird nichts gespeichert.
        if self.sitzung is None or self.zustand != "erfasst":
            return
        self.zustand = "verarbeitet"   # genau eine Probe je Anschlag
        fenster = self.ring.fenster_um_taste(self.t_taste)
        if fenster is None:
            self._verwerfen("kein_fenster")
            return

        anschlag = onset.analysiere(fenster, self.cfg)
        self.letztes_fenster = fenster

        if anschlag.brauchbar:
            self.sitzung.speichere(
                fenster, self.aktuelle_taste, anschlag,
                extra={
                    "prompt_index": self.prompt_nr,
                    "reaktionszeit_ms": round((self.t_taste - self.t_prompt) * 1000, 1),
                    "stream_latenz_ms": round(self.ring.latenz_s * 1000, 2),
                },
            )
            n = self.sitzung.zaehler[self.aktuelle_taste]
            self._status(f"gespeichert   {n}/{self.ziel}   "
                         f"Abstand {anschlag.snr_db:.0f} dB", theme.OK)
        else:
            self._verwerfen(anschlag.grund, still=True)

        self._zeichne_probe(fenster, anschlag)
        self._zeichne_fortschritt()
        self.after(self.cfg.pause_nach_probe_ms, self._naechster_prompt)

    def _verwerfen(self, grund: str, still: bool = False) -> None:
        if self.sitzung is not None:
            self.sitzung.notiere_verwurf(grund)
        # Probe kommt spaeter noch einmal dran, damit das Ziel erreicht wird.
        if self.warteschlange:
            self.warteschlange.insert(self.rng.randrange(0, len(self.warteschlange)),
                                      self.aktuelle_taste)
        else:
            self.warteschlange.append(self.aktuelle_taste)
        self._status(f"verworfen: {onset.GRUND_TEXT.get(grund, grund)}", theme.WARN)
        if not still:
            self.after(self.cfg.pause_nach_probe_ms, self._naechster_prompt)

    # -- Anzeige --------------------------------------------------------
    def _status(self, text: str, farbe: str) -> None:
        self.l_status.configure(text=text, fg=farbe)

    def _rec_anzeige(self) -> None:
        if self.zustand == "pause":
            self.l_rec.configure(text="■  PAUSE", fg=theme.WARN)
        elif self.zustand == "fertig":
            self.l_rec.configure(text="■  BEENDET", fg=theme.TEXT_SCHWACH)
        else:
            self.l_rec.configure(text="●  AUFNAHME", fg=theme.REC)

    def _zeichne_probe(self, fenster: np.ndarray, anschlag) -> None:
        sr = self.cfg.samplerate
        farbe = theme.farbe(self.aktuelle_taste)
        t = (np.arange(fenster.size) / sr - self.cfg.pre_roll_ms / 1000) * 1000

        self.ax_w.clear()
        self.ax_w.plot(t, fenster, color=farbe, lw=0.8)
        if anschlag.gefunden:
            t_on = t[min(anschlag.onset_sample, t.size - 1)]
            self.ax_w.axvline(t_on, color=theme.AKZENT2, lw=2.0)
            s0, s1 = onset.segment_grenzen(anschlag, fenster.size, self.cfg)
            self.ax_w.axvspan(t[s0], t[min(s1, t.size - 1)], color=theme.AKZENT2, alpha=0.12)
        self.ax_w.set_title(f"{anzeige(self.aktuelle_taste)}  -  aufgenommenes Fenster",
                            fontsize=self.pt(11))
        self.ax_w.set_xlim(t[0], t[-1])
        self.ax_w.tick_params(labelsize=self.pt(9))

        db = features.stft_db(fenster, NFFT, HOP)
        db, max_hz = features.bis_frequenz(db, sr, NFFT, self.cfg.spektro_max_hz)
        self.ax_s.clear()
        obergrenze = float(np.percentile(db, 99.9))
        self.ax_s.imshow(db.T, origin="lower", aspect="auto", cmap="magma",
                         extent=(t[0], t[-1], 0, max_hz / 1000),
                         vmin=obergrenze - 62, vmax=obergrenze, interpolation="nearest")
        self.ax_s.set_title("Spektrogramm", fontsize=self.pt(11))
        self.ax_s.set_xlabel("ms relativ zum Tastenereignis", fontsize=self.pt(9))
        self.ax_s.set_ylabel("kHz", fontsize=self.pt(9))
        self.ax_s.tick_params(labelsize=self.pt(9))
        self.ax_s.grid(False)
        self.canvas.draw_idle()

    def _zeichne_fortschritt(self) -> None:
        if self.sitzung is None:
            return
        c = self.fortschritt
        c.delete("all")
        breite = c.winfo_width()
        if breite < 100:
            breite = self.px(portrait.BREITE - 2 * portrait.INHALT_LINKS)
        # Raster an die Klassenzahl anpassen: hoechstens acht Zeilen, danach
        # kommen Spalten dazu. Ab vier Spalten ist fuer den Balken kein Platz
        # mehr - dann bleiben Kachel und Zaehler.
        n_klassen = len(TASTEN)
        spalten = max(1, -(-n_klassen // 8))
        zeilen = -(-n_klassen // spalten)
        spalte = breite / spalten
        zeilenhoehe = self.px(H_FORTSCHRITT) / zeilen
        mit_balken = spalten <= 3
        gross = zeilen <= 4

        for i, taste in enumerate(TASTEN):
            sp, zeile = divmod(i, zeilen)          # spaltenweise fuellen
            x0 = sp * spalte
            y = zeilenhoehe * (zeile + 0.5)
            n = self.sitzung.zaehler[taste]
            farbe = theme.farbe(taste)
            feld = self.px(34 if gross else 24)
            rundes_rechteck(c, x0, y - feld / 2, x0 + feld, y + feld / 2,
                            self.px(8), fill=farbe, outline="")
            c.create_text(x0 + feld / 2, y, text=anzeige(taste), fill=theme.BG,
                          font=(self.fam, self.schrift(15 if gross else 11), "bold"))
            zaehler_x = x0 + feld + self.px(14)
            if mit_balken:
                bx0 = zaehler_x
                bx1 = x0 + spalte - self.px(90)
                c.create_rectangle(bx0, y - self.px(10), bx1, y + self.px(10),
                                   fill=theme.PANEL, outline=theme.GRID)
                anteil = min(n / self.ziel, 1.0)
                if anteil > 0:
                    c.create_rectangle(bx0, y - self.px(10),
                                       bx0 + (bx1 - bx0) * anteil,
                                       y + self.px(10), fill=farbe, outline="")
                zaehler_x = bx1 + self.px(12)
            c.create_text(zaehler_x, y, text=f"{n}/{self.ziel}", anchor="w",
                          fill=theme.TEXT if n >= self.ziel else theme.TEXT_SCHWACH,
                          font=("Consolas", self.schrift(13 if gross else 10)))

        gesamt = self.sitzung.gesamt
        soll = self.ziel * len(TASTEN)
        dauer = max(time.perf_counter() - self.t_start, 1e-6)
        verworfen = sum(self.sitzung.verworfen.values())
        if self.buehne:
            self.l_summe.configure(text=f"{gesamt} / {soll}")
        else:
            self.l_summe.configure(
                text=f"{gesamt}/{soll} Proben    {gesamt / dauer * 60:5.1f}/min    "
                     f"verworfen {verworfen}    falsch {self.falsche}")

    # -- Steuerung ------------------------------------------------------
    def _pause_umschalten(self) -> None:
        # Auf der Startseite gibt es nichts zu pausieren - Esc beim Ausfuellen
        # der Felder darf den Zustand nicht verstellen.
        if self.zustand in ("start", "fertig"):
            return
        if self.zustand == "pause":
            try:
                self.ring.start()
            except RuntimeError as exc:
                self._status(f"Mikrofon nicht verfügbar: {exc}", theme.FEHLER)
                return
            self.b_pause.configure(text="Pause (Esc)")
            self._status("weiter", theme.TEXT_SCHWACH)
            if self.prompt_offen:
                self.prompt_offen = False
                self.zustand = "verarbeitet"
                self._naechster_prompt()
            elif self.vor_pause == "verarbeitet":
                # Der naechste Prompt ist schon angesetzt und kommt gleich.
                self.zustand = "verarbeitet"
            else:
                self.zustand = "warte"
        else:
            self.vor_pause = self.zustand
            self.ring.stop()
            self.zustand = "pause"
            self.b_pause.configure(text="Weiter (Esc)")
            self._status("pausiert - Mikrofon ist aus", theme.WARN)
        self._rec_anzeige()

    def _fokus_pruefen(self) -> None:
        """Verlaesst das Fenster den Vordergrund, stoppt die Aufnahme von selbst."""
        if self.zustand in ("warte", "erfasst"):
            try:
                im_vordergrund = self.focus_get() is not None
            except (KeyError, tk.TclError):
                im_vordergrund = False
            if not im_vordergrund:
                self._pause_umschalten()
                self._status("pausiert - Fenster war nicht im Vordergrund", theme.WARN)
        if self.zustand != "fertig":
            self.after(400, self._fokus_pruefen)

    def _zonen_umschalten(self) -> None:
        """Die von Shorts, Reels und TikTok ueberdeckten Raender einblenden."""
        if self.zonen:
            for z in self.zonen:
                z.destroy()
            self.zonen = []
            return
        for y_px in (portrait.SICHER_OBEN, portrait.HOEHE - portrait.SICHER_UNTEN):
            linie = tk.Frame(self, bg=theme.FEHLER, height=2)
            linie.place(x=0, y=self.px(y_px), relwidth=1.0, height=2)
            self.zonen.append(linie)
        senkrecht = tk.Frame(self, bg=theme.FEHLER, width=2)
        senkrecht.place(x=self.px(portrait.BREITE - portrait.SICHER_RECHTS), y=0,
                        width=2, relheight=1.0)
        self.zonen.append(senkrecht)

    def _asset_speichern(self) -> None:
        if self.letztes_fenster is None:
            return
        pfad = theme.speichere_asset(self.fig, "probe", "02_collector")
        self._status(f"gespeichert: {pfad.name}", theme.AKZENT)

    def _fertig(self) -> None:
        self.zustand = "fertig"
        self.ring.stop()
        if self.sitzung is not None:
            self.sitzung.abschliessen()
        self.aktuelle_taste = ""
        self.prompt.delete("all")
        self.prompt.create_text(
            self.prompt.winfo_width() / 2, self.px(H_PROMPT) / 2, text="✓",
            fill=theme.OK, font=(self.fam, self.schrift(140), "bold"))
        self._status("Alle Ziele erreicht - Sitzung gespeichert", theme.OK)
        self._rec_anzeige()
        self._bericht()

    def _beenden(self) -> None:
        self.zustand = "fertig"
        self.ring.stop()
        if self.sitzung is not None:
            self.sitzung.abschliessen()
        self._bericht()
        self.destroy()

    def _schliessen(self) -> None:
        if self.zustand not in ("start", "fertig"):
            self._beenden()
        else:
            self.ring.stop()
            self.destroy()

    def _bericht(self) -> None:
        if self.sitzung is None:
            return
        s = self.sitzung
        print(f"\nSitzung {s.id}")
        print(f"  Ordner      {s.ordner}")
        print(f"  Proben      {s.gesamt}")
        for taste in TASTEN:
            print(f"    {anzeige(taste)}  {s.zaehler[taste]:3d}")
        if s.verworfen:
            print("  verworfen   " + ", ".join(f"{k}: {v}" for k, v in s.verworfen.items()))
        print(f"  falsche Taste {self.falsche}")
        print("\n  Rolle vergeben (spaeter, nach Sitzungen getrennt):")
        print(f"    python werkzeuge/04_sitzungen.py --rolle {s.id}=train")


def main() -> int:
    p = argparse.ArgumentParser(description="Guided Data Collector")
    p.add_argument("--ziel", type=int, default=None, help="Proben je Taste")
    p.add_argument("--geraet", default=None, help="Index oder Namensfragment")
    p.add_argument("--seed", type=int, default=None, help="Seed fuer die Promptfolge")
    p.add_argument("--buehne", action="store_true",
                   help="ohne Sitzungskennung und Tastenhinweise - zum Filmen")
    args = p.parse_args()

    verzeichnisse_anlegen()
    portrait.dpi_bewusst()
    cfg = Config.laden()
    if args.geraet is not None:
        g = audio.geraet_finden(args.geraet)
        if g is None:
            print(f"Geraet {args.geraet!r} nicht gefunden.")
            return 1
        cfg.device, cfg.device_name, cfg.hostapi = g.index, g.name, g.hostapi
        cfg.samplerate, cfg.channels = audio.bestes_format(g, cfg.samplerate)
    if cfg.device is None:
        print("Keine Konfiguration gefunden. Bitte zuerst: python werkzeuge/01_systemcheck.py")
        return 1

    Collector(cfg, args.ziel, args.seed, args.buehne).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

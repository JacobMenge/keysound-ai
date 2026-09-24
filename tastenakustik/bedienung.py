"""Bedienung der beiden Live-Fenster - mit der Maus, nie ueber die Tastatur.

In der Kalibrierung und in der Live-Demo wird getippt, dafuer sind sie da.
Jede Taste, die dort nebenbei etwas ausloest, stoert genau das, was gerade
gemessen wird. Matplotlib bringt davon von Haus aus eine ganze Reihe mit: 's'
oeffnet den Speichern-Dialog, 'f' schaltet auf Vollbild, 'k' und 'l' stellen
die Achsen auf logarithmisch, 'q' schliesst das Fenster. Mit Klassen wie
"asdfjkl." trifft man sie beim Testen zwangslaeufig.

Deshalb gilt hier: Alle Tastenkuerzel sind abgeschaltet, und die Fenster
nehmen gar keine Tastatur-Ereignisse an. Bedient wird ueber eine Knopfleiste
unter dem Bild und ueber ein Rechtsklick-Menue. Das Menue funktioniert auch
auf der Buehne, wenn die Knopfleiste ausgeblendet ist.

Ausserdem sorgt dieses Modul dafuer, dass ein Live-Fenster ganz auf den
Bildschirm passt - auch auf einem Monitor im Querformat und bei einer
Windows-Skalierung von 125 oder 150 %.
"""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import font as tkfont
from typing import Callable

import matplotlib as mpl

from . import portrait, theme

# Hoehe, die ein Fenster ueber der Leinwand braucht (Titelleiste), und der
# Platz, der bei einem verkleinerten Fenster zusaetzlich fuer die Knopfleiste
# frei bleibt. In echten Pixeln bei 100 % Skalierung.
RAND_NATIV = 40
RAND_KLEIN = 130


def tastenkuerzel_aus() -> None:
    """Alle Matplotlib-Tastenkuerzel und die Werkzeugleiste abschalten.

    Muss vor dem Anlegen der Figur laufen - die Werkzeugleiste entsteht mit
    dem Fenster. Speichern geht danach ueber den Knopf "Standbild".
    """
    for schluessel in [k for k in mpl.rcParams if k.startswith("keymap.")]:
        mpl.rcParams[schluessel] = []
    mpl.rcParams["toolbar"] = "None"


def pixelfaktor(widget: tk.Misc | None = None) -> float:
    """Anzeigeskalierung des Systems: 1.0 bei 100 %, 1.25 bei 125 % und so fort.

    Matplotlib vergroessert eine Figur im Tk-Fenster um genau diesen Faktor.
    Wer echte Pixel will, muss ihn vorher herausrechnen. Gerechnet wird wie in
    Matplotlib selbst, damit sich beides genau aufhebt.
    """
    eigene = widget is None
    try:
        if eigene:
            widget = tk.Tk()
            widget.withdraw()
        if sys.platform == "win32":
            faktor = round(float(widget.tk.call("tk", "scaling")) / (96 / 72), 2)
        elif sys.platform.startswith("linux"):
            faktor = float(widget.winfo_fpixels("1i")) / 96
        else:
            faktor = 1.0
    except Exception:  # noqa: BLE001 - ohne Fenstersystem bleibt es bei 1
        faktor = 1.0
    finally:
        if eigene and widget is not None:
            try:
                widget.destroy()
            except Exception:  # noqa: BLE001, S110
                pass
    return faktor if faktor > 0 else 1.0


class LiveMasse:
    """Wie gross ein Live-Fenster wird und wohin es kommt.

    Passt die Leinwand in echten 1080 x 1920 Pixeln auf einen Monitor, bleibt
    es dabei - so wird gefilmt. Sonst wird die ganze Figur verkleinert,
    Schrift eingeschlossen, statt unten aus dem Bildschirm zu laufen.
    """

    def __init__(self) -> None:
        self.flaeche = portrait.beste_flaeche(RAND_NATIV)
        self.faktor = pixelfaktor()
        self.skala = 1.0
        if self.flaeche is not None:
            _, _, b, h = self.flaeche
            if b < portrait.BREITE or h < portrait.HOEHE + RAND_NATIV:
                # Titelleiste und Knopfleiste wachsen mit der Skalierung, und
                # auf einem schmalen Fenster bricht die Leiste zweizeilig um.
                rand = round(RAND_KLEIN * self.faktor)
                self.skala = max(min((h - rand) / portrait.HOEHE,
                                     (b - 24) / portrait.BREITE, 1.0), 0.25)
        # Matplotlib multipliziert die dpi spaeter mit dem Pixelfaktor. Hier
        # vorab geteilt, ergibt das genau skala * 1080 x 1920 echte Pixel.
        self.dpi = portrait.DPI * self.skala / self.faktor

    def platzieren(self, fenster: tk.Misc, leiste_hoehe: int = 0) -> None:
        """Fenster mittig auf die Arbeitsflaeche setzen - oben nie abgeschnitten."""
        if self.flaeche is None:
            return
        x0, y0, b, h = self.flaeche
        breite = round(portrait.BREITE * self.skala)
        hoehe = (round(portrait.HOEHE * self.skala) + leiste_hoehe
                 + round(RAND_NATIV * self.faktor))
        x = x0 + max((b - breite) // 2, 0)
        y = y0 + max((h - hoehe) // 2, 0)
        try:
            fenster.wm_geometry(f"+{x}+{y}")
        except Exception:  # noqa: BLE001, S110 - dann eben da, wo Windows es hinsetzt
            pass


class Bedienung:
    """Knopfleiste unter dem Bild und dieselben Befehle als Rechtsklick-Menue.

    Beim Anlegen wird das Fenster von der Tastatur getrennt: Der Standard-
    Handler von Matplotlib wird abgemeldet und die Tastenbindungen der
    Leinwand werden entfernt. Danach erreicht kein Tastendruck mehr den
    Python-Code dieses Fensters.
    """

    HINWEIS = "Rechtsklick: Menü"
    RAND_X, RAND_Y = 14, 6

    def __init__(self, fig, eintraege: list[tuple[str, Callable[[], None]]],
                 breite: int | None = None):
        self.fig = fig
        manager = fig.canvas.manager
        self.leinwand = fig.canvas.get_tk_widget()
        self.fenster = manager.window

        handler = getattr(manager, "key_press_handler_id", None)
        if handler is not None:
            fig.canvas.mpl_disconnect(handler)
        for sequenz in ("<Key>", "<KeyRelease>"):
            self.leinwand.unbind(sequenz)

        familie = "Segoe UI" if "Segoe UI" in tkfont.families(self.fenster) else "Arial"
        schrift = tkfont.Font(root=self.fenster, family=familie, size=10)
        klein = tkfont.Font(root=self.fenster, family=familie, size=9)

        # Die Leiste gibt ihre Breite nicht nach oben weiter: Sonst wuerde ein
        # schmales Fenster breiter als die Leinwand, und die Figur liefe aus
        # dem 9:16-Format. Die Knoepfe brechen stattdessen in Zeilen um.
        self.leiste = tk.Frame(self.fenster, bg=theme.BG, height=1)
        self.leiste.pack_propagate(False)
        self.innen = tk.Frame(self.leiste, bg=theme.BG)
        self.innen.place(x=self.RAND_X, y=self.RAND_Y)
        self.elemente = [self._knopf(beschriftung, befehl, schrift)
                         for beschriftung, befehl in eintraege]
        self.l_hinweis = tk.Label(self.innen, text=self.HINWEIS, font=klein,
                                  bg=theme.BG, fg=theme.TEXT_SCHWACH)
        self.elemente.append(self.l_hinweis)
        self._meldung_nr = 0
        self._breite = 0
        self._umbrechen(breite or self.leinwand.winfo_reqwidth())
        self.leiste.bind("<Configure>", lambda e: self._umbrechen(e.width))

        self.menue = tk.Menu(self.leinwand, tearoff=0)
        for beschriftung, befehl in eintraege:
            self.menue.add_command(label=beschriftung, command=befehl)
        self.leinwand.bind("<Button-3>", self._menue_zeigen, add="+")
        self.zeigen(True)

    def _knopf(self, text: str, befehl: Callable[[], None], schrift) -> tk.Label:
        """Flacher Knopf aus einem Label.

        Bewusst kein tk.Button: Der reagiert, sobald er den Fokus hat, auf die
        Leertaste - und die Leertaste kann eine Klasse sein.
        """
        knopf = tk.Label(self.innen, text=text, font=schrift, bg=theme.PANEL,
                         fg=theme.TEXT, padx=12, pady=5, cursor="hand2",
                         takefocus=0)
        knopf.bind("<Button-1>", lambda _e: befehl())
        knopf.bind("<Enter>", lambda _e: knopf.configure(bg=theme.GRID))
        knopf.bind("<Leave>", lambda _e: knopf.configure(bg=theme.PANEL))
        return knopf

    def _umbrechen(self, breite: int) -> None:
        """Knoepfe zeilenweise setzen, so viele wie in die Breite passen."""
        if breite <= 1 or abs(breite - self._breite) <= 2:
            return
        self._breite = breite
        verfuegbar = max(breite - 2 * self.RAND_X, 120)
        zeile = spalte = x = 0
        for element in self.elemente:
            element.grid_forget()
            noetig = element.winfo_reqwidth() + 8
            if spalte and x + noetig > verfuegbar:
                zeile, spalte, x = zeile + 1, 0, 0
            element.grid(row=zeile, column=spalte, sticky="w", padx=(0, 8), pady=2)
            spalte += 1
            x += noetig
        self.innen.update_idletasks()
        self.leiste.configure(height=self.innen.winfo_reqheight() + 2 * self.RAND_Y)

    def _menue_zeigen(self, ereignis) -> None:  # noqa: ANN001
        try:
            self.menue.tk_popup(ereignis.x_root, ereignis.y_root)
        finally:
            self.menue.grab_release()

    def melde(self, text: str, farbe: str = theme.AKZENT, dauer_ms: int = 4000) -> None:
        """Kurze Rueckmeldung rechts in der Leiste, danach wieder der Hinweis."""
        self._meldung_nr += 1
        nummer = self._meldung_nr
        self.l_hinweis.configure(text=text, fg=farbe)

        def zurueck() -> None:
            if nummer == self._meldung_nr:
                self.l_hinweis.configure(text=self.HINWEIS, fg=theme.TEXT_SCHWACH)

        try:
            self.l_hinweis.after(dauer_ms, zurueck)
        except tk.TclError:
            pass

    def zeigen(self, sichtbar: bool) -> None:
        """Knopfleiste ein- oder ausblenden (auf der Buehne aus)."""
        if sichtbar:
            # Vor der Leinwand einpacken: Wird der Platz knapp, schrumpft
            # sonst zuerst die Leiste weg.
            self.leiste.pack(side="bottom", fill="x", before=self.leinwand)
        else:
            self.leiste.pack_forget()

    def hoehe(self) -> int:
        """Hoehe der Knopfleiste in Pixeln - auch wenn sie gerade aus ist.

        Das Fenster wird so platziert, dass die Leiste immer noch Platz hat:
        Wer die Buehne verlaesst, soll die Knoepfe nicht unter der Taskleiste
        suchen muessen.
        """
        return int(self.leiste.cget("height"))

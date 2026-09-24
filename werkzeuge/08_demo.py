"""Schritt 8 - Live-Demo: aus dem Klang allein die Zeichenfolge aufbauen.

Es laufen KEINE Tastatur-Ereignisse in dieses Programm. Nichts davon wird
gelesen, nichts davon wird ausgewertet. Die Anschlaege werden ausschliesslich
im Audiosignal gefunden - mit demselben Transientendetektor, der auch beim
Aufnehmen die Onsets nachmisst.

Damit ist die Trennung nicht nur eine Absichtserklaerung, sondern eine
Eigenschaft des Programms: Es gibt hier gar keine Stelle, an der die
tatsaechlich gedrueckte Taste bekannt waere. Das gilt auch fuer die Bedienung
- sie laeuft nur ueber die Maus, damit beim Testen jede Taste frei ist (siehe
tastenakustik/bedienung.py).

Der optionale Vergleichstext (--soll) dient nur der Anzeige, wie viele Zeichen
getroffen wurden. Er beeinflusst die Vorhersage an keiner Stelle - er wird
erst nach der Klassifikation herangezogen.

Aufruf:
    python werkzeuge/08_demo.py
    python werkzeuge/08_demo.py --soll hallo
    python werkzeuge/08_demo.py --buehne          # ohne Titel und Knopfleiste
    python werkzeuge/08_demo.py --abstand 400     # laengere Sperrzeit
    python werkzeuge/08_demo.py --schwelle -40    # leisere Anschlaege zulassen

Abtastrate und Log-Mel-Parameter kommen aus der Modelldatei, nicht aus der
config.json: Das Modell muss dieselben Merkmale sehen wie beim Training.
Liefert das Mikrofon eine andere Rate, wird zuerst die Trainingsrate
angefordert, sonst jedes Segment umgerechnet - mit deutlicher Warnung.

Bedienung mit der Maus - Knoepfe unter dem Bild oder Rechtsklick ins Bild:
    Zuruecksetzen   Zeichenfolge leeren
    Buehne          Titel und Knopfleiste aus- und einblenden
    Standbild       aktuelles Bild nach ausgabe/05_demo/
    Beenden
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import numpy as np

from tastenakustik import portrait

portrait.dpi_bewusst()
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from tastenakustik import (audio, bedienung, datensatz, features, modell,  # noqa: E402
                           onset, storage, theme)
from tastenakustik.config import DATEN, TASTEN, Config, anzeige, setze_klassen, laden_oder_beenden  # noqa: E402

BILDRATE = 30
FENSTER_S = 3.0           # sichtbarer Ausschnitt der Wellenform
WELLE_SPALTEN = 900
MAX_MARKER = 24
# Ein Tastendruck macht zwei Geraeusche: das Anschlagen und das Loslassen,
# je nach Haltedauer 80 bis 250 ms auseinander. Ohne ausreichende Sperrzeit
# zaehlt das Loslassen als eigener Anschlag - genau das fuehrt zu doppelten
# Buchstaben. Der Preis: fluessiges Tippen wird nicht mehr aufgeloest.
MIN_ABSTAND_MS = 300.0
MAX_ZEICHEN = 16          # so viele passen in eine Zeile

# Mindestpegel eines Anschlags. -35 dBFS haelt Raumrauschen (um -42) sicher
# draussen. Nur wenn die eigenen Trainingsaufnahmen leiser waren, geht die
# Vorgabe tiefer - aber nie unter SCHWELLE_BODEN_DBFS.
SCHWELLE_DBFS = -35.0
SCHWELLE_BODEN_DBFS = -40.0

# Ergebnisfeld in Leinwandpixeln. Die Balkenliste darueber endet davor -
# beide haengen an denselben Zahlen, damit sie sich nie ueberlagern.
LISTE_OBEN = 706
ERGEBNIS_OBEN, ERGEBNIS_H = 1220, 110


def lade_modell():
    """Den zuletzt trainierten Stand laden - samt der Klassen, die er kennt.

    Die Klassenliste kommt aus der Modelldatei, nicht aus der config.json. Wer
    zwischendurch andere Klassen eingestellt hat, bekommt hier sonst entweder
    einen Formfehler oder - schlimmer - falsch beschriftete Vorhersagen.
    """
    dateien = sorted((DATEN / "modelle").glob("cnn_*.pt"))
    if not dateien:
        return None, None
    stand = torch.load(dateien[-1], map_location="cpu", weights_only=False)
    klassen = stand.get("klassen")
    if klassen:
        setze_klassen(klassen)
    netz = modell.KleinesCNN(len(TASTEN))
    netz.load_state_dict(stand["state_dict"])
    netz.eval()
    return netz, stand


def merkmale(stand: dict, cfg: Config) -> tuple[int, dict]:
    """Abtastrate und Log-Mel-Parameter, mit denen das Modell trainiert wurde.

    Das Netz akzeptiert jede Bildbreite - rechnet die Demo mit anderer Rate
    oder anderen Mel-Baendern, kommt kein Fehler, nur eine schlechtere
    Trefferquote. Deshalb gilt, was in der Modelldatei steht. Aeltere Modelle
    haben diese Angaben noch nicht, dann gilt die uebergebene Konfiguration.
    """
    sr = int(stand.get("samplerate") or cfg.samplerate)
    mel = {"nfft": cfg.mel_nfft, "hop": cfg.mel_hop,
           "baender": stand.get("mel_baender", cfg.mel_baender),
           "fmin": cfg.mel_fmin, "fmax": cfg.mel_fmax}
    mel.update(stand.get("mel") or {})
    for schluessel in ("nfft", "hop", "baender"):
        mel[schluessel] = int(mel[schluessel])
    return sr, mel


class Demo:
    def __init__(self, cfg: Config, netz, stand: dict, soll: str, buehne: bool,
                 mitschnitt: bool = False, schwelle_dbfs: float = SCHWELLE_DBFS,
                 merkmal: tuple[int, dict] | None = None, warnung: str = ""):
        self.cfg = cfg
        self.netz = netz
        self.sr_modell, self.mel = merkmal or merkmale(stand, cfg)
        # Segmentlaenge so, wie das Modell sie kennt. Laeuft das Mikrofon mit
        # einer anderen Rate, wird entsprechend mehr oder weniger geschnitten
        # und danach auf die Trainingsrate umgerechnet.
        self.segment_n_modell = int(stand["segment_ms"] / 1000 * self.sr_modell)
        self.umrechnen = self.sr_modell != cfg.samplerate
        self.segment_n = math.ceil(self.segment_n_modell * cfg.samplerate
                                   / self.sr_modell)
        self.vor_n = int(stand["vor_ms"] / 1000 * cfg.samplerate)
        self.warnung = warnung
        self.soll = soll.lower()
        self.buehne = buehne
        self.mitschnitt = mitschnitt
        self.mitschnitt_nr = 0
        # Je Lauf ein eigener Ordner: Sonst ueberschreibt ein zweiter Lauf die
        # gleich nummerierten Dateien des ersten, und die uebrigen bleiben
        # dazwischen liegen.
        self.mitschnitt_ordner = (DATEN / "demo_mitschnitt"
                                  / time.strftime("%Y%m%d-%H%M%S"))
        self.schwelle_dbfs = schwelle_dbfs
        self.erkannt_gesamt = 0

        self.ring = audio.Ringpuffer(cfg)
        # Alle vorhergesagten Zeichen seit dem letzten Zuruecksetzen. Angezeigt
        # werden nur die letzten MAX_ZEICHEN, verglichen wird mit allen - sonst
        # rutscht der Vergleich mit dem Zieltext nach dem 16. Zeichen weg.
        self.gefunden: list[str] = []
        self.letzte_p = {t: 0.0 for t in TASTEN}
        self.letzte_taste = ""
        self.verarbeitet_bis = 0               # absoluter Sample-Index
        self.marker_abs: list[int] = []
        self._laufen = True
        self._naechstes = 0.0
        self.stufe = 0.0
        self._baue_figur()

    # -- Aufbau ---------------------------------------------------------
    def _baue_figur(self) -> None:
        theme.anwenden("hochformat")
        bedienung.tastenkuerzel_aus()
        self.masse = bedienung.LiveMasse()
        self.fig = plt.figure(
            figsize=(portrait.BREITE / portrait.DPI, portrait.HOEHE / portrait.DPI),
            dpi=self.masse.dpi, facecolor=theme.BG)
        self.fig.canvas.manager.set_window_title("Live-Demo - Tastenakustik")
        P = portrait

        self.t = np.linspace(-FENSTER_S, 0.0, int(FENSTER_S * self.cfg.samplerate))
        self.kopf = P.kopf(self.fig, "Blindtest")

        # Wellenform
        self.ax_welle = P.achse(self.fig, oben=396, hoehe=250,
                                links=P.INHALT_LINKS, breite=P.INHALT_BREITE)
        self.welle = Polygon(np.zeros((2, 2)), closed=True,
                             facecolor=theme.AKZENT, edgecolor="none")
        self.ax_welle.add_patch(self.welle)
        self.ax_welle.set_xlim(-FENSTER_S, 0)
        self.ax_welle.set_ylim(-0.3, 0.3)
        P.achse_aufraeumen(self.ax_welle)
        self.marker = [self.ax_welle.axvline(0, color=theme.AKZENT2, lw=2.4,
                                             zorder=5, visible=False)
                       for _ in range(MAX_MARKER)]

        # Wahrscheinlichkeiten - Raster richtet sich nach der Klassenzahl,
        # damit auch ein ganzes Alphabet auf die Leinwand passt.
        self.balken, self.prozent, self.kacheln = [], [], []
        n = len(TASTEN)
        spalten = 1 if n <= 10 else 2
        zeilen = -(-n // spalten)
        # Die Liste endet mit etwas Luft ueber dem Ergebnisfeld. 18 px sind so
        # gewaehlt, dass acht Klassen genau wie bisher 62 px je Zeile haben.
        oben = LISTE_OBEN
        platz = ERGEBNIS_OBEN - oben - 18
        zeile = min(62.0, platz / zeilen)
        hoehe = zeile * 0.74
        spalten_luecke = 40
        spalte_b = (P.INHALT_BREITE - spalten_luecke * (spalten - 1)) / spalten
        kachel_b = min(46.0, hoehe, spalte_b * 0.14)
        zahl_b = 150 if spalten == 1 else 92
        # Prozentzahl nie hoeher als ihre Zeile (Punkt = 1,39 px), sonst
        # beruehren sich die Zahlen benachbarter Zeilen.
        zahl_gr = min(26 if spalten == 1 else 19, int(zeile * 0.9 / 1.39))
        self.balken_b = spalte_b - kachel_b - 20 - zahl_b
        for i, taste in enumerate(TASTEN):
            spalte, reihe = divmod(i, zeilen)      # spaltenweise fuellen
            links = P.INHALT_LINKS + spalte * (spalte_b + spalten_luecke)
            balken_x = links + kachel_b + 20
            y0 = oben + reihe * zeile
            farbe = theme.farbe(taste)
            self.fig.add_artist(FancyBboxPatch(
                (P.x(links), P.y(y0 + hoehe)), kachel_b / P.BREITE,
                hoehe / P.HOEHE, boxstyle="round,pad=0,rounding_size=0.01",
                transform=self.fig.transFigure, facecolor=farbe, edgecolor="none"))
            self.fig.text(P.x(links + kachel_b / 2), P.y(y0 + hoehe / 2),
                          anzeige(taste), ha="center", va="center",
                          fontsize=max(int(kachel_b * 0.56), 11),
                          fontweight="bold", color=theme.BG)
            self.fig.add_artist(FancyBboxPatch(
                (P.x(balken_x), P.y(y0 + hoehe)), self.balken_b / P.BREITE,
                hoehe / P.HOEHE, boxstyle="round,pad=0,rounding_size=0.006",
                transform=self.fig.transFigure, facecolor=theme.PANEL, edgecolor="none"))
            stueck = FancyBboxPatch(
                (P.x(balken_x), P.y(y0 + hoehe)), 0.0001, hoehe / P.HOEHE,
                boxstyle="round,pad=0,rounding_size=0.006",
                transform=self.fig.transFigure, facecolor=farbe, edgecolor="none",
                zorder=3)
            self.fig.add_artist(stueck)
            self.balken.append(stueck)
            self.prozent.append(self.fig.text(
                P.x(links + spalte_b), P.y(y0 + hoehe / 2), "", fontsize=zahl_gr,
                family="monospace", color=theme.TEXT_SCHWACH, ha="right",
                va="center", zorder=3))

        # Ergebnis
        self.ergebnis_feld = FancyBboxPatch(
            (P.x(P.INHALT_LINKS), P.y(ERGEBNIS_OBEN + ERGEBNIS_H)),
            P.INHALT_BREITE / P.BREITE, ERGEBNIS_H / P.HOEHE,
            boxstyle="round,pad=0,rounding_size=0.012",
            transform=self.fig.transFigure, facecolor=theme.PANEL, edgecolor="none")
        self.fig.add_artist(self.ergebnis_feld)
        self.ergebnis_text = self.fig.text(
            P.x(P.INHALT_LINKS + P.INHALT_BREITE / 2),
            P.y(ERGEBNIS_OBEN + ERGEBNIS_H / 2), "", ha="center",
            va="center", fontsize=52, fontweight="bold", color=theme.TEXT_SCHWACH,
            zorder=3)

        # Aufgebaute Zeichenfolge - beide Zeilen oberhalb von 1480 px, wo die
        # Plattformen Bildunterschrift und Fortschrittsbalken einblenden.
        self.zeichen_text = self.fig.text(
            P.x(P.INHALT_LINKS), P.y(1400), "", fontsize=46, family="monospace",
            fontweight="bold", color=theme.TEXT, va="center")
        self.treffer_text = self.fig.text(
            P.x(P.INHALT_LINKS), P.y(1455), "", fontsize=P.S_TICK,
            color=theme.TEXT_SCHWACH, va="center")

        self.chrome = [self.kopf]
        # Bedient wird nur mit der Maus. Beim Live-Test soll jede Taste frei
        # sein - auch s, q, f, l und k, die Matplotlib sonst selbst belegt.
        self.bedienung = bedienung.Bedienung(self.fig, [
            ("Zurücksetzen", self._zuruecksetzen),
            ("Bühne", self._buehne_umschalten),
            ("Standbild", self._standbild),
            ("Beenden", self._beenden),
        ], breite=round(portrait.BREITE * self.masse.skala),
            bei_klick=self._klick_ausblenden)
        self._setze_buehne()
        self.fig.canvas.mpl_connect("close_event", lambda _e: self.ring.stop())

    def _setze_buehne(self) -> None:
        for a in self.chrome:
            if a is not None:
                a.set_visible(not self.buehne)
        self.bedienung.zeigen(not self.buehne)
        self.fig.canvas.draw_idle()

    # -- Erkennung ------------------------------------------------------
    def _neue_anschlaege(self) -> None:
        """Im Audiostrom nach Anschlaegen suchen und klassifizieren."""
        gesamt = self.ring.gesamt
        blick = min(int(1.5 * self.cfg.samplerate), gesamt)
        if blick < self.segment_n * 2:
            return
        x = self.ring.letzte(blick / self.cfg.samplerate)
        basis = gesamt - x.size
        spitzen, _rausch = onset.finde_transienten(x, self.cfg, MIN_ABSTAND_MS)
        # Nur Spitzen, die laut genug fuer einen echten Anschlag sind. Ohne
        # diese Pruefung loest jedes Rascheln im Raum aus: der Detektor misst
        # relativ zum Rauschboden, und 12 dB darueber ist in einer stillen
        # Sekunde schnell erreicht. Echte Anschlaege liegen bei -15 bis -28,
        # Raumrauschen bei -42 und tiefer.
        umkreis = int(0.02 * self.cfg.samplerate)
        laut = []
        for i in spitzen:
            a, b = max(0, int(i) - umkreis), min(x.size, int(i) + umkreis)
            if b <= a:
                continue
            db = 20.0 * np.log10(max(float(np.max(np.abs(x[a:b]))), 1e-9))
            if db >= self.schwelle_dbfs:
                laut.append(int(i))
        spitzen = np.array(laut, dtype=int)

        # Die Suche sieht nur die letzten 1,5 s, die Welle zeigt aber 3 s.
        # Marker werden deshalb fortgeschrieben und erst entfernt, wenn sie
        # links aus dem Bild laufen. Dieselbe Spitze verrutscht zwischen zwei
        # Ausschnitten um ein paar Samples - innerhalb der Sperrzeit gilt sie
        # als schon markiert.
        mindest = int(MIN_ABSTAND_MS / 1000 * self.cfg.samplerate)
        for i in spitzen:
            m = basis + int(i)
            if not self.marker_abs or m > self.marker_abs[-1] + mindest:
                self.marker_abs.append(m)
        sichtbar_ab = gesamt - int(FENSTER_S * self.cfg.samplerate)
        self.marker_abs = [m for m in self.marker_abs if m >= sichtbar_ab][-MAX_MARKER:]

        vorlauf = int(self.cfg.pre_roll_ms / 1000 * self.cfg.samplerate)
        nachlauf = int(self.cfg.post_roll_ms / 1000 * self.cfg.samplerate)
        for i in spitzen:
            spitze_abs = basis + int(i)
            if spitze_abs <= self.verarbeitet_bis + mindest:
                continue
            if spitze_abs + nachlauf > gesamt:
                continue
            # Erst dasselbe Kontextfenster holen wie beim Aufnehmen und den
            # Onset genauso nachmessen. Sonst liegt der Schnitt ein paar
            # Millisekunden anders als beim Training - und das kostet
            # messbar Trefferquote.
            fenster = self.ring.fenster_absolut(spitze_abs - vorlauf,
                                                vorlauf + nachlauf)
            if fenster is None:
                continue
            # Dieselbe Analyse wie im Collector. Liegt ein zweiter Anschlag im
            # Fenster und stellt dort das Maximum, wird lokal ab der
            # erkannten Spitze zurueckverfolgt (siehe onset.py).
            onset_lokal = onset.onset_wie_beim_aufnehmen(fenster, self.cfg, vorlauf)
            start = max(0, min(onset_lokal - self.vor_n,
                               fenster.size - self.segment_n))
            segment = fenster[start:start + self.segment_n]
            if segment.size < self.segment_n:
                continue
            self.verarbeitet_bis = spitze_abs
            self.erkannt_gesamt += 1
            self._klassifiziere(segment)

    def auf_modellrate(self, segment: np.ndarray) -> np.ndarray:
        """Segment auf die Abtastrate des Trainings bringen, exakt passend lang.

        Nur noetig, wenn das Mikrofon die Trainingsrate nicht liefert. Ohne
        Umrechnung saehe das Netz ein gestauchtes oder gedehntes Spektrogramm
        und verloere spuerbar an Trefferquote.
        """
        if not self.umrechnen:
            return segment
        g = math.gcd(self.sr_modell, self.cfg.samplerate)
        neu = resample_poly(segment.astype(np.float64), self.sr_modell // g,
                            self.cfg.samplerate // g)[:self.segment_n_modell]
        if neu.size < self.segment_n_modell:
            neu = np.pad(neu, (0, self.segment_n_modell - neu.size))
        return neu.astype(np.float32)

    def _klassifiziere(self, segment: np.ndarray) -> None:
        segment = self.auf_modellrate(segment)
        m = self.mel
        mel = features.log_mel(segment, self.sr_modell, m["nfft"], m["hop"],
                               m["baender"], m["fmin"], m["fmax"]).T
        x = datensatz.normiere(mel[None].astype(np.float32))
        with torch.no_grad():
            p = torch.softmax(self.netz(torch.from_numpy(x).unsqueeze(1))[0], 0).numpy()
        self.letzte_p = {t: float(v) for t, v in zip(TASTEN, p)}
        self.letzte_taste = TASTEN[int(p.argmax())]
        self.gefunden.append(self.letzte_taste)
        if self.mitschnitt:
            import soundfile as sf
            ziel = self.mitschnitt_ordner
            ziel.mkdir(parents=True, exist_ok=True)
            self.mitschnitt_nr += 1
            sf.write(ziel / f"{self.mitschnitt_nr:03d}_{self.letzte_taste_datei()}.wav",
                     segment, self.sr_modell, subtype="FLOAT")

    def letzte_taste_datei(self) -> str:
        from tastenakustik.config import datei_token
        return datei_token(self.letzte_taste)

    # -- Anzeige --------------------------------------------------------
    def _aktualisiere(self) -> None:
        x = self.ring.letzte(FENSTER_S)
        n = self.t.size
        if x.size < n:
            x = np.concatenate([np.zeros(n - x.size, dtype=np.float32), x])
        idx, werte = features.min_max_huellkurve(x, WELLE_SPALTEN)
        zeit = self.t[np.minimum(idx.astype(int), n - 1)]
        unten, oben = werte[0::2], werte[1::2]
        tc = zeit[0::2]
        self.welle.set_xy(np.concatenate([
            np.column_stack([tc, oben]), np.column_stack([tc[::-1], unten[::-1]])]))

        spitze = float(np.max(np.abs(werte)))
        stufe = next((s for s in (0.05, 0.1, 0.2, 0.35, 0.6, 1.05)
                      if spitze * 1.3 <= s), 1.05)
        if stufe != self.stufe:
            self.stufe = stufe
            self.ax_welle.set_ylim(-stufe, stufe)

        basis = self.ring.gesamt - n
        for k, linie in enumerate(self.marker):
            rel = self.marker_abs[k] - basis if k < len(self.marker_abs) else -1
            if 0 <= rel < n:
                linie.set_xdata([self.t[rel], self.t[rel]])
                linie.set_visible(True)
            else:
                linie.set_visible(False)

        for i, taste in enumerate(TASTEN):
            wert = self.letzte_p.get(taste, 0.0)
            fuehrend = taste == self.letzte_taste
            self.balken[i].set_width(max(self.balken_b * wert, 0.5) / portrait.BREITE)
            self.prozent[i].set_text(f"{wert * 100:.0f}%" if wert >= 0.005 else "")
            self.prozent[i].set_color(theme.TEXT if fuehrend else theme.TEXT_SCHWACH)
            self.prozent[i].set_fontweight("bold" if fuehrend else "normal")

        if self.letzte_taste:
            farbe = theme.farbe(self.letzte_taste)
            self.ergebnis_feld.set_facecolor(farbe)
            self.ergebnis_text.set_text(
                f"{anzeige(self.letzte_taste)}   "
                f"{self.letzte_p[self.letzte_taste] * 100:.0f}%")
            self.ergebnis_text.set_color(theme.BG)

        text = "".join(anzeige(t) for t in self.gefunden[-MAX_ZEICHEN:])
        self.zeichen_text.set_text(text)
        if self.soll:
            n_treffer, n_verglichen = self.treffer()
            self.treffer_text.set_text(
                f"{n_treffer} von {n_verglichen} richtig   "
                f"(Ziel: {''.join(anzeige(z) for z in self.soll)})")
        else:
            self.treffer_text.set_text("")

    def treffer(self) -> tuple[int, int]:
        """(richtig, verglichen) - Zeichen fuer Zeichen gegen den Zieltext.

        Verglichen wird nur, wofuer es ein Zielzeichen gibt: Wer ueber das
        Zielwort hinaus weitertippt, verschlechtert damit nicht die Quote.
        """
        n_treffer = sum(1 for a, b in zip(self.gefunden, self.soll) if a == b)
        return n_treffer, min(len(self.gefunden), len(self.soll))

    # -- Bedienung (nur Maus) -------------------------------------------
    def _klick_ausblenden(self) -> None:
        """Das Klickgeraeusch der Maus nicht als Anschlag werten.

        Laeuft vor jedem Knopf, jedem Menuebefehl und beim Oeffnen des
        Rechtsklick-Menues. Alles bis jetzt gilt als verarbeitet, und die
        Sperrzeit deckt das Geraeusch ab, das gerade erst im Puffer ankommt.
        """
        self.verarbeitet_bis = max(self.verarbeitet_bis, self.ring.gesamt)

    def _zuruecksetzen(self) -> None:
        self.gefunden.clear()
        self.letzte_taste = ""
        self.letzte_p = {t: 0.0 for t in TASTEN}
        self.verarbeitet_bis = self.ring.gesamt
        self.ergebnis_feld.set_facecolor(theme.PANEL)
        self.ergebnis_text.set_text("")

    def _buehne_umschalten(self) -> None:
        self.buehne = not self.buehne
        self._setze_buehne()

    def _standbild(self) -> None:
        pfad = portrait.exportiere(self.fig, "demo", "05_demo")
        print(f"gespeichert: {pfad}")
        self.bedienung.melde("Standbild gespeichert")

    def _beenden(self) -> None:
        self._laufen = False
        plt.close(self.fig)

    def _takt(self) -> None:
        if not self._laufen:
            return
        try:
            self._neue_anschlaege()
            self._aktualisiere()
            self.fig.canvas.draw_idle()
        except Exception as exc:  # noqa: BLE001
            print("Fehler im Takt:", exc)
        takt = 1.0 / BILDRATE
        jetzt = time.perf_counter()
        self._naechstes += takt
        if self._naechstes < jetzt:
            self._naechstes = jetzt + takt
        try:
            self.fig.canvas.get_tk_widget().after(
                max(int((self._naechstes - jetzt) * 1000), 1), self._takt)
        except Exception:  # noqa: BLE001
            self._laufen = False

    def starten(self) -> None:
        self.ring.start()
        self.masse.platzieren(self.fig.canvas.manager.window,
                              self.bedienung.hoehe())
        self.verarbeitet_bis = 0
        if self.warnung:
            self.bedienung.melde(self.warnung, theme.WARN, dauer_ms=20000)
        self.fig.canvas.draw()
        self._naechstes = time.perf_counter()
        self._takt()
        try:
            plt.show()
        finally:
            self._laufen = False
            self.ring.stop()
            if self.gefunden:
                print("\nErkannt:", "".join(anzeige(t) for t in self.gefunden))
                if self.soll:
                    n, verglichen = self.treffer()
                    print(f"Ziel:    {''.join(anzeige(z) for z in self.soll)}")
                    print(f"Treffer: {n} von {verglichen}")


def rate_moeglich(cfg: Config, sr: int) -> bool:
    """Nimmt der Eingang diese Abtastrate an? Es wird nur gefragt, nichts geoeffnet.

    Das Geraet wird genau so aufgeloest wie beim Oeffnen des Ringpuffers
    (Name und Host-API, bei gleichem Namen der gespeicherte Index) - sonst
    koennte hier ein anderes Geraet gefragt werden als spaeter geoeffnet.
    Laesst es sich nicht aufloesen, gilt der gespeicherte Index; die klare
    Fehlermeldung kommt dann beim Start.
    """
    try:
        index = audio.geraet_aufloesen(cfg)
    except Exception:  # noqa: BLE001 - dann eben mit dem Index
        index = cfg.device
    for kanaele in dict.fromkeys([cfg.channels, 2]):
        try:
            audio.sd.check_input_settings(device=index, samplerate=sr,
                                          channels=kanaele, dtype="float32")
        except Exception:  # noqa: BLE001, PERF203
            continue
        return True
    return False


def abtastrate_abstimmen(cfg: Config, sr_modell: int) -> str:
    """Mikrofon auf die Trainingsrate bringen - oder deutlich warnen.

    Gibt einen kurzen Warntext fuer das Fenster zurueck, leer wenn alles
    passt. Laesst sich die Rate nicht einstellen, rechnet die Demo jedes
    Segment um: besser als ein still verzerrtes Spektrogramm, aber nicht
    ganz so gut wie gleiche Raten.
    """
    if cfg.samplerate == sr_modell:
        return ""
    if rate_moeglich(cfg, sr_modell):
        print(f"Abtastrate {cfg.samplerate} Hz -> {sr_modell} Hz umgestellt, "
              f"so wurde das Modell trainiert.")
        cfg.samplerate = sr_modell
        return ""
    print("!" * 70)
    print(f"ACHTUNG: Das Modell wurde mit {sr_modell} Hz trainiert, das Mikrofon")
    print(f"liefert {cfg.samplerate} Hz und laesst sich nicht umstellen.")
    print("Jedes Segment wird umgerechnet - die Trefferquote kann trotzdem")
    print(f"sinken. Besser: Eingang in den Windows-Soundeinstellungen auf "
          f"{sr_modell} Hz stellen")
    print("oder mit diesem Mikrofon neu aufnehmen und trainieren.")
    print("!" * 70)
    return f"Achtung: Modell {sr_modell} Hz, Mikrofon {cfg.samplerate} Hz"


def pegel_im_training(klassen: list[str]) -> list[float]:
    """Spitzenpegel aller Trainingsanschlaege mit denselben Klassen, in dBFS.

    Gelesen werden nur die Metadaten der Sitzungen, keine Audiodateien.
    Geht dabei etwas schief, bleibt die Liste leer - dann gilt die Vorgabe.
    """
    pegel: list[float] = []
    try:
        for ordner in storage.sitzungen():
            kopf, proben = storage.lade_sitzung(ordner)
            if kopf.get("rolle") != "train":
                continue
            if kopf.get("tasten") and list(kopf["tasten"]) != list(klassen):
                continue
            pegel += [float(p["peak_dbfs"]) for p in proben if "peak_dbfs" in p]
    except Exception:  # noqa: BLE001
        return []
    return pegel


def schwelle_waehlen(pegel: list[float]) -> float:
    """Mindestpegel der Demo, passend zu den eigenen Trainingsaufnahmen.

    Wer leise aufgenommen hat, trainiert ein brauchbares Modell - und die
    Demo wuerde mit -35 dBFS trotzdem fast nichts hoeren. Liegt das
    5. Perzentil der Trainingsspitzen weniger als 6 dB ueber der Vorgabe,
    geht die Schwelle entsprechend tiefer, aber nie unter -40 dBFS, wo
    Raumrauschen wieder ausloesen wuerde. Laute Aufnahmen aendern nichts.
    """
    if not pegel:
        return SCHWELLE_DBFS
    p5 = float(np.percentile(pegel, 5))
    return max(min(SCHWELLE_DBFS, p5 - 6.0), SCHWELLE_BODEN_DBFS)


def main() -> int:
    global MIN_ABSTAND_MS
    p = argparse.ArgumentParser(description="Live-Demo, nur aus dem Audiosignal")
    p.add_argument("--soll", default="", help="Vergleichstext, nur fuer die Anzeige")
    p.add_argument("--buehne", action="store_true",
                   help="ohne Titel und Knopfleiste starten (zum Filmen)")
    p.add_argument("--geraet", default=None,
                   help="anderes Eingangsgeraet als in config.json (Index oder Name)")
    p.add_argument("--abstand", type=float, default=MIN_ABSTAND_MS,
                   help="Sperrzeit nach einem Anschlag in ms")
    p.add_argument("--mitschnitt", action="store_true",
                   help="erkannte Segmente zur Fehlersuche mitschreiben")
    p.add_argument("--schwelle", type=float, default=None,
                   help=f"Mindestpegel eines Anschlags in dBFS (Vorgabe "
                        f"{SCHWELLE_DBFS:.0f}, tiefer bei leisen Trainingsaufnahmen)")
    args = p.parse_args()

    cfg = laden_oder_beenden()
    # Stand der config.json, bevor --geraet etwas aendert: Aeltere Modelle
    # kennen ihre Abtastrate nicht, dann gilt die dort eingestellte.
    cfg_datei = replace(cfg)
    if args.geraet is not None:
        g = audio.geraet_finden(args.geraet)
        if g is None:
            print(f"Geraet {args.geraet!r} nicht gefunden.")
            return 1
        cfg.device, cfg.device_name, cfg.hostapi = g.index, g.name, g.hostapi
        cfg.samplerate, cfg.channels = audio.bestes_format(g, cfg.samplerate)
    if cfg.device is None:
        print("Keine Konfiguration. Bitte zuerst: python werkzeuge/01_systemcheck.py")
        return 1

    netz, stand = lade_modell()
    if netz is None:
        print("Kein trainiertes Modell gefunden. Bitte zuerst: "
              "python werkzeuge/07_training.py")
        return 1

    ungueltig = [c for c in args.soll.lower() if c not in TASTEN]
    if ungueltig:
        print(f"Der Vergleichstext enthaelt Zeichen ausserhalb der Klassen: "
              f"{sorted(set(ungueltig))}")
        return 1

    merkmal = merkmale(stand, cfg_datei)
    parameter = f"{netz.parameterzahl:,}".replace(",", " ")
    print(f"Modell   {stand['segment_ms']:.0f} ms Segment, {merkmal[0]} Hz, "
          f"{parameter} Parameter")
    warnung = abtastrate_abstimmen(cfg, merkmal[0])
    print(f"Eingang  {cfg.device_name} ({cfg.samplerate} Hz)")
    print(f"Klassen  {'  '.join(anzeige(t) for t in TASTEN)}")
    print("\nEs werden keine Tastatur-Ereignisse gelesen. Anschlaege werden")
    print("ausschliesslich im Audiosignal gefunden.\n")
    MIN_ABSTAND_MS = args.abstand
    schwelle = args.schwelle
    if schwelle is None:
        pegel = pegel_im_training(TASTEN)
        schwelle = schwelle_waehlen(pegel)
        if schwelle < SCHWELLE_DBFS:
            print(f"Die Trainingsaufnahmen sind leise - Mindestpegel auf "
                  f"{schwelle:.0f} statt {SCHWELLE_DBFS:.0f} dBFS gesenkt.")
        unter = float(np.mean(np.array(pegel) < schwelle)) if pegel else 0.0
        if unter >= 0.02:
            print(f"Hinweis: {unter * 100:.0f} % der Trainingsanschlaege liegen unter "
                  f"{schwelle:.0f} dBFS und werden hier ueberhoert - mehr Gain oder")
            print("naeher ans Mikrofon, notfalls --schwelle tiefer setzen.")
    print(f"Sperrzeit nach einem Anschlag: {MIN_ABSTAND_MS:.0f} ms")
    print(f"Mindestpegel eines Anschlags:  {schwelle:.0f} dBFS")
    print()
    demo = Demo(cfg, netz, stand, args.soll, args.buehne, args.mitschnitt,
                schwelle, merkmal=merkmal, warnung=warnung)
    try:
        demo.starten()
    except RuntimeError as fehler:
        # Meist ist das Mikrofon belegt oder abgezogen - dann klar sagen,
        # statt mit einem Traceback und einem leeren Fenster zu enden.
        plt.close(demo.fig)
        print(fehler)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

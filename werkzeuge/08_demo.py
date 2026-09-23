"""Schritt 8 - Live-Demo: aus dem Klang allein die Zeichenfolge aufbauen.

Es laufen KEINE Tastatur-Ereignisse in dieses Programm. Nichts davon wird
gelesen, nichts davon wird ausgewertet. Die Anschlaege werden ausschliesslich
im Audiosignal gefunden - mit demselben Transientendetektor, der auch beim
Aufnehmen die Onsets nachmisst.

Damit ist die Trennung nicht nur eine Absichtserklaerung, sondern eine
Eigenschaft des Programms: Es gibt hier gar keine Stelle, an der die
tatsaechlich gedrueckte Taste bekannt waere.

Der optionale Vergleichstext (--soll) dient nur der Anzeige, wie viele Zeichen
getroffen wurden. Er beeinflusst die Vorhersage an keiner Stelle - er wird
erst nach der Klassifikation herangezogen.

Aufruf:
    python werkzeuge/08_demo.py
    python werkzeuge/08_demo.py --soll hallo
    python werkzeuge/08_demo.py --buehne          # ohne Bedienhinweise

Tasten im Fenster:
    r  Zeichenfolge zuruecksetzen
    b  Buehne an/aus
    s  Standbild nach ausgabe/
    q  beenden
"""

from __future__ import annotations

import argparse
import sys
import time
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

from tastenakustik import audio, datensatz, features, modell, onset, theme  # noqa: E402
from tastenakustik.config import DATEN, TASTEN, Config, anzeige, setze_klassen  # noqa: E402

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


class Demo:
    def __init__(self, cfg: Config, netz, stand: dict, soll: str, buehne: bool,
                 mitschnitt: bool = False, schwelle_dbfs: float = -35.0):
        self.cfg = cfg
        self.netz = netz
        self.segment_n = int(stand["segment_ms"] / 1000 * cfg.samplerate)
        self.vor_n = int(stand["vor_ms"] / 1000 * cfg.samplerate)
        self.soll = soll.lower()
        self.buehne = buehne
        self.mitschnitt = mitschnitt
        self.mitschnitt_nr = 0
        self.schwelle_dbfs = schwelle_dbfs
        self.erkannt_gesamt = 0

        self.ring = audio.Ringpuffer(cfg)
        self.gefunden: list[str] = []          # vorhergesagte Zeichen
        self.letzte_p = {t: 0.0 for t in TASTEN}
        self.letzte_taste = ""
        self.verarbeitet_bis = 0               # absoluter Sample-Index
        self.marker_abs: list[int] = []
        self._hintergrund = None
        self._laufen = True
        self._naechstes = 0.0
        self.stufe = 0.0
        self._baue_figur()

    # -- Aufbau ---------------------------------------------------------
    def _baue_figur(self) -> None:
        theme.anwenden("hochformat")
        self.fig = plt.figure(
            figsize=(portrait.BREITE / portrait.DPI, portrait.HOEHE / portrait.DPI),
            dpi=portrait.DPI, facecolor=theme.BG)
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
        oben, platz = 706, 600
        zeile = min(62.0, platz / zeilen)
        hoehe = zeile * 0.74
        spalten_luecke = 40
        spalte_b = (P.INHALT_BREITE - spalten_luecke * (spalten - 1)) / spalten
        kachel_b = min(46.0, hoehe, spalte_b * 0.14)
        zahl_b = 150 if spalten == 1 else 92
        zahl_gr = 26 if spalten == 1 else 19
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
            (P.x(P.INHALT_LINKS), P.y(1330)), P.INHALT_BREITE / P.BREITE,
            110 / P.HOEHE, boxstyle="round,pad=0,rounding_size=0.012",
            transform=self.fig.transFigure, facecolor=theme.PANEL, edgecolor="none")
        self.fig.add_artist(self.ergebnis_feld)
        self.ergebnis_text = self.fig.text(
            P.x(P.INHALT_LINKS + P.INHALT_BREITE / 2), P.y(1275), "", ha="center",
            va="center", fontsize=52, fontweight="bold", color=theme.TEXT_SCHWACH,
            zorder=3)

        # Aufgebaute Zeichenfolge
        self.zeichen_text = self.fig.text(
            P.x(P.INHALT_LINKS), P.y(1430), "", fontsize=46, family="monospace",
            fontweight="bold", color=theme.TEXT, va="center")
        self.treffer_text = self.fig.text(
            P.x(P.INHALT_LINKS), P.y(1500), "", fontsize=P.S_TICK,
            color=theme.TEXT_SCHWACH, va="center")

        self.chrome = [self.kopf, portrait.fuss(
            self.fig, "r  zuruecksetzen      b  Buehne      s  Standbild      q  beenden")]
        self._setze_buehne()
        self.fig.canvas.mpl_connect("key_press_event", self._taste)
        self.fig.canvas.mpl_connect("close_event", lambda _e: self.ring.stop())
        self.fig.canvas.mpl_connect(
            "draw_event", lambda _e: setattr(self, "_hintergrund", None))

    def _setze_buehne(self) -> None:
        for a in self.chrome:
            if a is not None:
                a.set_visible(not self.buehne)
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
        self.marker_abs = [basis + int(i) for i in spitzen[-MAX_MARKER:]]

        mindest = int(MIN_ABSTAND_MS / 1000 * self.cfg.samplerate)
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
            # Onset lokal ab der erkannten Spitze zurueckverfolgen. Ueber das
            # ganze Fenster zu suchen ginge schief, sobald ein zweiter
            # Anschlag darin liegt - dann landet man beim falschen.
            onset_lokal = onset.onset_zu_spitze(fenster, self.cfg, vorlauf)
            start = max(0, min(onset_lokal - self.vor_n,
                               fenster.size - self.segment_n))
            segment = fenster[start:start + self.segment_n]
            if segment.size < self.segment_n:
                continue
            self.verarbeitet_bis = spitze_abs
            self.erkannt_gesamt += 1
            self._klassifiziere(segment)

    def _klassifiziere(self, segment: np.ndarray) -> None:
        mel = features.log_mel(segment, self.cfg.samplerate, self.cfg.mel_nfft,
                               self.cfg.mel_hop, self.cfg.mel_baender,
                               self.cfg.mel_fmin, self.cfg.mel_fmax).T
        x = datensatz.normiere(mel[None].astype(np.float32))
        with torch.no_grad():
            p = torch.softmax(self.netz(torch.from_numpy(x).unsqueeze(1))[0], 0).numpy()
        self.letzte_p = {t: float(v) for t, v in zip(TASTEN, p)}
        self.letzte_taste = TASTEN[int(p.argmax())]
        self.gefunden.append(self.letzte_taste)
        self.gefunden = self.gefunden[-MAX_ZEICHEN:]
        if self.mitschnitt:
            import soundfile as sf
            ziel = DATEN / "demo_mitschnitt"
            ziel.mkdir(parents=True, exist_ok=True)
            self.mitschnitt_nr += 1
            sf.write(ziel / f"{self.mitschnitt_nr:03d}_{self.letzte_taste_datei()}.wav",
                     segment, self.cfg.samplerate, subtype="FLOAT")

    def letzte_taste_datei(self) -> str:
        from tastenakustik.config import datei_token
        return datei_token(self.letzte_taste)

    # -- Anzeige --------------------------------------------------------
    def _aktualisiere(self) -> bool:
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
        struktur = stufe != self.stufe
        if struktur:
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

        text = "".join(self.gefunden).upper()
        self.zeichen_text.set_text(text)
        if self.soll:
            n_treffer = sum(1 for a, b in zip(self.gefunden, self.soll) if a == b)
            self.treffer_text.set_text(
                f"{n_treffer} von {len(self.gefunden)} richtig   "
                f"(Ziel: {self.soll.upper()})")
        return struktur

    # -- Betrieb --------------------------------------------------------
    def _taste(self, event) -> None:  # noqa: ANN001
        if event.key == "r":
            self.gefunden.clear()
            self.letzte_taste = ""
            self.letzte_p = {t: 0.0 for t in TASTEN}
            self.verarbeitet_bis = self.ring.gesamt
        elif event.key == "b":
            self.buehne = not self.buehne
            self._setze_buehne()
        elif event.key == "s":
            print(f"gespeichert: {portrait.exportiere(self.fig, 'demo', '05_demo')}")
        elif event.key in ("q", "escape"):
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
        try:
            _b, _h, px, py, _ = portrait.fensterplatz(rand=90)
            self.fig.canvas.manager.window.wm_geometry(f"+{px}+{py}")
        except Exception:  # noqa: BLE001
            pass
        self.verarbeitet_bis = 0
        self.fig.canvas.draw()
        self._naechstes = time.perf_counter()
        self._takt()
        try:
            plt.show()
        finally:
            self._laufen = False
            self.ring.stop()
            if self.gefunden:
                print("\nErkannt:", "".join(self.gefunden).upper())
                if self.soll:
                    n = sum(1 for a, b in zip(self.gefunden, self.soll) if a == b)
                    print(f"Ziel:    {self.soll.upper()}")
                    print(f"Treffer: {n} von {len(self.gefunden)}")


def main() -> int:
    global MIN_ABSTAND_MS
    p = argparse.ArgumentParser(description="Live-Demo, nur aus dem Audiosignal")
    p.add_argument("--soll", default="", help="Vergleichstext, nur fuer die Anzeige")
    p.add_argument("--buehne", action="store_true")
    p.add_argument("--geraet", default=None)
    p.add_argument("--abstand", type=float, default=MIN_ABSTAND_MS,
                   help="Sperrzeit nach einem Anschlag in ms")
    p.add_argument("--mitschnitt", action="store_true",
                   help="erkannte Segmente zur Fehlersuche mitschreiben")
    p.add_argument("--schwelle", type=float, default=-35.0,
                   help="Mindestpegel eines Anschlags in dBFS")
    args = p.parse_args()

    cfg = Config.laden()
    if args.geraet is not None:
        g = audio.geraet_finden(args.geraet)
        if g is None:
            print(f"Geraet {args.geraet!r} nicht gefunden.")
            return 1
        cfg.device, cfg.device_name = g.index, g.name
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

    print(f"Modell   {stand['segment_ms']:.0f} ms Segment, "
          f"{netz.parameterzahl:,} Parameter".replace(",", " "))
    print(f"Eingang  {cfg.device_name} ({cfg.samplerate} Hz)")
    print(f"Klassen  {'  '.join(t.upper() for t in TASTEN)}")
    print("\nEs werden keine Tastatur-Ereignisse gelesen. Anschlaege werden")
    print("ausschliesslich im Audiosignal gefunden.\n")
    MIN_ABSTAND_MS = args.abstand
    print(f"Sperrzeit nach einem Anschlag: {MIN_ABSTAND_MS:.0f} ms")
    print(f"Mindestpegel eines Anschlags:  {args.schwelle:.0f} dBFS")
    print()
    Demo(cfg, netz, stand, args.soll, args.buehne, args.mitschnitt,
         args.schwelle).starten()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

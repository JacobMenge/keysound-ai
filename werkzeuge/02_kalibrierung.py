"""Schritt 2 - Mikrofon-Test und Live-Ansicht im Hochformat 9:16.

Zwei Ansichten, umschaltbar mit dem Knopf "Bühne":

  Werkzeug  Alles, was beim Einstellen hilft: Kennzahlen, Bewertung,
            Geraetename, Knopfleiste.
  Buehne    Fuer die Aufnahme: Wellenform, Spektrogramm und Pegel, jeweils
            mit Ueberschrift und beschrifteten Achsen. Sonst nichts.

Das Fenster ist 1080 x 1920 gross und legt sich von selbst auf einen
Hochformat-Monitor, falls einer vorhanden ist. Passt es nirgends in voller
Groesse hin, wird es als Ganzes verkleinert.

Aufruf:
    python werkzeuge/02_kalibrierung.py
    python werkzeuge/02_kalibrierung.py --buehne
    python werkzeuge/02_kalibrierung.py --geraet 39 --sekunden 3

Bedienung mit der Maus - Knoepfe unter dem Bild oder Rechtsklick ins Bild.
Die Tastatur bleibt frei, denn hier wird zum Pruefen getippt:
    Buehne       zwischen Werkzeug und Buehne umschalten
    Standbild    nach ausgabe/ speichern (exakt 1080 x 1920)
    Zonen        Sicherheitszonen fuer Shorts / Reels / TikTok einblenden
    Peak-Hold    Peak-Hold und Clip-Zaehler zuruecksetzen
    Skala        auf Vollaussteuerung umschalten
    Einfrieren   Bild anhalten und weiterlaufen lassen
    Beenden

Es wird nichts auf die Platte geschrieben ausser den Standbildern, die du
selbst ausloest.
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
from matplotlib import colormaps  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from tastenakustik import audio, bedienung, features, onset, theme  # noqa: E402
from tastenakustik.config import Config, TASTEN, verzeichnisse_anlegen  # noqa: E402

NFFT = 512
METER_MIN = -70.0

# Das Live-Bild wird auf Bildschirmbreite gerechnet, nicht auf Samplezahl:
# So kostet ein Bild gleich viel, egal wie lang das Fenster ist.
WELLE_SPALTEN = 900       # Stuetzstellen der Wellenform
SPEKTRO_SPALTEN = 300     # Spalten des Spektrogramms
MAX_MARKER = 24           # vorgehaltene Anschlagsmarkierungen
MARKER_TAKT = 3           # Anschlagssuche nur jedes n-te Bild
BILDRATE = 30             # Bilder je Sekunde, fest getaktet

# Feste Stufen fuer die Wellenform-Skala. Eine Skala, die sich bei jedem Bild
# neu einpasst, wirkt im Video nervoes - und bei Blitting muesste dafuer jedes
# Mal der Hintergrund neu aufgebaut werden.
SKALA_STUFEN = (0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.45, 0.70, 1.05)

# Platzierung beider Ansichten in Leinwandpixeln: (oben, hoehe, links, breite)
A_L, A_B = portrait.ACHSE_LINKS, portrait.ACHSE_BREITE
LAYOUT = {
    "werkzeug": {
        "welle": (478, 210, A_L, A_B), "h_welle": 452,
        "spek": (766, 258, A_L, A_B), "h_spek": 730,
        "meter": (1148, 56, A_L, A_B), "h_meter": 1112,
    },
    "buehne": {
        "welle": (352, 496, A_L, A_B), "h_welle": 302,
        "spek": (962, 420, A_L, A_B), "h_spek": 912,
        "meter": (1572, 58, A_L, A_B), "h_meter": 1522,
    },
}


class Kalibrierung:
    def __init__(self, cfg: Config, fensterlaenge: float, buehne: bool = False):
        self.cfg = cfg
        self.laenge = fensterlaenge
        self.buehne = buehne
        self.ring = audio.Ringpuffer(cfg)
        self.peak_hold = METER_MIN
        self.clips = 0
        self.eingefroren = False
        self.voll_skala = False
        self.zonen_sichtbar = False
        self.letzter_anschlag: tuple[float, float] | None = None
        self.rausch_verlauf: list[float] = []
        self.bild_nr = 0
        self.marker_abs: list[int] = []
        self.rausch_db = -120.0
        self.stufe = 0.0
        self._hintergrund = None
        self._laufen = True
        self._naechstes = 0.0
        self._baue_figur()

    # -- Aufbau ---------------------------------------------------------
    def _baue_figur(self) -> None:
        theme.anwenden("hochformat")
        bedienung.tastenkuerzel_aus()
        self.masse = bedienung.LiveMasse()
        self.fig = plt.figure(
            figsize=(portrait.BREITE / portrait.DPI, portrait.HOEHE / portrait.DPI),
            dpi=self.masse.dpi, facecolor=theme.BG,
        )
        self.fig.canvas.manager.set_window_title("Kalibrierung - Tastenakustik")
        self.t = np.linspace(-self.laenge, 0.0, int(self.laenge * self.cfg.samplerate))

        # --- Wellenform ---------------------------------------------------
        self.h_welle = portrait.abschnitt(self.fig, "Wellenform", 452)
        self.ax_welle = portrait.achse(self.fig, *self._masse("welle"))
        # Gefuellte Flaeche statt Zickzacklinie: sieht gleich aus, ist aber
        # rund hundertmal billiger zu zeichnen - und das entscheidet ueber die
        # Bildrate. Eine ruckelnde Aufnahme ist spaeter nicht zu retten.
        self.welle = Polygon(np.zeros((2, 2)), closed=True,
                             facecolor=theme.AKZENT, edgecolor="none")
        self.ax_welle.add_patch(self.welle)
        self.ax_welle.set_xlim(-self.laenge, 0)
        self.ax_welle.set_ylim(-1.05, 1.05)
        self.t_skala = self.fig.text(
            portrait.x(portrait.INHALT_RECHTS), portrait.y(452), "",
            fontsize=portrait.S_TICK, color=theme.TEXT_SCHWACH, ha="right", va="center")
        # Markierungen einmal anlegen und danach nur bewegen - Artists je Bild
        # zu erzeugen und wegzuwerfen ist teuer.
        self.marker = [
            self.ax_welle.axvline(0, color=theme.AKZENT2, lw=2.0, alpha=0.85,
                                  zorder=5, visible=False)
            for _ in range(MAX_MARKER)
        ]

        # --- Spektrogramm -------------------------------------------------
        self.h_spek = portrait.abschnitt(self.fig, "Spektrogramm", 730)
        self.ax_spek = portrait.achse(self.fig, *self._masse("spek"))
        # 256 fertige Farben statt eines Colormap-Aufrufs je Bild.
        self.farbtabelle = (colormaps["magma"](np.linspace(0, 1, 256)) * 255
                            ).astype(np.uint8)
        self.bild = self.ax_spek.imshow(
            np.zeros((2, 2, 4), dtype=np.uint8), origin="lower", aspect="auto",
            extent=(-self.laenge, 0, 0, self.cfg.spektro_max_hz / 1000),
            interpolation="none")
        self.ax_spek.set_xlim(-self.laenge, 0)
        self.ax_spek.grid(False)

        # --- Pegel --------------------------------------------------------
        self.h_meter = portrait.abschnitt(self.fig, "Pegel", 1112)
        self.ax_meter = portrait.achse(self.fig, *self._masse("meter"))
        self.ax_meter.set_xlim(METER_MIN, 0)
        self.ax_meter.set_ylim(0, 1)
        self.ax_meter.set_yticks([])
        self.ax_meter.set_xticks([-60, -40, -20, 0])
        self.balken = self.ax_meter.barh(0.5, 0, left=METER_MIN, height=0.82,
                                         color=theme.OK, zorder=3)[0]
        (self.hold_linie,) = self.ax_meter.plot([METER_MIN, METER_MIN], [0.05, 0.95],
                                                color=theme.TEXT, lw=3.0, zorder=4)
        for grenze, farbe in ((-9.0, theme.WARN), (-3.0, theme.FEHLER)):
            self.ax_meter.axvline(grenze, color=farbe, lw=1.6, ls="--", alpha=0.75)
        self.ax_meter.grid(axis="x", alpha=0.25)

        # --- Nur Werkzeugansicht ------------------------------------------
        self.werte: dict[str, object] = {}
        self.chrome: list = [portrait.kopf(self.fig, "Kalibrierung")]
        self.chrome.append(self.fig.text(
            portrait.x(portrait.INHALT_RECHTS), portrait.y(382),
            f"{self.cfg.device_name.split(' (')[0][:24]}   {self.cfg.samplerate} Hz",
            fontsize=portrait.S_TICK, color=theme.TEXT_SCHWACH, ha="right", va="center"))
        self.t_urteil = self.fig.text(
            portrait.x(portrait.INHALT_LINKS), portrait.y(1274), "",
            fontsize=portrait.S_LABEL, fontweight="bold",
            color=theme.TEXT_SCHWACH, va="center", ha="left")
        self.chrome.append(self.t_urteil)
        spalte = portrait.INHALT_BREITE / 3
        for i, (schluessel, beschriftung) in enumerate(
            (("spitze", "Spitze"), ("abstand", "Abstand"), ("clipping", "Clipping"))
        ):
            lx = portrait.INHALT_LINKS + i * spalte
            self.chrome.append(self.fig.text(
                portrait.x(lx), portrait.y(1356), beschriftung,
                fontsize=portrait.S_TICK, color=theme.TEXT_SCHWACH, va="center"))
            self.werte[schluessel] = self.fig.text(
                portrait.x(lx), portrait.y(1414), "-", fontsize=46,
                fontweight="bold", color=theme.TEXT, va="center")
            self.chrome.append(self.werte[schluessel])

        # Bedient wird nur mit der Maus: Zum Pruefen des Pegels wird getippt,
        # und jede belegte Taste wuerde dabei nebenher etwas ausloesen.
        self.bedienung = bedienung.Bedienung(self.fig, [
            ("Bühne", self._buehne_umschalten),
            ("Standbild", self._standbild),
            ("Zonen", self._zonen_umschalten),
            ("Peak-Hold", self._peak_zuruecksetzen),
            ("Skala", self._skala_umschalten),
            ("Einfrieren", self._einfrieren),
            ("Beenden", self._beenden),
        ], breite=round(portrait.BREITE * self.masse.skala))

        self._setze_layout()
        self.fig.canvas.mpl_connect("close_event", lambda _e: self.ring.stop())
        self.fig.canvas.mpl_connect(
            "draw_event", lambda _e: setattr(self, "_hintergrund", None))

    def _masse(self, name: str):
        return LAYOUT["buehne" if self.buehne else "werkzeug"][name]

    def _setze_layout(self) -> None:
        """Werkzeugansicht oder Buehne.

        Auf der Buehne bleiben genau drei Dinge: Wellenform, Spektrogramm und
        Pegel - jeweils mit Ueberschrift und beschrifteten Achsen. Kennzahlen,
        Bewertung und Tastenhinweise helfen beim Einstellen und waeren beim
        Filmen nur Beiwerk.
        """
        masse = LAYOUT["buehne" if self.buehne else "werkzeug"]
        for artist in self.chrome:
            if artist is not None:
                artist.set_visible(not self.buehne)
        self.bedienung.zeigen(not self.buehne)

        for ax, schluessel in ((self.ax_welle, "welle"), (self.ax_spek, "spek"),
                               (self.ax_meter, "meter")):
            ax.set_position(portrait.rechteck(*masse[schluessel]))
            ax.tick_params(labelsize=portrait.S_TICK, length=6, width=1.4, pad=8)
        for text, schluessel in ((self.h_welle, "h_welle"), (self.h_spek, "h_spek"),
                                 (self.h_meter, "h_meter")):
            text.set_position((portrait.x(portrait.INHALT_LINKS),
                               portrait.y(masse[schluessel])))
        self.t_skala.set_position((portrait.x(portrait.INHALT_RECHTS),
                                   portrait.y(masse["h_welle"])))

        if self.buehne:
            self.h_welle.set_text("Wellenform")
            self.h_spek.set_text("Spektrogramm")
            self.ax_welle.set_ylabel("Amplitude", fontsize=portrait.S_TICK)
            self.ax_spek.set_ylabel("kHz", fontsize=portrait.S_TICK)
            self.ax_spek.set_xlabel("Sekunden", fontsize=portrait.S_TICK)
            self.ax_spek.set_xticks(np.arange(-self.laenge, 0.5, 1.0))
            self.ax_meter.set_xlabel("dBFS", fontsize=portrait.S_TICK)
            self.ax_welle.grid(False)
        else:
            self.h_welle.set_text("Live-Wellenform")
            self.h_spek.set_text("Live-Spektrogramm")
            self.ax_welle.set_ylabel("")
            self.ax_spek.set_ylabel("kHz", fontsize=portrait.S_TICK)
            self.ax_spek.set_xlabel("")
            self.ax_spek.set_xticks([-self.laenge, 0])
            self.ax_meter.set_xlabel("")
            self.ax_welle.grid(True)
        self.ax_welle.set_xticks([])
        self.fig.canvas.draw_idle()

    # -- Bedienung (nur Maus) -------------------------------------------
    def _standbild(self) -> None:
        pfad = portrait.exportiere(self.fig, "kalibrierung", "01_kalibrierung")
        print(f"gespeichert: {pfad}")
        self.bedienung.melde("Standbild gespeichert")

    def _buehne_umschalten(self) -> None:
        self.buehne = not self.buehne
        self._setze_layout()
        self._animiert(self.buehne)
        self.fig.canvas.draw()

    def _zonen_umschalten(self) -> None:
        self.zonen_sichtbar = not self.zonen_sichtbar
        if self.zonen_sichtbar:
            portrait.sicherheitszonen(self.fig)
        else:
            for a in [a for a in self.fig.artists if getattr(a, "zorder", 0) == 50]:
                a.remove()
            for t in [t for t in self.fig.texts if getattr(t, "zorder", 0) == 51]:
                t.remove()
        self._hintergrund = None
        self.fig.canvas.draw()

    def _peak_zuruecksetzen(self) -> None:
        self.peak_hold = METER_MIN
        self.clips = 0

    def _skala_umschalten(self) -> None:
        self.voll_skala = not self.voll_skala

    def _einfrieren(self) -> None:
        self.eingefroren = not self.eingefroren
        self.bedienung.melde("eingefroren" if self.eingefroren else "läuft",
                             dauer_ms=2000)

    def _beenden(self) -> None:
        self._laufen = False
        plt.close(self.fig)

    # -- Aktualisierung -------------------------------------------------
    def _aktualisiere(self) -> bool:
        """Daten neu rechnen. True, wenn die Figur vollstaendig neu muss."""
        if self.eingefroren:
            return False

        x = self.ring.letzte(self.laenge)
        n = self.t.size
        if x.size < n:
            x = np.concatenate([np.zeros(n - x.size, dtype=np.float32), x])

        # Wellenform: je Bildspalte nur Minimum und Maximum, als Flaeche.
        idx, werte = features.min_max_huellkurve(x, WELLE_SPALTEN)
        zeit = self.t[np.minimum(idx.astype(int), n - 1)]
        unten, oben = werte[0::2], werte[1::2]
        tc = zeit[0::2]
        self.welle.set_xy(np.concatenate([
            np.column_stack([tc, oben]),
            np.column_stack([tc[::-1], unten[::-1]]),
        ]))

        spitze_abs = float(np.max(np.abs(werte)))
        stufe = 1.05 if self.voll_skala else next(
            (s for s in SKALA_STUFEN if spitze_abs * 1.3 <= s), SKALA_STUFEN[-1])
        struktur = stufe != self.stufe
        if struktur:
            self.stufe = stufe
            self.ax_welle.set_ylim(-stufe, stufe)
            self.t_skala.set_text("Vollaussteuerung" if self.voll_skala
                                  else f"Skala  +/-{stufe:.2f}")

        # Spektrogramm
        hop = max(int(x.size / SPEKTRO_SPALTEN), 64)
        db = features.stft_db(x, NFFT, hop)
        db, max_hz = features.bis_frequenz(db, self.cfg.samplerate, NFFT,
                                           self.cfg.spektro_max_hz)
        obergrenze = float(np.percentile(db[::3], 99.5))
        stufen = np.clip((db.T - (obergrenze - 62.0)) * (255.0 / 62.0),
                         0, 255).astype(np.uint8)
        self.bild.set_data(self.farbtabelle[stufen])
        self.bild.set_extent((-self.laenge, 0, 0, max_hz / 1000))

        # Pegel - die Spitze steht exakt in der Huellkurve, der Effektivwert
        # braucht keine volle Aufloesung.
        peak = float(20.0 * np.log10(max(spitze_abs, 1e-9)))
        rms = audio.rms_dbfs(x[::8])
        self.peak_hold = max(self.peak_hold, peak)
        if peak >= -0.2:
            self.clips += 1
        self.balken.set_width(max(rms, METER_MIN) - METER_MIN)
        self.balken.set_color(theme.pegel_farbe(peak))
        self.hold_linie.set_xdata([self.peak_hold, self.peak_hold])

        # Die Anschlagssuche filtert das ganze Fenster und kostet spuerbar.
        # Sie laeuft deshalb nur jedes n-te Bild; die Treffer werden als
        # absolute Samplepositionen gemerkt und danach nur mitgeschoben.
        basis = self.ring.gesamt - n
        self.bild_nr += 1
        if self.bild_nr % MARKER_TAKT == 1 or not self.marker_abs:
            spitzen, self.rausch_db = onset.finde_transienten(x, self.cfg)
            self.marker_abs = [basis + int(i) for i in spitzen[-MAX_MARKER:]]
            if spitzen.size:
                i = int(spitzen[-1])
                umfeld = x[max(0, i - 600): i + 2400]
                if umfeld.size:
                    s = audio.peak_dbfs(umfeld)
                    self.letzter_anschlag = (s, s - self.rausch_db)
        for k, linie in enumerate(self.marker):
            rel = self.marker_abs[k] - basis if k < len(self.marker_abs) else -1
            if 0 <= rel < n:
                linie.set_xdata([self.t[rel], self.t[rel]])
                linie.set_visible(True)
            else:
                linie.set_visible(False)

        self.rausch_verlauf.append(self.rausch_db)
        self.rausch_verlauf = self.rausch_verlauf[-60:]
        if not self.buehne:
            self._kennzahlen(peak)
        return struktur

    def _kennzahlen(self, peak: float) -> None:
        abstand = self.letzter_anschlag[1] if self.letzter_anschlag else None
        self.werte["spitze"].set_text(f"{peak:.0f}")
        self.werte["spitze"].set_color(theme.pegel_farbe(peak))
        self.werte["abstand"].set_text(f"{abstand:.0f} dB" if abstand else "-")
        self.werte["clipping"].set_text(str(self.clips))
        self.werte["clipping"].set_color(theme.FEHLER if self.clips else theme.TEXT)

        if self.clips:
            urteil, farbe = "übersteuert - Gain runter", theme.FEHLER
        elif abstand and abstand >= 25:
            urteil, farbe = "sehr gut - so aufnehmen", theme.OK
        elif abstand and abstand >= self.cfg.min_snr_db:
            urteil, farbe = "brauchbar - mehr Abstand wäre besser", theme.OK
        elif abstand:
            urteil, farbe = "zu wenig Abstand zum Rauschen", theme.WARN
        else:
            urteil, farbe = "Taste drücken zum Prüfen", theme.TEXT_SCHWACH
        self.t_urteil.set_text(urteil)
        self.t_urteil.set_color(farbe)

    # -- Zeichnen -------------------------------------------------------
    def _blit_gruppen(self) -> tuple:
        return ((self.ax_welle, [self.welle, *self.marker]),
                (self.ax_spek, [self.bild]),
                (self.ax_meter, [self.balken, self.hold_linie]))

    def _animiert(self, an: bool) -> None:
        """Artists, die per Blitting laufen, aus dem normalen Zeichnen nehmen."""
        for _ax, artists in self._blit_gruppen():
            for a in artists:
                a.set_animated(an)
        self._hintergrund = None

    def _bild(self) -> None:
        """Ein Bild zeichnen.

        Blitting heisst: Hintergrund einmal merken, danach je Bild nur noch
        die drei Datenflaechen darueberlegen. Das ist der Unterschied zwischen
        rund zehn und rund dreissig Bildern je Sekunde.
        """
        struktur = self._aktualisiere()
        leinwand = self.fig.canvas
        if not self.buehne:
            leinwand.draw_idle()
            return
        if struktur or self._hintergrund is None:
            leinwand.draw()
            self._hintergrund = [leinwand.copy_from_bbox(ax.bbox)
                                 for ax, _ in self._blit_gruppen()]
        for (ax, artists), hg in zip(self._blit_gruppen(), self._hintergrund):
            leinwand.restore_region(hg)
            for a in artists:
                if a.get_visible():
                    ax.draw_artist(a)
            leinwand.blit(ax.bbox)

    def _takt(self) -> None:
        """Feste Taktung auf einem Zeitraster.

        Wichtiger als eine hohe Bildrate ist eine gleichmaessige: ungleich
        lange Bilder sieht man in der Aufnahme sofort als Ruckeln.
        """
        if not self._laufen:
            return
        try:
            self._bild()
        except Exception:  # noqa: BLE001 - ein Aussetzer darf nicht alles abbrechen
            self._hintergrund = None
        takt = 1.0 / BILDRATE
        jetzt = time.perf_counter()
        self._naechstes += takt
        if self._naechstes < jetzt:
            self._naechstes = jetzt + takt
        try:
            self.fig.canvas.get_tk_widget().after(
                max(int((self._naechstes - jetzt) * 1000), 1), self._takt)
        except Exception:  # noqa: BLE001 - Fenster ist zu
            self._laufen = False

    def starten(self) -> None:
        self.ring.start()
        self.masse.platzieren(self.fig.canvas.manager.window,
                              self.bedienung.hoehe())
        self._animiert(self.buehne)
        self.fig.canvas.draw()
        self._naechstes = time.perf_counter()
        self._takt()
        try:
            plt.show()
        finally:
            self._laufen = False
            self.ring.stop()


def main() -> int:
    p = argparse.ArgumentParser(description="Kalibrierung / Mikrofon-Test")
    p.add_argument("--geraet", default=None, help="Index oder Namensfragment")
    p.add_argument("--sekunden", type=float, default=3.0, help="Laenge des Live-Fensters")
    p.add_argument("--buehne", action="store_true",
                   help="aufgeraeumte Ansicht fuer die Aufnahme")
    args = p.parse_args()

    verzeichnisse_anlegen()
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

    print(f"Geraet   {cfg.device_name} ({cfg.hostapi}, {cfg.samplerate} Hz)")
    print(f"Klassen  {'  '.join(t.upper() for t in TASTEN)}")
    print(f"Fenster  1080 x 1920, {BILDRATE} Bilder je Sekunde")
    print("Bedienung mit der Maus: Knoepfe unter dem Bild oder Rechtsklick.\n")
    kalibrierung = Kalibrierung(cfg, args.sekunden, args.buehne)
    try:
        kalibrierung.starten()
    except RuntimeError as fehler:
        plt.close(kalibrierung.fig)
        print(fehler)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

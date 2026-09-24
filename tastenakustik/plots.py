"""Alle Video-Grafiken im Hochformat 1080 x 1920.

Jede Funktion nimmt fertige Daten entgegen und gibt eine Figure zurueck. Damit
sind Darstellung und Auswertung getrennt: dieselben Funktionen zeichnen spaeter
die echten Trainingsergebnisse.

Gestaltungsregeln - bewusst streng, weil die Bilder im Video laufen und
nebenbei gesprochen erklaert werden:

  * Luft an allen vier Seiten. Inhalt nur zwischen x 140..920 und y 240..1480,
    der Bereich darunter bleibt leer fuer Bildunterschrift und Fortschritt.
  * So wenig Text wie moeglich. Kein Untertitel, keine Fussnoten, keine
    Achsenbeschriftung, die sich von selbst versteht. Was erklaert wird,
    gehoert nicht aufs Bild.
  * Was doch draufsteht, ist gross: kleinste Schrift 24 pt (33 px), Zahlen
    34 pt, die eine Kernzahl 92 pt.
  * Eine feste Farbe je Klasse, ueber alle Grafiken hinweg.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import features, icons, portrait, theme
from .config import Config, TASTEN, anzeige, beiname, zufall


P = portrait


# --- Bausteine ------------------------------------------------------------
def _feld(fig: Figure, oben: float, hoehe: float, links: float, breite: float,
          farbe: str, alpha: float = 1.0, radius: float = 0.012, zorder: int = 2):
    """Abgerundetes Feld in Leinwandpixeln."""
    stueck = FancyBboxPatch(
        (P.x(links), P.y(oben + hoehe)), breite / P.BREITE, hoehe / P.HOEHE,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        transform=fig.transFigure, facecolor=farbe, edgecolor="none",
        alpha=alpha, zorder=zorder,
    )
    fig.add_artist(stueck)
    return stueck


def _kachel(fig: Figure, text: str, oben: float, hoehe: float, farbe: str,
            links: float = P.INHALT_LINKS, breite: float | None = None,
            fontsize: int = P.S_KACHEL):
    """Farbiges Feld mit zentriertem Zeichen - so bleibt auch der Punkt sichtbar."""
    breite = breite if breite is not None else P.INHALT_BREITE
    _feld(fig, oben, hoehe, links, breite, farbe)
    fig.text(P.x(links + breite / 2), P.y(oben + hoehe / 2), text,
             ha="center", va="center", fontsize=fontsize, fontweight="bold",
             color=theme.BG, zorder=3)


def _pfeil(fig: Figure, oben: float, hoehe: float) -> None:
    mitte = P.INHALT_LINKS + P.INHALT_BREITE / 2
    fig.add_artist(FancyArrowPatch(
        (P.x(mitte), P.y(oben)), (P.x(mitte), P.y(oben + hoehe)),
        transform=fig.transFigure, arrowstyle="-|>", mutation_scale=44,
        color=theme.TEXT_SCHWACH, linewidth=3.0, zorder=2,
    ))


def vorschau_stempel(fig: Figure) -> None:
    """Markieren, dass hier keine Messung zu sehen ist."""
    fig.text(P.x(P.BREITE / 2), P.y(P.HOEHE / 2), "BEISPIELDATEN",
             ha="center", va="center", fontsize=60, fontweight="bold",
             color=theme.WARN, alpha=0.16, rotation=28, zorder=40)
    fig.text(P.x(P.INHALT_LINKS), P.y(1700), "BEISPIELDATEN", fontsize=22,
             fontweight="bold", color=theme.BG, va="center", ha="left", zorder=40,
             bbox=dict(boxstyle="round,pad=0.45", facecolor=theme.WARN,
                       edgecolor="none"))


def _mel_bild(ax, mel: np.ndarray, t0: float, t1: float, f0: float, f1: float,
              spanne_db: float = 55.0):
    oben = float(np.max(mel))
    ax.imshow(mel.T, origin="lower", aspect="auto", cmap="magma",
              extent=(t0, t1, f0, f1), vmin=oben - spanne_db, vmax=oben,
              interpolation="nearest")
    ax.grid(False)


def _mel_grenzen(cfg: Config) -> tuple[float, float]:
    f = features.mel_mittenfrequenzen(cfg.samplerate, cfg.mel_baender,
                                      cfg.mel_fmin, cfg.mel_fmax)
    return f[0] / 1000, f[-1] / 1000


# --- 1. Markierter Tastenschlag ------------------------------------------
def wellenform_mit_onset(fenster: np.ndarray, label: str, anschlag, cfg: Config) -> Figure:
    """Ein aufgenommenes Fenster mit Onset-Marke und markiertem Schnitt."""
    from . import onset as onset_modul

    fig = P.figur()
    P.kopf(fig, "Ein Tastenschlag")
    farbe = theme.farbe(label)
    _kachel(fig, anzeige(label), 396, 132, farbe, breite=132, fontsize=62)

    t = (np.arange(fenster.size) / cfg.samplerate - cfg.pre_roll_ms / 1000) * 1000
    ax = P.achse(fig, oben=628, hoehe=560)
    ax.plot(t, fenster, color=farbe, lw=1.4)
    ax.set_xlim(t[0], t[-1])
    grenze = float(np.max(np.abs(fenster))) * 1.25
    ax.set_ylim(-grenze, grenze)
    P.achse_aufraeumen(ax, x_ticks=[-200, 0, 200, 400])

    if anschlag.gefunden:
        t_on = t[min(anschlag.onset_sample, t.size - 1)]
        s0, s1 = onset_modul.segment_grenzen(anschlag, fenster.size, cfg)
        ax.axvspan(t[s0], t[min(s1, t.size - 1)], color=theme.AKZENT2,
                   alpha=0.16, zorder=1)
        ax.axvline(t_on, color=theme.AKZENT2, lw=3.2, zorder=5)
        fig.text(P.x(P.INHALT_LINKS), P.y(596), "Onset", fontsize=P.S_LABEL,
                 fontweight="bold", color=theme.AKZENT2, va="center")
        fig.text(P.x(P.INHALT_RECHTS), P.y(596), f"{cfg.segment_ms} ms",
                 fontsize=P.S_LABEL, fontweight="bold", color=theme.AKZENT2,
                 va="center", ha="right")

    P.hero(fig, f"{anschlag.snr_db:.0f} dB", 1340, theme.OK, fontsize=76)
    fig.text(P.x(P.INHALT_LINKS), P.y(1432), "über dem Rauschen",
             fontsize=P.S_TICK, color=theme.TEXT_SCHWACH, va="center")
    return fig


# --- 2. Wellenform zu Log-Mel --------------------------------------------
def wellenform_zu_mel(fenster: np.ndarray, label: str, cfg: Config) -> Figure:
    """Die Umrechnung, die vor jedem Modell steht."""
    fig = P.figur()
    P.kopf(fig, "Vom Klang zum Bild")
    farbe = theme.farbe(label)
    _kachel(fig, anzeige(label), 250, 104, farbe,
            links=P.INHALT_RECHTS - 104, breite=104, fontsize=50)

    t = (np.arange(fenster.size) / cfg.samplerate - cfg.pre_roll_ms / 1000) * 1000

    P.abschnitt(fig, "Wellenform", 452)
    ax1 = P.achse(fig, oben=492, hoehe=330)
    ax1.plot(t, fenster, color=farbe, lw=1.4)
    ax1.set_xlim(t[0], t[-1])
    P.achse_aufraeumen(ax1)

    _pfeil(fig, 872, 96)

    P.abschnitt(fig, "Log-Mel-Spektrogramm", 1036)
    ax2 = P.achse(fig, oben=1076, hoehe=364)
    mel = features.log_mel(fenster, cfg.samplerate, cfg.mel_nfft, cfg.mel_hop,
                           cfg.mel_baender, cfg.mel_fmin, cfg.mel_fmax)
    f0, f1 = _mel_grenzen(cfg)
    _mel_bild(ax2, mel, t[0], t[-1], f0, f1)
    P.achse_aufraeumen(ax2)
    return fig


# --- 2b. Die Klassen als Tafel -------------------------------------------
def klassen_tafel(titel: str = "", spalten: int = 2,
                  anteil: float = 1.0) -> Figure:
    """Alle Klassen als grosse Kacheln - das Alphabet dieses Experiments.

    Die Grafik fuer den Satz "um es einfach zu halten, erstmal nur diese acht".
    Jede Klasse in ihrer festen Farbe, damit sie in allen spaeteren Bildern
    wiedererkannt wird.

    Zeichen, die sich nicht selbst erklaeren, bekommen ihren Namen dazu - ein
    Punkt ist sonst nur ein Pixel neben lauter Buchstaben.

    anteil blendet die Kacheln nacheinander ein (fuer animation.py).
    """
    fig = P.figur()
    if titel:
        P.kopf(fig, titel)

    n = len(TASTEN)
    spalten = max(1, min(spalten, n))
    zeilen = -(-n // spalten)

    band_oben = 430 if titel else 320
    band_unten = 1460
    luecke = 34
    breite = (P.INHALT_BREITE - luecke * (spalten - 1)) / spalten
    hoehe = (band_unten - band_oben - luecke * (zeilen - 1)) / zeilen
    # Kacheln nicht hoeher als breit werden lassen: bei vier Spalten waeren
    # es sonst schmale Saeulen, in denen der Buchstabe verloren geht. Was an
    # Hoehe uebrig bleibt, wird oben und unten gleich verteilt.
    hoehe = min(hoehe, breite * 1.15)
    block = zeilen * hoehe + (zeilen - 1) * luecke
    # Oben ausrichten, nur mit etwas Luft unter dem Titel. Bleibt darunter
    # Platz frei, ist das so gewollt - der untere Rand gehoert der
    # Bildunterschrift.
    oben0 = band_oben + min((band_unten - band_oben - block) * 0.15, 60.0)

    schrift = max(int(min(breite, hoehe) * 0.64), 16)
    name_schrift = max(int(schrift * 0.22), P.S_TICK)

    fortschritt = float(np.clip(anteil, 0.0, 1.0)) * n
    for i, taste in enumerate(TASTEN):
        sichtbar = float(np.clip(fortschritt - i, 0.0, 1.0))
        if sichtbar <= 0.0:
            continue
        zeile, spalte = divmod(i, spalten)
        links = P.INHALT_LINKS + spalte * (breite + luecke)
        oben = oben0 + zeile * (hoehe + luecke)
        name = beiname(taste)

        _feld(fig, oben, hoehe, links, breite, theme.farbe(taste),
              alpha=sichtbar, radius=0.018)
        fig.text(P.x(links + breite / 2),
                 P.y(oben + hoehe / 2 - (hoehe * 0.07 if name else 0.0)),
                 anzeige(taste), ha="center", va="center", fontsize=schrift,
                 fontweight="bold", color=theme.BG, zorder=3, alpha=sichtbar)
        if name:
            fig.text(P.x(links + breite / 2), P.y(oben + hoehe * 0.82), name,
                     ha="center", va="center", fontsize=name_schrift,
                     fontweight="bold", color=theme.BG, zorder=3,
                     alpha=sichtbar * 0.65)
    return fig


# --- 3. Klassen nebeneinander --------------------------------------------
def klassen_vergleich(mittel: dict[str, np.ndarray], cfg: Config) -> Figure:
    """Je Klasse ein gemitteltes Log-Mel-Spektrogramm.

    Das Raster richtet sich nach der Klassenzahl: bei acht Klassen zwei
    Spalten mit grossen Kacheln, bei einem ganzen Alphabet vier Spalten.
    """
    n = len(TASTEN)
    fig = P.figur()
    P.kopf(fig, f"{n} Klassen")

    spalten = 2 if n <= 10 else (3 if n <= 18 else 4)
    zeilen = -(-n // spalten)

    # Verfuegbare Hoehe zwischen Kopf und dem leeren Bereich unten aufteilen.
    oben0, unten = 442, 1500
    luecke = max(int(66 * 4 / zeilen), 14)
    hoehe = max((unten - oben0 - luecke * (zeilen - 1)) / zeilen, 40)
    spalten_luecke = 56 if spalten == 2 else 30
    spalte_b = (P.INHALT_BREITE - spalten_luecke * (spalten - 1)) / spalten
    kachel_b = min(62, spalte_b * 0.26, hoehe * 0.9)
    f0, f1 = _mel_grenzen(cfg)

    for i, taste in enumerate(TASTEN):
        zeile, spalte = divmod(i, spalten)
        links = P.INHALT_LINKS + spalte * (spalte_b + spalten_luecke)
        oben = oben0 + zeile * (hoehe + luecke)
        _kachel(fig, anzeige(taste), oben + (hoehe - kachel_b) / 2, kachel_b,
                theme.farbe(taste), links=links, breite=kachel_b,
                fontsize=max(int(kachel_b * 0.52), 13))
        ax = P.achse(fig, oben=oben, hoehe=hoehe, links=links + kachel_b + 18,
                     breite=spalte_b - kachel_b - 18)
        mel = mittel.get(taste)
        if mel is None or mel.size == 0:
            P.achse_aufraeumen(ax, rahmen=True)
            continue
        _mel_bild(ax, mel, 0, mel.shape[0], f0, f1)
        P.achse_aufraeumen(ax)
    return fig


# --- 4. Pipeline ----------------------------------------------------------
PIPELINE_SCHRITTE = (
    ("taste", "Anschlag"),
    ("mikrofon", "Mikrofon"),
    ("welle", "400 ms Audio"),
    ("mel", "Log-Mel"),
    ("netz", "Neuronales Netz"),
    ("balken", "Klassen"),
)

_SCHRITT_OBEN = 424
_SCHRITT_ABSTAND = 172
_ICON = 140


def _pipeline_schritt(fig: Figure, nummer: int, anteil: float = 1.0) -> None:
    """Einen Schritt der Pipeline zeichnen. anteil steuert das Einblenden."""
    schluessel, beschriftung = PIPELINE_SCHRITTE[nummer]
    oben = _SCHRITT_OBEN + nummer * _SCHRITT_ABSTAND
    ax = P.achse(fig, oben=oben, hoehe=_ICON, links=P.INHALT_LINKS, breite=_ICON)
    if schluessel == "balken":
        icons.balken(ax, [theme.farbe(t) for t in TASTEN], anteil=anteil)
    else:
        icons.SCHRITTE[schluessel](ax, theme.AKZENT, anteil=anteil)
    fig.text(P.x(P.INHALT_LINKS + _ICON + 46), P.y(oben + _ICON / 2), beschriftung,
             fontsize=P.S_LABEL, fontweight="bold", color=theme.TEXT,
             va="center", ha="left", alpha=max(min(anteil * 1.4 - 0.4, 1.0), 0.0))


def _pipeline_verbindung(fig: Figure, nummer: int, anteil: float = 1.0) -> None:
    """Senkrechte Linie zwischen zwei Schritten."""
    if anteil <= 0:
        return
    x_mitte = P.INHALT_LINKS + _ICON / 2
    von = _SCHRITT_OBEN + nummer * _SCHRITT_ABSTAND + _ICON + 8
    bis = von + (_SCHRITT_ABSTAND - _ICON - 16) * min(anteil, 1.0)
    fig.add_artist(Line2D([P.x(x_mitte), P.x(x_mitte)], [P.y(von), P.y(bis)],
                          transform=fig.transFigure, color=theme.GRID,
                          linewidth=3.0, zorder=1))


def pipeline(cfg: Config, bis_schritt: int | None = None, anteil: float = 1.0) -> Figure:
    """Der Weg vom Anschlag zur Wahrscheinlichkeit.

    Senkrechte Abfolge mit gezeichneten Icons statt farbiger Kaesten: nur ein
    Akzentton, die Klassenfarben bleiben dem letzten Schritt vorbehalten. So
    bleibt das Bild ruhig genug, um davor zu sprechen und darauf zu zeigen.

    bis_schritt und anteil dienen der Animation: Schritte werden nacheinander
    eingeblendet, der Rest bleibt unsichtbar.
    """
    fig = P.figur()
    P.kopf(fig, "Die Pipeline")
    letzter = len(PIPELINE_SCHRITTE) - 1 if bis_schritt is None else bis_schritt
    for i in range(len(PIPELINE_SCHRITTE)):
        if i > letzter:
            break
        teil = 1.0 if i < letzter else anteil
        if i > 0:
            _pipeline_verbindung(fig, i - 1, 1.0 if i < letzter else anteil * 1.6)
        _pipeline_schritt(fig, i, teil)
    return fig


# --- 4b. Was das Netz macht ----------------------------------------------
def modell_erklaerung(wahrscheinlichkeiten: dict[str, float] | None = None,
                      mel_bild: np.ndarray | None = None, cfg: Config | None = None,
                      anteil: float = 1.0) -> Figure:
    """Was im Modell passiert - in einem Bild, ohne Fachbegriffe.

    Oben das Log-Mel als Eingang, in der Mitte das Netz, unten die Klassen mit
    der Wahrscheinlichkeit des Gewinners. Genau das ist die Aussage: ein Bild
    vom Klang rein, eine Zahl pro Klasse raus.

    mel_bild ist optional - liegt ein echtes Log-Mel vor, wird es gezeigt,
    sonst ein stilisiertes Raster.
    """
    fig = P.figur()
    P.kopf(fig, "Was das Netz macht")

    if not wahrscheinlichkeiten:
        rest = 0.1 / max(len(TASTEN) - 1, 1)
        wahrscheinlichkeiten = {t: (0.9 if t == TASTEN[0] else rest) for t in TASTEN}
    p = wahrscheinlichkeiten
    beste = max(p, key=p.get)
    a = float(np.clip(anteil, 0.0, 1.0))

    # 1 - Eingang
    P.abschnitt(fig, "Bild vom Klang", 452)
    ax_in = P.achse(fig, oben=492, hoehe=200)
    if mel_bild is not None and cfg is not None:
        _mel_bild(ax_in, mel_bild, 0, mel_bild.shape[0], *_mel_grenzen(cfg))
        P.achse_aufraeumen(ax_in)
        ax_in.set_alpha(min(a * 3, 1.0))
    else:
        icons.mel(ax_in, theme.AKZENT, anteil=min(a * 3, 1.0))

    _pfeil(fig, 722, 74)

    # 2 - Netz
    P.abschnitt(fig, "sucht Muster", 838)
    ax_netz = P.achse(fig, oben=878, hoehe=214)
    icons.netz(ax_netz, theme.AKZENT2, anteil=min(max(a * 3 - 0.9, 0.0), 1.0))

    _pfeil(fig, 1122, 74)

    # 3 - Ausgang: eine Saeule je Klasse, die Kernzahl gross daneben
    auf = min(max(a * 3 - 1.8, 0.0), 1.0)
    P.abschnitt(fig, "und entscheidet", 1238)
    if auf > 0.25:
        fig.text(P.x(P.INHALT_RECHTS), P.y(1238), f"{p[beste] * 100:.0f}%",
                 fontsize=52, fontweight="bold", color=theme.farbe(beste),
                 ha="right", va="center", alpha=min(auf * 2, 1.0))

    boden, max_hoch = 1418, 132
    breite = P.INHALT_BREITE / len(TASTEN)
    # Beschriftung an die Saeulenbreite koppeln, sonst ueberlappen die
    # Buchstaben, sobald mehr als ein Dutzend Klassen nebeneinander stehen.
    label_gr = int(min(32, max(breite * 0.58, 11)))
    for i, taste in enumerate(TASTEN):
        wert = float(p.get(taste, 0.0)) * auf
        x_mitte = P.INHALT_LINKS + breite * (i + 0.5)
        hoch = max_hoch * wert
        if hoch > 2:
            _feld(fig, boden - hoch, hoch, x_mitte - breite * 0.28, breite * 0.56,
                  theme.farbe(taste), radius=0.005)
        fig.text(P.x(x_mitte), P.y(boden + 44), anzeige(taste), fontsize=label_gr,
                 fontweight="bold", color=theme.farbe(taste),
                 ha="center", va="center")
    fig.add_artist(Line2D([P.x(P.INHALT_LINKS), P.x(P.INHALT_RECHTS)],
                          [P.y(boden), P.y(boden)], transform=fig.transFigure,
                          color=theme.GRID, linewidth=2.4))
    return fig


# --- 5. Trainingsverlauf --------------------------------------------------
def trainingsverlauf(epochen, train_loss, val_loss, train_acc, val_acc,
                     anteil: float = 1.0) -> Figure:
    """Loss und Trefferquote uebereinander."""
    fig = P.figur()
    P.kopf(fig, "Training")

    epochen = np.asarray(epochen)
    n = max(int(round(len(epochen) * float(np.clip(anteil, 0.0, 1.0)))), 1)
    train_loss, val_loss = np.asarray(train_loss)[:n], np.asarray(val_loss)[:n]
    train_acc, val_acc = np.asarray(train_acc)[:n], np.asarray(val_acc)[:n]
    sichtbar = epochen[:n]
    beste = float(np.max(val_acc))
    P.hero(fig, f"{beste * 100:.0f} %", 472, theme.OK, fontsize=88)
    fig.text(P.x(P.INHALT_LINKS), P.y(566), "beste Validation",
             fontsize=P.S_TICK, color=theme.TEXT_SCHWACH, va="center")

    # Farbcode statt Legendenkasten: zwei Woerter, direkt in der Linienfarbe.
    fig.text(P.x(P.INHALT_RECHTS), P.y(452), "Training", fontsize=P.S_TICK,
             fontweight="bold", color=theme.AKZENT, ha="right", va="center")
    fig.text(P.x(P.INHALT_RECHTS), P.y(504), "Validation", fontsize=P.S_TICK,
             fontweight="bold", color=theme.AKZENT2, ha="right", va="center")

    # Keine x-Ticks: die Epochenzahl ist nicht die Aussage, die Form ist es.
    P.abschnitt(fig, "Loss", 662)
    ax1 = P.achse(fig, oben=722, hoehe=300, links=P.ACHSE_LINKS, breite=P.ACHSE_BREITE)
    ax1.plot(sichtbar, train_loss, color=theme.AKZENT, lw=3.4)
    ax1.plot(sichtbar, val_loss, color=theme.AKZENT2, lw=3.4)
    ax1.set_xlim(epochen[0], epochen[-1])
    hoch = max(float(np.max(train_loss)), float(np.max(val_loss)))
    ax1.set_ylim(0, hoch * 1.08)
    P.achse_aufraeumen(ax1, y_ticks=[0, round(hoch, 1)])

    P.abschnitt(fig, "Trefferquote", 1100)
    ax2 = P.achse(fig, oben=1160, hoehe=300, links=P.ACHSE_LINKS, breite=P.ACHSE_BREITE)
    ax2.plot(sichtbar, train_acc * 100, color=theme.AKZENT, lw=3.4)
    ax2.plot(sichtbar, val_acc * 100, color=theme.AKZENT2, lw=3.4)
    ax2.set_xlim(epochen[0], epochen[-1])
    ax2.axhline(zufall() * 100, color=theme.TEXT_SCHWACH, ls="--", lw=2.4)
    ax2.set_ylim(0, 100)
    P.achse_aufraeumen(ax2, y_ticks=[0, 50, 100])
    # y in Datenkoordinaten - die Achse laeuft in Prozent.
    ax2.text(0.985, zufall() * 100 + 2, "Zufall", color=theme.TEXT_SCHWACH,
             fontsize=P.S_TICK, ha="right", va="bottom",
             transform=ax2.get_yaxis_transform(which="grid"))
    return fig


# --- 6. Confusion Matrix --------------------------------------------------
def konfusionsmatrix(matrix: np.ndarray, genauigkeit: float,
                     anteil: float = 1.0) -> Figure:
    """Welche Klasse wird mit welcher verwechselt."""
    fig = P.figur()
    P.kopf(fig, "Verwechslungen")

    m = np.asarray(matrix, dtype=float)
    zeilen_auf = float(np.clip(anteil, 0.0, 1.0)) * len(TASTEN)
    verteilung = np.divide(m, np.maximum(m.sum(axis=1, keepdims=True), 1e-9))
    maske = (np.arange(len(TASTEN))[:, None] < zeilen_auf).astype(float)
    anteil_bild = verteilung * maske

    P.hero(fig, f"{genauigkeit * 100 * min(anteil * 1.2, 1.0):.0f} %", 430,
           theme.OK, fontsize=88)
    fig.text(P.x(P.INHALT_LINKS), P.y(524), "richtig erkannt",
             fontsize=P.S_TICK, color=theme.TEXT_SCHWACH, va="center")
    fig.text(P.x(P.INHALT_RECHTS), P.y(430),
             f"Zufall  {zufall() * 100:.0f} %", fontsize=P.S_TICK,
             color=theme.TEXT_SCHWACH, ha="right", va="center")

    kante = P.ACHSE_BREITE
    ax = P.achse(fig, oben=636, hoehe=kante, links=P.ACHSE_LINKS, breite=kante)
    ax.imshow(anteil_bild, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    # Schrift an die Zellengroesse koppeln - bei 26 Klassen ist eine Zelle
    # nur noch 26 Leinwandpixel breit.
    zellen_px = kante / len(TASTEN)
    label_gr = int(min(28, max(zellen_px * 0.46, 9)))
    ax.set_xticks(range(len(TASTEN)), [anzeige(t) for t in TASTEN])
    ax.set_yticks(range(len(TASTEN)), [anzeige(t) for t in TASTEN])
    ax.tick_params(labelsize=label_gr, length=0, pad=12)
    ax.grid(False)
    for seite in ("top", "right", "bottom", "left"):
        ax.spines[seite].set_visible(False)
    for i, taste in enumerate(TASTEN):
        ax.get_yticklabels()[i].set_color(theme.farbe(taste))
        ax.get_yticklabels()[i].set_fontweight("bold")
        ax.get_xticklabels()[i].set_color(theme.farbe(taste))
        ax.get_xticklabels()[i].set_fontweight("bold")

    # Ab etwa zwoelf Klassen passt keine Zahl mehr in eine Zelle - dann
    # traegt die Farbflaeche die Aussage allein.
    if zellen_px >= 54:
        for i in range(len(TASTEN)):
            for j in range(len(TASTEN)):
                if m[i, j] <= 0 or i >= zeilen_auf:
                    continue
                treffer = i == j
                ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center",
                        fontsize=26 if treffer else 21,
                        fontweight="bold" if treffer else "normal",
                        color=theme.BG if verteilung[i, j] > 0.5 else theme.TEXT,
                        alpha=1.0 if treffer else 0.75)

    fig.text(P.x(P.ACHSE_LINKS + kante / 2), P.y(636 + kante + 74), "erkannt",
             fontsize=P.S_TICK, color=theme.TEXT_SCHWACH, ha="center", va="center")
    fig.text(P.x(P.INHALT_LINKS - 8), P.y(636 + kante / 2), "gedrückt",
             fontsize=P.S_TICK, color=theme.TEXT_SCHWACH, ha="center", va="center",
             rotation=90)
    return fig


# --- 7. Einzelvorhersage / Blindtest -------------------------------------
def vorhersage(vorgabe: str | None, wahrscheinlichkeiten: dict[str, float],
               sequenz: str = "", titel: str = "Blindtest",
               anteil: float = 1.0) -> Figure:
    """Ein Anschlag, eine Wahrscheinlichkeit je Klasse.

    Alle Klassen werden gezeigt, auch die mit null Prozent - ein Bild, das nur
    die besten drei zeigt, laesst offen, wie knapp die Entscheidung war. Bei
    vielen Klassen ruecken die Zeilen zusammen und bei ueber vierzehn in zwei
    Spalten, statt aus der Leinwand zu laufen.

    vorgabe dient nur dem Abgleich - in die Vorhersage geht ausschliesslich
    das Audiosignal ein.
    """
    fig = P.figur()
    P.kopf(fig, titel)

    a = float(np.clip(anteil, 0.0, 1.0))
    fuellung = min(a * 1.8, 1.0)          # Balken laufen zuerst hoch
    ergebnis_auf = max(a * 3.0 - 2.0, 0.0)  # dann kommt das Ergebnis
    beste = max(wahrscheinlichkeiten, key=wahrscheinlichkeiten.get)
    p = float(wahrscheinlichkeiten[beste])

    n = len(TASTEN)
    spalten = 1 if n <= 14 else 2
    zeilen = -(-n // spalten)
    oben, platz = 430, 880
    zeile = min(80.0, platz / zeilen)
    hoehe = zeile * 0.72
    spalten_luecke = 44
    spalte_b = (P.INHALT_BREITE - spalten_luecke * (spalten - 1)) / spalten
    kachel_b = min(58.0, hoehe, spalte_b * 0.14)
    zahl_b = 190 if spalten == 1 else 104
    balken_b = spalte_b - kachel_b - 22 - zahl_b
    zahl_gr = P.S_ZAHL if spalten == 1 else 24

    for i, taste in enumerate(TASTEN):
        spalte, reihe = divmod(i, zeilen)      # spaltenweise fuellen
        links = P.INHALT_LINKS + spalte * (spalte_b + spalten_luecke)
        y0 = oben + reihe * zeile
        wert = float(wahrscheinlichkeiten.get(taste, 0.0)) * fuellung
        farbe = theme.farbe(taste)
        fuehrend = taste == beste
        balken_x = links + kachel_b + 22
        _kachel(fig, anzeige(taste), y0, hoehe, farbe,
                links=links, breite=kachel_b, fontsize=max(int(kachel_b * 0.52), 12))
        _feld(fig, y0, hoehe, balken_x, balken_b, theme.PANEL, radius=0.008)
        if wert > 0.004:
            _feld(fig, y0, hoehe, balken_x, balken_b * wert, farbe,
                  alpha=1.0 if fuehrend else 0.5, radius=0.008, zorder=3)
        # Nullzeilen bleiben leer: der leere Balken sagt es schon, und jede
        # weggelassene Zahl macht die uebrigen lesbarer.
        if wert >= 0.005:
            fig.text(P.x(links + spalte_b), P.y(y0 + hoehe / 2), f"{wert * 100:.0f}%",
                     fontsize=zahl_gr if fuehrend else zahl_gr - 6,
                     family="monospace",
                     fontweight="bold" if fuehrend else "normal",
                     color=theme.TEXT if fuehrend else theme.TEXT_SCHWACH,
                     ha="right", va="center", zorder=3)

    ergebnis = oben + zeilen * zeile + 42
    if ergebnis_auf > 0:
        _feld(fig, ergebnis, 138, P.INHALT_LINKS, P.INHALT_BREITE,
              theme.farbe(beste), alpha=ergebnis_auf)
        fig.text(P.x(P.INHALT_LINKS + P.INHALT_BREITE / 2), P.y(ergebnis + 69),
                 f"{anzeige(beste)}   {p * 100:.0f}%", ha="center", va="center",
                 fontsize=62, fontweight="bold", color=theme.BG,
                 alpha=ergebnis_auf, zorder=3)

    if vorgabe and ergebnis_auf >= 1.0:
        richtig = vorgabe == beste
        fig.text(P.x(P.INHALT_LINKS), P.y(ergebnis + 190),
                 "richtig" if richtig else f"falsch  -  war {anzeige(vorgabe)}",
                 fontsize=P.S_LABEL, fontweight="bold",
                 color=theme.OK if richtig else theme.FEHLER, va="center")

    if sequenz:
        fig.text(P.x(P.INHALT_LINKS), P.y(1422), sequenz.upper(), fontsize=54,
                 family="monospace", fontweight="bold", color=theme.TEXT, va="center")
    return fig


# --- 8. Der ehrliche Vergleich -------------------------------------------
def ergebnis_vergleich(stufen: list[tuple[str, float, str]],
                       hero: tuple[str, float] | None = None,
                       anteil: float = 1.0) -> Figure:
    """Alle gemessenen Trefferquoten nebeneinander.

    Die wichtigste Grafik des Projekts: Sie zeigt nicht die beste Zahl, sondern
    alle - vom Zufall ueber das Verfahren ohne Lernen bis zu dem, was bei
    fluessigem Tippen uebrig bleibt. Wer nur die 99 % zeigt, erzaehlt die
    Haelfte.

    stufen: Liste aus (Beschriftung, Anteil 0..1, Farbe).
    """
    fig = P.figur()
    P.kopf(fig, "Was das Modell kann")
    a = float(np.clip(anteil, 0.0, 1.0))

    if hero is not None:
        name, wert = hero
        P.hero(fig, f"{wert * 100:.0f} %", 452, theme.OK, fontsize=88)
        fig.text(P.x(P.INHALT_LINKS), P.y(546), name, fontsize=P.S_TICK,
                 color=theme.TEXT_SCHWACH, va="center")

    oben, zeile, hoehe = 660, 128, 58
    breite_max = P.INHALT_BREITE - 190
    for i, (beschriftung, wert, farbe) in enumerate(stufen):
        y0 = oben + i * zeile
        auf = min(max(a * len(stufen) - i, 0.0), 1.0)
        fig.text(P.x(P.INHALT_LINKS), P.y(y0 - 26), beschriftung,
                 fontsize=P.S_TICK, color=theme.TEXT_SCHWACH, va="center")
        _feld(fig, y0, hoehe, P.INHALT_LINKS, breite_max, theme.PANEL, radius=0.008)
        if wert * auf > 0.002:
            _feld(fig, y0, hoehe, P.INHALT_LINKS, breite_max * wert * auf, farbe,
                  radius=0.008, zorder=3)
        if auf > 0.3:
            fig.text(P.x(P.INHALT_RECHTS), P.y(y0 + hoehe / 2),
                     f"{wert * 100:.0f}%", fontsize=P.S_ZAHL, family="monospace",
                     fontweight="bold", color=theme.TEXT, ha="right", va="center",
                     alpha=min((auf - 0.3) / 0.4, 1.0), zorder=3)
    return fig

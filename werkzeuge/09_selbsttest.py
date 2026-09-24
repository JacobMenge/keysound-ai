"""Schritt 9 - Selbsttest: laeuft die Installation, und stimmt die Rechnung?

Braucht kein Mikrofon und ruehrt deine eigenen Aufnahmen nicht an. Der Test
erzeugt kuenstliche Anschlaege mit einer klassenspezifischen Klangfarbe,
schreibt daraus echte Sitzungen in einen Wegwerfordner, laedt sie ueber
dieselbe Pipeline wie im Ernstfall, trainiert kurz, prueft das Modell an der
Testsitzung und rendert jede Grafik.

Damit ist zweierlei geprueft:

  1. Alle Abhaengigkeiten sind da und passen zusammen - auch Tk fuer die
     Fenster und PortAudio fuer das Mikrofon. Geladen wird beides nur, ein
     Geraet wird nicht geoeffnet: Ob dein Mikrofon ankommt, zeigt erst
     Schritt 1 im Studio (oder 01_systemcheck.py).
  2. Die Kette Label -> Datei -> Datensatz -> Ausgang des Netzes stimmt.
     Waere sie vertauscht, koennte das Netz die kuenstlichen Klassen nicht
     trennen. Bestanden heisst deutlich ueber Zufall, nicht jede Klasse
     sauber: Bei sehr vielen Klassen liegen die kuenstlichen Toene eng, und
     einzelne werden verwechselt - das Fazit nennt sie.

Aufruf:
    python werkzeuge/09_selbsttest.py
    python werkzeuge/09_selbsttest.py --klassen abcdefghijklmnopqrstuvwxyz.,-?
    python werkzeuge/09_selbsttest.py --behalten      # Bilder nicht loeschen
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

import numpy as np

from tastenakustik import config

GRUEN, ROT, GRAU, AUS = "\033[92m", "\033[91m", "\033[90m", "\033[0m"

STANDARD_TEST_KLASSEN = "abcdefghijklmnopqrstuvwxyz.,-?"


class Pruefung:
    """Sammelt Ergebnisse und haelt fest, ob etwas schiefging."""

    def __init__(self) -> None:
        self.fehler: list[str] = []

    def __call__(self, name: str, bedingung: bool, info: str = "") -> bool:
        marke = f"{GRUEN}ok  {AUS}" if bedingung else f"{ROT}FEHL{AUS}"
        zusatz = f"   {GRAU}{info}{AUS}" if info else ""
        print(f"   {marke}  {name}{zusatz}")
        if not bedingung:
            self.fehler.append(name)
        return bedingung


def main() -> int:
    # Der Aufruf-Block aus dem Docstring erscheint unter --help als Beispiel.
    p = argparse.ArgumentParser(
        description="Selbsttest ohne Mikrofon",
        epilog=__doc__[__doc__.index("Aufruf:"):],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--klassen", default=STANDARD_TEST_KLASSEN,
                   help="Zeichensatz, mit dem getestet wird (2 bis 40 Zeichen)")
    p.add_argument("--proben", type=int, default=8,
                   help="Trainingsproben je Klasse (Standard 8)")
    p.add_argument("--epochen", type=int, default=150,
                   help="Anzahl Trainingsdurchlaeufe (Standard 150)")
    p.add_argument("--behalten", action="store_true",
                   help="Wegwerfordner stehen lassen")
    args = p.parse_args()
    # Vor dem Wegwerfordner pruefen: Ungueltige Angaben sollen eine klare
    # Meldung geben und keinen Ordner in %TEMP% zuruecklassen.
    try:
        config.pruefe_klassen(args.klassen)
    except config.KlassenFehler as fehler:
        p.error(f"--klassen: {fehler}")
    if args.proben < 1:
        p.error("--proben muss mindestens 1 sein")
    if args.epochen < 1:
        p.error("--epochen muss mindestens 1 sein")

    tmp = Path(tempfile.mkdtemp(prefix="tastenakustik_test_"))
    # Umlenken, bevor irgendein Modul die echten Ordner anfasst.
    config.DATEN = tmp / "daten"
    config.ROH = tmp / "daten" / "roh"
    config.MODELLE = tmp / "daten" / "modelle"
    config.AUSGABE = tmp / "ausgabe"
    config.CONFIG_PFAD = tmp / "config.json"
    config.verzeichnisse_anlegen()

    from tastenakustik import (datensatz, modell, onset, plots, portrait,
                               storage, theme, training)
    for modul in (storage, datensatz, training):
        modul.ROH = config.ROH
    training.MODELLE = config.MODELLE
    portrait.AUSGABE = config.AUSGABE
    theme.AUSGABE = config.AUSGABE

    ok = Pruefung()

    # --- 0. Umgebung ------------------------------------------------------
    # Studio und Collector brauchen Tk und PortAudio. Beides fehlt unter Linux
    # gern als Systempaket - dann startet start.py nicht, und der Rest dieses
    # Tests wuerde trotzdem bestehen. Kein Stream, kein Mikrofon: nur laden.
    print("\n0  Umgebung")
    try:
        import tkinter
        tkinter.Tcl()
        ok("tkinter / Tcl geladen", True, f"Tk {tkinter.TkVersion}")
    except Exception as fehler:                           # noqa: BLE001
        ok("tkinter / Tcl geladen", False, f"{type(fehler).__name__}: {fehler} - "
           "Linux: sudo apt install python3-tk, macOS: brew install python-tk")
    else:
        # Ein echtes Fenster braucht ein Display. Ohne (etwa auf einem Server)
        # wird nur uebersprungen - dort laeuft das Studio ohnehin nicht.
        if sys.platform in ("win32", "darwin") or os.environ.get("DISPLAY") \
                or os.environ.get("WAYLAND_DISPLAY"):
            try:
                wurzel = tkinter.Tk()
                wurzel.withdraw()
                wurzel.destroy()
                ok("Fenster (Tk) startbar", True)
            except Exception as fehler:                   # noqa: BLE001
                ok("Fenster (Tk) startbar", False, f"{type(fehler).__name__}: {fehler}")
        else:
            print(f"   {GRAU}----  Fenster (Tk) nicht geprueft - kein Display{AUS}")
    try:
        import sounddevice as sd
        hostapis = [h["name"] for h in sd.query_hostapis()]
        eingaenge = sum(1 for d in sd.query_devices() if d["max_input_channels"] > 0)
        ok("sounddevice / PortAudio geladen", True,
           f"{len(hostapis)} Schnittstellen, {eingaenge} Eingaenge")
    except Exception as fehler:                           # noqa: BLE001
        # Unter Windows kommt auch eine kaputte DLL als OSError an.
        ok("sounddevice / PortAudio geladen", False,
           f"{type(fehler).__name__}: {fehler} - "
           "PortAudio fehlt? Linux: sudo apt install libportaudio2")

    # --- 1. Klassen -------------------------------------------------------
    print("\n1  Klassen setzen")
    n = len(args.klassen)
    cfg = config.Config(klassen=args.klassen, ziel_pro_taste=args.proben)
    cfg.anwenden()
    # Auch in den Wegwerfordner schreiben: Wer spaeter Config.laden() ruft,
    # etwa das Training fuer den Schnitt, sieht dann dieselben Werte.
    cfg.speichern()
    ok(f"{n} Klassen aktiv", len(config.TASTEN) == n, "".join(config.TASTEN))
    ok("Zufallsniveau", abs(config.zufall() - 1 / n) < 1e-9,
       f"{config.zufall() * 100:.1f} %")
    ok("Farben eindeutig", len(set(theme.TASTEN_FARBEN.values())) == n)
    ok("Dateinamen eindeutig",
       len({config.datei_token(t) for t in config.TASTEN}) == n)
    ok("Fremdes Zeichen wird abgelehnt",
       not config.ist_demo_taste("§"))

    # --- 2. Kuenstliche Aufnahmen ----------------------------------------
    print("\n2  Sitzungen schreiben")
    sr = cfg.samplerate
    rng = np.random.default_rng(7)

    # Grundtoene in 420-Hz-Schritten ab 1800 Hz. Bei mehr als etwa dreissig
    # Klassen laegen die obersten sonst ueber mel_fmax - im Log-Mel-Bild
    # unsichtbar und damit nie zu treffen. Dann rueckt das Raster so weit
    # zusammen, dass auch die zweite Resonanz unter 0,9 * mel_fmax bleibt.
    # (Gleichmaessig auf der Mel-Skala verteilt trennte das Netz bei dreissig
    # Klassen schlechter - die tiefen Klassen verloren ihren grossen Abstand.)
    f_oben = 0.9 * min(cfg.mel_fmax, sr / 2)
    schritt = min(420.0, (f_oben - 1800 - 0.31 * 900) / max(n - 1, 1))

    def anschlag(klasse: str, sitzung_nr: int) -> np.ndarray:
        """Ein Kontextfenster mit genau einem kuenstlichen Anschlag."""
        laenge = int((cfg.pre_roll_ms + cfg.post_roll_ms) / 1000 * sr)
        t = np.arange(laenge) / sr
        x = rng.normal(0, 0.0009, laenge)
        i = config.TASTEN.index(klasse)
        start = int(cfg.pre_roll_ms / 1000 * sr)
        dauer = int(0.05 * sr)
        huelle = np.zeros(laenge)
        huelle[start:start + dauer] = np.exp(-np.linspace(0, 14, dauer))
        # Zwei Resonanzen je Klasse; der Sitzungsversatz sorgt dafuer, dass
        # train, val und test nicht dieselben Zahlen enthalten.
        for k, ab in ((1, 0.0), (2, 0.31)):
            f = 1800 + i * schritt + ab * 900 + sitzung_nr * 12
            x += 0.22 / k * huelle * np.sin(2 * np.pi * f * t + i)
        return (x * rng.uniform(0.8, 1.2)).astype(np.float32)

    geschrieben: dict[str, int] = {}
    for rolle, je_klasse, nr in (("train", args.proben, 0),
                                 ("val", max(args.proben // 2, 4), 1),
                                 ("test", max(args.proben // 2, 4), 2)):
        s = storage.Sitzung(cfg, notiz=f"Selbsttest {rolle}", rolle=rolle)
        for klasse in config.TASTEN:
            for _ in range(je_klasse):
                fenster = anschlag(klasse, nr)
                a = onset.analysiere(fenster, cfg)
                if a.gefunden:
                    s.speichere(fenster, klasse, a)
        s.abschliessen()
        geschrieben[rolle] = s.gesamt
        ok(f"Sitzung {rolle}", s.gesamt == n * je_klasse, f"{s.gesamt} Proben")

    # --- 3. Datensatz -----------------------------------------------------
    print("\n3  Datensatz laden")
    _, d_train = datensatz.lade("train")
    _, d_test = datensatz.lade("test")
    ok("Alle Klassen im Training", len(set(d_train.y.tolist())) == n)
    ok("Keine Sitzung in zwei Rollen",
       not (set(d_train.sitzungen) & set(d_test.sitzungen)))

    # Eine garantiert andere Liste - auch wenn nur zwei Klassen im Spiel sind.
    # Bei vierzig Klassen wird ersetzt statt angehaengt, mehr sind nicht erlaubt.
    merker = list(config.TASTEN)
    fremd = next(c for c in map(chr, range(0x21, 0x250))
                 if c.lower() == c and len(c) == 1 and c not in merker)
    config.setze_klassen((merker if len(merker) < 40 else merker[:-1]) + [fremd])
    try:
        datensatz.lade("train")
        ok("Klassenwechsel faellt auf", False)
    except ValueError:
        ok("Klassenwechsel faellt auf", True)
    config.setze_klassen(merker)

    # --- 4. Training ------------------------------------------------------
    print(f"\n4  Training ({args.epochen} Epochen)")
    ergebnis = None
    for stand in training.trainiere(epochen=args.epochen, seed=0):
        if isinstance(stand, training.Ergebnis):
            ergebnis = stand
            break
        # Alle zehn Epochen ein Lebenszeichen - bei vierzig Klassen dauern
        # fuenfzig Epochen schon eine halbe Minute.
        if stand.epoche % 10 == 0:
            print(f"   {GRAU}Epoche {stand.epoche:>3}   "
                  f"Val {stand.val_acc * 100:5.1f} %{AUS}")
    ok("Ausgabeschicht passt zur Klassenzahl",
       ergebnis.konfusion.shape == (n, n))
    # Die kuenstlichen Klassen sind sehr deutlich getrennt. Bleibt das Netz
    # hier nahe am Zufall, stimmt die Zuordnung Label -> Ausgang nicht. Die
    # Latte haengt am Zufallsniveau: bei zwei Klassen sind 50 % nichts, bei
    # dreissig waeren sie ein deutliches Signal.
    latte = min(0.9, max(2 * config.zufall(), config.zufall() + 0.15))
    ok("Lernt deutlich ueber Zufall", ergebnis.beste_val >= latte,
       f"{ergebnis.beste_val * 100:.1f} % statt {config.zufall() * 100:.1f} % "
       f"(nötig: {latte * 100:.0f} %)")
    ok("Modell gespeichert",
       ergebnis.modell_pfad is not None and ergebnis.modell_pfad.exists())
    # Derselbe Weg wie "Auf Testsitzung pruefen" im Studio und 07 --test.
    # Bewusst ohne Schwelle auf die Quote: Nach kurzem Training kann sie
    # deutlich unter der Val-Quote liegen, ohne dass etwas kaputt ist.
    try:
        t = training.teste(ergebnis.modell_pfad, speichern=False)
        ok("Testsitzung pruefbar",
           t.n == geschrieben["test"] and t.konfusion.shape == (n, n)
           and t.val_quote is not None
           and list(getattr(t, "klassen", config.TASTEN)) == list(config.TASTEN),
           f"n={t.n}, {t.quote * 100:.1f} %")
    except Exception as fehler:                           # noqa: BLE001
        ok("Testsitzung pruefbar", False, f"{type(fehler).__name__}: {fehler}")

    # --- 5. Grafiken ------------------------------------------------------
    print("\n5  Grafiken rendern")
    import matplotlib.pyplot as plt
    from PIL import Image

    theme.anwenden("hochformat")
    bilder = tmp / "bilder"
    bilder.mkdir(exist_ok=True)

    beispiel_klasse = config.TASTEN[min(3, n - 1)]
    fenster = anschlag(beispiel_klasse, 0)
    a = onset.analysiere(fenster, cfg)
    mittel = {t: np.random.default_rng(i).normal(-40, 6, (40, cfg.mel_baender))
              for i, t in enumerate(config.TASTEN)}
    quoten = ergebnis.je_klasse
    p_bsp = {t: (0.72 if t == beispiel_klasse else 0.28 / max(n - 1, 1))
             for t in config.TASTEN}
    e = np.arange(1, args.epochen + 1)
    v = ergebnis.verlauf

    aufgaben = {
        "01_anschlag": lambda: plots.wellenform_mit_onset(
            fenster, beispiel_klasse, a, cfg),
        "02_mel": lambda: plots.wellenform_zu_mel(fenster, beispiel_klasse, cfg),
        "03_klassen": lambda: plots.klassen_vergleich(mittel, cfg),
        "04_pipeline": lambda: plots.pipeline(cfg),
        "05_modell": lambda: plots.modell_erklaerung(p_bsp),
        "06_verlauf": lambda: plots.trainingsverlauf(
            e, v["train_loss"], v["val_loss"], v["train_acc"], v["val_acc"]),
        "07_konfusion": lambda: plots.konfusionsmatrix(
            ergebnis.konfusion, ergebnis.beste_val),
        "08_vorhersage": lambda: plots.vorhersage(beispiel_klasse, p_bsp),
        "09_ergebnis": lambda: plots.ergebnis_vergleich(
            [("Zufall", config.zufall(), theme.TEXT_SCHWACH),
             ("Validation", ergebnis.beste_val, theme.OK)]),
    }
    for name, bauen in aufgaben.items():
        try:
            fig = bauen()
            pfad = bilder / f"{name}.png"
            fig.savefig(pfad, dpi=portrait.DPI, facecolor=fig.get_facecolor())
            plt.close(fig)
            with Image.open(pfad) as bild:
                groesse = bild.size
            ok(name, groesse == (portrait.BREITE, portrait.HOEHE),
               f"{groesse[0]}x{groesse[1]}")
        except Exception as fehler:                       # noqa: BLE001
            ok(name, False, f"{type(fehler).__name__}: {fehler}")

    # --- Fazit ------------------------------------------------------------
    print()
    if ok.fehler:
        print(f"{ROT}Nicht bestanden:{AUS} {', '.join(ok.fehler)}")
        print(f"{GRAU}Testdaten liegen in {tmp}{AUS}")
        return 1

    # Ehrlich bleiben: Bestanden heisst "deutlich ueber Zufall", nicht
    # "jede Klasse sauber". Schwache Klassen nur als Hinweis - bei vier
    # Val-Proben je Klasse waere eine harte Grenze je Klasse reines Rauschen.
    print(f"{GRUEN}Alles in Ordnung.{AUS} Die Installation laeuft, und die Zuordnung "
          f"Label -> Ausgang stimmt ({ergebnis.beste_val * 100:.0f} % statt "
          f"{config.zufall() * 100:.0f} % Zufall bei {n} Klassen).")
    print(f"{GRAU}Je Klasse: "
          f"{min(quoten.values()) * 100:.0f}-{max(quoten.values()) * 100:.0f} %{AUS}")
    schwach = [config.anzeige(t) for t, q in quoten.items() if q < 0.5]
    if schwach:
        print(f"{GRAU}Schwach bei den kuenstlichen Toenen (unter 50 %): "
              f"{' '.join(schwach)}{AUS}")
    if args.behalten:
        print(f"{GRAU}Bilder und Testdaten: {tmp}{AUS}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

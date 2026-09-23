"""Layout-Vorschau aller Video-Grafiken im Hochformat - mit Beispieldaten.

Damit laesst sich die Darstellung festzurren, bevor echte Messungen vorliegen.
Jedes Bild traegt den Hinweis "LAYOUT-VORSCHAU - BEISPIELDATEN". Sobald Daten
da sind, zeichnen dieselben Funktionen aus tastenakustik/plots.py die echten
Ergebnisse - dann ohne Stempel.

Aufruf:
    python werkzeuge/90_layoutvorschau.py
    python werkzeuge/90_layoutvorschau.py --zonen     # Sicherheitszonen einblenden
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

import numpy as np

from tastenakustik import beispiel, features, onset, plots, portrait
from tastenakustik.config import Config, TASTEN, AUSGABE

ORDNER = "_layout"
rng = np.random.default_rng(20260922)

def sichern(fig, name: str, zonen: bool) -> Path:
    if zonen:
        portrait.sicherheitszonen(fig)
    pfad = portrait.exportiere(fig, name, ORDNER)
    print(f"  {pfad.relative_to(AUSGABE.parent)}")
    return pfad


def main() -> int:
    p = argparse.ArgumentParser(description="Layout-Vorschau der Video-Grafiken")
    p.add_argument("--zonen", action="store_true",
                   help="Sicherheitszonen von Shorts / Reels / TikTok einblenden")
    args = p.parse_args()

    cfg = Config.laden()
    print(f"Hochformat {portrait.BREITE} x {portrait.HOEHE} nach "
          f"{AUSGABE / ORDNER}\n")

    # 02 Markierter Tastenschlag
    erste, zweite = TASTEN[0], TASTEN[min(1, len(TASTEN) - 1)]
    w, _ = beispiel.fenster(cfg, erste, rng=rng)
    fig = plots.wellenform_mit_onset(w, erste, onset.analysiere(w, cfg), cfg)
    plots.vorschau_stempel(fig)
    sichern(fig, "02_tastenschlag", args.zonen)

    # 03 Wellenform zu Log-Mel
    fig = plots.wellenform_zu_mel(beispiel.fenster(cfg, zweite, rng=rng)[0],
                                  zweite, cfg)
    plots.vorschau_stempel(fig)
    sichern(fig, "03_wellenform_zu_mel", args.zonen)

    # 04 Klassenvergleich - je Klasse ueber mehrere Proben gemittelt
    mittel = {}
    for taste in TASTEN:
        stapel = [
            features.log_mel(beispiel.fenster(cfg, taste,
                                              versatz_ms=rng.uniform(4, 10),
                                              rng=rng)[0],
                             cfg.samplerate, cfg.mel_nfft, cfg.mel_hop,
                             cfg.mel_baender, cfg.mel_fmin, cfg.mel_fmax)
            for _ in range(12)
        ]
        mittel[taste] = np.mean(stapel, axis=0)
    fig = plots.klassen_vergleich(mittel, cfg)
    plots.vorschau_stempel(fig)
    sichern(fig, "04_klassenvergleich", args.zonen)

    # 04b Die Klassen als Tafel - keine Messung, deshalb ohne Stempel
    fig = plots.klassen_tafel(f"{len(TASTEN)} Tasten")
    sichern(fig, "04b_klassentafel", args.zonen)

    # 05 Pipeline
    fig = plots.pipeline(cfg)
    sichern(fig, "05_pipeline", args.zonen)

    # 05b Was das Netz macht
    p_modell = beispiel.wahrscheinlichkeiten(erste, 0.90, rng)
    fig = plots.modell_erklaerung(p_modell)
    plots.vorschau_stempel(fig)
    sichern(fig, "05b_modell", args.zonen)

    # 06/07 Trainingsverlauf
    epochen = np.arange(1, 31)
    train_loss = 2.08 * np.exp(-epochen / 7.5) + 0.17 + rng.normal(0, 0.02, 30)
    val_loss = 2.08 * np.exp(-epochen / 8.5) + 0.46 + rng.normal(0, 0.05, 30)
    train_acc = np.clip(1 - 0.86 * np.exp(-epochen / 7.0) + rng.normal(0, .008, 30), 0, 1)
    val_acc = np.clip(1 - 0.94 * np.exp(-epochen / 8.5) + rng.normal(0, .02, 30), 0, 1)
    fig = plots.trainingsverlauf(epochen, train_loss, val_loss, train_acc, val_acc)
    plots.vorschau_stempel(fig)
    sichern(fig, "06_trainingsverlauf", args.zonen)

    # 08 Confusion Matrix
    n = len(TASTEN)
    matrix = np.zeros((n, n))
    for i in range(n):
        richtig = rng.integers(28, 38)
        matrix[i, i] = richtig
        rest = 40 - richtig
        for _ in range(rest):
            matrix[i, rng.choice([j for j in range(n) if j != i])] += 1
    fig = plots.konfusionsmatrix(matrix, float(np.trace(matrix) / matrix.sum()))
    plots.vorschau_stempel(fig)
    sichern(fig, "08_confusion", args.zonen)

    # 09 Einzelvorhersage
    p_dict = beispiel.wahrscheinlichkeiten(erste, 0.91, rng)
    fig = plots.vorhersage(erste, p_dict, sequenz=beispiel.beispiel_sequenz(1))
    plots.vorschau_stempel(fig)
    sichern(fig, "09_vorhersage", args.zonen)

    # 10 Blindtest mitten in der Sequenz
    p_dict = beispiel.wahrscheinlichkeiten(zweite, 0.84, rng)
    fig = plots.vorhersage(zweite, p_dict, sequenz=beispiel.beispiel_sequenz(8))
    plots.vorschau_stempel(fig)
    sichern(fig, "10_blindtest_sequenz", args.zonen)

    print("\nAlle Bilder sind 1080 x 1920 und tragen den Vorschau-Hinweis.")
    print("Sobald Daten vorliegen, zeichnen dieselben Funktionen die echten "
          "Ergebnisse - ohne Stempel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

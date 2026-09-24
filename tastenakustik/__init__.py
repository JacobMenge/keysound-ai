"""Tastenakustik - ein kontrolliertes Experiment zum Klang einzelner Tasten.

Umfang dieses Pakets (bewusst eng gehalten):
  * Aufnahme NUR innerhalb eines ausdruecklich gestarteten Recording-Modus
  * ausschliesslich die vorher festgelegten Klassen (siehe config.TASTEN)
  * jede gespeicherte Probe gehoert zu einem vorher angezeigten Prompt
  * kein globaler Tastatur-Hook, kein Mitschnitt normaler Texteingaben
  * kein Decoding freier Texte - das Modell kennt nur einzelne Anschlaege
"""

import sys

__version__ = "1.0.1"

# Unter Windows schreibt Python in eine umgeleitete Ausgabe (Datei, Pipe) mit
# der alten Codepage. Ein einziges Zeichen, das dort fehlt - ein Pfeil in der
# Sitzungsnotiz, das Zeichen fuer die Leertaste -, bricht sonst das ganze
# Werkzeug ab. Lieber ein Fragezeichen in der Ausgabe als ein Absturz.
for _strom in (sys.stdout, sys.stderr):
    try:
        _strom.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

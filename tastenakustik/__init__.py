"""Tastenakustik - ein kontrolliertes Experiment zum Klang einzelner Tasten.

Umfang dieses Pakets (bewusst eng gehalten):
  * Aufnahme NUR innerhalb eines ausdruecklich gestarteten Recording-Modus
  * ausschliesslich die vorher festgelegten Klassen (siehe config.TASTEN)
  * jede gespeicherte Probe gehoert zu einem vorher angezeigten Prompt
  * kein globaler Tastatur-Hook, kein Mitschnitt normaler Texteingaben
  * kein Decoding freier Texte - das Modell kennt nur einzelne Anschlaege
"""

__version__ = "1.0.0"

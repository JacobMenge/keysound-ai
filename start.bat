@echo off
REM Tastenakustik unter Windows starten - auch per Doppelklick.
REM Nimmt die venv im Projektordner, falls es eine gibt, sonst das
REM System-Python. Das Fenster bleibt bei einem Fehler offen.

cd /d "%~dp0"

set PYTHON=python
if exist ".venv\Scripts\python.exe" set PYTHON=.venv\Scripts\python.exe

"%PYTHON%" start.py
if errorlevel 1 goto ende
goto raus

:ende
echo.
echo Beendet. Fenster schliesst sich nicht von selbst.
pause

:raus

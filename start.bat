@echo off
REM Tastenakustik unter Windows starten - auch per Doppelklick.
REM Nimmt die venv im Projektordner, falls es eine gibt, sonst das
REM System-Python. Das Fenster bleibt bei einem Fehler offen.

cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    ".venv\Scripts\python.exe" start.py
    goto ende
)

python start.py
if errorlevel 1 goto ende
goto raus

:ende
echo.
echo Beendet. Fenster schliesst sich nicht von selbst.
pause

:raus

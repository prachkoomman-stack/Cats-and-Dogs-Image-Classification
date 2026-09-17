@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo The local Python environment was not found.
    echo Create it and install the requirements before launching the app.
    pause
    exit /b 1
)

start "Oxford Pets server" /b ".venv\Scripts\pythonw.exe" app.py
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:7860"

endlocal
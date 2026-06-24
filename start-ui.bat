@echo off
setlocal

cd /d "%~dp0"

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
set "UI_URL=http://127.0.0.1:8765"

if not exist "%VENV_PYTHON%" (
    echo Cannot find .venv\Scripts\python.exe.
    echo.
    echo Please run setup-venv.bat first, then double-click start-ui.bat again.
    echo.
    pause
    exit /b 1
)

echo Starting Remove Watermark UI...
echo.
echo Browser URL: %UI_URL%
echo.
echo Keep this window open while using the UI.
echo Press Ctrl+C in this window to stop the UI.
echo.

"%VENV_PYTHON%" -m remove_watermark.web

echo.
echo UI stopped.
echo.
pause

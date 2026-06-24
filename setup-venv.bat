@echo off
setlocal

cd /d "%~dp0"

set "EXTRA_ARGS="

if not "%~1"=="" (
    goto run_setup
)

echo Select setup profile:
echo   1. Basic environment, no SAM3
echo   2. Basic environment plus SAM3 tools, no model download
echo   3. Basic environment plus SAM3 tools and SAM 3.1 model download
echo.
set /p "SETUP_PROFILE=Choose 1-3 [1]: "
if "%SETUP_PROFILE%"=="" set "SETUP_PROFILE=1"

if "%SETUP_PROFILE%"=="1" (
    set "EXTRA_ARGS="
) else if "%SETUP_PROFILE%"=="2" (
    set "EXTRA_ARGS=-InstallAiTools"
) else if "%SETUP_PROFILE%"=="3" (
    set "EXTRA_ARGS=-InstallAiTools -DownloadAiModels"
) else (
    echo Invalid setup profile: %SETUP_PROFILE%
    echo.
    pause
    exit /b 1
)

:run_setup
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup-venv.ps1" %EXTRA_ARGS% %*

echo.
if errorlevel 1 (
    echo Setup failed. See the messages above.
) else (
    echo Setup completed successfully.
)
echo.
pause

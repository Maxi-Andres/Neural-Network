@echo off
:: Creates a local virtual environment (.venv) and installs everything the
:: notebooks need. Only Python 3 has to be installed beforehand.
setlocal
cd /d "%~dp0"

set "PY="
py -3 --version >nul 2>&1
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    python --version >nul 2>&1
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    echo Python 3 was not found.
    echo Install it from https://www.python.org/downloads/ and check
    echo "Add python.exe to PATH" during the installation, then run this again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo Could not create the virtual environment.
        pause
        exit /b 1
    )
)

echo Installing dependencies, this can take a few minutes ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Installation failed. Check the messages above.
    pause
    exit /b 1
)

echo.
echo Done! Double-click run.bat to open the notebook.
pause

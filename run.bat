@echo off
:: Opens the main notebook in Jupyter using the local virtual environment.
:: Runs install.bat first if the environment does not exist yet.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    call install.bat
    if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" -m notebook Neural_Network.ipynb

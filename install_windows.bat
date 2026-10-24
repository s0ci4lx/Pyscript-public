@echo off
setlocal
cd /d "%~dp0"
set "CCTV_PROFILE=vehicle"
set "PYTHONUTF8=1"
echo Installing CCTV Vehicle and LPR for NVIDIA GPU...
py -3.11 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
if errorlevel 1 (
    echo Install Python 3.11 64-bit from python.org, including Python Launcher.
    echo Then extract the ZIP and run this file again.
    pause
    exit /b 1
)
py -3.11 bootstrap.py --profile vehicle
if errorlevel 1 (
    echo Installation failed. Read the error above. Do not start the application yet.
    pause
    exit /b 1
)
echo Installation and NVIDIA GPU checks completed.
echo Open start_windows.bat to run. Start Satun-AI-Cam separately before selecting its cameras.
pause

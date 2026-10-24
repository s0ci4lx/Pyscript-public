@echo off
setlocal
cd /d "%~dp0"
set "CCTV_PROFILE=vehicle"
set "PYTHONUTF8=1"
echo =====================================
echo    Starting CCTV AI System...
echo =====================================

echo Preparing isolated environment...
py -3.11 bootstrap.py --profile vehicle
if errorlevel 1 goto :error
.venv\Scripts\python.exe check_ports.py
if errorlevel 1 goto :error

echo [1/2] Starting Backend (Port 8081)...
start "CCTV Backend" cmd /k ".venv\Scripts\python.exe rtsp_yolo_backend.py"

echo [2/2] Starting Frontend (Port 8000)...
start "CCTV Frontend" cmd /k ".venv\Scripts\python.exe serve_frontend.py"

echo Opening Web Browser...
.venv\Scripts\python.exe wait_for_backend.py
if errorlevel 1 goto :error
start http://localhost:8000

echo =====================================
echo System is running! (Two windows opened in background)
echo Close the two command prompt windows to stop the servers.
echo =====================================
pause
goto :eof

:error
echo Failed to prepare the Python environment. See the error above.
pause
exit /b 1

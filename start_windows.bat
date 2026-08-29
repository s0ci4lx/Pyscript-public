@echo off
echo =====================================
echo    Starting CCTV AI System...
echo =====================================

echo Checking and installing required packages from requirements.txt...
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install -r requirements.txt

echo [1/2] Starting Backend (Port 8081)...
start cmd /k "python rtsp_yolo_backend.py"

echo [2/2] Starting Frontend (Port 8000)...
start cmd /k "python -m http.server 8000"

echo Opening Web Browser...
timeout /t 2 >nul
start http://localhost:8000

echo =====================================
echo System is running! (Two windows opened in background)
echo Close the two command prompt windows to stop the servers.
echo =====================================
pause

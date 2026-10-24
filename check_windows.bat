@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist ".venv\Scripts\python.exe" (
    echo Run install_windows.bat first.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m pip check
if errorlevel 1 goto :error
.venv\Scripts\python.exe check_windows_gpu.py
if errorlevel 1 goto :error
.venv\Scripts\python.exe -c "from satun_bridge import list_cameras; print('Satun cameras:',len(list_cameras()))"
if errorlevel 1 echo Satun is not reachable. Start Satun-AI-Cam on this PC and check its port.
pause
exit /b 0
:error
echo Check failed. Read the error above.
pause
exit /b 1

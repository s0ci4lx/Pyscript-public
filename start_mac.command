#!/bin/bash
cd "$(dirname "$0")"

# เมื่อปิดหน้าต่าง Terminal ให้ปิดโปรเซสทั้งหมดด้วย
trap "kill 0" EXIT

echo "====================================="
echo "   🚀 Starting CCTV AI System..."
echo "====================================="

echo "[0/2] Cleaning up old processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8081 | xargs kill -9 2>/dev/null
sleep 1

echo "Checking and installing required packages from requirements.txt..."
python3 -m pip install --upgrade pip wheel "setuptools<81"
python3 -m pip install -r requirements.txt

export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "[1/2] Starting Backend (Port 8081)..."
while true; do
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python3 rtsp_yolo_backend.py
    echo "Backend process exited, restarting in 1s..."
    sleep 1
done &

echo "[2/2] Starting Frontend (Port 8000)..."
python3 -m http.server 8000 &

echo "Opening Web Browser..."
sleep 2
open http://localhost:8000

echo "====================================="
echo "✅ System is running!"
echo "❌ ปิดหน้าต่างนี้ (หรือกด Ctrl+C) เพื่อหยุดการทำงานทั้งหมด"
echo "====================================="
wait

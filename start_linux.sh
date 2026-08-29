#!/bin/bash
cd "$(dirname "$0")"

# เมื่อปิดหน้าต่างหรือกด Ctrl+C ให้หยุดโปรเซสทั้งหมด
trap "kill 0" EXIT

echo "====================================="
echo "   🚀 Starting CCTV AI System (Linux)"
echo "====================================="

echo "[0/2] Cleaning up old processes on ports 8000 & 8081..."
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8081/tcp 2>/dev/null || true
sleep 1

echo "Checking and installing required packages from requirements.txt..."
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt

echo "[1/2] Starting Backend (Port 8081)..."
python3 rtsp_yolo_backend.py &

echo "[2/2] Starting Frontend (Port 8000)..."
python3 -m http.server 8000 &

echo "Opening Web Browser..."
sleep 2
if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:8000 &
fi

echo "====================================="
echo "✅ System is running!"
echo "📍 Frontend: http://localhost:8000"
echo "📍 Backend:  http://localhost:8081"
echo "❌ กด Ctrl+C เพื่อหยุดการทำงานทั้งหมด"
echo "====================================="
wait

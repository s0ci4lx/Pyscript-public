#!/bin/bash
cd "$(dirname "$0")"

cleanup() {
    [ -n "${BACKEND_PID:-}" ] && kill "$BACKEND_PID" 2>/dev/null || true
    [ -n "${FRONTEND_PID:-}" ] && kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "====================================="
echo "   🚀 Starting CCTV AI System..."
echo "====================================="

echo "🧹 ตรวจสอบและปิดโปรเซสเดิมที่ค้างอยู่..."
for port in 8081 8000; do
    pids=$(lsof -ti :$port 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "   - เคลียร์พอร์ต $port (PID: $pids)"
        echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
done
pkill -f "rtsp_yolo_backend.py" 2>/dev/null || true
pkill -f "serve_frontend.py" 2>/dev/null || true
sleep 0.5

echo "[0/2] Preparing isolated environment..."
python3 bootstrap.py || exit 1
PYTHON_BIN=".venv/bin/python"
"$PYTHON_BIN" check_ports.py || exit 1

export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "[1/2] Starting Backend (Port 8081)..."
"$PYTHON_BIN" rtsp_yolo_backend.py &
BACKEND_PID=$!

echo "[2/2] Starting Frontend (Port 8000)..."
"$PYTHON_BIN" serve_frontend.py &
FRONTEND_PID=$!

echo "Opening Web Browser..."
"$PYTHON_BIN" wait_for_backend.py || exit 1
open http://localhost:8000

echo "====================================="
echo "✅ System is running!"
echo "❌ ปิดหน้าต่างนี้ (หรือกด Ctrl+C) เพื่อหยุดการทำงานทั้งหมด"
echo "====================================="
wait

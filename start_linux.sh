#!/bin/bash
cd "$(dirname "$0")"

cleanup() {
    [ -n "${BACKEND_PID:-}" ] && kill "$BACKEND_PID" 2>/dev/null || true
    [ -n "${FRONTEND_PID:-}" ] && kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "====================================="
echo "   🚀 Starting CCTV AI System (Linux)"
echo "====================================="

echo "[0/2] Preparing isolated environment..."
python3 bootstrap.py || exit 1
PYTHON_BIN=".venv/bin/python"
"$PYTHON_BIN" check_ports.py || exit 1

echo "[1/2] Starting Backend (Port 8081)..."
"$PYTHON_BIN" rtsp_yolo_backend.py &
BACKEND_PID=$!

echo "[2/2] Starting Frontend (Port 8000)..."
"$PYTHON_BIN" serve_frontend.py &
FRONTEND_PID=$!

echo "Opening Web Browser..."
"$PYTHON_BIN" wait_for_backend.py || exit 1
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

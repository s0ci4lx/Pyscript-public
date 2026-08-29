import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import time
import numpy as np
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, Body, Query, Request

from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import face_recognition
import threading
import torch

if torch.cuda.is_available():
    DEVICE = "0"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"🔥 [Backend] Using acceleration device: {DEVICE}")
import json
import re
import sqlite3
import urllib.parse
import yt_dlp
import mss
try:
    from sklearn.neighbors import KNeighborsClassifier
except ImportError:
    pass

# กำหนดฐานข้อมูล SQLite
DB_FILE = 'history.db'

def get_db_connection():
    conn = sqlite3.connect(DB_FILE, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
    return conn

knn_color_model = None

def train_color_model():
    global knn_color_model
    try:
        from sklearn.neighbors import KNeighborsClassifier
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT h, s, v, correct_color FROM color_training")
            rows = cursor.fetchall()
            
            if len(rows) >= 3:
                X = [[r[0], r[1], r[2]] for r in rows]
                y = [r[3] for r in rows]
                knn = KNeighborsClassifier(n_neighbors=min(3, len(rows)))
                knn.fit(X, y)
                knn_color_model = knn
                print(f"[ML] Color model trained with {len(rows)} samples.")
            else:
                knn_color_model = None
                print(f"[ML] Not enough data to train color model (need >=3, got {len(rows)}).")
    except Exception as e:
        print(f"[ML] Failed to train color model: {e}")
        knn_color_model = None

def init_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_url TEXT,
                name TEXT,
                timestamp DATETIME,
                confidence INTEGER,
                img_url TEXT,
                full_url TEXT,
                is_color_edited INTEGER DEFAULT 0,
                plate_img_url TEXT,
                license_plate TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS color_training (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                h INTEGER,
                s INTEGER,
                v INTEGER,
                correct_color TEXT,
                timestamp DATETIME
            )
        ''')
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hits_timestamp ON hits(timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hits_name ON hits(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hits_camera_url ON hits(camera_url)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hits_license_plate ON hits(license_plate)")
        
        # DB Migration: Check if is_color_edited exists and add it if not
        cursor.execute("PRAGMA table_info(hits)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'is_color_edited' not in columns:
            cursor.execute("ALTER TABLE hits ADD COLUMN is_color_edited INTEGER DEFAULT 0")
        if 'plate_img_url' not in columns:
            cursor.execute("ALTER TABLE hits ADD COLUMN plate_img_url TEXT")
        if 'license_plate' not in columns:
            cursor.execute("ALTER TABLE hits ADD COLUMN license_plate TEXT")
        conn.commit()

init_db()

# Initialize model on startup
train_color_model()
from PIL import Image, ImageDraw, ImageFont

app = FastAPI()
DEFAULT_CORS_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8081",
    "http://127.0.0.1:8081",
]
allowed_cors_origins = [
    origin.strip()
    for origin in os.getenv("CCTV_ALLOWED_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS)).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;2000000"

os.makedirs('targets', exist_ok=True)
os.makedirs('alerts', exist_ok=True)
app.mount("/alerts_img", StaticFiles(directory="alerts"), name="alerts_img")
os.makedirs("targets", exist_ok=True)
app.mount("/targets_img", StaticFiles(directory="targets"), name="targets_img")
os.makedirs("uploads", exist_ok=True)
os.makedirs("vehicle_targets", exist_ok=True)
app.mount("/vehicle_targets_img", StaticFiles(directory="vehicle_targets"), name="vehicle_targets_img")

CONFIG_FILE = "config.json"
ALLOWED_YOLO_MODELS = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8s-world.pt"}
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_VIDEO_UPLOAD_BYTES = int(os.getenv("CCTV_MAX_VIDEO_UPLOAD_MB", "1024")) * 1024 * 1024
MAX_IMAGE_UPLOAD_BYTES = int(os.getenv("CCTV_MAX_IMAGE_UPLOAD_MB", "20")) * 1024 * 1024
MAX_MULTI_IMAGE_UPLOAD_BYTES = int(os.getenv("CCTV_MAX_MULTI_IMAGE_UPLOAD_MB", "200")) * 1024 * 1024

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {"yolo_model": "yolov8n.pt"}

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def mask_secret(value):
    value = str(value or "").strip()
    if not value:
        return ""
    return f"***{value[-4:]}" if len(value) > 4 else "***"

def is_loopback_client(request):
    host = request.client.host if request.client else ""
    return host in {"127.0.0.1", "::1", "localhost"}

def get_api_key():
    return str(os.getenv("CCTV_API_KEY") or config.get("api_key", "")).strip()

def request_origin_allowed(request):
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")
    candidate = origin or referer
    if not candidate:
        return True
    parsed = urllib.parse.urlparse(candidate)
    request_origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else candidate.rstrip("/")
    return request_origin in allowed_cors_origins

@app.middleware("http")
async def require_api_key_for_remote_writes(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        if not request_origin_allowed(request):
            return JSONResponse(status_code=403, content={"status": "error", "message": "Forbidden origin"})
        if not is_loopback_client(request):
            api_key = get_api_key()
            if not api_key or request.headers.get("X-API-Key") != api_key:
                return JSONResponse(status_code=401, content={"status": "error", "message": "Unauthorized"})
    return await call_next(request)

def sanitized_stem(value, default="item", max_len=80):
    stem = os.path.basename(str(value or "").strip())
    stem = os.path.splitext(stem)[0]
    stem = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", stem).strip(" ._")
    return (stem or default)[:max_len]

def sanitized_text(value, default="", max_len=160):
    text = str(value or "").strip()
    text = re.sub(r"[\x00-\x1f<>]", "", text)
    return (text or default)[:max_len]

def validate_camera_source(source):
    source = str(source or "").strip()
    if not source:
        return False
    if source.isdigit():
        return True
    if re.fullmatch(r"screen(?::\d+)?", source, flags=re.IGNORECASE):
        return True
    parsed = urllib.parse.urlparse(source)
    return parsed.scheme.lower() in {"rtsp", "rtsps", "http", "https"} and bool(parsed.netloc)

def validate_youtube_url(url):
    parsed = urllib.parse.urlparse(str(url or "").strip())
    host = parsed.netloc.lower().split(":")[0]
    allowed_hosts = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}
    return parsed.scheme in {"http", "https"} and host in allowed_hosts

async def save_upload_limited(upload_file, filepath, max_bytes):
    total = 0
    with open(filepath, "wb") as buffer:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                buffer.close()
                try:
                    os.remove(filepath)
                except Exception:
                    pass
                raise ValueError(f"File too large. Limit is {max_bytes // (1024 * 1024)} MB")
            buffer.write(chunk)
    return total

config = load_config()
current_yolo_model = config.get("yolo_model", "yolov8n.pt")
if current_yolo_model not in ALLOWED_YOLO_MODELS:
    print(f"[Security] Unsupported YOLO model in config: {current_yolo_model}; falling back to yolov8n.pt")
    current_yolo_model = "yolov8n.pt"
    config["yolo_model"] = current_yolo_model
    save_config(config)
save_unknown_faces = config.get("save_unknown_faces", False)
show_only_targets = config.get("show_only_targets", False)
min_confidence = config.get("min_confidence", 50)
# Minimum face similarity (%) required to accept a face as a known target.
# similarity% = (1 - face_distance) * 100, so 55% == a face_distance of 0.45.
# Higher = stricter (fewer false matches). Adjustable from the UI.
face_match_threshold = config.get("face_match_threshold", 55)
# A candidate must beat the best *different-person* match by at least this face_distance
# margin, otherwise the frame is treated as ambiguous and rejected. This stops one real
# person from flip-flopping between several enrolled targets across frames.
FACE_MATCH_MARGIN = 0.04
# Minimum appearance similarity (%) to confirm a detected vehicle is a specific enrolled
# "this-car-only" target, via the VeRi-776 Re-ID model (cosine). In testing same car scores
# ~95% (across viewpoints) and different cars ~63%, so 80 sits in the clean gap. Adjustable.
vehicle_match_threshold = config.get("vehicle_match_threshold", 80)
enable_lpr = config.get("enable_lpr", False)

# Telegram Alert Settings
telegram_enabled = config.get("telegram_enabled", False)
telegram_bot_token = config.get("telegram_bot_token", "")
telegram_chat_id = config.get("telegram_chat_id", "")
telegram_cooldown = config.get("telegram_cooldown", 30)
telegram_notify_faces = config.get("telegram_notify_faces", True)
telegram_notify_vehicles = config.get("telegram_notify_vehicles", True)
telegram_notify_plates = config.get("telegram_notify_plates", True)

telegram_last_sent = {} # key -> timestamp (float)
telegram_lock = threading.Lock()

def find_target_face_image_path(base_target_name):
    if not os.path.exists("targets"): return None
    clean_target = str(base_target_name).strip()
    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
        p = os.path.join("targets", f"{clean_target}{ext}")
        if os.path.exists(p): return p
    for fn in os.listdir("targets"):
        if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            name_part = os.path.splitext(fn)[0]
            base_name = re.sub(r'_[0-9]+$', '', name_part)
            if base_name == clean_target or name_part == clean_target or clean_target in name_part:
                return os.path.join("targets", fn)
    return None

def find_target_vehicle_image_path(vehicle_name):
    if not os.path.exists("vehicle_targets"): return None
    clean_name = str(vehicle_name).strip()
    try:
        with vehicle_targets_lock:
            for vt in vehicle_targets:
                raw = str(vt.get("raw", "")).strip()
                fn = vt.get("filename")
                if fn and (raw == clean_name or clean_name in raw or fn == clean_name):
                    p = os.path.join("vehicle_targets", fn)
                    if os.path.exists(p): return p
    except Exception:
        pass
    return None

def rounded_rect(img, pt1, pt2, color, radius=18, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if radius == 0:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)

def build_telegram_comparison_card(target_type, target_name, match_pct, camera_name, live_crop_img, target_img_path=None, plate_text=None):
    card_w, card_h = 820, 700
    canvas = np.zeros((card_h, card_w, 3), dtype=np.uint8)
    canvas[:] = (46, 27, 15)

    safe_target = telegram_image_text(target_name, max_len=34)
    safe_camera = telegram_image_text(camera_name, max_len=22)
    safe_plate = telegram_image_text(plate_text, max_len=24) if plate_text else ""
    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    score_color = (114, 186, 86) if match_pct >= 80 else (210, 145, 42) if match_pct >= 60 else (41, 33, 178)

    rounded_rect(canvas, (20, 20), (800, 680), (48, 29, 16), radius=24)
    rounded_rect(canvas, (20, 20), (800, 680), (122, 82, 54), radius=24, thickness=2)
    rounded_rect(canvas, (44, 44), (776, 116), (66, 45, 31), radius=16)
    rounded_rect(canvas, (620, 60), (752, 100), score_color, radius=20)
    canvas = put_thai_text(canvas, "ตรวจพบเป้าหมาย", (68, 60), (250, 252, 255), font_size=28)
    canvas = put_thai_text(canvas, f"{match_pct}%", (663, 65), (255, 255, 255), font_size=27)

    chip_y = 140
    rounded_rect(canvas, (44, chip_y), (304, chip_y + 44), (54, 31, 15), radius=22, thickness=-1)
    rounded_rect(canvas, (320, chip_y), (588, chip_y + 44), (54, 31, 15), radius=22, thickness=-1)
    rounded_rect(canvas, (604, chip_y), (776, chip_y + 44), (54, 31, 15), radius=22, thickness=-1)
    rounded_rect(canvas, (44, chip_y), (304, chip_y + 44), (155, 102, 66), radius=22, thickness=1)
    rounded_rect(canvas, (320, chip_y), (588, chip_y + 44), (155, 102, 66), radius=22, thickness=1)
    rounded_rect(canvas, (604, chip_y), (776, chip_y + 44), (155, 102, 66), radius=22, thickness=1)
    canvas = put_thai_text(canvas, f"เป้าหมาย: {safe_target}", (60, chip_y + 10), (220, 230, 245), font_size=17)
    canvas = put_thai_text(canvas, f"กล้อง: {safe_camera}", (336, chip_y + 10), (220, 230, 245), font_size=17)
    canvas = put_thai_text(canvas, f"เวลา: {ts_now[-8:]}", (620, chip_y + 10), (220, 230, 245), font_size=17)

    img_y1, img_y2 = 260, 565
    box_h = img_y2 - img_y1
    box_w = 350

    left_x1, left_x2 = 44, 44 + box_w
    right_x1, right_x2 = 426, 426 + box_w

    canvas = put_thai_text(canvas, "ภาพเป้าหมาย", (left_x1 + 16, img_y1 - 34), (255, 160, 70), font_size=21)
    canvas = put_thai_text(canvas, "ภาพตรวจพบ", (right_x1 + 16, img_y1 - 34), (0, 184, 255), font_size=21)
    rounded_rect(canvas, (left_x1, img_y1), (left_x2, img_y2), (66, 48, 36), radius=16)
    rounded_rect(canvas, (right_x1, img_y1), (right_x2, img_y2), (66, 48, 36), radius=16)
    rounded_rect(canvas, (left_x1, img_y1), (left_x2, img_y2), (148, 103, 71), radius=16, thickness=1)
    rounded_rect(canvas, (right_x1, img_y1), (right_x2, img_y2), (148, 103, 71), radius=16, thickness=1)

    target_loaded = None
    if target_img_path and os.path.exists(target_img_path):
        try:
            target_loaded = cv2.imread(target_img_path)
        except Exception:
            target_loaded = None
            
    if target_loaded is not None and target_loaded.size > 0:
        th, tw = target_loaded.shape[:2]
        scale = min((box_w - 30) / tw, (box_h - 30) / th)
        nw, nh = max(1, int(tw * scale)), max(1, int(th * scale))
        resized_target = cv2.resize(target_loaded, (nw, nh), interpolation=cv2.INTER_AREA)
        off_x = left_x1 + (box_w - nw) // 2
        off_y = img_y1 + (box_h - nh) // 2
        canvas[off_y:off_y+nh, off_x:off_x+nw] = resized_target
    else:
        canvas = put_thai_text(canvas, "ไม่มีภาพอ้างอิง", (left_x1 + 88, img_y1 + 132), (185, 160, 145), font_size=23)
        if safe_plate:
            canvas = put_thai_text(canvas, f"ทะเบียน: {safe_plate}", (left_x1 + 84, img_y1 + 168), (240, 205, 35), font_size=21)

    if live_crop_img is not None and live_crop_img.size > 0:
        lh, lw = live_crop_img.shape[:2]
        scale = min((box_w - 30) / lw, (box_h - 30) / lh)
        nw, nh = max(1, int(lw * scale)), max(1, int(lh * scale))
        resized_live = cv2.resize(live_crop_img, (nw, nh), interpolation=cv2.INTER_AREA)
        off_x = right_x1 + (box_w - nw) // 2
        off_y = img_y1 + (box_h - nh) // 2
        canvas[off_y:off_y+nh, off_x:off_x+nw] = resized_live
    else:
        canvas = put_thai_text(canvas, "ไม่มีภาพจากกล้อง", (right_x1 + 82, img_y1 + 132), (185, 160, 145), font_size=23)

    rounded_rect(canvas, (44, 600), (776, 654), (66, 45, 31), radius=14)
    note = "ความเหมือนต่ำกว่า 60%: ควรตรวจสอบด้วยสายตาอีกครั้ง" if match_pct < 60 else f"ตรวจพบเมื่อ {ts_now}"
    canvas = put_thai_text(canvas, telegram_image_text(note, max_len=62), (68, 615), (38, 202, 255) if match_pct < 60 else (235, 218, 205), font_size=22)

    return canvas

def send_telegram_alert_async(target_type, target_name, match_pct, camera_name, live_crop_img, target_img_path=None, plate_text=None):
    if not config.get("telegram_enabled", False):
        return
        
    bot_token = str(config.get("telegram_bot_token", "")).strip()
    chat_id = str(config.get("telegram_chat_id", "")).strip()
    if not bot_token or not chat_id:
        return
        
    if target_type == "face" and not config.get("telegram_notify_faces", True):
        return
    if target_type == "vehicle" and not config.get("telegram_notify_vehicles", True):
        return
    if target_type == "plate" and not config.get("telegram_notify_plates", True):
        return
        
    cooldown = int(config.get("telegram_cooldown", 30))
    dedup_key = f"{target_type}:{target_name}:{camera_name}"
    now_ts = time.time()
    
    with telegram_lock:
        last_ts = telegram_last_sent.get(dedup_key, 0.0)
        if now_ts - last_ts < cooldown:
            return
        telegram_last_sent[dedup_key] = now_ts

    crop_copy = live_crop_img.copy() if (live_crop_img is not None and live_crop_img.size > 0) else None

    def _worker():
        try:
            comparison_img = build_telegram_comparison_card(
                target_type=target_type,
                target_name=target_name,
                match_pct=match_pct,
                camera_name=camera_name,
                live_crop_img=crop_copy,
                target_img_path=target_img_path,
                plate_text=plate_text
            )
            
            ok, buf = cv2.imencode(".jpg", comparison_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                return
                
            ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            icon = "🧑" if target_type == "face" else "🚗" if target_type == "vehicle" else "🔤"
            caption = (
                f"🚨 แจ้งเตือนตรวจพบเป้าหมาย! {icon}\n"
                f"🎯 เป้าหมาย: {telegram_caption_text(target_name, max_len=48)}\n"
                f"📊 ความเหมือน: {match_pct}%\n"
                f"📹 กล้อง: {telegram_caption_text(camera_name, max_len=28)}\n"
                f"⏰ เวลา: {ts_str}\n"
            )
            
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            files = {"photo": ("alert_comparison.jpg", buf.tobytes(), "image/jpeg")}
            data = {"chat_id": chat_id, "caption": caption}
            
            import requests
            resp = requests.post(url, data=data, files=files, timeout=10)
            if resp.status_code == 200:
                print(f"✈️ [Telegram] ส่งการแจ้งเตือนสำเร็จ: {target_name} ({match_pct}%)")
            else:
                print(f"⚠️ [Telegram Error]: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"⚠️ [Telegram Exception]: {e}")

    threading.Thread(target=_worker, daemon=True).start()

yolo_world_classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
custom_search_terms = []

try:
    model = YOLO(current_yolo_model).to(DEVICE)
    if "world" in current_yolo_model:
        model.set_classes(yolo_world_classes + custom_search_terms)
except Exception as e:
    print("Error loading YOLO:", e)
    model = None

# Globals
known_face_encodings = []
known_face_names = []
known_face_cache = {}
targets_lock = threading.Lock()
vehicle_targets_lock = threading.Lock()
plate_targets_lock = threading.Lock()
tracked_ids = set()
tracked_target_ids = {} # track_id -> target name
track_last_seen = {} # track_id -> timestamp (float)
track_history = {} # track_id -> (cx, cy)
track_lpr_attempts = {} # track_id -> int
track_lpr_last_time = {} # track_id -> float
track_lpr_results = {} # track_id -> dict
track_lpr_max_size = {} # track_id -> int (width * height)
track_lpr_logged = set() # track_ids that have been logged as LPR hits
detection_zone = None
crossing_line = None
crossing_direction = "any"
TRACK_TTL_SECONDS = 60

def prune_stale_tracks(now=None):
    """Drop track state for ids not seen within TRACK_TTL_SECONDS."""
    if now is None:
        now = time.time()
    stale = [tid for tid, seen in track_last_seen.items() if now - seen > TRACK_TTL_SECONDS]
    for tid in stale:
        track_last_seen.pop(tid, None)
        track_history.pop(tid, None)
        tracked_target_ids.pop(tid, None)
        tracked_ids.discard(tid)
        track_lpr_attempts.pop(tid, None)
        track_lpr_last_time.pop(tid, None)
        track_lpr_results.pop(tid, None)
        track_lpr_max_size.pop(tid, None)
        track_lpr_logged.discard(tid)

line_cross_count = 0
show_detection_zone = config.get("show_detection_zone", True)
unknown_persons_count = 0
latest_hits = {} # Changed to dictionary to separate by camera
last_alert_time = {}
target_hit_counts = {} # cam_url -> {name: count}

def load_latest_hits_from_db():
    global latest_hits, target_hit_counts
    try:
        if not os.path.exists(DB_FILE):
            return
        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 1. Load latest 10 hits per camera_url
            cursor.execute('''
                SELECT id, camera_url, name, timestamp, confidence, img_url, full_url, is_color_edited, plate_img_url, license_plate
                FROM hits
                ORDER BY timestamp DESC
            ''')
            rows = cursor.fetchall()
            for row in rows:
                cam_url = row["camera_url"]
                if not cam_url:
                    continue
                if cam_url not in latest_hits:
                    latest_hits[cam_url] = []
                if len(latest_hits[cam_url]) < 10:
                    ts = row["timestamp"]
                    time_str = ts.split(" ")[1] if (ts and " " in ts) else ts
                    hit_info = {
                        "id": row["id"],
                        "name": row["name"],
                        "time": time_str,
                        "confidence": row["confidence"],
                        "img_url": row["img_url"],
                        "full_url": row["full_url"],
                        "is_color_edited": row["is_color_edited"] if "is_color_edited" in row.keys() else 0,
                        "plate_img_url": row["plate_img_url"] if "plate_img_url" in row.keys() else None,
                        "license_plate": row["license_plate"] if "license_plate" in row.keys() else None
                    }
                    latest_hits[cam_url].append(hit_info)
            
            # 2. Load target hit counts
            cursor.execute('''
                SELECT camera_url, name, COUNT(*) as count 
                FROM hits 
                WHERE name != 'บุคคลทั่วไป' AND name NOT LIKE '[LPR]%'
                GROUP BY camera_url, name
            ''')
            count_rows = cursor.fetchall()
            for r in count_rows:
                c_url = r["camera_url"]
                t_name = r["name"]
                t_count = r["count"]
                
                base_name = t_name.split(" (#")[0] if " (#" in t_name else t_name
                if c_url not in target_hit_counts:
                    target_hit_counts[c_url] = {}
                target_hit_counts[c_url][base_name] = target_hit_counts[c_url].get(base_name, 0) + t_count
                
            print(f"[DB] Loaded latest hits for {len(latest_hits)} cameras, target stats for {len(target_hit_counts)} cameras.")
    except Exception as e:
        print(f"[DB] Error loading latest hits from DB: {e}")

load_latest_hits_from_db()

CAMERAS_FILE = "cameras.json"
DEFAULT_CAMERAS = [
    {"id": "webcam", "name": "Webcam (กล้องคอมพิวเตอร์)", "url": "0"},
    {"id": "screen1", "name": "แคปหน้าจอ (Monitor 1)", "url": "screen:1"},
    {"id": "cctv_sample", "name": "RTSP Sample Stream", "url": "rtsp://admin:password@192.168.1.100:554/stream1"}
]

def load_cameras():
    if not os.path.exists(CAMERAS_FILE):
        with open(CAMERAS_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CAMERAS, f, indent=4, ensure_ascii=False)
        return DEFAULT_CAMERAS
    try:
        with open(CAMERAS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return DEFAULT_CAMERAS

def save_cameras(cameras):
    with open(CAMERAS_FILE, "w", encoding="utf-8") as f:
        json.dump(cameras, f, indent=4, ensure_ascii=False)

init_cameras = load_cameras()
RTSP_URL = init_cameras[0]["url"] if init_cameras else "0"
RESTART_STREAM = False
STOP_STREAM = False
PAUSE_STREAM = True
current_fps = 0.0
current_frame_skip = 0
save_unknown_faces = config.get("save_unknown_faces", False)

# Playback progress for local video-file sources (0 / unknown for live cameras).
is_video_source = False        # True when the active source is a seekable video file
video_native_fps = 0.0         # frames-per-second reported by the file
video_total_frames = 0         # total frames in the file (0 if unknown)
video_current_frame = 0        # last decoded frame index
seek_request = None            # target frame index requested by the UI, applied in the loop

global_frame = None
global_raw_frame = None
current_cap = None
global_frame_lock = threading.Lock()
inference_lock = threading.Lock()

def load_known_faces():
    global known_face_encodings, known_face_names, known_face_cache
    temp_encodings = []
    temp_names = []
    new_cache = {}
    
    if not os.path.exists('targets'):
        os.makedirs('targets', exist_ok=True)
        
    for filename in os.listdir('targets'):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            path = os.path.join('targets', filename)
            # If it's already in our cache, reuse it
            if filename in known_face_cache:
                entry = known_face_cache[filename]
                new_cache[filename] = entry
                temp_encodings.append(entry["encoding"])
                temp_names.append(entry["base_name"])
                continue
                
            try:
                img = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    encoding = encodings[0]
                    name = os.path.splitext(filename)[0]
                    base_name = re.sub(r'_[0-9]+$', '', name)
                    new_cache[filename] = {"encoding": encoding, "base_name": base_name}
                    temp_encodings.append(encoding)
                    temp_names.append(base_name)
                    print(f"✅ โหลดใบหน้าเป้าหมาย: {name} (กลุ่ม: {base_name})")
            except Exception as e:
                print(f"❌ โหลด {filename} ไม่ได้: {e}")
                
    with targets_lock:
        known_face_encodings = temp_encodings
        known_face_names = temp_names
        known_face_cache = new_cache

def append_known_face(filepath):
    """Encode a single target image and append it to the in-memory face DB.

    This is the incremental alternative to load_known_faces(): adding one target no
    longer re-encodes every existing target (which froze the UI once the DB grew
    large). Returns True only if a usable face was found in the image."""
    global known_face_cache, known_face_encodings, known_face_names
    try:
        img = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(img)
        if not encodings:
            return False
        filename = os.path.basename(filepath)
        name = os.path.splitext(filename)[0]
        base_name = re.sub(r'_[0-9]+$', '', name)
        encoding = encodings[0]
        with targets_lock:
            known_face_encodings.append(encoding)
            known_face_names.append(base_name)
            known_face_cache[filename] = {"encoding": encoding, "base_name": base_name}
        print(f"✅ เพิ่มใบหน้าเป้าหมาย (incremental): {name} (กลุ่ม: {base_name})")
        return True
    except Exception as e:
        print(f"❌ เพิ่มใบหน้า {filepath} ไม่ได้: {e}")
        return False

load_known_faces()

# ---- Vehicle appearance embedding: dedicated Vehicle Re-ID model ----
# ResNet-34 trained on VeRi-776 (vehicle re-id dataset), run via onnxruntime. Purpose-built
# to tell *which* vehicle it is (viewpoint/lighting robust), unlike a generic ImageNet CNN.
# Lazily initialised; the app still runs (type+color only) if the model/runtime is missing.
VEHICLE_REID_MODEL = os.path.join("models", "resnet34_veri776.onnx")
VEHICLE_EMBED_DIM = 512   # output dim of the VeRi model; used to invalidate stale embeddings
_REID_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_REID_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_vehicle_session = None
_vehicle_input_name = None
_vehicle_embedder_failed = False

def _get_vehicle_embedder():
    global _vehicle_session, _vehicle_input_name, _vehicle_embedder_failed
    if _vehicle_session is not None or _vehicle_embedder_failed:
        return _vehicle_session
    try:
        import onnxruntime as ort
        if not os.path.exists(VEHICLE_REID_MODEL):
            raise FileNotFoundError(VEHICLE_REID_MODEL)
        _vehicle_session = ort.InferenceSession(VEHICLE_REID_MODEL, providers=["CPUExecutionProvider"])
        _vehicle_input_name = _vehicle_session.get_inputs()[0].name
        print("[ReID] Vehicle Re-ID model (VeRi-776) ready")
    except Exception as e:
        print(f"[ReID] Vehicle Re-ID model unavailable ({e}); using type+color matching only")
        _vehicle_embedder_failed = True
    return _vehicle_session

def get_vehicle_embedding(bgr):
    """Return an L2-normalized 512-d Re-ID embedding for a vehicle crop, or None."""
    sess = _get_vehicle_embedder()
    if sess is None or bgr is None or getattr(bgr, "size", 0) == 0:
        return None
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        rgb = (rgb - _REID_MEAN) / _REID_STD
        x = np.transpose(rgb, (2, 0, 1))[None]  # NCHW
        f = sess.run(None, {_vehicle_input_name: x})[0][0].astype(np.float32)
        n = float(np.linalg.norm(f))
        if n < 1e-9:
            return None
        return f / n
    except Exception as e:
        print(f"[ReID] embedding error: {e}")
        return None

def vehicle_similarity(det_emb, target_views):
    """Best cosine similarity between a detected embedding and a target's stored views.

    target_views may be a single (512,) vector or a stacked (N, 512) array of multiple
    enrolled viewpoints (multi-shot). Returns the max similarity over all views."""
    if det_emb is None or target_views is None:
        return 0.0
    views = np.atleast_2d(target_views)
    if views.size == 0:
        return 0.0
    return float(np.max(views @ det_emb))

# Two vehicle crops with cosine >= this are treated as the same car when enrolling, so a
# second uploaded photo of an existing target is merged as an extra viewpoint (multi-shot)
# rather than creating a duplicate target. VeRi gives ~0.95 same-car / ~0.63 different-car.
VEHICLE_MERGE_SIM = 0.80

VEHICLE_TARGETS_FILE = "vehicle_targets.json"
vehicle_targets = []
# filename -> np.ndarray of shape (N, 512): one row per enrolled viewpoint. Kept out of the
# JSON-serialised target dicts.
vehicle_target_embeddings = {}

def load_vehicle_targets():
    global vehicle_targets, vehicle_target_embeddings
    temp_targets = []
    temp_embeddings = {}

    if os.path.exists(VEHICLE_TARGETS_FILE):
        try:
            with open(VEHICLE_TARGETS_FILE, "r", encoding="utf-8") as f:
                temp_targets = json.load(f)
                print(f"✅ โหลดข้อมูลยานพาหนะเป้าหมาย: {len(temp_targets)} รายการ")
        except Exception as e:
            print("Error loading vehicle targets:", e)

    # Load companion embedding files (.npy) saved at enroll time. Backfill targets with no
    # embedding, and regenerate any whose dimension doesn't match the current Re-ID model
    # (e.g. embeddings saved by the previous ResNet50 backbone) — stored as (N, 512).
    json_dirty = False
    for t in temp_targets:
        emb_file = t.get("embedding_file")
        crop_file = t.get("filename")
        emb = None
        if emb_file and os.path.exists(os.path.join("vehicle_targets", emb_file)):
            try:
                loaded = np.load(os.path.join("vehicle_targets", emb_file))
                if np.atleast_2d(loaded).shape[-1] == VEHICLE_EMBED_DIM:
                    emb = np.atleast_2d(loaded)
                else:
                    print(f"[ReID] Embedding {emb_file} has wrong dim -> regenerating")
            except Exception as e:
                print(f"[ReID] Failed to load embedding {emb_file}: {e}")

        if emb is None and crop_file and os.path.exists(os.path.join("vehicle_targets", crop_file)):
            # Compute (or regenerate) an embedding from the saved crop.
            crop = cv2.imread(os.path.join("vehicle_targets", crop_file))
            v = get_vehicle_embedding(crop)
            if v is not None:
                emb = np.atleast_2d(v)
                new_emb_file = os.path.splitext(crop_file)[0] + ".npy"
                try:
                    np.save(os.path.join("vehicle_targets", new_emb_file), emb)
                    t["embedding_file"] = new_emb_file
                    json_dirty = True
                    print(f"[ReID] (Re)generated embedding for {crop_file}")
                except Exception as e:
                    print(f"[ReID] Failed to save embedding: {e}")
        if emb is not None and crop_file:
            temp_embeddings[crop_file] = emb

    if json_dirty:
        try:
            with open(VEHICLE_TARGETS_FILE, "w", encoding="utf-8") as f:
                json.dump(temp_targets, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[ReID] Failed to persist backfilled targets: {e}")

    with targets_lock:
        vehicle_targets = temp_targets
        vehicle_target_embeddings = temp_embeddings

load_vehicle_targets()


# --- License Plate Target Management ---
PLATE_TARGETS_FILE = "plate_targets.json"
plate_targets = []

def normalize_plate(text):
    if not text:
        return ""
    thai_digits = "๐๑๒๓๔๕๖๗๘๙"
    arabic_digits = "0123456789"
    tr = str.maketrans(thai_digits, arabic_digits)
    text = str(text).translate(tr)
    # Keep Thai letters, English letters, and numbers
    cleaned = re.sub(r'[^ก-ฮ0-9a-zA-Zก-๙]', '', text).strip().upper()
    return cleaned

def extract_digits(text):
    if not text:
        return ""
    thai_digits = "๐๑๒๓๔๕๖๗๘๙"
    arabic_digits = "0123456789"
    tr = str.maketrans(thai_digits, arabic_digits)
    text = str(text).translate(tr)
    digits = re.sub(r'[^0-9]', '', text)
    return digits


def load_plate_targets():
    global plate_targets
    temp_targets = []
    if os.path.exists(PLATE_TARGETS_FILE):
        try:
            with open(PLATE_TARGETS_FILE, "r", encoding="utf-8") as f:
                temp_targets = json.load(f)
                print(f"✅ โหลดข้อมูลป้ายทะเบียนเป้าหมาย: {len(temp_targets)} รายการ")
        except Exception as e:
            print("Error loading plate targets:", e)
    with targets_lock:
        plate_targets = temp_targets

def save_plate_targets():
    global plate_targets
    try:
        with open(PLATE_TARGETS_FILE, "w", encoding="utf-8") as f:
            json.dump(plate_targets, f, indent=4, ensure_ascii=False)
        load_plate_targets()
    except Exception as e:
        print("Error saving plate targets:", e)

load_plate_targets()



def check_bbox_polygon_intersection(x1, y1, x2, y2, pts_arr):
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2), (cx, cy)]
    for c in corners:
        if cv2.pointPolygonTest(pts_arr, c, False) >= 0:
            return True
    for pt in pts_arr:
        px, py = int(pt[0]), int(pt[1])
        if x1 <= px <= x2 and y1 <= py <= y2:
            return True
    return False

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# A chromatic cluster must be at least this saturated (median chroma in CIELAB,
# where a*/b* are centred at 0) before a body is called a colour instead of a
# neutral white/silver/grey/black. Sky/greenery reflections on a grey car sit
# well below this.
CHROMA_MIN = 15.0

def extract_vehicle_color_stats(image):
    """Robust body-panel colour statistics in CIELAB."""
    try:
        h, w = image.shape[:2]
        if h < 10 or w < 10:
            return None

        # Sample central body zones (doors, lower hood, quarter panels)
        # Avoid the very top (windscreen/roof/cargo) and bottom (wheels/shadow/tires)
        bands = [(0.40, 0.65), (0.50, 0.75)]
        patches = []
        for y0, y1 in bands:
            p = image[int(h * y0):int(h * y1), int(w * 0.25):int(w * 0.75)]
            if p is not None and p.size > 0:
                patches.append(p.reshape(-1, 3))
        if not patches:
            patches = [image.reshape(-1, 3)]

        body = np.vstack(patches).astype(np.uint8).reshape(-1, 1, 3)
        if body.shape[0] < 20:
            return None

        lab = cv2.cvtColor(body, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)

        # OpenCV LAB: L in 0..255 (scale to 0..100), a/b in 0..255 centred at 128
        L = lab[:, 0] * (100.0 / 255.0)
        a = lab[:, 1] - 128.0
        b = lab[:, 2] - 128.0
        chroma = np.sqrt(a * a + b * b)

        # Filter out extreme specular glare (>92) and deep wheel/underbody shadows (<12)
        valid_paint = (L >= 12.0) & (L <= 92.0)
        if np.count_nonzero(valid_paint) < 15:
            valid_paint = np.ones_like(L, dtype=bool)

        L_valid = L[valid_paint]
        L50 = float(np.median(L_valid))
        L25 = float(np.percentile(L_valid, 25))
        L75 = float(np.percentile(L_valid, 75))

        chroma_med = float(np.median(chroma[valid_paint]))
        h_med = float(np.median(hsv[valid_paint, 0]))
        s_med = float(np.median(hsv[valid_paint, 1]))
        v_med = float(np.median(hsv[valid_paint, 2]))

        return {
            "L50": L50, "L25": L25, "L75": L75, "chroma": chroma_med,
            "h": int(h_med), "s": int(s_med), "v": int(v_med),
        }
    except Exception as e:
        print(f"[Color] Extraction error: {e}")
        return None

def extract_vehicle_hsv(image):
    """Backward-compatible (h, s, v) for the colour-correction training table."""
    stats = extract_vehicle_color_stats(image)
    if stats is None:
        return None
    return stats["h"], stats["s"], stats["v"]

def get_vehicle_color(image):
    try:
        stats = extract_vehicle_color_stats(image)
        if stats is None:
            return "ไม่ระบุสี"

        L50, L25, L75, chroma = stats["L50"], stats["L25"], stats["L75"], stats["chroma"]
        h_val, s_val, v_val = stats["h"], stats["s"], stats["v"]

        # 1) Neutral / Achromatic body (White, Silver, Grey, Black)
        if chroma < 14.0:
            if L50 >= 60.0 and L25 >= 42.0:
                return "ขาว"
            elif L50 < 30.0 and L75 < 45.0:
                return "ดำ"
            elif L50 >= 48.0:
                return "เงิน"
            else:
                return "เทา"

        # 2) Chromatic body by hue
        if (h_val < 10) or (h_val > 168):
            return "แดง"
        elif h_val < 25:
            return "น้ำตาล" if (L50 < 45 or s_val < 80) else "ส้ม"
        elif h_val < 35:
            return "เหลือง"
        elif h_val < 85:
            return "เขียว"
        elif h_val < 135:
            return "ฟ้า" if (L50 >= 55 and s_val < 130) else "น้ำเงิน"
        else:
            return "ม่วง"
    except Exception as e:
        print(f"Color clustering error: {e}")
        return "ไม่ระบุสี"

COLOR_CANONICAL = {
    "ขาว": "white",
    "เงิน": "silver",
    "เทา": "grey",
    "ดำ": "black",
    "แดง": "red",
    "ชมพู": "pink",
    "ส้ม": "orange",
    "น้ำตาล": "brown",
    "เหลือง": "yellow",
    "ทอง": "yellow",
    "เขียว": "green",
    "น้ำเงิน": "blue",
    "ฟ้า": "blue",
    "ม่วง": "purple",
}

def normalize_color(c):
    if not c:
        return ""
    return COLOR_CANONICAL.get(c.strip(), c.strip())

def is_color_compatible(c1, c2):
    if not c1 or not c2 or c1 in ["ไม่ระบุ", "ไม่ระบุสี"] or c2 in ["ไม่ระบุ", "ไม่ระบุสี"]:
        return True

    n1, n2 = normalize_color(c1), normalize_color(c2)
    if n1 == n2:
        return True

    # Allow closely related shades (e.g. silver & grey, red & pink, blue & skyblue)
    compatible_groups = [
        {"silver", "grey"},
        {"red", "pink"},
        {"blue", "cyan", "sky_blue"},
        {"yellow", "gold"},
        {"orange", "brown"}
    ]
    for group in compatible_groups:
        if n1 in group and n2 in group:
            return True

    return False

def is_type_compatible(target_type, detected_type):
    if not target_type or not detected_type:
        return False
    target_type = target_type.strip()
    detected_type = detected_type.strip()
    
    if target_type == detected_type:
        return True
        
    # กลุ่มรถยนต์ขนาดเล็ก-ปานกลาง (รถเก๋ง มักจะโดนตรวจจับสลับกับรถตู้ได้บางกรณี)
    if target_type in ["รถเก๋ง", "รถตู้"] and detected_type in ["รถเก๋ง", "รถตู้"]:
        return True
        
    # กลุ่มรถบรรทุกและรถกระบะ (YOLO มักจะเห็นรถกระบะ/รถบรรทุกเป็น class เดียวกัน)
    if target_type in ["รถกระบะ", "รถบรรทุก"] and detected_type in ["รถกระบะ", "รถบรรทุก"]:
        return True
        
    return False

vehicle_types_compatible = is_type_compatible
vehicle_colors_compatible = is_color_compatible

_lpr_reader = None

def _get_lpr_reader():
    global _lpr_reader
    if _lpr_reader is None:
        try:
            print("[LPR] Initializing EasyOCR Reader (th, en) on CPU...")
            import easyocr
            _lpr_reader = easyocr.Reader(['th', 'en'], gpu=False)
            print("[LPR] EasyOCR Reader initialized successfully.")
        except Exception as e:
            print(f"[LPR] Failed to initialize EasyOCR Reader: {e}")
    return _lpr_reader

import re

def correct_thai_license_plate(text):
    if not text:
        return ""
        
    # Remove any non-alphanumeric characters
    cleaned = re.sub(r'[^ก-ฮ0-9a-zA-Zก-๙]', '', text)
    if len(cleaned) < 3:
        return cleaned
        
    eng_to_thai = {
        'A': 'ล', 'a': 'ล', 'B': 'ข', 'b': 'ข', 'C': 'ง', 'c': 'ง',
        'D': 'ด', 'd': 'ด', 'E': 'ธ', 'e': 'ธ', 'F': 'ร', 'f': 'ร',
        'G': 'ต', 'g': 'ต', 'H': 'ห', 'h': 'ห', 'I': 'เ', 'i': 'เ',
        'J': 'ง', 'j': 'ง', 'K': 'ห', 'k': 'ห', 'L': 'เ', 'l': 'เ',
        'M': 'ม', 'm': 'ม', 'N': 'ก', 'n': 'ก', 'O': 'อ', 'o': 'อ',
        'P': 'ร', 'p': 'ร', 'Q': 'อ', 'q': 'อ', 'R': 'ร', 'r': 'ร',
        'S': 'ส', 's': 'ส', 'T': 'ต', 't': 'ต', 'U': 'ข', 'u': 'ข',
        'V': 'น', 'v': 'น', 'W': 'พ', 'w': 'พ', 'X': 'ต', 'x': 'ต',
        'Y': 'ญ', 'y': 'ญ', 'Z': 'ว', 'z': 'ว'
    }
    
    char_to_digit = {
        '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
        'ร': '3', 'R': '3', 'r': '3', 'P': '3', 'p': '3', 'F': '3', 'f': '3',
        'ข': '8', 'B': '8', 'b': '8',
        'ว': '2', 'Z': '2', 'z': '2',
        'จ': '7', 'J': '7', 'j': '7',
        'ต': '5', 'T': '5', 't': '5',
        'เ': '1', 'แ': '1', 'I': '1', 'i': '1', 'L': '1', 'l': '1',
        'ง': '4', 'อ': '0', 'O': '0', 'o': '0'
    }
    
    digit_to_char = {
        '3': 'ร', '8': 'ข', '2': 'ว', '7': 'จ', '5': 'ต', '1': 'เ', '0': 'อ'
    }
    
    num_digits_at_end = 0
    for char in reversed(cleaned):
        if char.isdigit() or char in ['O', 'o', 'I', 'i', 'L', 'l', 'B', 'b', 'Z', 'z', 'S', 's', 'T', 't', 'Q', 'q']:
            num_digits_at_end += 1
        else:
            break
            
    num_digits = min(4, max(1, num_digits_at_end))
    
    prefix_part = cleaned[:-num_digits]
    number_part = cleaned[-num_digits:]
    
    corrected_prefix = ""
    if len(prefix_part) == 3:
        first_char = prefix_part[0]
        if first_char.isdigit():
            corrected_prefix += first_char
        elif first_char in char_to_digit:
            corrected_prefix += char_to_digit[first_char]
        else:
            if '\u0e00' <= first_char <= '\u0e7f':
                corrected_prefix += first_char
            elif first_char in eng_to_thai:
                corrected_prefix += eng_to_thai[first_char]
            else:
                corrected_prefix += first_char
            
        for char in prefix_part[1:]:
            if '\u0e00' <= char <= '\u0e7f':
                if char in digit_to_char:
                    corrected_prefix += digit_to_char[char]
                else:
                    corrected_prefix += char
            elif char in eng_to_thai:
                corrected_prefix += eng_to_thai[char]
            elif char in digit_to_char:
                corrected_prefix += digit_to_char[char]
            else:
                corrected_prefix += char
    elif len(prefix_part) == 2:
        for char in prefix_part:
            if '\u0e00' <= char <= '\u0e7f':
                if char in digit_to_char:
                    corrected_prefix += digit_to_char[char]
                else:
                    corrected_prefix += char
            elif char in eng_to_thai:
                corrected_prefix += eng_to_thai[char]
            elif char in digit_to_char:
                corrected_prefix += digit_to_char[char]
            else:
                corrected_prefix += char
    else:
        for char in prefix_part:
            if '\u0e00' <= char <= '\u0e7f':
                corrected_prefix += char
            elif char in eng_to_thai:
                corrected_prefix += eng_to_thai[char]
            else:
                corrected_prefix += char
                
    corrected_number = ""
    for char in number_part:
        if char.isdigit():
            corrected_number += char
        elif char in char_to_digit:
            corrected_number += char_to_digit[char]
        else:
            corrected_number += char
            
    return f"{corrected_prefix} {corrected_number}".strip()

def sort_detections_by_reading_order(detections):
    if not detections:
        return []
    
    items = []
    for idx, (bbox, text, conf) in enumerate(detections):
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        min_y = min(ys)
        max_y = max(ys)
        min_x = min(xs)
        height = max_y - min_y
        center_y = (min_y + max_y) / 2
        items.append({
            'index': idx,
            'bbox': bbox,
            'text': text,
            'conf': conf,
            'min_y': min_y,
            'max_y': max_y,
            'min_x': min_x,
            'height': height,
            'center_y': center_y
        })
        
    items.sort(key=lambda x: x['center_y'])
    
    lines = []
    for item in items:
        placed = False
        for line in lines:
            ref = line[0]
            h_overlap = min(item['max_y'], ref['max_y']) - max(item['min_y'], ref['min_y'])
            avg_height = (item['height'] + ref['height']) / 2
            if h_overlap > 0.4 * avg_height:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])
            
    lines.sort(key=lambda line: sum(item['center_y'] for item in line) / len(line))
    
    sorted_detections = []
    for line in lines:
        line.sort(key=lambda item: item['min_x'])
        for item in line:
            sorted_detections.append((item['bbox'], item['text'], item['conf']))
            
    return sorted_detections

def clean_license_plate(text):
    if not text:
        return ""
    cleaned = re.sub(r'[^ก-ฮ0-9a-zA-Zก-๙]', '', text)
    corrected = correct_thai_license_plate(cleaned)
    return corrected.strip()

def is_potential_plate_part(text):
    if not text:
        return False
    cleaned = re.sub(r'[^ก-ฮ0-9a-zA-Zก-๙]', '', text)
    if len(cleaned) < 2 or len(cleaned) > 10:
        return False
    return True

def is_valid_combined_plate(text):
    if not text:
        return False
    cleaned = re.sub(r'[^ก-ฮ0-9a-zA-Zก-๙]', '', text)
    if len(cleaned) < 3 or len(cleaned) > 12:
        return False
    has_digit = any(c.isdigit() or '๐' <= c <= '๙' for c in cleaned)
    return has_digit

def is_valid_license_plate(cleaned_text):
    return is_valid_combined_plate(cleaned_text)


def get_black_frame(text="Connecting to camera...", base_frame=None):
    if base_frame is not None:
        img = base_frame.copy()
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        img = put_thai_text(img, text, (50, img.shape[0]//2), (0, 255, 255))
        return img
    else:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        blank = put_thai_text(blank, text, (50, 240), (0, 255, 255))
        return blank

def save_high_quality_crop(img, path):
    try:
        if img is None or img.size == 0: return
        h, w = img.shape[:2]
        if h < 10 or w < 10: return
        
        target_min_dim = 600
        if w < target_min_dim and h < target_min_dim:
            scale = target_min_dim / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    except Exception as e:
        print(f"Error saving high quality crop: {e}")

thai_font = None
thai_font_path = None
thai_font_cache = {}
font_candidates = [
    os.path.join(os.path.dirname(__file__), "Kanit-Regular.ttf"),
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/SukhumvitSet.ttc",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/leelawad.ttf",
    "C:/Windows/Fonts/cordia.ttf",
    "/usr/share/fonts/truetype/thai/Loma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
for font_path in font_candidates:
    if os.path.exists(font_path):
        try:
            thai_font = ImageFont.truetype(font_path, 24)
            thai_font_path = font_path
            thai_font_cache[24] = thai_font
            break
        except Exception:
            continue

if thai_font is None:
    thai_font = ImageFont.load_default()
    thai_font_cache[24] = thai_font

def get_thai_font(font_size=24):
    font_size = int(font_size or 24)
    if font_size in thai_font_cache:
        return thai_font_cache[font_size]
    if thai_font_path:
        try:
            font = ImageFont.truetype(thai_font_path, font_size)
            thai_font_cache[font_size] = font
            return font
        except Exception:
            pass
    return thai_font

def compact_text(value, max_len=48):
    text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_len:
        return text
    return text[:max(0, max_len - 3)].rstrip() + "..."

def telegram_image_text(value, max_len=48):
    text = compact_text(value, max_len=max_len)
    allowed = []
    for ch in text:
        code = ord(ch)
        is_thai = 0x0E00 <= code <= 0x0E7F
        is_ascii_text = ch.isascii() and (ch.isalnum() or ch in " -_.,:()/#@%+|")
        if is_thai or is_ascii_text:
            allowed.append(ch)
        elif ch.isspace():
            allowed.append(" ")
    return re.sub(r"\s+", " ", "".join(allowed)).strip()

def telegram_caption_text(value, max_len=48):
    return telegram_image_text(value, max_len=max_len)

def put_thai_text(img, text, position, text_color=(0, 0, 255), font_size=24):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # OpenCV uses BGR, PIL uses RGB. 
        fill_color = (text_color[2], text_color[1], text_color[0])
        draw.text(position, text, font=get_thai_font(font_size), fill=fill_color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        # Fallback to OpenCV if PIL fails
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        return img
class ScreenGrabber:
    def __init__(self, monitor_index=1):
        self.sct = mss.mss()
        # ใช้มอนิเตอร์ที่ระบุ
        if monitor_index < len(self.sct.monitors):
            self.base_monitor = self.sct.monitors[monitor_index]
        else:
            self.base_monitor = self.sct.monitors[1] if len(self.sct.monitors) > 1 else self.sct.monitors[0]
        self.monitor = self.base_monitor.copy()

    def set_region(self, left_pct, top_pct, width_pct, height_pct):
        self.monitor["left"] = int(self.base_monitor["left"] + (self.base_monitor["width"] * left_pct))
        self.monitor["top"] = int(self.base_monitor["top"] + (self.base_monitor["height"] * top_pct))
        self.monitor["width"] = max(10, int(self.base_monitor["width"] * width_pct))
        self.monitor["height"] = max(10, int(self.base_monitor["height"] * height_pct))
        
    def reset_region(self):
        self.monitor = self.base_monitor.copy()

    def read(self):
        try:
            img = np.array(self.sct.grab(self.monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return True, frame
        except Exception as e:
            print(f"[ScreenGrabber Error]: {e}")
            return False, None

    def isOpened(self):
        return True

    def release(self):
        try:
            self.sct.close()
        except Exception:
            pass

class CameraWorker:
    def __init__(self, cam_id, name, url, cam_config=None):
        self.cam_id = str(cam_id)
        self.name = str(name)
        self.url = str(url)
        self.cam_config = cam_config or {}
        
        self.is_enabled = True
        self.stream_state = "stopped"  # Only active/requested cameras run AI inference
        self.frame_skip = int(self.cam_config.get("frame_skip", current_frame_skip))
        self.fps = 0.0
        
        # Seekable / video file properties
        is_vid = str(self.url).startswith("uploads/") or str(self.url).lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))
        self.is_video_source = is_vid
        self.video_native_fps = 25.0
        self.video_total_frames = 0
        self.video_current_frame = 0
        self.seek_request = None
        
        if is_vid and os.path.exists(self.url):
            try:
                probe_cap = cv2.VideoCapture(self.url)
                if probe_cap.isOpened():
                    fps_val = float(probe_cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    if fps_val > 0:
                        self.video_native_fps = fps_val
                    self.video_total_frames = int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    probe_cap.release()
            except Exception:
                pass
        
        # Detection zone and crossing line per camera
        self.detection_zone = self.cam_config.get("detection_zone", None)
        self.crossing_line = self.cam_config.get("crossing_line", None)
        self.crossing_direction = self.cam_config.get("crossing_direction", "any")
        self.show_detection_zone = self.cam_config.get("show_detection_zone", True)
        
        # Tracking states
        self.tracked_ids = set()
        self.tracked_target_ids = {} # track_id -> target name
        self.track_last_seen = {} # track_id -> timestamp (float)
        self.track_history = {} # track_id -> (cx, cy)
        self.track_lpr_attempts = {}
        self.track_lpr_last_time = {}
        self.track_lpr_results = {}
        self.track_lpr_max_size = {}
        self.track_lpr_logged = set()
        self.unknown_persons_count = 0
        self.line_cross_count = 0
        
        # Frames
        self.latest_annotated_frame = None
        self.latest_raw_frame = None
        self.frame_lock = threading.Lock()
        
        # Thread & connection control
        self.cap = None
        self.running = True
        self.reconnect_requested = False
        self.thread = None

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.stream_state = "stopped"
        with self.frame_lock:
            self.latest_annotated_frame = get_black_frame(f"{self.name} (Stopped)")
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def close(self):
        self.running = False
        self.stop()

    def play(self):
        self.stream_state = "playing"
        if self.cap is None or not getattr(self.cap, "isOpened", lambda: False)():
            self.reconnect_requested = True

    def pause(self):
        self.stream_state = "paused"

    def restart(self):
        self.reconnect_requested = True
        self.stream_state = "playing"

    def seek(self, target_frame):
        self.seek_request = target_frame

    def prune_stale(self, now=None):
        if now is None:
            now = time.time()
        stale = [tid for tid, seen in self.track_last_seen.items() if now - seen > TRACK_TTL_SECONDS]
        for tid in stale:
            self.track_last_seen.pop(tid, None)
            self.track_history.pop(tid, None)
            self.tracked_target_ids.pop(tid, None)
            self.tracked_ids.discard(tid)
            self.track_lpr_attempts.pop(tid, None)
            self.track_lpr_last_time.pop(tid, None)
            self.track_lpr_results.pop(tid, None)
            self.track_lpr_max_size.pop(tid, None)
            self.track_lpr_logged.discard(tid)

    def get_status_dict(self):
        fps_for_time = self.video_native_fps if self.video_native_fps and self.video_native_fps > 0 else 25.0
        video_duration = (self.video_total_frames / fps_for_time) if (self.is_video_source and self.video_total_frames > 0) else 0
        video_position = (self.video_current_frame / fps_for_time) if self.is_video_source else 0
        if video_duration > 0:
            video_position = min(video_position, video_duration)
            
        cam_url_str = str(self.url)
        return {
            "id": self.cam_id,
            "name": self.name,
            "url": self.url,
            "is_enabled": self.is_enabled,
            "stream_state": self.stream_state,
            "fps": round(self.fps, 1),
            "unknown_count": self.unknown_persons_count,
            "line_cross_count": self.line_cross_count,
            "target_stats": target_hit_counts.get(cam_url_str, {}),
            "is_video": self.is_video_source,
            "video_position": round(video_position, 1),
            "video_duration": round(video_duration, 1),
            "video_total_frames": self.video_total_frames,
            "video_current_frame": self.video_current_frame,
            "frame_skip": self.frame_skip,
            "has_zone": self.detection_zone is not None and len(self.detection_zone) >= 3,
            "has_line": self.crossing_line is not None and len(self.crossing_line) == 2
        }

    def _run_loop(self):
        global target_hit_counts, latest_hits, known_face_encodings, known_face_names, model, current_yolo_model
        cap = None
        
        while self.running:
            if self.stream_state == "stopped":
                with self.frame_lock:
                    self.latest_annotated_frame = get_black_frame(f"{self.name} (Stopped)")
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                time.sleep(0.5)
                continue
                
            if self.reconnect_requested or cap is None or not getattr(cap, "isOpened", lambda: False)():
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                    
                disp = self.url[:25] + "..." if len(self.url) > 25 else self.url
                with self.frame_lock:
                    self.latest_annotated_frame = get_black_frame(f"Connecting to {self.name} ({disp})")
                time.sleep(1)
                
                if str(self.url).lower().startswith("screen"):
                    parts = str(self.url).split(":")
                    monitor_idx = 1
                    if len(parts) > 1:
                        try:
                            monitor_idx = int(parts[1])
                        except:
                            pass
                    cap = ScreenGrabber(monitor_index=monitor_idx)
                else:
                    cam_source = int(self.url) if str(self.url).isdigit() else self.url
                    cap = cv2.VideoCapture(cam_source, cv2.CAP_FFMPEG if not str(self.url).isdigit() else cv2.CAP_ANY)
                
                self.cap = cap
                self.reconnect_requested = False
                
                # Detect seekable local video file
                if str(self.url).startswith("uploads/") and cap.isOpened():
                    self.is_video_source = True
                    self.video_native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    self.video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    self.video_current_frame = 0
                    
                    success, frame = cap.read()
                    if success:
                        with self.frame_lock:
                            self.latest_raw_frame = frame.copy()
                            annotated_frame = frame.copy()
                            if self.show_detection_zone:
                                h_f, w_f = frame.shape[:2]
                                if self.crossing_line:
                                    lpts = [(int(p["x"] * w_f), int(p["y"] * h_f)) for p in self.crossing_line]
                                    cv2.line(annotated_frame, lpts[0], lpts[1], (0, 255, 255), 4)
                                elif self.detection_zone:
                                    pts = [(int(p["x"] * w_f), int(p["y"] * h_f)) for p in self.detection_zone]
                                    pts_arr = np.array(pts, np.int32)
                                    overlay = annotated_frame.copy()
                                    cv2.fillPoly(overlay, [pts_arr], (255, 0, 255))
                                    cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
                                    cv2.polylines(annotated_frame, [pts_arr], True, (255, 0, 255), 2)
                            
                            annotated_frame = get_black_frame("ระบบหยุดชั่วคราว (รอคำสั่งเริ่ม)", base_frame=annotated_frame)
                            self.latest_annotated_frame = annotated_frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.video_current_frame = 0
                else:
                    self.is_video_source = False
                    self.video_native_fps = 0.0
                    self.video_total_frames = 0
                    self.video_current_frame = 0
                self.seek_request = None
                
                if not cap.isOpened():
                    with self.frame_lock:
                        self.latest_annotated_frame = get_black_frame(f"{self.name}: Connection Failed. Retrying...")
                    time.sleep(2)
                    continue

            prev_time = time.time()
            frame_counter = 0
            
            while cap.isOpened() and not self.reconnect_requested and self.stream_state != "stopped" and self.running:
                # Seek when paused
                if self.is_video_source and self.stream_state == "paused":
                    if self.seek_request is not None:
                        target = self.seek_request
                        self.seek_request = None
                        if self.video_total_frames > 0:
                            target = max(0, min(target, self.video_total_frames - 1))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                        self.video_current_frame = target
                        success, frame = cap.read()
                        if success:
                            with self.frame_lock:
                                self.latest_raw_frame = frame.copy()
                                annotated_frame = frame.copy()
                                if self.show_detection_zone:
                                    h_f, w_f = frame.shape[:2]
                                    if self.crossing_line:
                                        lpts = [(int(p["x"] * w_f), int(p["y"] * h_f)) for p in self.crossing_line]
                                        cv2.line(annotated_frame, lpts[0], lpts[1], (0, 255, 255), 4)
                                    elif self.detection_zone:
                                        pts = [(int(p["x"] * w_f), int(p["y"] * h_f)) for p in self.detection_zone]
                                        pts_arr = np.array(pts, np.int32)
                                        overlay = annotated_frame.copy()
                                        cv2.fillPoly(overlay, [pts_arr], (255, 0, 255))
                                        cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
                                        cv2.polylines(annotated_frame, [pts_arr], True, (255, 0, 255), 2)
                                annotated_frame = get_black_frame("ระบบหยุดชั่วคราว (รอคำสั่งเริ่ม)", base_frame=annotated_frame)
                                self.latest_annotated_frame = annotated_frame
                    else:
                        time.sleep(0.1)
                    continue

                if self.is_video_source and self.seek_request is not None:
                    target = self.seek_request
                    self.seek_request = None
                    if self.video_total_frames > 0:
                        target = max(0, min(target, self.video_total_frames - 1))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    self.video_current_frame = target

                try:
                    success, frame = cap.read()
                except Exception as read_ex:
                    print(f"[{self.name}] Exception on cap.read(): {read_ex}")
                    success, frame = False, None

                frame_counter += 1
                if self.is_video_source and cap is not None:
                    try:
                        self.video_current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or (self.video_current_frame + 1))
                    except:
                        self.video_current_frame += 1
                    
                if not success or frame is None or getattr(frame, 'size', 0) == 0:
                    if self.is_video_source or str(self.url).startswith("uploads/") or "googlevideo.com" in str(self.url):
                        print(f"[{self.name}] Video ended or reached end of stream. Resetting cleanly.")
                        try:
                            if cap is not None:
                                cap.release()
                        except:
                            pass
                        cam_source = int(self.url) if str(self.url).isdigit() else self.url
                        cap = cv2.VideoCapture(cam_source)
                        self.cap = cap
                        self.video_current_frame = 0
                        self.stream_state = "paused"
                        time.sleep(0.5)
                        continue
                    else:
                        print(f"[{self.name}] Stream lost. Reconnecting in 2s...")
                        if cap is not None:
                            try:
                                cap.release()
                            except:
                                pass
                        cap = None
                        time.sleep(2)
                        break

                with self.frame_lock:
                    self.latest_raw_frame = frame.copy()
                    
                target_delay = (1.0 / self.video_native_fps) if (self.is_video_source and self.video_native_fps > 0) else 0.033

                # Frame skipping for AI
                eff_frame_skip = self.frame_skip if self.frame_skip > 0 else current_frame_skip
                skip_ai = (eff_frame_skip > 0 and frame_counter % (eff_frame_skip + 1) != 0)
                
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if current_time - prev_time > 0 else 0
                prev_time = current_time
                self.fps = fps
                
                if frame_counter % 300 == 0:
                    self.prune_stale(current_time)
                    
                target_classes = [0, 1, 2, 3, 5, 7]
                if len(custom_search_terms) > 0:
                    target_classes.extend(range(80, 80 + len(custom_search_terms)))
                    
                if self.stream_state == "paused":
                    annotated_frame = frame.copy()
                    if self.show_detection_zone:
                        h_f, w_f = frame.shape[:2]
                        if self.crossing_line:
                            lpts = [(int(p["x"] * w_f), int(p["y"] * h_f)) for p in self.crossing_line]
                            cv2.line(annotated_frame, lpts[0], lpts[1], (0, 255, 255), 4)
                        elif self.detection_zone:
                            pts = [(int(p["x"] * w_f), int(p["y"] * h_f)) for p in self.detection_zone]
                            pts_arr = np.array(pts, np.int32)
                            overlay = annotated_frame.copy()
                            cv2.fillPoly(overlay, [pts_arr], (255, 0, 255))
                            cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
                            cv2.polylines(annotated_frame, [pts_arr], True, (255, 0, 255), 2)
                    annotated_frame = get_black_frame("ระบบหยุดชั่วคราว (รอคำสั่งเริ่ม)", base_frame=annotated_frame)
                    with self.frame_lock:
                        self.latest_annotated_frame = annotated_frame
                    time.sleep(0.05)
                    continue
                    
                if skip_ai:
                    annotated_frame = frame.copy()
                    if self.show_detection_zone:
                        h_f, w_f = frame.shape[:2]
                        if self.crossing_line:
                            lpts = [(int(p["x"] * w_f), int(p["y"] * h_f)) for p in self.crossing_line]
                            cv2.line(annotated_frame, lpts[0], lpts[1], (0, 255, 255), 4)
                        elif self.detection_zone:
                            pts = [(int(p["x"] * w_f), int(p["y"] * h_f)) for p in self.detection_zone]
                            pts_arr = np.array(pts, np.int32)
                            overlay = annotated_frame.copy()
                            cv2.fillPoly(overlay, [pts_arr], (255, 0, 255))
                            cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
                            cv2.polylines(annotated_frame, [pts_arr], True, (255, 0, 255), 2)
                    with self.frame_lock:
                        self.latest_annotated_frame = annotated_frame
                    if self.is_video_source:
                        elapsed = time.time() - current_time
                        if elapsed < target_delay:
                            time.sleep(target_delay - elapsed)
                    continue

                # Run shared AI model
                if model is None:
                    with self.frame_lock:
                        self.latest_annotated_frame = get_black_frame("YOLO model is not loaded", base_frame=frame)
                    time.sleep(0.5)
                    continue
                with inference_lock:
                    with torch.inference_mode():
                        results = model.track(frame, classes=target_classes, conf=min_confidence / 100.0, persist=True, verbose=False, device=DEVICE)
                
                if frame_counter % 120 == 0 and DEVICE == "mps":
                    try:
                        torch.mps.empty_cache()
                    except:
                        pass
                
                annotated_frame = frame.copy()
                
                # Draw detection zone or crossing line
                if self.show_detection_zone:
                    h_f, w_f = frame.shape[:2]
                    if self.crossing_line:
                        lpts = [(int(p["x"] * w_f), int(p["y"] * h_f)) for p in self.crossing_line]
                        cv2.line(annotated_frame, lpts[0], lpts[1], (0, 255, 255), 4)
                    elif self.detection_zone:
                        pts = [(int(p["x"] * w_f), int(p["y"] * h_f)) for p in self.detection_zone]
                        pts_arr = np.array(pts, np.int32)
                        overlay = annotated_frame.copy()
                        cv2.fillPoly(overlay, [pts_arr], (255, 0, 255))
                        cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
                        cv2.polylines(annotated_frame, [pts_arr], True, (255, 0, 255), 2)
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    cls_ids = results[0].boxes.cls.int().cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    
                    for box, track_id, cls_id, conf in zip(boxes, track_ids, cls_ids, confs):
                        x1, y1, x2, y2 = map(int, box)
                        track_id = int(track_id)
                        cls_id = int(cls_id)
                        conf = float(conf)
                        self.track_last_seen[track_id] = current_time

                        # Line crossing check
                        if self.crossing_line:
                            h_f, w_f = frame.shape[:2]
                            line_pts = [(int(p["x"] * w_f), int(p["y"] * h_f)) for p in self.crossing_line]
                            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            crossed = False
                            if track_id in self.track_history:
                                prev_cx, prev_cy = self.track_history[track_id]
                                if intersect((prev_cx, prev_cy), (cx, cy), line_pts[0], line_pts[1]):
                                    dx = cx - prev_cx
                                    dy = cy - prev_cy
                                    valid_cross = True
                                    if self.crossing_direction == "left_to_right" and dx <= 0:
                                        valid_cross = False
                                    elif self.crossing_direction == "right_to_left" and dx >= 0:
                                        valid_cross = False
                                    elif self.crossing_direction == "top_to_bottom" and dy <= 0:
                                        valid_cross = False
                                    elif self.crossing_direction == "bottom_to_top" and dy >= 0:
                                        valid_cross = False
                                    if valid_cross:
                                        crossed = True
                                        self.line_cross_count += 1
                            self.track_history[track_id] = (cx, cy)
                            if not crossed:
                                continue
                        elif self.detection_zone:
                            h_f, w_f = frame.shape[:2]
                            pts = [(int(p["x"] * w_f), int(p["y"] * h_f)) for p in self.detection_zone]
                            pts_arr = np.array(pts, np.int32)
                            if not check_bbox_polygon_intersection(x1, y1, x2, y2, pts_arr):
                                continue

                        label_name = model.names.get(cls_id, "Object")
                        if not show_only_targets:
                            cv2.line(annotated_frame, (x1, y2), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"#{track_id} {label_name} {conf:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Custom YOLO-world classes
                        if cls_id >= 80:
                            name = label_name
                            match_percentage = int(conf * 100)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            annotated_frame = put_thai_text(annotated_frame, f"TARGET: {name} (#{track_id})", (x1, y2 + 40), (0, 0, 255))
                            
                            if track_id not in self.tracked_target_ids or self.tracked_target_ids[track_id] != name:
                                self.tracked_target_ids[track_id] = name
                                padding = 100
                                h_img, w_img = frame.shape[:2]
                                cx1, cy1 = max(0, x1 - padding), max(0, y1 - padding)
                                cx2, cy2 = min(w_img, x2 + padding), min(h_img, y2 + padding)
                                v_crop_img = frame[cy1:cy2, cx1:cx2]
                                
                                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                                img_name = f"{name}_{track_id}_{timestamp_str}.jpg"
                                img_name_full = f"{name}_{track_id}_{timestamp_str}_full.jpg"
                                
                                save_high_quality_crop(v_crop_img, os.path.join('alerts', img_name))
                                cv2.imwrite(os.path.join('alerts', img_name_full), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                
                                cam_url_str = str(self.url)
                                if cam_url_str not in target_hit_counts: target_hit_counts[cam_url_str] = {}
                                target_hit_counts[cam_url_str][name] = target_hit_counts[cam_url_str].get(name, 0) + 1
                                
                                hit_info_name = f"{name} (#{track_id})"
                                hit_info = {
                                    "name": hit_info_name,
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "confidence": match_percentage,
                                    "img_url": f"http://localhost:8081/alerts_img/{img_name}",
                                    "full_url": f"http://localhost:8081/alerts_img/{img_name_full}",
                                    "is_color_edited": 0,
                                    "camera_id": self.cam_id,
                                    "camera_name": self.name
                                }
                                if cam_url_str not in latest_hits: latest_hits[cam_url_str] = []
                                latest_hits[cam_url_str].insert(0, hit_info)
                                if len(latest_hits[cam_url_str]) > 10: latest_hits[cam_url_str].pop()
                                
                                try:
                                    with get_db_connection() as conn:
                                        cursor = conn.cursor()
                                        cursor.execute('''
                                            INSERT INTO hits (camera_url, name, timestamp, confidence, img_url, full_url)
                                            VALUES (?, ?, ?, ?, ?, ?)
                                        ''', (cam_url_str, hit_info_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), match_percentage, hit_info["img_url"], hit_info["full_url"]))
                                        hit_info["id"] = cursor.lastrowid
                                        conn.commit()
                                except Exception as e:
                                    print(f"Error saving to DB: {e}")
                                    
                                send_telegram_alert_async(
                                    target_type="object",
                                    target_name=name,
                                    match_pct=match_percentage,
                                    camera_name=self.name,
                                    live_crop_img=v_crop_img
                                )

                        # Vehicle & LPR
                        if cls_id != 0:
                            is_target_vehicle = False
                            is_legacy_match = False
                            v_name = ""
                            match_percentage = int(round(conf * 100))
                            
                            if cls_id in [1, 2, 3, 5, 7]:
                                vehicle_types = {1: "รถจักรยาน", 2: "รถเก๋ง", 3: "รถมอเตอร์ไซค์", 5: "รถบัส", 7: "รถกระบะ"}
                                v_type = vehicle_types.get(cls_id, "รถยนต์")
                                v_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                                
                                if v_crop.size > 0:
                                    v_color = get_vehicle_color(v_crop)
                                    plate_text = None
                                    plate_img_name = None
                                    
                                    if enable_lpr:
                                        if track_id in self.track_lpr_results:
                                            plate_text = self.track_lpr_results[track_id]["plate"]
                                            plate_img_name = self.track_lpr_results[track_id]["plate_img_name"]
                                        else:
                                            curr_size = v_crop.shape[0] * v_crop.shape[1]
                                            prev_size = self.track_lpr_max_size.get(track_id, 0)
                                            if curr_size > prev_size * 1.3:
                                                self.track_lpr_attempts[track_id] = 0
                                                self.track_lpr_max_size[track_id] = curr_size
                                            if (self.track_lpr_attempts.get(track_id, 0) < 5 and
                                                v_crop.shape[1] >= 100 and v_crop.shape[0] >= 100 and
                                                (time.time() - self.track_lpr_last_time.get(track_id, 0.0)) >= 0.4):
                                                self.track_lpr_last_time[track_id] = time.time()
                                                self.track_lpr_attempts[track_id] = self.track_lpr_attempts.get(track_id, 0) + 1
                                                if track_id not in self.track_lpr_max_size:
                                                    self.track_lpr_max_size[track_id] = curr_size
                                                reader = _get_lpr_reader()
                                                if reader is not None:
                                                    try:
                                                        vh, vw = v_crop.shape[:2]
                                                        lower_half = v_crop[int(vh * 0.45):vh, 0:vw]
                                                        upscaled = cv2.resize(lower_half, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                                                        ycrcb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2YCrCb)
                                                        ch = list(cv2.split(ycrcb))
                                                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                                                        ch[0] = clahe.apply(ch[0])
                                                        ycrcb = cv2.merge(ch)
                                                        processed_plate = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                                                        processed_plate = cv2.bilateralFilter(processed_plate, 9, 75, 75)
                                                        
                                                        with inference_lock:
                                                            ocr_res = reader.readtext(processed_plate)
                                                        
                                                        raw_runs = []
                                                        for bbox, text, ocr_conf in ocr_res:
                                                            cleaned_seg = re.sub(r'[^ก-ฮ0-9a-zA-Zก-๙]', '', text).strip()
                                                            if len(cleaned_seg) >= 1 and ocr_conf >= 0.20:
                                                                raw_runs.append((bbox, cleaned_seg, ocr_conf))
                                                                
                                                        if raw_runs:
                                                            valid_runs = sort_detections_by_reading_order(raw_runs)
                                                            combined_raw = "".join([x[1] for x in valid_runs])
                                                            combined_text = correct_thai_license_plate(combined_raw)
                                                            avg_conf = sum([x[2] for x in valid_runs]) / len(valid_runs)
                                                            
                                                            if is_valid_combined_plate(combined_text) and avg_conf >= 0.30:
                                                                all_pts = []
                                                                for bbox_pts, _, _ in valid_runs:
                                                                    for pt in bbox_pts:
                                                                        all_pts.append(pt)
                                                                all_pts = np.array(all_pts)
                                                                px1, py1 = int(np.min(all_pts[:, 0]) / 2.0), int(np.min(all_pts[:, 1]) / 2.0)
                                                                px2, py2 = int(np.max(all_pts[:, 0]) / 2.0), int(np.max(all_pts[:, 1]) / 2.0)
                                                                
                                                                pad_px, pad_py = 10, 10
                                                                px1 = max(0, px1 - pad_px)
                                                                py1 = max(0, py1 - pad_py)
                                                                px2 = min(vw, px2 + pad_px)
                                                                py2 = min(vh - int(vh * 0.45), py2 + pad_py)
                                                                
                                                                plate_crop = lower_half[py1:py2, px1:px2]
                                                                if plate_crop.size > 0:
                                                                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                                                                    p_name = f"plate_{track_id}_{timestamp_str}.jpg"
                                                                    save_high_quality_crop(plate_crop, os.path.join('alerts', p_name))
                                                                    plate_img_name = p_name
                                                                    
                                                                self.track_lpr_results[track_id] = {
                                                                    "plate": combined_text,
                                                                    "plate_img_name": plate_img_name,
                                                                    "conf": int(avg_conf * 100)
                                                                }
                                                                plate_text = combined_text
                                                    except Exception as e:
                                                        print(f"LPR Error: {e}")

                                    # Vehicle matching
                                    with vehicle_targets_lock:
                                        curr_v_targets = list(vehicle_targets)
                                        curr_v_embeds = dict(vehicle_target_embeddings)
                                    with plate_targets_lock:
                                        curr_p_targets = list(plate_targets)
                                        
                                    is_plate_match = False
                                    matched_plate_target = None
                                    if plate_text and len(curr_p_targets) > 0:
                                        for pt in curr_p_targets:
                                            target_plate = pt.get("plate", "").replace(" ", "").strip()
                                            if target_plate and target_plate in plate_text.replace(" ", "").strip():
                                                is_plate_match = True
                                                matched_plate_target = pt
                                                break
                                                
                                    if is_plate_match:
                                        is_target_vehicle = True
                                        v_name = f"{matched_plate_target.get('name', 'เป้าหมายป้ายทะเบียน')} ({plate_text})"
                                        match_percentage = 99
                                    elif len(curr_v_targets) > 0:
                                        v_emb = get_vehicle_embedding(v_crop)
                                        best_visual = None
                                        legacy_match = None
                                        for vt in curr_v_targets:
                                            target_type = vt.get("type", "")
                                            target_color = vt.get("color", "")
                                            fn = vt.get("filename", "")
                                            has_embed = fn and fn in curr_v_embeds
                                            type_ok = vehicle_types_compatible(v_type, target_type)
                                            color_ok = vehicle_colors_compatible(v_color, target_color)
                                            
                                            if has_embed and v_emb is not None:
                                                if not type_ok:
                                                    continue
                                                sim = vehicle_similarity(v_emb, curr_v_embeds[fn])
                                                if sim >= (vehicle_match_threshold / 100.0):
                                                    if best_visual is None or sim > best_visual[0]:
                                                        best_visual = (sim, vt.get("raw") or f"{target_type}{target_color}")
                                            elif type_ok and color_ok:
                                                if legacy_match is None:
                                                    legacy_match = vt.get("raw") or f"{target_type}{target_color}"
                                                    
                                        if best_visual is not None:
                                            is_target_vehicle = True
                                            v_name = best_visual[1]
                                            match_percentage = int(round(best_visual[0] * 100))
                                        elif legacy_match is not None:
                                            is_target_vehicle = True
                                            is_legacy_match = True
                                            v_name = legacy_match
                                            match_percentage = 50

                                    if is_target_vehicle:
                                        name = v_name
                                        display_name = name
                                        if is_legacy_match:
                                            display_name = f"{name} (สี+ประเภท?)"
                                        if plate_text:
                                            display_name = f"{display_name} [ทะเบียน: {plate_text}]"
                                            
                                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                        annotated_frame = put_thai_text(annotated_frame, f"TARGET: {display_name} (#{track_id})", (x1, y2 + 40), (0, 0, 255))
                                        
                                        # Check if this vehicle track_id has already been logged
                                        already_logged = (track_id in self.tracked_target_ids)
                                        
                                        # Filter out degraded edge-clipped snapshots when car is leaving the frame
                                        h_img, w_img = frame.shape[:2]
                                        is_at_border = (x1 <= 15 or y1 <= 15 or x2 >= w_img - 15 or y2 >= h_img - 15)
                                        
                                        if not already_logged and not is_at_border:
                                            self.tracked_target_ids[track_id] = name
                                            self.track_lpr_logged.add(track_id)
                                            padding = 100
                                            cx1, cy1 = max(0, x1 - padding), max(0, y1 - padding)
                                            cx2, cy2 = min(w_img, x2 + padding), min(h_img, y2 + padding)
                                            v_crop_img = frame[cy1:cy2, cx1:cx2]
                                            
                                            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            img_name = f"vehicle_{track_id}_{timestamp_str}.jpg"
                                            img_name_full = f"vehicle_{track_id}_{timestamp_str}_full.jpg"
                                            
                                            save_high_quality_crop(v_crop_img, os.path.join('alerts', img_name))
                                            cv2.imwrite(os.path.join('alerts', img_name_full), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                            
                                            cam_url_str = str(self.url)
                                            if cam_url_str not in target_hit_counts: target_hit_counts[cam_url_str] = {}
                                            target_hit_counts[cam_url_str][name] = target_hit_counts[cam_url_str].get(name, 0) + 1
                                            
                                            hit_info_name = f"{name} (#{track_id})"
                                            hit_info = {
                                                "name": hit_info_name,
                                                "time": datetime.now().strftime("%H:%M:%S"),
                                                "confidence": match_percentage,
                                                "img_url": f"http://localhost:8081/alerts_img/{img_name}",
                                                "full_url": f"http://localhost:8081/alerts_img/{img_name_full}",
                                                "is_color_edited": 0,
                                                "plate_img_url": f"http://localhost:8081/alerts_img/{plate_img_name}" if plate_img_name else None,
                                                "license_plate": plate_text,
                                                "camera_id": self.cam_id,
                                                "camera_name": self.name
                                            }
                                            if cam_url_str not in latest_hits: latest_hits[cam_url_str] = []
                                            latest_hits[cam_url_str].insert(0, hit_info)
                                            if len(latest_hits[cam_url_str]) > 10: latest_hits[cam_url_str].pop()
                                            
                                            try:
                                                with get_db_connection() as conn:
                                                    cursor = conn.cursor()
                                                    cursor.execute('''
                                                        INSERT INTO hits (camera_url, name, timestamp, confidence, img_url, full_url, plate_img_url, license_plate)
                                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                                    ''', (cam_url_str, hit_info_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), match_percentage, hit_info["img_url"], hit_info["full_url"], hit_info["plate_img_url"], hit_info["license_plate"]))
                                                    hit_info["id"] = cursor.lastrowid
                                                    conn.commit()
                                            except Exception as e:
                                                print(f"Error saving to DB: {e}")
                                                
                                            target_veh_img = find_target_vehicle_image_path(name)
                                            send_telegram_alert_async(
                                                target_type="vehicle",
                                                target_name=display_name,
                                                match_pct=match_percentage,
                                                camera_name=self.name,
                                                live_crop_img=v_crop_img,
                                                target_img_path=target_veh_img,
                                                plate_text=plate_text
                                            )
                                    
                                    elif plate_text and track_id not in self.track_lpr_logged and track_id not in self.tracked_target_ids:
                                        self.track_lpr_logged.add(track_id)
                                        padding = 100
                                        h_img, w_img = frame.shape[:2]
                                        cx1, cy1 = max(0, x1 - padding), max(0, y1 - padding)
                                        cx2, cy2 = min(w_img, x2 + padding), min(h_img, y2 + padding)
                                        v_crop_img = frame[cy1:cy2, cx1:cx2]
                                        
                                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        img_name = f"lpr_{track_id}_{timestamp_str}.jpg"
                                        img_name_full = f"lpr_{track_id}_{timestamp_str}_full.jpg"
                                        
                                        save_high_quality_crop(v_crop_img, os.path.join('alerts', img_name))
                                        cv2.imwrite(os.path.join('alerts', img_name_full), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                        
                                        cam_url_str = str(self.url)
                                        hit_info_name = f"[LPR] {plate_text} (#{track_id})"
                                        hit_info = {
                                            "name": hit_info_name,
                                            "time": datetime.now().strftime("%H:%M:%S"),
                                            "confidence": self.track_lpr_results.get(track_id, {}).get("conf", 80),
                                            "img_url": f"http://localhost:8081/alerts_img/{img_name}",
                                            "full_url": f"http://localhost:8081/alerts_img/{img_name_full}",
                                            "is_color_edited": 0,
                                            "plate_img_url": f"http://localhost:8081/alerts_img/{plate_img_name}" if plate_img_name else None,
                                            "license_plate": plate_text,
                                            "camera_id": self.cam_id,
                                            "camera_name": self.name
                                        }
                                        if cam_url_str not in latest_hits: latest_hits[cam_url_str] = []
                                        latest_hits[cam_url_str].insert(0, hit_info)
                                        if len(latest_hits[cam_url_str]) > 10: latest_hits[cam_url_str].pop()
                                        
                                        try:
                                            with get_db_connection() as conn:
                                                cursor = conn.cursor()
                                                cursor.execute('''
                                                    INSERT INTO hits (camera_url, name, timestamp, confidence, img_url, full_url, plate_img_url, license_plate)
                                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                                ''', (cam_url_str, hit_info_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), hit_info["confidence"], hit_info["img_url"], hit_info["full_url"], hit_info["plate_img_url"], hit_info["license_plate"]))
                                                hit_info["id"] = cursor.lastrowid
                                                conn.commit()
                                        except Exception as e:
                                            print(f"Error saving LPR hit to DB: {e}")
                                        
                                        send_telegram_alert_async(
                                            target_type="plate",
                                            target_name=f"ป้ายทะเบียน {plate_text}",
                                            match_pct=self.track_lpr_results.get(track_id, {}).get("conf", 80),
                                            camera_name=self.name,
                                            live_crop_img=v_crop_img,
                                            plate_text=plate_text
                                        )

                                if track_id not in self.tracked_ids:
                                    self.tracked_ids.add(track_id)
                            continue

                        # Person Face Recognition (cls_id == 0)
                        person_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        if person_crop.size == 0:
                            continue
                            
                        rgb_person = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                        crop_min_side = min(person_crop.shape[0], person_crop.shape[1])
                        if crop_min_side >= 300:
                            fr_upsample = 0
                        elif crop_min_side >= 160:
                            fr_upsample = 1
                        else:
                            fr_upsample = 2
                            
                        face_locations = face_recognition.face_locations(rgb_person, model="hog", number_of_times_to_upsample=fr_upsample)
                        is_target = False
                        
                        if face_locations:
                            with targets_lock:
                                curr_face_encodings = list(known_face_encodings)
                                curr_face_names = list(known_face_names)
                                
                            if len(curr_face_encodings) > 0:
                                face_encodings = face_recognition.face_encodings(rgb_person, face_locations)
                                match_tolerance = 1.0 - (face_match_threshold / 100.0)
                                for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                                    face_distances = face_recognition.face_distance(curr_face_encodings, face_encoding)
                                    if len(face_distances) > 0:
                                        best_match_index = int(np.argmin(face_distances))
                                        best_distance = float(face_distances[best_match_index])
                                        name = curr_face_names[best_match_index]
                                        
                                        other_distances = [d for d, nm in zip(face_distances, curr_face_names) if nm != name]
                                        is_ambiguous = bool(other_distances) and (min(other_distances) - best_distance) < FACE_MATCH_MARGIN
                                        
                                        if best_distance <= match_tolerance and not is_ambiguous:
                                            match_percentage = round((1 - best_distance) * 100)
                                            is_target = True
                                            
                                            fx1, fy1 = x1 + left, y1 + top
                                            fx2, fy2 = x1 + right, y1 + bottom
                                            
                                            cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 3)
                                            annotated_frame = put_thai_text(annotated_frame, f"TARGET: {name} (#{track_id})", (fx1, fy2 + 25), (0, 0, 255))
                                            
                                            if track_id not in self.tracked_target_ids or self.tracked_target_ids[track_id] != name:
                                                self.tracked_target_ids[track_id] = name
                                                padding = 100
                                                h, w = frame.shape[:2]
                                                cx1, cy1 = max(0, x1 - padding), max(0, y1 - padding)
                                                cx2, cy2 = min(w, x2 + padding), min(h, y2 + padding)
                                                face_crop_img = frame[cy1:cy2, cx1:cx2]
                                                
                                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                                img_name = f"{name}_{track_id}_{timestamp}.jpg"
                                                img_name_full = f"{name}_{track_id}_{timestamp}_full.jpg"
                                                
                                                save_high_quality_crop(face_crop_img, os.path.join('alerts', img_name))
                                                cv2.imwrite(os.path.join('alerts', img_name_full), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                                
                                                cam_url_str = str(self.url)
                                                if cam_url_str not in target_hit_counts:
                                                    target_hit_counts[cam_url_str] = {}
                                                target_hit_counts[cam_url_str][name] = target_hit_counts[cam_url_str].get(name, 0) + 1
                                                
                                                hit_info_name = f"{name} (#{track_id})"
                                                hit_info = {
                                                    "name": hit_info_name,
                                                    "time": datetime.now().strftime("%H:%M:%S"),
                                                    "confidence": match_percentage,
                                                    "img_url": f"http://localhost:8081/alerts_img/{img_name}",
                                                    "full_url": f"http://localhost:8081/alerts_img/{img_name_full}",
                                                    "is_color_edited": 0,
                                                    "camera_id": self.cam_id,
                                                    "camera_name": self.name
                                                }
                                                if cam_url_str not in latest_hits:
                                                    latest_hits[cam_url_str] = []
                                                latest_hits[cam_url_str].insert(0, hit_info)
                                                if len(latest_hits[cam_url_str]) > 10: latest_hits[cam_url_str].pop()
                                                
                                                try:
                                                    with get_db_connection() as conn:
                                                        cursor = conn.cursor()
                                                        cursor.execute('''
                                                            INSERT INTO hits (camera_url, name, timestamp, confidence, img_url, full_url)
                                                            VALUES (?, ?, ?, ?, ?, ?)
                                                        ''', (cam_url_str, hit_info_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), match_percentage, hit_info["img_url"], hit_info["full_url"]))
                                                        hit_info["id"] = cursor.lastrowid
                                                        conn.commit()
                                                except Exception as e:
                                                    print(f"Error saving to DB: {e}")
                                                    
                                                target_face_img = find_target_face_image_path(name)
                                                send_telegram_alert_async(
                                                    target_type="face",
                                                    target_name=name,
                                                    match_pct=match_percentage,
                                                    camera_name=self.name,
                                                    live_crop_img=face_crop_img,
                                                    target_img_path=target_face_img
                                                )

                        if track_id not in self.tracked_ids:
                            self.tracked_ids.add(track_id)
                            if not is_target:
                                self.unknown_persons_count += 1
                                if save_unknown_faces:
                                    try:
                                        timestamp_unknown = int(time.time())
                                        img_name_unknown = f"unknown_{track_id}_{timestamp_unknown}.jpg"
                                        full_path_unknown = os.path.join('alerts', img_name_unknown)
                                        cv2.imwrite(full_path_unknown, frame)
                                        
                                        cam_url_str = str(self.url)
                                        url_unknown = f"http://localhost:8081/alerts_img/{img_name_unknown}"
                                        with get_db_connection() as conn:
                                            cursor = conn.cursor()
                                            cursor.execute('''
                                                INSERT INTO hits (camera_url, name, timestamp, confidence, img_url, full_url)
                                                VALUES (?, ?, ?, ?, ?, ?)
                                            ''', (cam_url_str, "บุคคลทั่วไป", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0.0, url_unknown, url_unknown))
                                            conn.commit()
                                    except Exception as e:
                                        print(f"Error saving unknown to DB: {e}")

                with self.frame_lock:
                    self.latest_annotated_frame = annotated_frame.copy()

                if self.is_video_source:
                    elapsed = time.time() - current_time
                    if elapsed < target_delay:
                        time.sleep(target_delay - elapsed)

class CameraManager:
    def __init__(self):
        self.workers = {} # cam_id -> CameraWorker
        self.active_cam_id = None
        self.lock = threading.Lock()

    def initialize_from_json(self):
        cams = load_cameras()
        if not cams:
            cams = DEFAULT_CAMERAS
            save_cameras(cams)
            
        with self.lock:
            for cam in cams:
                cid = cam.get("id")
                if not cid:
                    cid = f"cam_{int(time.time())}_{len(self.workers)}"
                    cam["id"] = cid
                worker = CameraWorker(cid, cam.get("name", cid), cam.get("url", "0"), cam.get("config", {}))
                self.workers[cid] = worker
                worker.start()
                
            if cams:
                self.active_cam_id = cams[0]["id"]
                if self.active_cam_id in self.workers:
                    self.workers[self.active_cam_id].play()

    def get_worker(self, cam_id_or_url=None):
        with self.lock:
            if not self.workers:
                return None
            if cam_id_or_url is None or cam_id_or_url == "" or cam_id_or_url == "default":
                if self.active_cam_id and self.active_cam_id in self.workers:
                    return self.workers[self.active_cam_id]
                return next(iter(self.workers.values()))
            
            # Lookup by id
            if cam_id_or_url in self.workers:
                return self.workers[cam_id_or_url]
            
            # Lookup by url
            for w in self.workers.values():
                if str(w.url) == str(cam_id_or_url):
                    return w
            return None

    def get_active_worker(self):
        return self.get_worker(self.active_cam_id)

    def set_active_camera(self, cam_id_or_url):
        with self.lock:
            w = None
            if cam_id_or_url in self.workers:
                w = self.workers[cam_id_or_url]
            else:
                for worker in self.workers.values():
                    if str(worker.url) == str(cam_id_or_url):
                        w = worker
                        break
            if w:
                self.active_cam_id = w.cam_id
                return w
            return None

    def add_camera(self, name, url):
        with self.lock:
            # Check if url already exists
            for w in self.workers.values():
                if str(w.url) == str(url):
                    w.name = name
                    return w
            cid = f"cam_{int(time.time())}"
            worker = CameraWorker(cid, name, url)
            self.workers[cid] = worker
            worker.start()
            
            cams = load_cameras()
            cams.append({"id": cid, "name": name, "url": url})
            save_cameras(cams)
            return worker

    def remove_camera(self, cam_id):
        with self.lock:
            if cam_id in self.workers:
                w = self.workers.pop(cam_id)
                w.close()
                
                cams = load_cameras()
                cams = [c for c in cams if c.get("id") != cam_id]
                save_cameras(cams)
                
                if self.active_cam_id == cam_id:
                    self.active_cam_id = next(iter(self.workers.keys())) if self.workers else None
                return True
            return False

    def play_all(self):
        with self.lock:
            for w in self.workers.values():
                w.play()

    def play_selected(self, cam_ids):
        cam_id_set = set(str(cid) for cid in cam_ids if cid)
        with self.lock:
            for cid, w in self.workers.items():
                if str(cid) in cam_id_set:
                    w.play()
                else:
                    w.stop()

    def pause_all(self):
        with self.lock:
            for w in self.workers.values():
                w.pause()

    def pause_selected(self, cam_ids):
        cam_id_set = set(str(cid) for cid in cam_ids if cid)
        with self.lock:
            for cid, w in self.workers.items():
                if str(cid) in cam_id_set:
                    w.pause()

    def stop_all(self):
        with self.lock:
            for w in self.workers.values():
                w.stop()

    def stop_selected(self, cam_ids):
        cam_id_set = set(str(cid) for cid in cam_ids if cid)
        with self.lock:
            for cid, w in self.workers.items():
                if str(cid) in cam_id_set:
                    w.stop()

    def get_all_statuses(self):
        with self.lock:
            return [w.get_status_dict() for w in self.workers.values()]

camera_manager = CameraManager()
camera_manager.initialize_from_json()

def generate_frames_for_cam(cam_id=None):
    while True:
        worker = camera_manager.get_worker(cam_id)
        frame_to_send = None
        if worker:
            with worker.frame_lock:
                if worker.latest_annotated_frame is not None:
                    frame_to_send = worker.latest_annotated_frame.copy()
        
        if frame_to_send is None:
            name = worker.name if worker else (cam_id or "Camera")
            frame_to_send = get_black_frame(f"Connecting to {name}...")
            
        ret, buffer = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04) # ~25 FPS limit for stream output

@app.get("/video_feed")
def video_feed(camera_id: str = None):
    return StreamingResponse(generate_frames_for_cam(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed/{camera_id}")
def video_feed_cam(camera_id: str):
    return StreamingResponse(generate_frames_for_cam(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/snapshot")
def get_snapshot(camera_id: str = None):
    worker = camera_manager.get_worker(camera_id)
    if not worker:
        placeholder = get_black_frame("Camera not found")
        ret, buffer = cv2.imencode('.jpg', placeholder)
        if ret:
            return Response(content=buffer.tobytes(), media_type="image/jpeg")
        return JSONResponse(status_code=404, content={"message": "Camera not found"})
        
    with worker.frame_lock:
        raw_frame = worker.latest_raw_frame if worker.latest_raw_frame is not None else worker.latest_annotated_frame
        if raw_frame is None:
            placeholder = get_black_frame(f"Waiting for {worker.name}...")
            ret, buffer = cv2.imencode('.jpg', placeholder)
            if ret:
                return Response(content=buffer.tobytes(), media_type="image/jpeg")
            return JSONResponse(status_code=500, content={"message": "Error encoding image"})
            
        ret, buffer = cv2.imencode('.jpg', raw_frame)
        if ret:
            return Response(content=buffer.tobytes(), media_type="image/jpeg")
        return JSONResponse(status_code=500, content={"message": "Error encoding image"})

@app.get("/snapshot/{camera_id}")
def get_snapshot_cam(camera_id: str):
    return get_snapshot(camera_id)

@app.post("/set_screen_region")
def set_screen_region(payload: dict = Body(...)):
    cam_id = payload.get("camera_id")
    worker = camera_manager.get_worker(cam_id)
    if not worker:
        return {"status": "error", "message": "Camera not found"}
        
    if isinstance(worker.cap, ScreenGrabber):
        if payload.get("reset"):
            worker.cap.reset_region()
            return {"status": "success", "message": "คืนค่าแคปเต็มหน้าจอเรียบร้อยแล้ว"}
        
        worker.cap.set_region(
            payload.get("left", 0.0),
            payload.get("top", 0.0),
            payload.get("width", 1.0),
            payload.get("height", 1.0)
        )
        return {"status": "success", "message": "กำหนดพื้นที่แคปหน้าจอสำเร็จ"}
    return {"status": "error", "message": "กล้องปัจจุบันไม่ใช่การแคปหน้าจอ"}

@app.get("/stats")
def get_stats(camera_id: str = None):
    worker = camera_manager.get_worker(camera_id)
    if not worker:
        return {
            "fps": 0.0,
            "known_faces_count": len(known_face_names),
            "unknown_persons_count": 0,
            "line_cross_count": 0,
            "camera_url": "",
            "camera": "None",
            "camera_id": "",
            "target_stats": {}
        }
    st = worker.get_status_dict()
    return {
        "fps": st["fps"],
        "known_faces_count": len(known_face_names),
        "unknown_persons_count": st["unknown_count"],
        "line_cross_count": st["line_cross_count"],
        "camera_url": str(st["url"]),
        "camera": st["name"],
        "camera_id": st["id"],
        "target_stats": st["target_stats"]
    }

@app.get("/stats/{camera_id}")
def get_stats_cam(camera_id: str):
    return get_stats(camera_id)

@app.get("/system_status")
def system_status(camera_id: str = None):
    worker = camera_manager.get_worker(camera_id)
    active_st = worker.get_status_dict() if worker else {
        "id": "none", "name": "None", "url": "0", "stream_state": "stopped",
        "fps": 0, "unknown_count": 0, "line_cross_count": 0, "target_stats": {},
        "is_video": False, "video_position": 0, "video_duration": 0,
        "video_total_frames": 0, "video_current_frame": 0, "frame_skip": 0
    }
    all_cams = camera_manager.get_all_statuses()
    
    total_unknown = sum(c.get("unknown_count", 0) for c in all_cams)
    total_line_cross = sum(c.get("line_cross_count", 0) for c in all_cams)
    
    return {
        "status": "running",
        "stream_state": active_st["stream_state"],
        "fps": active_st["fps"],
        "active_camera": str(active_st["url"]),
        "active_camera_id": active_st["id"],
        "active_camera_name": active_st["name"],
        "db_size": len(set(known_face_names)),
        "unknown_count": active_st["unknown_count"],
        "total_unknown_count": total_unknown,
        "line_cross_count": active_st["line_cross_count"],
        "total_line_cross_count": total_line_cross,
        "target_stats": active_st["target_stats"],
        "is_video": active_st["is_video"],
        "video_position": active_st["video_position"],
        "video_duration": active_st["video_duration"],
        "video_total_frames": active_st["video_total_frames"],
        "video_current_frame": active_st["video_current_frame"],
        "frame_skip": active_st["frame_skip"],
        "cameras": all_cams
    }

@app.post("/seek_video")
def seek_video(payload: dict = Body(...)):
    cam_id = payload.get("camera_id")
    worker = camera_manager.get_worker(cam_id)
    if not worker:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Camera not found"})
        
    fraction = payload.get("fraction")
    seconds = payload.get("seconds")
    if fraction is None and seconds is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Missing fraction or seconds"})

    if worker.video_total_frames <= 0:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Current source is not seekable"})

    fps = worker.video_native_fps if worker.video_native_fps and worker.video_native_fps > 0 else 25.0
    if fraction is not None:
        fraction = max(0.0, min(1.0, float(fraction)))
        target_frame = int(round(fraction * (worker.video_total_frames - 1)))
    else:
        seconds = max(0.0, float(seconds))
        target_frame = int(round(seconds * fps))
        target_frame = max(0, min(target_frame, worker.video_total_frames - 1))

    worker.seek(target_frame)
    return {"status": "success", "target_frame": target_frame}

@app.post("/play_stream")
def play_stream(camera_id: str = None, payload: dict = Body(None)):
    cid = camera_id or (payload.get("camera_id") if payload else None)
    w = camera_manager.get_worker(cid)
    if w:
        w.play()
        return {"status": "success", "message": f"เริ่มประมวลผล {w.name} แล้ว"}
    return {"status": "error", "message": "Camera not found"}

@app.post("/pause_stream")
def pause_stream(camera_id: str = None, payload: dict = Body(None)):
    cid = camera_id or (payload.get("camera_id") if payload else None)
    w = camera_manager.get_worker(cid)
    if w:
        w.pause()
        return {"status": "success", "message": f"พักการประมวลผล {w.name} แล้ว"}
    return {"status": "error", "message": "Camera not found"}

@app.post("/stop_stream")
def stop_stream(camera_id: str = None, payload: dict = Body(None)):
    cid = camera_id or (payload.get("camera_id") if payload else None)
    w = camera_manager.get_worker(cid)
    if w:
        w.stop()
        return {"status": "success", "message": f"หยุดประมวลผล {w.name} แล้ว"}
    return {"status": "error", "message": "Camera not found"}

@app.post("/cameras/{camera_id}/play")
def play_specific_camera(camera_id: str):
    w = camera_manager.get_worker(camera_id)
    if not w:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Camera not found"})
    w.play()
    return {"status": "success", "message": f"Started {w.name}"}

@app.post("/cameras/{camera_id}/pause")
def pause_specific_camera(camera_id: str):
    w = camera_manager.get_worker(camera_id)
    if not w:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Camera not found"})
    w.pause()
    return {"status": "success", "message": f"Paused {w.name}"}

@app.post("/cameras/{camera_id}/stop")
def stop_specific_camera(camera_id: str):
    w = camera_manager.get_worker(camera_id)
    if not w:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Camera not found"})
    w.stop()
    return {"status": "success", "message": f"Stopped {w.name}"}

@app.post("/cameras/play_all")
def play_all_cameras():
    camera_manager.play_all()
    return {"status": "success", "message": "เริ่มประมวลผลทุกกล้องแล้ว"}

@app.post("/cameras/play_selected")
def play_selected_cameras(payload: dict = Body(...)):
    ids = payload.get("ids", [])
    camera_manager.play_selected(ids)
    return {"status": "success", "message": f"เริ่มประมวลผล {len(ids)} กล้องที่เลือกแล้ว"}

@app.post("/cameras/pause_all")
def pause_all_cameras():
    camera_manager.pause_all()
    return {"status": "success", "message": "พักการประมวลผลทุกกล้องแล้ว"}

@app.post("/cameras/pause_selected")
def pause_selected_cameras(payload: dict = Body(...)):
    ids = payload.get("ids", [])
    camera_manager.pause_selected(ids)
    return {"status": "success", "message": f"พักการประมวลผล {len(ids)} กล้องที่เลือกแล้ว"}

@app.post("/cameras/stop_all")
def stop_all_cameras():
    camera_manager.stop_all()
    return {"status": "success", "message": "หยุดประมวลผลทุกกล้องแล้ว"}
@app.post("/cameras/stop_selected")
def stop_selected_cameras(payload: dict = Body(...)):
    ids = payload.get("ids", [])
    camera_manager.stop_selected(ids)
    return {"status": "success", "message": f"หยุดประมวลผล {len(ids)} กล้องที่เลือกแล้ว"}

@app.post("/set_detection_zone")
async def set_detection_zone(payload: dict = Body(...)):
    cam_id = payload.get("camera_id")
    worker = camera_manager.get_worker(cam_id)
    if not worker:
        return {"status": "error", "message": "Camera not found"}
        
    if payload.get("reset"):
        worker.detection_zone = None
        return {"status": "success", "message": "ยกเลิกพื้นที่ตรวจจับแล้ว"}
        
    zone = payload.get("points", [])
    if len(zone) < 3:
        worker.detection_zone = None
        return {"status": "error", "message": "กรุณาเลือกจุดอย่างน้อย 3 จุด"}
    else:
        worker.detection_zone = zone
        return {"status": "success", "message": "ตั้งค่าพื้นที่ตรวจจับเรียบร้อย"}

@app.post("/set_crossing_line")
async def set_crossing_line(payload: dict = Body(...)):
    cam_id = payload.get("camera_id")
    worker = camera_manager.get_worker(cam_id)
    if not worker:
        return {"status": "error", "message": "Camera not found"}
        
    if payload.get("reset"):
        worker.crossing_line = None
        worker.crossing_direction = "any"
        worker.track_history.clear()
        worker.line_cross_count = 0
        return {"status": "success", "message": "ยกเลิกเส้นตรวจจับแล้ว"}
        
    line_pts = payload.get("points", [])
    direction = payload.get("direction", "any")
    
    if len(line_pts) != 2:
        return {"status": "error", "message": "กรุณาลากเส้น 2 จุดเท่านั้น"}
    else:
        worker.crossing_line = line_pts
        worker.crossing_direction = direction
        worker.track_history.clear()
        worker.line_cross_count = 0
        return {"status": "success", "message": "ตั้งค่าเส้นข้ามสำเร็จ"}

@app.post("/toggle_show_zone")
async def toggle_show_zone(payload: dict = Body(...)):
    cam_id = payload.get("camera_id")
    worker = camera_manager.get_worker(cam_id)
    show = payload.get("show", True)
    if worker:
        worker.show_detection_zone = show
    global show_detection_zone, config
    show_detection_zone = show
    config["show_detection_zone"] = show_detection_zone
    save_config(config)
    status_msg = "แสดง" if show else "ซ่อน"
    return {"status": "success", "message": f"{status_msg}พื้นที่ตรวจจับแล้ว"}

@app.get("/history")
def get_history(
    start_date: str = None, 
    end_date: str = None, 
    name: str = None, 
    person_type: str = None,
    license_plate: str = None,
    vehicle_type: str = None,
    vehicle_color: str = None
):
    try:
        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM hits WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(f"{start_date} 00:00:00")
            if end_date:
                query += " AND timestamp <= ?"
                params.append(f"{end_date} 23:59:59")
            if name:
                query += " AND name LIKE ?"
                params.append(f"%{name}%")
            if license_plate:
                plate_digits = extract_digits(license_plate)
                if plate_digits and len(plate_digits) >= 3:
                    query += " AND (license_plate LIKE ? OR license_plate LIKE ? OR name LIKE ? OR name LIKE ?)"
                    params.append(f"%{license_plate}%")
                    params.append(f"%{plate_digits}%")
                    params.append(f"%{license_plate}%")
                    params.append(f"%{plate_digits}%")
                else:
                    query += " AND (license_plate LIKE ? OR name LIKE ?)"
                    params.append(f"%{license_plate}%")
                    params.append(f"%{license_plate}%")

            if vehicle_type:
                query += " AND name LIKE ?"
                params.append(f"%{vehicle_type}%")
            if vehicle_color:
                query += " AND name LIKE ?"
                params.append(f"%{vehicle_color}%")
            
            if person_type == "target":
                query += " AND name != ? AND name NOT LIKE ?"
                params.append("บุคคลทั่วไป")
                params.append("[LPR]%")
            elif person_type == "plate_target":
                query += " AND (name LIKE ? OR name LIKE ?)"
                params.append("%ทะเบียนเป้าหมาย%")
                params.append("%💳%")
            elif person_type == "unknown":
                query += " AND (name = ? OR name LIKE ?)"
                params.append("บุคคลทั่วไป")
                params.append("[LPR]%")

            query += " ORDER BY timestamp DESC LIMIT 500"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    "id": row["id"],
                    "camera_url": row["camera_url"],
                    "name": row["name"],
                    "timestamp": row["timestamp"],
                    "confidence": row["confidence"],
                    "img_url": row["img_url"],
                    "full_url": row["full_url"],
                    "is_color_edited": row["is_color_edited"] if "is_color_edited" in row.keys() else 0,
                    "plate_img_url": row["plate_img_url"] if "plate_img_url" in row.keys() else None,
                    "license_plate": row["license_plate"] if "license_plate" in row.keys() else None
                })
            
            return {"status": "success", "data": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.delete("/history/bulk_delete")
def bulk_delete_history(payload: dict = Body(...)):
    raw_ids = payload.get("ids", [])
    if not raw_ids:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No IDs provided"})
    try:
        ids = [int(i) for i in raw_ids if str(i).lstrip('-').isdigit()]
        if not ids:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid IDs provided"})
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 1. ค้นหา url ของรูปภาพทั้งหมดที่จะถูกลบ
            placeholders = ','.join(['?'] * len(ids))
            cursor.execute(f'SELECT img_url, full_url, plate_img_url FROM hits WHERE id IN ({placeholders})', ids)
            rows = cursor.fetchall()
            
            urls_to_remove = set()
            for row in rows:
                for col_idx in range(len(row)):
                    url = row[col_idx]
                    if url:
                        urls_to_remove.add(url)
                        if "alerts_img/" in url:
                            img_name = os.path.basename(url.split("alerts_img/")[-1])
                            img_path = os.path.join("alerts", img_name)
                            if os.path.exists(img_path):
                                try:
                                    os.remove(img_path)
                                except Exception:
                                    pass
            
            # 2. ลบออกจาก DB
            cursor.execute(f'DELETE FROM hits WHERE id IN ({placeholders})', ids)
            conn.commit()
            
            # 3. ลบออกจาก latest_hits (เพื่อให้หน้า Dashboard หายไปด้วย)
            for cam_url in list(latest_hits.keys()):
                latest_hits[cam_url] = [hit for hit in latest_hits[cam_url] if hit.get("img_url") not in urls_to_remove]
                
            return {"status": "success", "message": f"Deleted {len(ids)} records"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/train_color")
def train_color(data: dict = Body(...)):
    hit_id = data.get("hit_id")
    correct_color = data.get("correct_color")
    
    if hit_id is not None:
        try:
            hit_id = int(hit_id)
        except (ValueError, TypeError):
            pass
            
    if not hit_id or not correct_color:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Missing hit_id or correct_color"})
        
    try:
        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT img_url FROM hits WHERE id = ?", (hit_id,))
            row = cursor.fetchone()
            
            if not row or not row['img_url']:
                return JSONResponse(status_code=404, content={"status": "error", "message": "Hit not found or no image available"})
                
            img_url = row['img_url']
            if "alerts_img/" in img_url:
                filename = os.path.basename(img_url.split("alerts_img/")[-1])
                filepath = os.path.join("alerts", filename)
                
                if not os.path.exists(filepath):
                    return JSONResponse(status_code=404, content={"status": "error", "message": "Image file not found"})
                    
                # Read image and extract HSV
                img = cv2.imread(filepath)
                if img is None:
                    return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to read image"})
                    
                # We need to extract dominant color like in get_vehicle_color
                hsv_vals = extract_vehicle_hsv(img)
                if hsv_vals is None:
                    return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to extract color from vehicle image"})
                h_val, s_val, v_val = hsv_vals
                
                # Insert into color_training
                cursor.execute('''
                    INSERT INTO color_training (h, s, v, correct_color, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (h_val, s_val, v_val, correct_color, datetime.now()))
                
                # Update hits table: mark is_color_edited = 1 and correct the vehicle name
                cursor.execute("SELECT name FROM hits WHERE id = ?", (hit_id,))
                row_hit = cursor.fetchone()
                new_name = None
                if row_hit and row_hit['name']:
                    old_name = row_hit['name']
                    colors_list = ['ขาว', 'ดำ', 'เทา', 'แดง', 'น้ำเงิน', 'เขียว', 'เหลือง', 'ส้ม', 'ม่วง', 'ไม่ระบุสี']
                    new_name = old_name
                    for c in colors_list:
                        if c in old_name:
                            new_name = old_name.replace(f"สี{c}", f"สี{correct_color}").replace(c, correct_color)
                            break
                    else:
                        if " (#" in old_name:
                            parts = old_name.split(" (#")
                            new_name = f"{parts[0]}สี{correct_color} (#{parts[1]}"
                    
                    cursor.execute("UPDATE hits SET name = ?, is_color_edited = 1 WHERE id = ?", (new_name, hit_id))
                else:
                    cursor.execute("UPDATE hits SET is_color_edited = 1 WHERE id = ?", (hit_id,))
                
                conn.commit()
                
                # Sync in-memory latest_hits list
                sync_count = 0
                for cam_url in latest_hits:
                    for hit in latest_hits[cam_url]:
                        if hit.get("id") is not None and str(hit.get("id")) == str(hit_id):
                            hit["is_color_edited"] = 1
                            if new_name:
                                hit["name"] = new_name
                            sync_count += 1
                print(f"[DEBUG] Synced corrected name to {sync_count} in-memory hits for hit_id={hit_id}")
                
                return {"status": "success", "message": f"แก้ไขสีรถยนต์เป็นสี {correct_color} เรียบร้อยแล้ว!"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.delete("/history/{hit_id}")
def delete_history(hit_id: int):
    try:
        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get the record to delete physical files
            cursor.execute("SELECT img_url, full_url, plate_img_url FROM hits WHERE id = ?", (hit_id,))
            row = cursor.fetchone()
            
            if not row:
                return JSONResponse(status_code=404, content={"status": "error", "message": "ไม่พบข้อมูลประวัตินี้"})
                
            url_to_remove = row['img_url']
            
            # Attempt to delete physical files from alerts/ directory
            for url_field in ['img_url', 'full_url', 'plate_img_url']:
                if url_field in row.keys():
                    url = row[url_field]
                    if url and "alerts_img/" in url:
                        filename = os.path.basename(url.split("alerts_img/")[-1])
                        filepath = os.path.join("alerts", filename)
                        if os.path.exists(filepath):
                            try:
                                os.remove(filepath)
                            except Exception:
                                pass
                            
            # Delete from database
            cursor.execute("DELETE FROM hits WHERE id = ?", (hit_id,))
            conn.commit()
            
            # Update latest_hits memory
            if url_to_remove:
                for cam_url in list(latest_hits.keys()):
                    latest_hits[cam_url] = [hit for hit in latest_hits[cam_url] if hit.get("img_url") != url_to_remove]
            
        return {"status": "success", "message": "ลบข้อมูลเรียบร้อยแล้ว"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/target_hits")
def get_target_hits(camera_id: str = None):
    worker = camera_manager.get_worker(camera_id)
    cam_key = str(worker.url) if worker else str(RTSP_URL)
    
    # 1. Look up in-memory latest_hits with exact key or partial/prefix key match
    hits = latest_hits.get(cam_key)
    if not hits:
        for k, v in latest_hits.items():
            if v and (cam_key in k or k in cam_key or (cam_key.startswith("screen") and k.startswith("screen"))):
                hits = v
                break
    
    if not hits:
        hits = []

    # 2. If memory is empty, query DB for latest 10 hits matching camera_url or prefix
    if not hits:
        try:
            with get_db_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                clean_prefix = cam_key.split(":")[0] if ":" in cam_key else cam_key
                cursor.execute('''
                    SELECT id, camera_url, name, timestamp, confidence, img_url, full_url, is_color_edited, plate_img_url, license_plate
                    FROM hits
                    WHERE camera_url = ? OR camera_url LIKE ? OR camera_url LIKE ?
                    ORDER BY timestamp DESC LIMIT 10
                ''', (cam_key, f"{cam_key}%", f"%{clean_prefix}%"))
                rows = cursor.fetchall()
                loaded = []
                for row in rows:
                    ts = row["timestamp"]
                    time_str = ts.split(" ")[1] if (ts and " " in ts) else ts
                    loaded.append({
                        "id": row["id"],
                        "name": row["name"],
                        "time": time_str,
                        "confidence": row["confidence"],
                        "img_url": row["img_url"],
                        "full_url": row["full_url"],
                        "is_color_edited": row["is_color_edited"] if "is_color_edited" in row.keys() else 0,
                        "plate_img_url": row["plate_img_url"] if "plate_img_url" in row.keys() else None,
                        "license_plate": row["license_plate"] if "license_plate" in row.keys() else None
                    })
                hits = loaded
                if loaded:
                    latest_hits[cam_key] = loaded
        except Exception as e:
            print(f"Error querying DB for latest hits: {e}")

    # 3. Fallback: If still empty, fetch the 10 most recent hits overall from DB
    if not hits:
        try:
            with get_db_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, camera_url, name, timestamp, confidence, img_url, full_url, is_color_edited, plate_img_url, license_plate
                    FROM hits
                    ORDER BY timestamp DESC LIMIT 10
                ''')
                rows = cursor.fetchall()
                loaded = []
                for row in rows:
                    ts = row["timestamp"]
                    time_str = ts.split(" ")[1] if (ts and " " in ts) else ts
                    loaded.append({
                        "id": row["id"],
                        "name": row["name"],
                        "time": time_str,
                        "confidence": row["confidence"],
                        "img_url": row["img_url"],
                        "full_url": row["full_url"],
                        "is_color_edited": row["is_color_edited"] if "is_color_edited" in row.keys() else 0,
                        "plate_img_url": row["plate_img_url"] if "plate_img_url" in row.keys() else None,
                        "license_plate": row["license_plate"] if "license_plate" in row.keys() else None
                    })
                hits = loaded
        except Exception as e:
            print(f"Error querying fallback DB hits: {e}")

    if show_only_targets:
        return [h for h in hits if not h.get("name", "").startswith("[LPR]") and h.get("name", "") != "บุคคลทั่วไป"]
    return hits

@app.post("/change_camera")
async def change_camera(camera_url: str = Form(...), frame_skip: int = Form(0), camera_id: str = Form(None)):
    global RTSP_URL, current_frame_skip
    
    worker = None
    if camera_id:
        worker = camera_manager.get_worker(camera_id)
    if not worker:
        worker = camera_manager.get_worker(camera_url)
    if not worker:
        if not validate_camera_source(camera_url):
            return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid camera source"})
        worker = camera_manager.add_camera(f"Camera ({sanitized_text(camera_url, max_len=80)})", camera_url)
        
    camera_manager.set_active_camera(worker.cam_id)
    frame_skip = max(0, min(30, int(frame_skip)))
    worker.frame_skip = frame_skip
    worker.play()
    RTSP_URL = worker.url
    current_frame_skip = frame_skip
    return {"status": "success", "message": f"Changed camera to {worker.name}", "camera_id": worker.cam_id}

@app.post("/set_frame_skip")
async def set_frame_skip(frame_skip: int = Form(...), camera_id: str = Form(None)):
    global current_frame_skip
    frame_skip = max(0, min(30, int(frame_skip)))
    current_frame_skip = frame_skip
    if camera_id:
        worker = camera_manager.get_worker(camera_id)
        if worker:
            worker.frame_skip = frame_skip
    else:
        for w in camera_manager.workers.values():
            w.frame_skip = frame_skip
    return {"status": "success", "message": f"Frame skip set to {frame_skip}"}

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    global RTSP_URL
    try:
        timestamp = int(time.time())
        raw_ext = os.path.splitext(file.filename or "")[1].lower()
        if raw_ext not in ALLOWED_VIDEO_EXTS:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Unsupported video file type"})
        ext = raw_ext
        
        filename = f"video_{timestamp}{ext}"
        filepath = os.path.join("uploads", filename)
        
        try:
            await save_upload_limited(file, filepath, MAX_VIDEO_UPLOAD_BYTES)
        except ValueError as e:
            return JSONResponse(status_code=413, content={"status": "error", "message": str(e)})
            
        safe_display_name = sanitized_text(os.path.basename(file.filename or filename), default=filename, max_len=120)
        worker = camera_manager.add_camera(f"Video: {safe_display_name}", filepath)
        camera_manager.set_active_camera(worker.cam_id)
        worker.pause() # Wait for play button
        RTSP_URL = filepath
        
        return {"status": "success", "message": f"อัปโหลดและสลับไปยังวิดีโอ {safe_display_name} เรียบร้อยแล้ว", "camera_id": worker.cam_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/add_youtube")
async def add_youtube(youtube_url: str = Form(...), frame_skip: int = Form(0)):
    global RTSP_URL
    try:
        if not validate_youtube_url(youtube_url):
            return JSONResponse(status_code=400, content={"status": "error", "message": "รองรับเฉพาะลิงก์ YouTube เท่านั้น"})
        ydl_opts = {'format': 'best', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            video_url = info_dict.get('url', None)
            title = info_dict.get('title', 'YouTube Video')
            
        if not video_url:
            return {"status": "error", "message": "ไม่สามารถดึงสตรีมวิดีโอจากลิงก์นี้ได้"}

        worker = camera_manager.add_camera(f"YT: {sanitized_text(title, default='YouTube Video', max_len=100)}", video_url)
        camera_manager.set_active_camera(worker.cam_id)
        worker.frame_skip = max(0, min(30, int(frame_skip)))
        worker.pause()
        RTSP_URL = video_url
        
        return {"status": "success", "message": f"เริ่มดึงภาพจาก {title} เรียบร้อยแล้ว", "camera_id": worker.cam_id}
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}

@app.get("/system_settings")
def get_system_settings():
    return {
        "save_unknown_faces": save_unknown_faces,
        "show_only_targets": show_only_targets,
        "show_detection_zone": show_detection_zone,
        "min_confidence": min_confidence,
        "face_match_threshold": face_match_threshold,
        "vehicle_match_threshold": vehicle_match_threshold,
        "frame_skip": current_frame_skip,
        "enable_lpr": enable_lpr,
        "telegram_enabled": telegram_enabled,
        "telegram_bot_token": "",
        "telegram_bot_token_configured": bool(telegram_bot_token),
        "telegram_bot_token_masked": mask_secret(telegram_bot_token),
        "telegram_chat_id": telegram_chat_id,
        "telegram_cooldown": telegram_cooldown,
        "telegram_notify_faces": telegram_notify_faces,
        "telegram_notify_vehicles": telegram_notify_vehicles,
        "telegram_notify_plates": telegram_notify_plates
    }

@app.post("/update_settings")
def update_settings(payload: dict = Body(...)):
    global save_unknown_faces, show_only_targets, min_confidence, face_match_threshold, vehicle_match_threshold, enable_lpr, config
    global telegram_enabled, telegram_bot_token, telegram_chat_id, telegram_cooldown, telegram_notify_faces, telegram_notify_vehicles, telegram_notify_plates
    updated = False
    
    if "save_unknown_faces" in payload:
        save_unknown_faces = bool(payload["save_unknown_faces"])
        config["save_unknown_faces"] = save_unknown_faces
        updated = True
        
    if "show_only_targets" in payload:
        show_only_targets = bool(payload["show_only_targets"])
        config["show_only_targets"] = show_only_targets
        updated = True
        
    if "min_confidence" in payload:
        min_confidence = max(1, min(99, int(payload["min_confidence"])))
        config["min_confidence"] = min_confidence
        updated = True

    if "face_match_threshold" in payload:
        face_match_threshold = max(30, min(99, int(payload["face_match_threshold"])))
        config["face_match_threshold"] = face_match_threshold
        updated = True

    if "vehicle_match_threshold" in payload:
        vehicle_match_threshold = max(50, min(99, int(payload["vehicle_match_threshold"])))
        config["vehicle_match_threshold"] = vehicle_match_threshold
        updated = True

    if "enable_lpr" in payload:
        enable_lpr = bool(payload["enable_lpr"])
        config["enable_lpr"] = enable_lpr
        updated = True

    if "telegram_enabled" in payload:
        telegram_enabled = bool(payload["telegram_enabled"])
        config["telegram_enabled"] = telegram_enabled
        updated = True

    if "telegram_bot_token" in payload:
        incoming_token = str(payload["telegram_bot_token"]).strip()
        if incoming_token and not incoming_token.startswith("***"):
            telegram_bot_token = incoming_token
            config["telegram_bot_token"] = telegram_bot_token
            updated = True

    if "telegram_chat_id" in payload:
        telegram_chat_id = sanitized_text(payload["telegram_chat_id"], max_len=80)
        config["telegram_chat_id"] = telegram_chat_id
        updated = True

    if "telegram_cooldown" in payload:
        telegram_cooldown = max(5, int(payload["telegram_cooldown"]))
        config["telegram_cooldown"] = telegram_cooldown
        updated = True

    if "telegram_notify_faces" in payload:
        telegram_notify_faces = bool(payload["telegram_notify_faces"])
        config["telegram_notify_faces"] = telegram_notify_faces
        updated = True

    if "telegram_notify_vehicles" in payload:
        telegram_notify_vehicles = bool(payload["telegram_notify_vehicles"])
        config["telegram_notify_vehicles"] = telegram_notify_vehicles
        updated = True

    if "telegram_notify_plates" in payload:
        telegram_notify_plates = bool(payload["telegram_notify_plates"])
        config["telegram_notify_plates"] = telegram_notify_plates
        updated = True

    if updated:
        save_config(config)
        
    return {"status": "success", "message": "อัปเดตการตั้งค่าเรียบร้อยแล้ว"}

@app.post("/test_telegram")
def test_telegram(payload: dict = Body(...)):
    bot_token = str(payload.get("bot_token", "")).strip() or str(config.get("telegram_bot_token", "")).strip()
    chat_id = str(payload.get("chat_id", "")).strip() or str(config.get("telegram_chat_id", "")).strip()
    
    if not bot_token or not chat_id:
        return JSONResponse(status_code=400, content={"status": "error", "message": "กรุณาระบุ Bot Token และ Chat ID ให้ครบถ้วน"})
        
    try:
        import requests
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = (
            f"✅ *ทดสอบการเชื่อมต่อ Telegram สำเร็จ!*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🛡️ *CCTV AI Surveillance System*\n"
            f"⏰ *เวลาทดสอบ:* {ts_now}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"ระบบพร้อมส่งภาพเปรียบเทียบและแจ้งเตือนเมื่อตรวจพบเป้าหมายแล้วครับ"
        )
        resp = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=10)
        res_json = resp.json()
        if resp.status_code == 200 and res_json.get("ok"):
            return {"status": "success", "message": "ส่งข้อความทดสอบไปยัง Telegram สำเร็จแล้ว!"}
        else:
            err_desc = res_json.get("description", resp.text)
            return JSONResponse(status_code=400, content={"status": "error", "message": f"Telegram แจ้งข้อผิดพลาด: {err_desc}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"ไม่สามารถเชื่อมต่อ Telegram ได้: {str(e)}"})

@app.get("/cameras")
def get_cameras():
    return camera_manager.get_all_statuses()

@app.get("/video_thumbnail")
def video_thumbnail(path: str):
    """Return a representative JPEG frame from an uploaded video file so the UI can
    preview which clip is selected before starting processing."""
    # Only allow files that actually live inside the uploads directory (no traversal).
    uploads_dir = os.path.realpath("uploads")
    target = os.path.realpath(path)
    if os.path.commonpath([uploads_dir, target]) != uploads_dir or not os.path.isfile(target):
        return JSONResponse(status_code=404, content={"status": "error", "message": "Video not found"})

    cap = None
    try:
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            return JSONResponse(status_code=422, content={"status": "error", "message": "Cannot open video"})

        # Seek a little into the clip (~10%) to skip black/intro frames, capped at 30 frames.
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        seek = min(30, int(total * 0.1)) if total > 0 else 0
        if seek > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, seek)

        success, frame = cap.read()
        if not success or frame is None:
            # Fall back to the very first frame.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = cap.read()
        if not success or frame is None:
            return JSONResponse(status_code=422, content={"status": "error", "message": "Cannot read frame"})

        # Downscale wide frames to keep the thumbnail light.
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return JSONResponse(status_code=500, content={"status": "error", "message": "Encode failed"})
        return Response(content=buf.tobytes(), media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if cap is not None:
            cap.release()

@app.post("/add_camera")
async def add_camera(name: str = Form(...), url: str = Form(...)):
    if not validate_camera_source(url):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid camera source"})
    name = sanitized_text(name, default="Camera", max_len=80)
    worker = camera_manager.add_camera(name, url)
    return {"status": "success", "message": f"Added camera {name}", "camera_id": worker.cam_id}

@app.delete("/remove_camera/{camera_id}")
async def remove_camera(camera_id: str):
    if camera_id == "webcam":
        return JSONResponse(status_code=400, content={"status": "error", "message": "Cannot remove default webcam"})
    success = camera_manager.remove_camera(camera_id)
    if success:
        return {"status": "success", "message": "Camera removed"}
    return JSONResponse(status_code=404, content={"status": "error", "message": "Camera not found"})

@app.get("/targets_list")
def get_targets_list():
    targets = []
    if not os.path.exists('targets'):
        return targets
    for filename in os.listdir('targets'):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            name = os.path.splitext(filename)[0]
            base_name = re.sub(r'_[0-9]+$', '', name)
            targets.append({"name": name, "base_name": base_name, "filename": filename})
    return targets

@app.post("/rename_target")
async def rename_target(old_filename: str = Form(...), new_name: str = Form(...)):
    global known_face_cache, known_face_encodings, known_face_names
    if not old_filename or not new_name:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Missing data"})
    
    safe_old = os.path.basename(old_filename.strip())
    safe_new = sanitized_stem(new_name, default="target")
    ext = os.path.splitext(safe_old)[1]
    old_path = os.path.join('targets', safe_old)
    new_filename = f"{safe_new}{ext}"
    new_path = os.path.join('targets', new_filename)
    
    if not os.path.exists(old_path):
        return JSONResponse(status_code=404, content={"status": "error", "message": "File not found"})
        
    os.rename(old_path, new_path)
    
    # Update cache incrementally without re-encoding!
    with targets_lock:
        if safe_old in known_face_cache:
            entry = known_face_cache.pop(safe_old)
            base_name = re.sub(r'_[0-9]+$', '', safe_new)
            entry["base_name"] = base_name
            known_face_cache[new_filename] = entry
            
            # Rebuild arrays from cache
            known_face_encodings = [item["encoding"] for item in known_face_cache.values()]
            known_face_names = [item["base_name"] for item in known_face_cache.values()]
            print(f"✅ เปลี่ยนชื่อเป้าหมายใบหน้า (incremental): {safe_old} -> {new_filename}")
        else:
            # Fallback
            load_known_faces()
            
    return {"status": "success", "message": f"Renamed to {safe_new}"}

@app.delete("/remove_target/{filename}")
async def remove_target(filename: str):
    global known_face_cache, known_face_encodings, known_face_names
    safe_filename = os.path.basename(filename.strip())
    path = os.path.join('targets', safe_filename)
    if os.path.exists(path):
        os.remove(path)
        
        # Update cache incrementally without re-encoding!
        with targets_lock:
            if safe_filename in known_face_cache:
                known_face_cache.pop(safe_filename)
                known_face_encodings = [item["encoding"] for item in known_face_cache.values()]
                known_face_names = [item["base_name"] for item in known_face_cache.values()]
                print(f"✅ ลบเป้าหมายใบหน้า (incremental): {safe_filename}")
            else:
                # Fallback
                load_known_faces()
                
        return {"status": "success", "message": "Target removed"}
    return JSONResponse(status_code=404, content={"status": "error", "message": "File not found"})

@app.post("/add_target")
async def add_target(name: str = Form(...), file: UploadFile = File(...)):
    raw_ext = os.path.splitext(file.filename or "")[1].lower()
    if raw_ext not in ALLOWED_IMAGE_EXTS:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Unsupported image file type"})
    ext = raw_ext
    
    safe_name = sanitized_stem(name, default="target")
    filepath = os.path.join('targets', f"{safe_name}{ext}")
    
    # Auto-increment filename if it already exists to avoid overwriting
    counter = 2
    while os.path.exists(filepath):
        filepath = os.path.join('targets', f"{safe_name}_{counter}{ext}")
        counter += 1
        
    try:
        await save_upload_limited(file, filepath, MAX_IMAGE_UPLOAD_BYTES)
    except ValueError as e:
        return JSONResponse(status_code=413, content={"status": "error", "message": str(e)})

    # Encode only the new image and append it, instead of re-encoding the whole DB.
    added = append_known_face(filepath)
    saved_filename = os.path.basename(filepath)
    if not added:
        return {"status": "warning", "message": f"บันทึก {saved_filename} แล้ว แต่ตรวจไม่พบใบหน้าในรูป"}
    return {"status": "success", "message": f"เพิ่ม {saved_filename} เรียบร้อย"}

@app.post("/add_multiple_targets")
async def add_multiple_targets(files: list[UploadFile] = File(...)):
    import urllib.parse
    saved_files = []
    faces_found = 0
    total_uploaded = 0

    for file in files:
        raw_ext = os.path.splitext(file.filename or "")[1].lower()
        if raw_ext not in ALLOWED_IMAGE_EXTS:
            continue
        ext = raw_ext
        
        # Extract base name from filename (e.g. "Witchapol_3" -> "Witchapol")
        original_name = os.path.splitext(file.filename or "target")[0]
        # Decode URL-encoded characters in filename if any
        original_name = urllib.parse.unquote(original_name)
        original_name = os.path.basename(original_name)
        
        # Remove trailing _number if present
        name_clean = sanitized_stem(re.sub(r'_[0-9]+$', '', original_name), default="target")
        
        filepath = os.path.join('targets', f"{name_clean}{ext}")
        counter = 2
        while os.path.exists(filepath):
            filepath = os.path.join('targets', f"{name_clean}_{counter}{ext}")
            counter += 1
            
        try:
            written = await save_upload_limited(file, filepath, MAX_IMAGE_UPLOAD_BYTES)
        except ValueError as e:
            return JSONResponse(status_code=413, content={"status": "error", "message": str(e)})
        total_uploaded += written
        if total_uploaded > MAX_MULTI_IMAGE_UPLOAD_BYTES:
            try:
                os.remove(filepath)
            except Exception:
                pass
            return JSONResponse(status_code=413, content={"status": "error", "message": f"Total upload too large. Limit is {MAX_MULTI_IMAGE_UPLOAD_BYTES // (1024 * 1024)} MB"})
        saved_files.append(os.path.basename(filepath))
        # Incrementally encode each new file rather than rebuilding the whole DB at the end.
        if append_known_face(filepath):
            faces_found += 1

    msg = f"นำเข้าสำเร็จ {len(saved_files)} รูปภาพ"
    if faces_found < len(saved_files):
        msg += f" (พบใบหน้า {faces_found} รูป, ที่เหลือตรวจไม่พบใบหน้า)"
    return {"status": "success", "message": msg}

@app.get("/vehicle_targets_list")
def get_vehicle_targets_list():
    return vehicle_targets

@app.post("/add_vehicle_target")
async def add_vehicle_target(text: str = Form(...)):
    global vehicle_targets
    v_type = "รถ" # Default to any car
    v_color = "ไม่ระบุ"
    text = sanitized_text(text, default="รถ", max_len=120)
    text_lower = text.lower()
    
    # Improve parse type
    if "มอเตอร์ไซค์" in text_lower or "จักรยานยนต์" in text_lower:
        v_type = "รถมอเตอร์ไซค์"
    elif "จักรยาน" in text_lower:
        v_type = "รถจักรยาน"
    elif "กระบะ" in text_lower or "บรรทุก" in text_lower:
        v_type = "รถกระบะ"
    elif "เก๋ง" in text_lower or "ยนต์" in text_lower:
        v_type = "รถเก๋ง"
    elif "บัส" in text_lower or "ตู้" in text_lower:
        v_type = "รถบัส"
            
    # Simple parse color
    for c in ["ขาว", "ดำ", "เทา", "แดง", "น้ำเงิน", "เหลือง", "เขียว"]:
        if f"สี{c}" in text or f" {c}" in text or text.endswith(c):
            v_color = c
            break
            
    new_target = {"type": v_type, "color": v_color, "raw": text}
    vehicle_targets.append(new_target)
    
    try:
        with open(VEHICLE_TARGETS_FILE, "w", encoding="utf-8") as f:
            json.dump(vehicle_targets, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(e)
        
    return {"status": "success", "message": f"เพิ่มเป้าหมาย: {text} ({v_type} สี{v_color})"}

@app.post("/add_vehicle_image_target")
async def add_vehicle_image_target(file: UploadFile = File(...)):
    global vehicle_targets
    
    raw_ext = os.path.splitext(file.filename or "")[1].lower()
    if raw_ext not in ALLOWED_IMAGE_EXTS:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Unsupported image file type"})
    contents = await file.read(MAX_IMAGE_UPLOAD_BYTES + 1)
    if len(contents) > MAX_IMAGE_UPLOAD_BYTES:
        return JSONResponse(status_code=413, content={"status": "error", "message": f"File too large. Limit is {MAX_IMAGE_UPLOAD_BYTES // (1024 * 1024)} MB"})
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "ไฟล์รูปภาพไม่ถูกต้อง"})
    if model is None:
        return JSONResponse(status_code=503, content={"status": "error", "message": "YOLO model is not loaded"})
        
    # Run YOLO to find vehicle
    results = model(img, classes=[1, 2, 3, 5, 7], conf=0.25, verbose=False)
    
    found_vehicle = False
    v_type = ""
    v_color = ""
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        found_vehicle = True
        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls_ids = results[0].boxes.cls.int().cpu().numpy()
        
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        max_idx = np.argmax(areas)
        
        best_box = boxes[max_idx]
        best_cls = cls_ids[max_idx]
        
        x1, y1, x2, y2 = map(int, best_box)
        v_crop = img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
        
        vehicle_types = {1: "รถจักรยาน", 2: "รถเก๋ง", 3: "รถมอเตอร์ไซค์", 5: "รถบัส", 7: "รถกระบะ"}
        v_type = vehicle_types.get(best_cls, "รถยนต์")
        
        v_color = get_vehicle_color(v_crop)

        emb = get_vehicle_embedding(v_crop)  # (512,) or None

        # Multi-shot: if this photo clearly matches an existing target, append it as an extra
        # viewpoint instead of creating a duplicate. Otherwise create a new target.
        merged_into = None
        if emb is not None:
            with targets_lock:
                existing = list(vehicle_targets)
                existing_embs = dict(vehicle_target_embeddings)
            best_sim, best_t = 0.0, None
            for et in existing:
                ev = existing_embs.get(et.get("filename"))
                sim = vehicle_similarity(emb, ev) if ev is not None else 0.0
                if sim > best_sim:
                    best_sim, best_t = sim, et
            if best_t is not None and best_sim >= VEHICLE_MERGE_SIM:
                merged_into = best_t

        if merged_into is not None:
            # Append this viewpoint to the matched target's embedding file.
            emb_file = merged_into.get("embedding_file")
            try:
                if emb_file and os.path.exists(os.path.join("vehicle_targets", emb_file)):
                    stack = np.atleast_2d(np.load(os.path.join("vehicle_targets", emb_file)))
                    stack = np.vstack([stack, emb[None]])
                else:
                    emb_file = os.path.splitext(merged_into.get("filename"))[0] + ".npy"
                    merged_into["embedding_file"] = emb_file
                    stack = emb[None]
                np.save(os.path.join("vehicle_targets", emb_file), stack)
                print(f"[ReID] Added viewpoint #{len(stack)} to existing target {merged_into.get('raw')}")
            except Exception as e:
                print(f"[ReID] Failed to append viewpoint: {e}")
        else:
            # Save a new target (crop + embedding).
            timestamp = int(time.time())
            v_filename = f"vehicle_{timestamp}.jpg"
            cv2.imwrite(os.path.join("vehicle_targets", v_filename), v_crop)
            new_target = {
                "type": v_type,
                "color": v_color,
                "raw": f"{v_type}สี{v_color} (เฉพาะคัน)",
                "filename": v_filename
            }
            if emb is not None:
                emb_filename = f"vehicle_{timestamp}.npy"
                try:
                    np.save(os.path.join("vehicle_targets", emb_filename), emb[None])  # (1, 512)
                    new_target["embedding_file"] = emb_filename
                except Exception as e:
                    print(f"[ReID] Failed to save embedding: {e}")
            vehicle_targets.append(new_target)

        try:
            with open(VEHICLE_TARGETS_FILE, "w", encoding="utf-8") as f:
                json.dump(vehicle_targets, f, indent=4, ensure_ascii=False)
            load_vehicle_targets() # Reload targets + embeddings in memory
        except Exception as e:
            print("Error saving vehicle target JSON:", e)

    # Now detect faces in the SAME image
    face_msg = ""
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # upsample to find smaller faces on bikes
    face_locations = face_recognition.face_locations(rgb_img, model="hog", number_of_times_to_upsample=2)
    
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        if face_encodings:
            timestamp = int(time.time())
            face_name = f"ผู้ขับขี่รถจากภาพ_{timestamp}"
            face_filename = f"{face_name}.jpg"
            face_path = os.path.join('targets', face_filename)
            
            # Save the face crop to targets directory
            top, right, bottom, left = face_locations[0]
            h, w = img.shape[:2]
            padding = 40
            cx1, cy1 = max(0, left - padding), max(0, top - padding)
            cx2, cy2 = min(w, right + padding), min(h, bottom + padding)
            face_crop = img[cy1:cy2, cx1:cx2]
            
            if face_crop.size > 0:
                cv2.imwrite(face_path, face_crop)
                append_known_face(face_path) # Incrementally add the new face
                face_msg = "และดึงใบหน้าบุคคลเพิ่มเป็นเป้าหมายด้วย"
    
    if not found_vehicle and not face_msg:
         return JSONResponse(status_code=400, content={"status": "error", "message": "ไม่พบยานพาหนะหรือใบหน้าในรูปภาพนี้"})
         
    if found_vehicle:
        return {"status": "success", "message": f"ตรวจพบ '{v_type}สี{v_color}' {face_msg} สำเร็จ!"}
    else:
        return {"status": "success", "message": f"ไม่พบรถ แต่พบใบหน้าบุคคลและเพิ่มเป้าหมายสำเร็จ!"}

@app.delete("/remove_vehicle_target/{index}")
async def remove_vehicle_target(index: int):
    global vehicle_targets
    if 0 <= index < len(vehicle_targets):
        t = vehicle_targets[index]
        filename = t.get("filename")
        if filename:
            safe_filename = os.path.basename(filename)
            path = os.path.join("vehicle_targets", safe_filename)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error removing vehicle target image {safe_filename}: {e}")
        emb_file = t.get("embedding_file")
        if emb_file:
            safe_emb = os.path.basename(emb_file)
            emb_path = os.path.join("vehicle_targets", safe_emb)
            if os.path.exists(emb_path):
                try:
                    os.remove(emb_path)
                except Exception as e:
                    print(f"Error removing vehicle target embedding {safe_emb}: {e}")
        del vehicle_targets[index]
        try:
            with open(VEHICLE_TARGETS_FILE, "w", encoding="utf-8") as f:
                json.dump(vehicle_targets, f, indent=4, ensure_ascii=False)
            load_vehicle_targets() # Reload in memory
        except Exception as e:
            print(e)
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
        return {"status": "success", "message": "ลบเป้าหมายสำเร็จ"}
@app.post("/edit_vehicle_target/{index}")
async def edit_vehicle_target(index: int, type: str = Form(...), color: str = Form(...), raw: str = Form(...)):
    global vehicle_targets
    if 0 <= index < len(vehicle_targets):
        vehicle_targets[index]["type"] = sanitized_text(type, default="รถ", max_len=40)
        vehicle_targets[index]["color"] = sanitized_text(color, default="ไม่ระบุ", max_len=30)
        vehicle_targets[index]["raw"] = sanitized_text(raw, default="รถ", max_len=120)
        try:
            with open(VEHICLE_TARGETS_FILE, "w", encoding="utf-8") as f:
                json.dump(vehicle_targets, f, indent=4, ensure_ascii=False)
            load_vehicle_targets() # Reload SIFT descriptors in memory
        except Exception as e:
            print(e)
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
        return {"status": "success", "message": "แก้ไขเป้าหมายสำเร็จ"}
    return JSONResponse(status_code=400, content={"status": "error", "message": "ไม่พบเป้าหมาย"})

# --- Plate Targets API Endpoints & Route Tracing ---
@app.get("/plate_targets_list")
def get_plate_targets_list():
    return plate_targets

@app.post("/add_plate_target")
async def add_plate_target(plate: str = Form(...), note: str = Form(default="")):
    global plate_targets
    plate_clean = sanitized_text(plate, max_len=32)
    if not plate_clean:
        return JSONResponse(status_code=400, content={"status": "error", "message": "กรุณาระบุหมายเลขป้ายทะเบียน"})
    
    note_clean = sanitized_text(note, max_len=120)
    raw_str = f"{plate_clean} ({note_clean})" if note_clean else f"ป้ายทะเบียน {plate_clean}"
    
    new_target = {
        "plate": plate_clean,
        "note": note_clean,
        "raw": raw_str,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    plate_targets.append(new_target)
    save_plate_targets()
    return {"status": "success", "message": f"เพิ่มเป้าหมายทะเบียน: {plate_clean} สำเร็จ"}

@app.delete("/remove_plate_target/{index}")
async def remove_plate_target(index: int):
    global plate_targets
    if 0 <= index < len(plate_targets):
        del plate_targets[index]
        save_plate_targets()
        return {"status": "success", "message": "ลบเป้าหมายทะเบียนสำเร็จ"}
    return JSONResponse(status_code=400, content={"status": "error", "message": "ไม่พบเป้าหมาย"})

@app.post("/edit_plate_target/{index}")
async def edit_plate_target(index: int, plate: str = Form(...), note: str = Form(default="")):
    global plate_targets
    if 0 <= index < len(plate_targets):
        plate_clean = sanitized_text(plate, max_len=32)
        note_clean = sanitized_text(note, max_len=120)
        raw_str = f"{plate_clean} ({note_clean})" if note_clean else f"ป้ายทะเบียน {plate_clean}"
        plate_targets[index]["plate"] = plate_clean
        plate_targets[index]["note"] = note_clean
        plate_targets[index]["raw"] = raw_str
        save_plate_targets()
        return {"status": "success", "message": "แก้ไขเป้าหมายทะเบียนสำเร็จ"}
    return JSONResponse(status_code=400, content={"status": "error", "message": "ไม่พบเป้าหมาย"})

@app.get("/route_trace")
def get_route_trace(license_plate: str = Query(...)):
    norm = normalize_plate(license_plate)
    digits = extract_digits(license_plate)
    if not norm and not digits:
        return JSONResponse(status_code=400, content={"status": "error", "message": "ต้องระบุหมายเลขป้ายทะเบียน"})
    try:
        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, camera_url, name, timestamp, confidence, img_url, full_url, plate_img_url, license_plate
                FROM hits
                WHERE (license_plate IS NOT NULL AND license_plate != '') OR name LIKE '%ทะเบียน%'
                ORDER BY timestamp ASC
            ''')
            rows = cursor.fetchall()
            matched = []
            for r in rows:
                lp = r["license_plate"] or ""
                hit_name = r["name"] or ""
                lp_norm = normalize_plate(lp)
                lp_digits = extract_digits(lp) or extract_digits(hit_name)
                
                is_match = False
                if norm and (lp_norm == norm or norm in lp_norm or norm in normalize_plate(hit_name)):
                    is_match = True
                elif digits and len(digits) >= 3 and (digits == lp_digits or digits in lp_digits or lp_digits.endswith(digits)):
                    is_match = True
                
                if is_match:
                    matched.append(dict(r))
            return {"status": "success", "data": matched, "query_plate": license_plate}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/")

def read_root(): return {"message": "YOLO Backend กำลังทำงาน"}

@app.post("/set_yolo_model")
def set_yolo_model(model_name: str = Form(...)):
    global model, current_yolo_model, config
    try:
        model_name = os.path.basename(str(model_name or "").strip())
        if model_name not in ALLOWED_YOLO_MODELS:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Unsupported YOLO model"})
        # Load in a temp variable to avoid breaking current stream while downloading
        new_model = YOLO(model_name).to(DEVICE)
        if "world" in model_name:
            new_model.set_classes(yolo_world_classes + custom_search_terms)
            
        # Swap globally after successful load
        model = new_model
        current_yolo_model = model_name
        config["yolo_model"] = current_yolo_model
        save_config(config)
        return {"status": "success", "message": f"YOLO Model changed to {model_name}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/get_yolo_model")
def get_yolo_model():
    return {"yolo_model": current_yolo_model}

@app.post("/set_custom_search")
def set_custom_search(search_term: str = Form(...)):
    global custom_search_terms, model
    search_term = sanitized_text(search_term, max_len=200)
    if not search_term.strip():
        custom_search_terms = []
    else:
        # Split by comma if multiple
        custom_search_terms = [t.strip() for t in search_term.split(",") if t.strip()]
    
    if "world" in current_yolo_model:
        try:
            if model is None:
                return JSONResponse(status_code=503, content={"status": "error", "message": "YOLO model is not loaded"})
            model.set_classes(yolo_world_classes + custom_search_terms)
            return {"status": "success", "message": "Custom search updated"}
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    else:
        return {"status": "success", "message": "Search updated (Inactive, model is not YOLO-World)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("CCTV_BACKEND_HOST", "127.0.0.1"), port=int(os.getenv("CCTV_BACKEND_PORT", "8081")))

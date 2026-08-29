# 🛡️ CCTV AI Surveillance & Intelligence System
### ระบบตรวจจับอัจฉริยะแบบเรียลไทม์ (YOLOv8 + Face Recognition + Vehicle Re-ID + LPR)

ระบบตรวจจับและวิเคราะห์ภาพจากกล้องวงจรปิดแบบครบวงจร รองรับกล้อง **RTSP (IP Camera)**, **Webcam**, **บันทึกหน้าจอคอมพิวเตอร์ (Screen Capture)**, **ไฟล์วิดีโอ (MP4/MKV)** และ **YouTube Live Stream** พร้อมระบบตรวจจับใบหน้า, จำแนกยานพาหนะพร้อมสี, อ่านป้ายทะเบียน (LPR) และระบบแจ้งเตือนแบบเรียลไทม์

---

## ✨ ฟีเจอร์หลัก (Key Features)

- 🎯 **Object Detection (YOLOv8):** ตรวจจับวัตถุ บุคคล ยานพาหนะ (รถยนต์, รถจักรยานยนต์, รถบรรทุก ฯลฯ) พร้อมปรับแต่ง Threshold และเลือกสลับโมเดลได้ตามความเร็วเครื่อง (`yolov8n`, `yolov8s`, `yolov8m`, `yolov8s-world`)
- 🧑 **Face Recognition (ระบบจดจำใบหน้า):**
  - ลงทะเบียนบุคคลเป้าหมาย (Watchlist) ได้หลายมุมมอง
  - ปรับระดับความเหมือน (Similarity Threshold) และระบบป้องกันบุคคลสับสนข้ามเฟรม
  - บันทึกภาพ Snapshot ใบหน้าอัตโนมัติเมื่อตรวจพบ
- 🚗 **Vehicle Re-ID & Color Detection (ตรวจจับและแยกแยะรถยนต์):**
  - ระบุประเภทยานพาหนะและสีรถ (ขาว, ดำ, เทา/เงิน, แดง, น้ำเงิน, น้ำตาล, เหลือง ฯลฯ)
  - รองรับการจับคู่รถเป้าหมายเฉพาะคัน (Vehicle Feature Embedding)
- 🔤 **License Plate Recognition (LPR - ระบบอ่านป้ายทะเบียน):**
  - อ่านป้ายทะเบียนภาษาไทยและตัวเลขอัตโนมัติด้วย OCR
  - ระบบเฝ้าระวังป้ายทะเบียนเป้าหมาย (Blacklist / Target Plates) แจ้งเตือนทันทีที่พบ
- 📡 **Multi-Source Video Input:**
  - **RTSP Stream:** เชื่อมต่อกล้อง IP Camera / NVR ทุกแบรนด์ (Hikvision, Dahua, Ezviz, TP-Link VIGI ฯลฯ)
  - **Webcam:** รองรับกล้องติดคอมพิวเตอร์ (Device `0`, `1`)
  - **Screen Capture:** ดึงภาพจากหน้าจอ Monitor หรือโปรแกรม CCTV Client อื่นๆ มาวิเคราะห์ได้ทันที (`screen:1`, `screen:2`)
  - **Video Files & Timeline:** อัปโหลดคลิปวิดีโอย้อนหลังเพื่อสแกนค้นหา พร้อมแถบ Seek Timeline ควบคุมการเล่น
  - **YouTube Streams:** ดึงสตรีมสดหรือคลิปจาก YouTube มาวิเคราะห์ได้โดยตรง
- 🔔 **Real-Time Alert & Evidence History:**
  - แจ้งเตือนด้วยเสียงและ Visual Indicator บนหน้าจอแบบเรียลไทม์
  - บันทึกประวัติการตรวจพบลงฐานข้อมูล SQLite พร้อมรูป Crop เป้าหมายและภาพมุมกว้าง (Full Frame)
  - หน้าระบบค้นหาประวัติย้อนหลัง (History), กรองตามวันเวลา/ประเภท/กล้อง พร้อมส่งออกข้อมูล
- ✈️ **Telegram Alert Notifications (แจ้งเตือนผ่าน Telegram พร้อมรูปเปรียบเทียบ):**
  - ส่งภาพการ์ดเปรียบเทียบแบบ Side-by-Side (ภาพเป้าหมายที่ลงทะเบียน VS ภาพตรวจพบสดจากกล้อง)
  - ระบุชื่อเป้าหมาย, ค่า % ความเหมือน (Similarity Score), ชื่อกล้อง, วันเวลาที่ตรวจพบ
  - รองรับการแจ้งเตือนทั้งบุคคลเป้าหมาย, ยานพาหนะ, และป้ายทะเบียน พร้อมระบบ Cooldown ป้องกันการส่งข้อความซ้ำ
- 💻 **Modern Web Dashboard:**
  - หน้าควบคุม Responsive Dashboard แสดงผลสดผ่านเว็บเบราว์เซอร์
  - รองรับการทำงานทั้งบน macOS (Apple Silicon MPS Acceleration), Windows (NVIDIA CUDA / CPU) และ Linux

---

## 📋 ข้อกำหนดของระบบ (System Requirements)

- **Python:** แนะนำ **Python 3.10** หรือ **Python 3.11** (เพื่อความเข้ากันได้สูงสุดกับ `dlib` และ `face_recognition`)
- **ระบบปฏิบัติการ:**
  - 🍎 **macOS:** รองรับทั้ง Apple Silicon (M1/M2/M3/M4) และ Intel
  - 🪟 **Windows:** Windows 10 / 11 (64-bit)
  - 🐧 **Linux:** Ubuntu 20.04 / 22.04 / 24.04 หรือเทียบเท่า
- **ฮาร์ดแวร์ที่แนะนำ:**
  - RAM: ขั้นต่ำ 8 GB (แนะนำ 16 GB ขึ้นไป)
  - GPU (ถ้ามี): NVIDIA GPU ที่รองรับ CUDA หรือ Apple Silicon (MPS)

---

## 🚀 คู่มือการติดตั้งตั้งแต่เริ่มต้น (Step-by-Step Installation)

### 1. ติดตั้ง Prerequisite (ก่อนลงโปรแกรม)

เนื่องจากระบบใช้ไลบรารี `face_recognition` และ `dlib` ซึ่งต้องใช้ C++ Compiler และ CMake ในการ Build:

<details>
<summary><b>🍎 สำหรับ macOS</b></summary>

1. ติดตั้ง **Homebrew** (หากยังไม่มี):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. ติดตั้ง **CMake**:
   ```bash
   brew install cmake
   ```
3. ติดตั้ง Xcode Command Line Tools:
   ```bash
   xcode-select --install
   ```
</details>

<details>
<summary><b>🪟 สำหรับ Windows</b></summary>

1. ติดตั้ง **Python 3.10 หรือ 3.11** จาก [python.org](https://www.python.org/downloads/) *(⚠️ อย่าลืมติ๊กถูก **"Add Python to PATH"** ตอนติดตั้ง)*
2. ติดตั้ง **CMake** จาก [cmake.org/download](https://cmake.org/download/) *(เลือก Add CMake to the system PATH)*
3. ติดตั้ง **Visual Studio C++ Build Tools** จาก [Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) โดยเลือก Package: **"Desktop development with C++"**
</details>

<details>
<summary><b>🐧 สำหรับ Linux / Ubuntu</b></summary>

ติดตั้ง Package พื้นฐานผ่าน APT:
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev ffmpeg
```
</details>

---

### 2. Clone โปรเจกต์ & สร้าง Virtual Environment

เปิด Terminal / Command Prompt แล้วรันคำสั่ง:

```bash
# 1. เข้ามายังโฟลเดอร์ของโปรเจกต์
git clone https://github.com/s0ci4lx/Pyscript-public.git
cd Pyscript-public

# 2. สร้าง Virtual Environment (แนะนำ)
python3 -m venv venv

# สำหรับ Windows ใช้:
# python -m venv venv
```

เปิดใช้งาน Virtual Environment:
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```
- **Windows (Command Prompt):**
  ```cmd
  venv\Scripts\activate
  ```
- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```

---

### 3. ติดตั้ง Dependencies (Python Packages)

อัปเกรด pip และติดตั้งแพ็กเกจทั้งหมดผ่าน `requirements.txt`:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> 💡 **หมายเหตุสำหรับผู้ใช้ GPU (NVIDIA CUDA บน Windows/Linux):**
> หากต้องการใช้งาน CUDA ให้ติดตั้ง PyTorch รุ่นที่รองรับ CUDA ก่อน:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 🎬 วิธีการเปิดใช้งานระบบ (How to Run)

คุณสามารถเปิดใช้งานได้ทั้งแบบ **One-Click Script** หรือ **รันคำสั่งด้วยตนเอง**:

### วิธีที่ 1: One-Click Script (ง่ายที่สุด)

- 🍎 **macOS:** ดับเบิลคลิกไฟล์ `start_mac.command` หรือรัน:
  ```bash
  ./start_mac.command
  ```
- 🪟 **Windows:** ดับเบิลคลิกไฟล์ `start_windows.bat`
- 🐧 **Linux:** รันสคริปต์:
  ```bash
  ./start_linux.sh
  ```

ระบบจะเปิด Backend Server (Port 8081) และ Frontend Server (Port 8000) พร้อมเปิดเว็บเบราว์เซอร์ไปยัง `http://localhost:8000` ให้อัตโนมัติ

---

### วิธีที่ 2: รันคำสั่งด้วยตนเอง (Manual Start)

เปิด Terminal 2 หน้าต่าง:

**หน้าต่างที่ 1: รัน Backend AI Server (FastAPI)**
```bash
python rtsp_yolo_backend.py
# Backend จะทำงานที่ http://localhost:8081
```

**หน้าต่างที่ 2: รัน Frontend Web Server**
```bash
python -m http.server 8000
# เข้าใช้งาน Dashboard ผ่านเว็บเบราว์เซอร์: http://localhost:8000
```

---

## 📖 คู่มือการใช้งานระบบ (User Guide)

```
┌────────────────────────────────────────────────────────┐
│                   CCTV AI Dashboard                    │
├─────────────────┬──────────────────────────────────────┤
│  หน้าเว็บ       │  หน้าที่ / การใช้งาน                 │
├─────────────────┼──────────────────────────────────────┤
│  index.html     │  จอมอนิเตอร์สด, สลับกล้อง, แจ้งเตือน │
│  targets.html   │  จัดการเป้าหมาย (ใบหน้า, รถ, ป้าย)   │
│  history.html   │  ค้นหาประวัติการตรวจพบย้อนหลัง       │
│  settings.html  │  ตั้งค่ากล้อง, ปรับแต่งโมเดล AI      │
└─────────────────┴──────────────────────────────────────┘
```

### 1. การเพิ่มและเลือกแหล่งสัญญาณภาพ (Camera Sources)
เข้าไปที่เมนู **ตั้งค่า (Settings)** หรือเลือกจาก Dropdown หน้าแรก:
- **กล้อง IP Camera:** ระบุ URL ในรูปแบบ `rtsp://username:password@IP_ADDRESS:554/stream`
- **กล้อง Webcam:** ระบุค่าเป็น `0` หรือ `1`
- **แคปหน้าจอ (Screen Grabber):** ระบุเป็น `screen:1` (สำหรับจอหลัก) หรือ `screen:2`
- **อัปโหลดไฟล์วิดีโอ:** สามารถอัปโหลดไฟล์ `.mp4` ผ่านหน้า Dashboard เพื่อสแกนคลิปย้อนหลัง

### 2. การลงทะเบียนเป้าหมายเฝ้าระวัง (Targets Management)
เข้าไปที่หน้า **เป้าหมาย (Targets)** เพื่อเพิ่มข้อมูลเฝ้าระวัง:
- **🧑 บุคคลเป้าหมาย:** อัปโหลดรูปใบหน้าที่ชัดเจน พร้อมตั้งชื่อบุคคล (สามารถอัปโหลดได้หลายรูปต่อคนเพื่อความแม่นยำ)
- **🚗 ยานพาหนะเป้าหมาย:** ระบุประเภทยานพาหนะ + สี (หรืออัปโหลดรูปทรงรถเพื่อจับคู่เฉพาะคัน)
- **🔤 ป้ายทะเบียนเป้าหมาย:** พิมพ์เลขทะเบียนที่ต้องการเฝ้าระวัง เช่น `9กภ2600` พร้อมบันทึกช่วยจำ

### 3. การตรวจสอบประวัติย้อนหลัง (History Log)
- ทุกครั้งที่ตรวจพบเป้าหมาย ระบบจะบันทึกรูป Crop, รูปเต็มเฟรม, วันเวลา, และระดับความเชื่อมั่นลงฐานข้อมูล
- หน้า **ประวัติ (History)** สามารถกรองตามชื่อ, เลขทะเบียน, สียานพาหนะ, ช่วงเวลา และกดคลิกเพื่อดูภาพหลักฐานขนาดเต็มได้

---

## 🛠️ โครงสร้างโปรเจกต์ (Project Structure)

```text
├── index.html               # หน้าจอมอนิเตอร์หลัก (Live Stream Dashboard)
├── targets.html             # หน้าจัดการเป้าหมาย (Face / Vehicle / License Plate)
├── history.html             # หน้าระบบค้นหาและดูประวัติย้อนหลัง
├── settings.html            # หน้าตั้งค่ากล้องและพารามิเตอร์ระบบ
├── app.css / style.css      # สไตล์และธีม Dark UI ของระบบ
├── rtsp_yolo_backend.py     # Backend Server หลัก (FastAPI + YOLO + Re-ID + OCR)
├── requirements.txt         # รายการแพ็กเกจ Python ที่จำเป็น
├── start_mac.command        # สคริปต์รันระบบ One-Click สำหรับ macOS
├── start_windows.bat        # สคริปต์รันระบบ One-Click สำหรับ Windows
├── start_linux.sh           # สคริปต์รันระบบ One-Click สำหรับ Linux
├── targets/                 # โฟลเดอร์เก็บภาพใบหน้าเป้าหมาย
├── vehicle_targets/         # โฟลเดอร์เก็บภาพและ Feature Embeddings รถเป้าหมาย
├── uploads/                 # โฟลเดอร์เก็บไฟล์วิดีโอที่อัปโหลด
├── alerts/                  # โฟลเดอร์บันทึกภาพ Snapshot เหตุการณ์แจ้งเตือน
└── history.db               # ฐานข้อมูล SQLite เก็บประวัติการตรวจพบ
```

---

## ❓ คำถามที่พบบ่อย & การแก้ปัญหา (Troubleshooting / FAQ)

### Q: ติดตั้ง `dlib` หรือ `face_recognition` ไม่ผ่าน?
- **สาเหตุ:** ขาด CMake หรือ C++ Compiler ในเครื่อง
- **วิธีแก้:**
  - **macOS:** รัน `brew install cmake` และ `xcode-select --install`
  - **Windows:** ตรวจสอบว่าได้ติดตั้ง Visual Studio C++ Build Tools และติ๊กเลือก "Desktop development with C++" หรือติดตั้ง pre-built wheel ของ dlib สำหรับ Python เวอร์ชั่นนั้นๆ
  - แนะนำให้ใช้ **Python 3.10 หรือ 3.11**

### Q: กล้อง RTSP กระตุก หรือหลุดบ่อย?
- ตรวจสอบความเสถียรของระบบเครือข่าย (แนะนำต่อสาย LAN กับกล้อง NVR/IP Cam)
- สามารถปรับ Resolution สตรีมย่อย (Sub-stream) ของกล้อง IP Camera เพื่อลดการใช้แบนด์วิดท์และการประมวลผล

### Q: พอร์ต 8000 หรือ 8081 ซ้ำ (Port in use)?
- ปิดโปรเซสที่ค้างอยู่ หรือรีสตาร์ทด้วยสคริปต์ `start_mac.command` / `start_linux.sh` (สคริปต์จะทำการเคลียร์พอร์ตเดิมให้อัตโนมัติ)

---

## 📄 License & Disclaimer

โปรเจกต์นี้พัฒนาขึ้นเพื่อการศึกษาและการเฝ้าระวังความปลอดภัย การนำไปใช้งานในการบันทึกภาพหรือตรวจจับบุคคลกรุณาปฏิบัติตามกฎหมายคุ้มครองข้อมูลส่วนบุคคล (PDPA) และกฎหมายที่เกี่ยวข้องในพื้นที่ของท่าน

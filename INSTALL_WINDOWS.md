# ติดตั้งบน Windows จาก GitHub ZIP

ชุด Windows สำหรับ NVIDIA ใช้ตรวจยานพาหนะเป้าหมายและ LPR ไม่มีข้อบังคับติดตั้ง dlib/ใบหน้า
Mac ยังเปิดผ่าน start_mac.command ตามเดิม และใช้ requirements.txt ที่รวมส่วนใบหน้า

## ก่อนติดตั้งครั้งแรก

1. ติดตั้ง Python 3.11 แบบ Windows 64-bit จาก https://www.python.org/downloads/windows/
   เลือกติดตั้ง Python Launcher (คำสั่ง py) ด้วย
2. ติดตั้ง/อัปเดตไดรเวอร์ NVIDIA จาก https://www.nvidia.com/en-us/drivers/
3. ดาวน์โหลด Code → Download ZIP จาก GitHub แล้ว **แตก ZIP ก่อน**
   แนะนำ C:\CCTV\Pyscript หลีกเลี่ยงโฟลเดอร์ที่ซิงค์ OneDrive
4. เปิด install_windows.bat รอจนขึ้น Installation and NVIDIA GPU checks completed
   ต้องต่ออินเทอร์เน็ตเพื่อดาวน์โหลดแพ็กเกจ CUDA และโมเดล OCR ครั้งแรก
5. เปิด start_windows.bat เปิดหน้า http://localhost:8000 อัตโนมัติ
   จะมีหน้าต่าง backend/frontend สองหน้าต่าง เปิดค้างไว้ขณะใช้งาน

ตัวติดตั้งใช้ PyTorch/torchvision ที่เข้าคู่กันจากคลัง CUDA 12.6 ทางการ:
https://download.pytorch.org/whl/cu126
ขั้นตรวจใช้ทั้ง CUDA tensor และ torchvision NMS จริง หากไม่ผ่านจะไม่เปิดระบบแบบ CPU เงียบ ๆ
ไม่จำเป็นต้องติดตั้ง CUDA Toolkit เต็มชุดสำหรับ PyTorch wheel นี้
หากมีข้อความ DLL ขาด ให้ติดตั้ง Microsoft Visual C++ Redistributable x64 จาก Microsoft
แล้วลองใหม่ ไม่ต้องติดตั้ง Visual Studio/dlib สำหรับโหมดรถ

## กล้องสตูล

ต้องติดตั้งและเปิด **Satun-AI-Cam แยกต่างหากบน Windows เครื่องเดียวกัน**
โดยติดตั้ง FFmpeg/ส่วนประกอบตามคู่มือของ Satun-AI-Cam ด้วย
Pyscript ไม่ได้ติดตั้งหรือย้ายระบบสตูลให้อัตโนมัติ
ค่าเริ่มต้น API คือ http://127.0.0.1:8765
หากเปลี่ยนพอร์ต ให้ตั้ง SATUN_API_URL ในเครื่องก่อนเริ่มโปรแกรม
จากหน้าตั้งค่า Pyscript เลือกกล้องสตูลและติ๊กเพิ่มหลายตัวได้
สตรีมหลักใช้ main; ตรวจ resolution อีกครั้งบน Windows เพราะขึ้นกับ NVR จริง

## ข้อมูลจาก Mac

รุ่นสาธารณะไม่รวมข้อมูลส่วนตัวด้านล่าง ให้เพิ่มกล้องและตั้งค่า Telegram ของเครื่องที่จะติดตั้งเอง:
- cameras.json: กล้องที่เพิ่มไว้ — ไม่ต้องคัดลอกไป Windows เพราะมีรายชื่อกล้องที่บ้าน
  ให้เลือกกล้องสตูลใหม่จากหน้าตั้งค่าบน Windows แทน
- vehicle_targets.json และ vehicle_targets/: เป้าหมายรถและรูปต้นฉบับ
- plate_targets.json: เป้าหมายทะเบียน
- config.json: การตั้งค่า รวมข้อมูล Telegram
- history.db, alerts/: ประวัติและภาพหลักฐาน

ถ้าจะย้ายข้อมูล ให้หยุดโปรแกรมก่อนแล้วคัดลอกไฟล์/โฟลเดอร์เหล่านี้แยกอย่างเป็นส่วนตัว
ห้ามคัดลอก .venv จาก Mac; สร้างใหม่ด้วยตัวติดตั้ง Windows
ควรสำรองข้อมูลก่อนอัปเดต ZIP และตรวจเส้นทางวิดีโอ/กล้องที่อ้างถึงเครื่อง Mac
หากไม่ย้าย config.json ตัวติดตั้งจะสร้างค่าเริ่มต้นสำหรับรถ/LPR โดยปิดบุคคล
แม้คัดลอก config.json โหมด Windows vehicle ก็ไม่เปิดใบหน้า

## ตรวจสอบหลังติดตั้ง

เปิด check_windows.bat จะตรวจแพ็กเกจ CUDA และความพร้อม Satun API
ทดสอบ 2 กล้องก่อน ดูว่ามี CUDA ในหน้าต่าง backend จากนั้นเพิ่มเป็น 4 กล้อง
ตรวจการอ่านป้าย/แจ้งเตือนเป้าหมายจริงและดูภาพ 1080p
ยังไม่มีผลทดสอบ Windows จริงในงานนี้ จึงไม่รับรองจำนวนกล้องหรือ FPS ล่วงหน้า

อัปเดตรุ่น: ปิดสองหน้าต่าง สำรองข้อมูล แตก ZIP ใหม่ แล้วรัน install_windows.bat
เปิด start_windows.bat อีกครั้ง; อย่านำไฟล์ส่วนตัวขึ้น GitHub

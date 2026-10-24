import io
import csv

def format_history_csv(rows):
    csv_output = io.StringIO()
    csv_output.write('\ufeff')
    writer = csv.writer(csv_output, quoting=csv.QUOTE_MINIMAL)
    writer.writerow([
        "ID",
        "วัน-เวลา",
        "กล้อง",
        "เป้าหมาย",
        "เลขทะเบียน",
        "สถานะป้าย",
        "ความมั่นใจ (%)",
        "จำนวนเฟรม",
        "ภาพตัดรถ",
        "ภาพมุมกว้าง",
        "ภาพป้าย"
    ])
    for row in rows:
        writer.writerow([
            row.get("id", ""),
            row.get("timestamp", ""),
            row.get("camera_url", ""),
            row.get("name", ""),
            row.get("license_plate", ""),
            row.get("lpr_status", ""),
            row.get("confidence", ""),
            row.get("lpr_sample_count", ""),
            row.get("img_url", ""),
            row.get("full_url", ""),
            row.get("plate_img_url", "")
        ])
    return csv_output.getvalue().encode('utf-8')

def test_history_csv_bom_and_content():
    sample_rows = [
        {
            "id": 15853,
            "timestamp": "2026-09-27 19:31:18",
            "camera_url": "screen:2",
            "name": "[LPR] กง 5764 (#940)",
            "license_plate": "กง 5764",
            "lpr_status": "verified",
            "confidence": 94,
            "lpr_sample_count": 5,
            "img_url": "alerts_img/crop1.jpg",
            "full_url": "alerts_img/scene1.jpg",
            "plate_img_url": "alerts_img/plate1.jpg"
        },
        {
            "id": 15690,
            "timestamp": "2026-09-27 18:45:00",
            "camera_url": "screen:2",
            "name": "[LPR] 80-4058 (#850)",
            "license_plate": "80-4058",
            "lpr_status": "verified",
            "confidence": 90,
            "lpr_sample_count": 4,
            "img_url": "alerts_img/crop2.jpg",
            "full_url": "alerts_img/scene2.jpg",
            "plate_img_url": "alerts_img/plate2.jpg"
        }
    ]
    raw_bytes = format_history_csv(sample_rows)
    # Check UTF-8 BOM
    assert raw_bytes.startswith(b'\xef\xbb\xbf')
    text = raw_bytes.decode('utf-8-sig')
    assert "วัน-เวลา" in text
    assert "กง 5764" in text
    assert "80-4058" in text
    assert "verified" in text

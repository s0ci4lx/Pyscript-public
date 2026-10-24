import sqlite3
import pytest

def query_target_hits_page(conn, camera_url=None, limit=12, before_id=None, after_id=None, show_only_targets=False):
    clauses = []
    params = []

    if camera_url and camera_url not in ("all", "default"):
        raw_urls = [u.strip() for u in camera_url.split(",") if u.strip()]
        if len(raw_urls) == 1:
            clauses.append("camera_url = ?")
            params.append(raw_urls[0])
        else:
            placeholders = ",".join("?" for _ in raw_urls)
            clauses.append(f"camera_url IN ({placeholders})")
            params.extend(raw_urls)

    if before_id is not None:
        clauses.append("id < ?")
        params.append(before_id)
    elif after_id is not None:
        clauses.append("id > ?")
        params.append(after_id)

    if show_only_targets:
        clauses.append("name NOT LIKE ?")
        params.append("บุคคลทั่วไป%")
        clauses.append("name NOT LIKE ?")
        params.append("[LPR]%")

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    order_direction = "ASC" if after_id is not None else "DESC"
    query = f"""
        SELECT id, camera_url, name, timestamp
        FROM hits
        {where_sql}
        ORDER BY id {order_direction}
        LIMIT ?
    """
    params.append(limit + 1)
    rows = conn.execute(query, params).fetchall()
    has_more = len(rows) > limit
    page_rows = rows[:limit]
    return page_rows, has_more

def setup_test_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE hits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_url TEXT,
            name TEXT,
            timestamp TEXT
        )
    """)
    # Insert test data
    conn.execute("INSERT INTO hits (camera_url, name, timestamp) VALUES ('rtsp://cam1', 'บุคคลทั่วไป (#101)', '2026-10-03 00:01:00')")
    conn.execute("INSERT INTO hits (camera_url, name, timestamp) VALUES ('rtsp://cam1', 'บุคคลทั่วไป (#102)', '2026-10-03 00:02:00')")
    conn.execute("INSERT INTO hits (camera_url, name, timestamp) VALUES ('rtsp://cam2', 'เป้าหมาย A', '2026-10-03 00:03:00')")
    conn.execute("INSERT INTO hits (camera_url, name, timestamp) VALUES ('rtsp://cam2', '[LPR] 1กก1234 (#103)', '2026-10-03 00:04:00')")
    conn.execute("INSERT INTO hits (camera_url, name, timestamp) VALUES ('rtsp://cam1', 'บุคคลทั่วไป (#104)', '2026-10-03 00:05:00')")
    conn.commit()
    return conn

def test_query_all_cameras_when_url_none():
    conn = setup_test_db()
    rows, has_more = query_target_hits_page(conn, camera_url=None, limit=10)
    assert len(rows) == 5
    assert rows[0]["name"] == "บุคคลทั่วไป (#104)"
    assert not has_more

def test_query_single_camera():
    conn = setup_test_db()
    rows, _ = query_target_hits_page(conn, camera_url="rtsp://cam1", limit=10)
    assert len(rows) == 3
    assert all(r["camera_url"] == "rtsp://cam1" for r in rows)

def test_query_multi_camera():
    conn = setup_test_db()
    rows, _ = query_target_hits_page(conn, camera_url="rtsp://cam1,rtsp://cam2", limit=10)
    assert len(rows) == 5

def test_show_only_targets_filters_general_person():
    conn = setup_test_db()
    rows, _ = query_target_hits_page(conn, camera_url=None, limit=10, show_only_targets=True)
    assert len(rows) == 1
    assert rows[0]["name"] == "เป้าหมาย A"

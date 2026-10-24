"""Read Satun's camera catalog and use its authenticated MJPEG gateway."""
import json
import os
from urllib.request import urlopen
from urllib.parse import urlencode, urlsplit

def base_url():
    value = os.environ.get("SATUN_API_URL", "http://127.0.0.1:8765").rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("SATUN_API_URL must point to the local Satun service")
    return value

def list_cameras():
    with urlopen(base_url() + "/api/nvr-cameras", timeout=8) as response:
        catalog = json.load(response)
    result = []
    for nvr in catalog.get("nvrs", []):
        for camera in nvr.get("cameras", []):
            channel = camera.get("channel")
            if channel is None:
                continue
            result.append({
                "id": f'{int(nvr["port"])}:{int(channel)}',
                "name": camera.get("name") or camera.get("code") or str(channel),
                "nvr": nvr.get("short_name") or nvr.get("name", ""),
                "code": camera.get("code", ""),
                "is_lpr": camera.get("is_lpr", False),
                "port": int(nvr["port"]),
                "channel": int(channel),
            })
    return result

def stream_url(camera):
    return base_url() + "/api/live/stream?" + urlencode({
        "port": camera["port"], "channel": camera["channel"], "quality": "main",
    })

"""Read Satun's camera catalog and use its authenticated MJPEG gateway."""
import json
import os
from urllib.request import urlopen, Request
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


def reconnect_nvr(port):
    request = Request(base_url() + '/api/live/reconnect?' + urlencode({'port': port}), data=b'', method='POST')
    with urlopen(request, timeout=90) as response:
        return json.load(response)


def playback_url(camera_source, detected_at, public_web_url=""):
    """Link a Satun stream to its NVR clip using the captured event timestamp."""
    from urllib.parse import parse_qs
    public_web_url = (os.environ.get("SATUN_PUBLIC_WEB_URL") or public_web_url or "").strip().rstrip("/")
    public = urlsplit(public_web_url)
    if public.scheme not in {"http", "https"} or not public.hostname:
        return ""
    source = urlsplit(str(camera_source or ""))
    local = urlsplit(base_url())
    if source.netloc != local.netloc or source.path != "/api/live/stream":
        return ""
    try:
        params = parse_qs(source.query)
        port, channel = int(params["port"][0]), int(params["channel"][0])
        if not (1 <= port <= 65535 and 1 <= channel <= 32):
            return ""
        if detected_at.tzinfo is None:
            from zoneinfo import ZoneInfo
            detected_at = detected_at.replace(tzinfo=ZoneInfo("Asia/Bangkok"))
        query = urlencode(dict(nvr_port=port, channel=channel,
                               timestamp=int(detected_at.timestamp()), seconds=30))
        return public_web_url + "/nvr/playback?" + query
    except (KeyError, ValueError, TypeError, IndexError):
        return ""

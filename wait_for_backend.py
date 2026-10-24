"""Wait for the local backend to become ready before opening the dashboard."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request


def main() -> int:
    port = int(os.getenv("CCTV_BACKEND_PORT", "8081"))
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                payload = json.load(response)
            if payload.get("status") in {"ok", "degraded"}:
                print(f"[ready] Backend is {payload['status']} at {url}")
                return 0
        except Exception:
            pass
        time.sleep(1)
    print(f"[error] Backend did not become ready within 90 seconds: {url}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Serve only the public dashboard files on the loopback interface."""

from __future__ import annotations

import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent
PUBLIC_FILES = {
    "index.html",
    "live_hits.html",
    "targets.html",
    "history.html",
    "settings.html",
    "app.css",
    "style.css",
    "theme.js",
    "dashboard.css",
    "tokens.css",
}


def requested_file(request_target: str) -> str:
    path = unquote(urlsplit(request_target).path).lstrip("/")
    return path or "index.html"


def is_public_request(request_target: str) -> bool:
    return requested_file(request_target) in PUBLIC_FILES


class DashboardHandler(SimpleHTTPRequestHandler):
    server_version = "CCTVLocal"
    sys_version = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _requested_file(self) -> str:
        return requested_file(self.path)

    def _is_public(self) -> bool:
        return is_public_request(self.path)

    def do_GET(self):
        if not self._is_public():
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self):
        if not self._is_public():
            self.send_error(404)
            return
        super().do_HEAD()

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Content-Security-Policy",
            "frame-ancestors 'none'; base-uri 'none'; object-src 'none'",
        )
        if self._requested_file().endswith(".html"):
            self.send_header("Cache-Control", "no-cache")
        else:
            self.send_header("Cache-Control", "public, max-age=3600")
        super().end_headers()


def main() -> None:
    host = "127.0.0.1"
    port = int(os.getenv("CCTV_FRONTEND_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Dashboard running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

"""Serve only the public dashboard files on the loopback interface."""

from __future__ import annotations

import os
import http.client
import json
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
        if self.path.startswith('/api/'):
            return self._proxy_api()
        if not self._is_public():
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self):
        if self.path.startswith('/api/'):
            return self._proxy_api()
        if not self._is_public():
            self.send_error(404)
            return
        super().do_HEAD()

    def do_POST(self):
        if self.path.startswith('/api/'):
            return self._proxy_api()
        self.send_error(404)

    do_DELETE = do_POST

    def _proxy_api(self):
        """Forward only to the fixed loopback backend; never accept a target URL."""
        connection = http.client.HTTPConnection('127.0.0.1', 8081, timeout=100)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 0 or length > 512 * 1024 * 1024:
                self.send_error(413)
                return
            headers = {key: self.headers[key] for key in ('Content-Type', 'X-API-Key') if key in self.headers}
            body = self.rfile.read(length) if length else None
            connection.request(self.command, self.path[4:], body=body, headers=headers)
            response = connection.getresponse()
            content_type = response.getheader('Content-Type', '')
            payload = None
            if 'application/json' in content_type:
                payload = response.read()
                def public_urls(value):
                    if isinstance(value, dict):
                        return {key: public_urls(item) for key, item in value.items()}
                    if isinstance(value, list):
                        return [public_urls(item) for item in value]
                    if isinstance(value, str):
                        for prefix in ('http://localhost:8081/', 'http://127.0.0.1:8081/'):
                            if value.startswith(prefix):
                                return '/api/' + value[len(prefix):]
                    return value
                payload = json.dumps(public_urls(json.loads(payload)), ensure_ascii=False).encode('utf-8')
            self.send_response(response.status)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            if payload is not None:
                self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            if self.command == 'HEAD':
                return
            if payload is not None:
                self.wfile.write(payload)
            else:
                while True:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, ValueError, http.client.HTTPException):
            self.close_connection = True
        finally:
            connection.close()

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

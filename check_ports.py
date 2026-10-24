"""Fail fast when another application already owns a required local port."""

from __future__ import annotations

import os
import socket
import sys


def port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def main() -> int:
    host = "127.0.0.1"
    ports = {
        "frontend": int(os.getenv("CCTV_FRONTEND_PORT", "8000")),
        "backend": int(os.getenv("CCTV_BACKEND_PORT", "8081")),
    }
    unavailable = [f"{name} ({port})" for name, port in ports.items() if not port_is_available(host, port)]
    if unavailable:
        print("[error] Port already in use: " + ", ".join(unavailable), file=sys.stderr)
        print("Close the existing application and try again.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

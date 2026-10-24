"""Small security helpers shared by the local CCTV application."""

from __future__ import annotations

import ipaddress
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_loopback_host(host: str | None) -> bool:
    """Return True only for local loopback clients, including IPv4-mapped IPv6."""
    if not host:
        return False
    normalized = host.strip().strip("[]")
    if normalized.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def redact_source_url(value: object) -> str:
    """Mask credentials embedded in camera URLs before logging or displaying them."""
    source = str(value or "")
    try:
        parsed = urlsplit(source)
    except ValueError:
        return "[invalid source]"
    if not parsed.scheme or parsed.username is None:
        return source

    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port:
        host = f"{host}:{parsed.port}"
    username = parsed.username or "user"
    safe_netloc = f"{username}:***@{host}"
    return urlunsplit((parsed.scheme, safe_netloc, parsed.path, parsed.query, parsed.fragment))


def atomic_write_json(path: str | os.PathLike[str], data: object, *, mode: int = 0o600) -> None:
    """Atomically replace a JSON file and keep local secrets owner-readable only."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=4, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def protect_private_file(path: str | os.PathLike[str], *, mode: int = 0o600) -> None:
    """Restrict an existing local data file without failing on unsupported platforms."""
    try:
        destination = Path(path)
        if destination.exists():
            os.chmod(destination, mode)
    except OSError:
        # Windows and some network filesystems may not implement POSIX modes.
        pass

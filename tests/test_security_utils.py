import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from security_utils import (
    atomic_write_json,
    env_flag,
    is_loopback_host,
    protect_private_file,
    redact_source_url,
)


class SecurityUtilsTests(unittest.TestCase):
    def test_loopback_detection(self):
        for host in ("127.0.0.1", "127.10.20.30", "::1", "::ffff:127.0.0.1", "localhost"):
            self.assertTrue(is_loopback_host(host), host)
        for host in (None, "", "0.0.0.0", "192.168.1.10", "example.com"):
            self.assertFalse(is_loopback_host(host), host)

    def test_redacts_camera_password(self):
        source = "rtsp://admin:secret@example.test:554/live?profile=1"
        redacted = redact_source_url(source)
        self.assertEqual(redacted, "rtsp://admin:***@example.test:554/live?profile=1")
        self.assertNotIn("secret", redacted)
        self.assertEqual(redact_source_url("screen:1"), "screen:1")

    def test_env_flag(self):
        old = os.environ.get("TEST_LOCAL_FLAG")
        try:
            os.environ["TEST_LOCAL_FLAG"] = "yes"
            self.assertTrue(env_flag("TEST_LOCAL_FLAG"))
            os.environ["TEST_LOCAL_FLAG"] = "no"
            self.assertFalse(env_flag("TEST_LOCAL_FLAG", True))
        finally:
            if old is None:
                os.environ.pop("TEST_LOCAL_FLAG", None)
            else:
                os.environ["TEST_LOCAL_FLAG"] = old

    def test_atomic_json_write_and_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            atomic_write_json(path, {"token": "secret"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"token": "secret"})
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_protect_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.db"
            path.write_bytes(b"sqlite")
            protect_private_file(path)
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()

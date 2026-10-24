import unittest

from serve_frontend import is_public_request, requested_file


class FrontendServerTests(unittest.TestCase):
    def test_dashboard_files_are_public(self):
        for path in ("/", "/index.html", "/live_hits.html", "/app.css?v=2", "/theme.js"):
            self.assertTrue(is_public_request(path), path)

    def test_sensitive_and_unknown_files_are_blocked(self):
        for path in (
            "/config.json",
            "/cameras.json",
            "/history.db",
            "/alerts/example.jpg",
            "/%2e%2e/config.json",
            "/README.md",
        ):
            self.assertFalse(is_public_request(path), path)

    def test_root_maps_to_index(self):
        self.assertEqual(requested_file("/?theme=dark"), "index.html")


if __name__ == "__main__":
    unittest.main()

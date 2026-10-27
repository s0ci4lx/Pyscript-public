# Security policy

## Supported deployment

This project is a local desktop-style application. The supported setup is:

- dashboard: `http://127.0.0.1:8000`
- backend: `http://127.0.0.1:8081`
- no router port-forwarding, public reverse proxy, or LAN exposure

Windows dashboards and backend APIs require a named administrator account.
When installed beside Satun AI Cam, they share its owner account file; passwords
are salted PBKDF2 hashes, never plaintext. A standalone installation creates its
first owner with `python -m dashboard_security --setup`. Missing or invalid
accounts fail closed. A browser may cache HTTP Basic credentials until its
profile closes; deleting the account file is not logout.

Tailscale Serve may expose the dashboard through the exact configured HTTPS
hostname, while the backend remains loopback-only. Authentication is still
required. Arbitrary Host/Origin values and forwarded peer spoofing are rejected.

Satun camera integration uses a restricted service account whose generated
password lives in Windows Credential Manager under the same Windows account.
It is sent only in a header to the fixed local camera gateway, never in camera
URLs, subprocess arguments or plaintext configuration. Native vault failures
do not fall back to plaintext. The dashboard proxy forwards browser credentials
only to its fixed loopback backend.

The application stores sensitive camera URLs, face images, alert evidence, license plates, and optional Telegram credentials on the local disk. Use an encrypted disk and a password-protected operating-system account.

## Before sharing diagnostics

Do not attach these files or directories to bug reports:

- `config.json`
- `cameras.json`
- `history.db*`
- `alerts/`, `targets/`, `uploads/`, `vehicle_targets/`
- terminal output containing camera URLs

Replace any accidentally shared RTSP or Telegram credential immediately.

## Reporting a vulnerability

Report security issues privately to the project owner. Do not include real camera credentials, biometric images, or surveillance footage in the initial report.

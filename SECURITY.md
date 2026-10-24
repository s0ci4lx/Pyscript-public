# Security policy

## Supported deployment

This project is a local desktop-style application. The supported setup is:

- dashboard: `http://127.0.0.1:8000`
- backend: `http://127.0.0.1:8081`
- no router port-forwarding, public reverse proxy, or LAN exposure

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

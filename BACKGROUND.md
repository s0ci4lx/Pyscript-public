# Windows background detection and web login

After the one-time installation, run `start_windows.bat` or
`start_background.bat`. The existing hidden watchdog starts the backend and
frontend without opening a browser or rerunning CUDA setup. Use
`open_dashboard.bat` when you want to view or control the cameras.

Detection runs in the backend. Closing the tab/browser stops previews, not AI.
Keep Windows running and awake. Explicit Play/Pause/Stop actions are saved in
`background-camera-state.json`; restart restores only those saved camera states,
not every configured camera. Source URLs and target settings remain in their
existing files. Removed camera IDs are not recreated by state restoration.

The dashboard on port 8000 and backend on 8081 require an administrator login.
When Satun exists at `SATUN_PROJECT_DIR` (or the standard Windows install path),
the dashboard shares its account file and owner password. No password is copied
into Pyscript configuration. The dashboard uses its same-origin `/api/` proxy so
previews and controls work without separate backend login dialogs.

For an independent installation, create the first account locally:

```bat
.venv\Scripts\python.exe -m dashboard_security --setup --username owner
```

Input is hidden; use at least 14 characters. Existing account files are never
overwritten by this command. HTTP Basic credentials may remain cached until the
browser profile closes. `CCTV_WEB_AUTH_PATH` may specify an absolute account path;
`CCTV_BACKGROUND_STATE_PATH` may specify a private persistent state path.

Satun camera access needs the matching `pyscript-bridge` service account and its
`PYSCRIPT_BRIDGE_PASSWORD` entry in Windows Credential Manager. It uses Satun's
installation-path service name. Provision it locally under the same Windows
account as both services; never place the password in URLs, arguments or Git.
Normal direct camera inputs do not require the optional Satun integration.

With Tailscale, the frontend accepts only the HTTPS hostname derived from
Satun's configured `WEB_TAILSCALE_ORIGIN`, using the dashboard's existing Serve
route on port 443. Web login remains required; the application ports bind to
loopback. Keep public tunnels and router forwarding disabled.

Validated on Windows: both saved live cameras automatically resumed after a
backend restart; their AI inference counters advanced during a 15-second window
without the test requesting video/preview data. Authenticated dashboard/API and
Tailscale requests pass; anonymous, foreign-origin and private-file requests are
rejected. Camera configuration and target settings were preserved.

"""Create/update the project virtual environment only when requirements change."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
STAMP = VENV_DIR / ".requirements.sha256"


def requirements_digest() -> str:
    files = sorted(ROOT.glob("requirements*.txt"))
    content = b"".join(path.name.encode() + path.read_bytes() for path in files)
    return hashlib.sha256(content + os.environ.get("CCTV_PROFILE", "full").encode()).hexdigest()


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["full", "vehicle"], default="vehicle" if os.name == "nt" else "full")
    args = parser.parse_args()
    os.environ["CCTV_PROFILE"] = args.profile
    windows_gpu = os.name == "nt" and args.profile == "vehicle"
    if windows_gpu and (sys.version_info[:2] != (3, 11) or sys.maxsize <= 2**32):
        print("[setup] Windows vehicle setup requires Python 3.11 (64-bit).")
        return 1
    python = venv_python()
    if not python.exists():
        print("[setup] Creating isolated environment in .venv ...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)

    wanted = requirements_digest()
    current = STAMP.read_text(encoding="utf-8").strip() if STAMP.exists() else ""
    if current == wanted:
        dependency_check = subprocess.run(
            [str(python), "-m", "pip", "check"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if dependency_check.returncode != 0:
            current = ""
    if current != wanted:
        print("[setup] Installing project dependencies (one-time or requirements changed) ...")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools<81"],
            cwd=ROOT,
            check=True,
        )
        if windows_gpu:
            subprocess.run(
                [str(python), "-m", "pip", "install", "--upgrade", "--force-reinstall", "torch>=2.6,<3",
                 "torchvision>=0.21,<1", "--index-url", "https://download.pytorch.org/whl/cu126"],
                cwd=ROOT, check=True)
        selected = ROOT / "requirements-windows.txt" if args.profile == "vehicle" else REQUIREMENTS
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(selected)],
            cwd=ROOT,
            check=True,
        )
        STAMP.write_text(wanted + "\n", encoding="utf-8")
    else:
        print("[setup] Dependencies are already up to date.")

    if windows_gpu:
        subprocess.run([str(python), str(ROOT / "check_windows_gpu.py")], cwd=ROOT, check=True)
        import json
        profile = {"profile": "vehicle", "python": "3.11", "cuda_index": "cu126"}
        (VENV_DIR / ".install-profile.json").write_text(json.dumps(profile), encoding="utf-8")
        config_file = ROOT / "config.json"
        if not config_file.exists():
            config_file.write_text(json.dumps({
                "enable_person_detection": False, "detect_any_person": False,
                "enable_lpr": True, "show_only_targets": True,
                "detect_any_vehicle_zone": False, "telegram_notify_faces": False,
                "telegram_notify_vehicles": True, "telegram_notify_plates": True,
                "satun_async_capture": False
            }, indent=2), encoding="utf-8")
    print(python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

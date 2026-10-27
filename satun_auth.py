"""Install as satun_auth.py in Pyscript. Credentials stay in the native vault."""
import base64
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

USERNAME = 'pyscript-bridge'
SECRET_KEY = 'PYSCRIPT_BRIDGE_PASSWORD'


def base_url():
    return os.environ.get('SATUN_API_URL', 'http://127.0.0.1:8765').rstrip('/')


def is_satun_stream(value):
    parsed, local = urlsplit(str(value)), urlsplit(base_url())
    return (parsed.scheme == local.scheme == 'http' and parsed.netloc == local.netloc
            and parsed.hostname in {'localhost', '127.0.0.1', '::1'}
            and parsed.path == '/api/live/stream' and not parsed.username)


def vault_password():
    import keyring
    vault = keyring.get_keyring()
    if type(vault).__module__ != 'keyring.backends.Windows':
        raise RuntimeError('Windows Credential Manager is required for the Satun bridge')
    root = Path(os.environ.get('SATUN_PROJECT_DIR') or
                (os.environ.get('SystemDrive', 'C:') + r'\Satun-AI-Cam-main\Satun-AI-Cam-main')).resolve()
    password = vault.get_password('Satun AI Cam:' + str(root), SECRET_KEY)
    if not password:
        raise RuntimeError('Satun bridge credential is not configured')
    return password


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError('Satun credential requests must not redirect')


def open_satun(value, *, timeout=20, data=None):
    parsed, local = urlsplit(value), urlsplit(base_url())
    allowed = {'/api/nvr-cameras', '/api/live/stream', '/api/live/snapshot', '/api/live/reconnect'}
    if (parsed.scheme != 'http' or parsed.hostname not in {'localhost','127.0.0.1','::1'}
            or parsed.netloc != local.netloc or parsed.path not in allowed or parsed.username):
        raise ValueError('Satun credentials may only be sent to the local camera gateway')
    authorization = base64.b64encode((USERNAME + ':' + vault_password()).encode()).decode()
    request = Request(value, data=data, headers={'Authorization': 'Basic ' + authorization})
    return build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=timeout)


class SatunCapture:
    """OpenCV-compatible MJPEG reader without credentials in URLs or subprocess args."""
    def __init__(self, source):
        self.response = None
        self.buffer = b''
        self.shape = None
        try:
            if not is_satun_stream(source):
                raise ValueError('Not a local Satun camera stream')
            self.response = open_satun(source)
        except Exception:
            self.response = None

    def isOpened(self):
        return self.response is not None

    def read(self):
        import cv2
        import numpy as np
        try:
            while self.response:
                start = self.buffer.find(b'\xff\xd8')
                end = self.buffer.find(b'\xff\xd9', start + 2) if start >= 0 else -1
                if start >= 0 and end >= 0:
                    frame = cv2.imdecode(np.frombuffer(self.buffer[start:end+2], dtype=np.uint8), cv2.IMREAD_COLOR)
                    self.buffer = self.buffer[end+2:]
                    if frame is not None:
                        self.shape = frame.shape
                        return True, frame
                chunk = self.response.read1(65536)
                if not chunk or len(self.buffer) + len(chunk) > 16_000_000:
                    break
                self.buffer += chunk
        except Exception:
            pass
        self.release()
        return False, None

    def release(self):
        response, self.response = self.response, None
        if response:
            response.close()
        self.buffer = b''

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FPS:
            return 15.0
        if self.shape and prop in {cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT}:
            return float(self.shape[1 if prop == cv2.CAP_PROP_FRAME_WIDTH else 0])
        return 0.0

    def set(self, prop, value):
        return False

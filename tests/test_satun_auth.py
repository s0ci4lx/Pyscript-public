import importlib.util
import io
from pathlib import Path
from unittest.mock import Mock
import cv2
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('satun_auth',Path(__file__).resolve().parents[1]/'satun_auth.py')
auth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auth)


def test_auth_header_uses_vault_and_never_url(monkeypatch):
    monkeypatch.setattr(auth,'vault_password',lambda:'synthetic-bridge-password')
    opener = Mock()
    monkeypatch.setattr(auth,'build_opener',lambda *args:opener)
    url = 'http://127.0.0.1:8765/api/live/stream?port=1011&channel=1'
    auth.open_satun(url)
    request = opener.open.call_args.args[0]
    assert request.full_url == url
    assert request.get_header('Authorization').startswith('Basic ')
    for target in ('http://attacker.example/api/live/stream','http://127.0.0.1:8000/api/live/stream','http://127.0.0.1:8765/settings'):
        with pytest.raises(ValueError): auth.open_satun(target)


def test_mjpeg_reader_decodes_chunked_frames_and_closes(monkeypatch):
    frame = np.full((12,16,3),120,dtype=np.uint8)
    ok,jpeg = cv2.imencode('.jpg',frame)
    assert ok
    stream = io.BytesIO(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpeg.tobytes()+b'\r\n')
    monkeypatch.setattr(auth,'open_satun',lambda *args,**kwargs:stream)
    capture = auth.SatunCapture('http://127.0.0.1:8765/api/live/stream?port=1011&channel=1')
    success, decoded = capture.read()
    assert success and decoded.shape == frame.shape
    assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 16
    capture.release()
    assert stream.closed and not capture.isOpened()


def test_mjpeg_failure_has_no_secret_and_fails_closed(monkeypatch):
    def unavailable(*args,**kwargs): raise RuntimeError('synthetic-private-value')
    monkeypatch.setattr(auth,'open_satun',unavailable)
    capture = auth.SatunCapture('http://127.0.0.1:8765/api/live/stream')
    assert not capture.isOpened()
    assert capture.read() == (False,None)
    with pytest.raises(RuntimeError): auth.NoRedirect().redirect_request()

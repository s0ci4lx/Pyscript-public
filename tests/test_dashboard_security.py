import importlib.util
import json
from pathlib import Path
import threading

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def security(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.setenv('SATUN_PROJECT_DIR',str(root))
    monkeypatch.setenv('CCTV_WEB_AUTH_PATH',str(tmp_path/'users.json'))
    monkeypatch.setenv('CCTV_BACKGROUND_STATE_PATH',str(tmp_path/'background.json'))
    spec=importlib.util.spec_from_file_location('dashboard_security',root/'dashboard_security.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def account(security):
    users={name:security.shared.password_record('synthetic-dashboard-password',role)
           for name,role in [('owner','admin'),('viewer','viewer'),('pyscript-bridge','bridge')]}
    security.shared.atomic_private_write(security.ACCOUNTS,json.dumps(users))


def test_backend_requires_login_and_admin(security):
    app=FastAPI(); app.add_middleware(security.PyscriptSecurity)
    @app.get('/health')
    def health(): return {'ok':True}
    client=TestClient(app,base_url='http://localhost',client=('127.0.0.1',12345))
    with client:
        assert client.get('/health').status_code==503
    assert client.get('/health').status_code==503
    account(security)
    assert client.get('/health').status_code==401
    assert client.get('/health',auth=('owner','synthetic-dashboard-password')).status_code==200
    assert client.get('/health',auth=('viewer','synthetic-dashboard-password')).status_code==403
    assert client.get('/health',auth=('pyscript-bridge','synthetic-dashboard-password')).status_code==403
    assert client.get('/health',headers={'host':'attacker.example'},auth=('owner','synthetic-dashboard-password')).status_code==403


def test_camera_state_survives_reloads_and_rejects_invalid_states(security):
    security.persist_camera_state('camera-a','playing')
    security.persist_camera_state('camera-b','stopped')
    assert security.background_states()=={'camera-a':'playing','camera-b':'stopped'}
    security.persist_camera_state('camera-a','paused')
    assert security.background_states()['camera-a']=='paused'
    with pytest.raises(ValueError): security.persist_camera_state('camera-c','unknown')
    security.STATE_PATH.write_text('[]')
    with pytest.raises(ValueError): security.background_states()


def test_frontend_html_requires_login_and_private_files_stay_private(security,tmp_path,monkeypatch):
    import sys
    from http.server import ThreadingHTTPServer
    monkeypatch.setitem(sys.modules,'dashboard_security',security)
    # Exercise the prepared companion frontend without loading camera/AI code.
    source=Path(__file__).resolve().parents[1]/'serve_frontend.py'
    spec=importlib.util.spec_from_file_location('test_dashboard_frontend',source)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ROOT=tmp_path
    (tmp_path/'index.html').write_text('<html>dashboard</html>')
    (tmp_path/'config.json').write_text('{"secret":"synthetic-private-config"}')
    account(security)
    server=ThreadingHTTPServer(('127.0.0.1',0),module.DashboardHandler)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{server.server_port}') as client:
            assert client.get('/index.html').status_code==401
            response=client.get('/index.html',auth=('owner','synthetic-dashboard-password'))
            assert response.status_code==200 and 'no-store' in response.headers['cache-control']
            assert client.get('/config.json',auth=('owner','synthetic-dashboard-password')).status_code==404
            assert client.get('/index.html',auth=('owner','synthetic-dashboard-password'),headers={'origin':'https://attacker.example'}).status_code==403
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)

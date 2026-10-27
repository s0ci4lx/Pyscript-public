"""Install as dashboard_security.py: shared owner login and persistent camera state."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlsplit, unquote

SATUN_ROOT = Path(os.environ.get('SATUN_PROJECT_DIR') or
                  (os.environ.get('SystemDrive','C:') + r'\Satun-AI-Cam-main\Satun-AI-Cam-main')).resolve()
ROOT = Path(__file__).resolve().parent
security_source = ROOT/'web_security.py'
if not security_source.is_file(): security_source = SATUN_ROOT/'src/web_security.py'
spec = importlib.util.spec_from_file_location('satun_shared_web_security', security_source)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
from dotenv import dotenv_values
from starlette.responses import JSONResponse, Response
values = dotenv_values(SATUN_ROOT/'.env')
default_accounts = SATUN_ROOT/'data/web-users.json' if (SATUN_ROOT/'.env').is_file() else ROOT/'data/web-users.json'
ACCOUNTS = Path(os.environ.get('CCTV_WEB_AUTH_PATH') or values.get('WEB_AUTH_PATH') or default_accounts)
if not ACCOUNTS.is_absolute(): ACCOUNTS = SATUN_ROOT/ACCOUNTS
origin = urlsplit(values.get('WEB_TAILSCALE_ORIGIN') or '')
TAILSCALE_ORIGIN = 'https://' + origin.hostname if origin.hostname else None
STATE_PATH = Path(os.environ.get('CCTV_BACKGROUND_STATE_PATH') or Path(__file__).resolve().parent/'background-camera-state.json')
_state_lock = threading.RLock()
_auth_lock = threading.RLock()


def background_states():
    if not STATE_PATH.exists(): return {}
    states = json.loads(STATE_PATH.read_text(encoding='utf-8'))
    if not isinstance(states,dict) or any(s not in {'playing','paused','stopped'} for s in states.values()):
        raise ValueError('Invalid background camera state; refusing to start unknown cameras')
    return states


def persist_camera_state(camera_id, state):
    if state not in {'playing','paused','stopped'}: raise ValueError('Invalid camera state')
    with _state_lock:
        states = background_states()
        states[str(camera_id)] = state
        shared.atomic_private_write(STATE_PATH,json.dumps(states))


class PyscriptSecurity:
    def __init__(self, app):
        async def admin_only(scope, receive, send):
            if scope['type'] not in {'http','websocket'}:
                return await app(scope,receive,send)
            if scope['state']['web_role'] != 'admin':
                return await JSONResponse({'error':'Administrator account required'},403)(scope,receive,send)
            return await app(scope,receive,send)
        self.app = shared.LocalWebSecurity(admin_only,accounts_path=ACCOUNTS)

    async def __call__(self, scope, receive, send):
        if scope['type'] in {'http','websocket'} and not ACCOUNTS.is_file():
            if scope['type']=='websocket': return await send({'type':'websocket.close','code':1008})
            return await JSONResponse({'error':'Configure the owner account locally first'},503)(scope,receive,send)
        await self.app(scope,receive,send)


async def _frontend_admin(scope,receive,send):
    if scope['state']['web_role'] != 'admin':
        return await JSONResponse({'error':'Administrator account required'},403)(scope,receive,send)
    return await Response(status_code=204)(scope,receive,send)


_frontend = shared.LocalWebSecurity(_frontend_admin,accounts_path=ACCOUNTS,tailscale_origin=TAILSCALE_ORIGIN)


def authorize_dashboard(handler):
    """Authenticate before serving HTML or proxying data; never read request bodies."""
    if not ACCOUNTS.is_file():
        handler.send_error(503,'Configure the owner account locally first')
        return False
    target = urlsplit(handler.path)
    # Actual mutation outcomes are audited by backend middleware after forwarding.
    scope = dict(type='http',http_version='1.1',method='GET',scheme='http',
                 path=unquote(target.path),raw_path=target.path.encode('ascii'),
                 query_string=target.query.encode('ascii'),root_path='',
                 client=handler.client_address,server=handler.server.server_address,
                 headers=[(k.lower().encode('latin1'),v.encode('latin1')) for k,v in handler.headers.items()])
    messages=[]
    async def receive(): return {'type':'http.request','body':b'','more_body':False}
    async def send(message): messages.append(message)
    with _auth_lock:
        asyncio.run(_frontend(scope,receive,send))
    start = next(message for message in messages if message['type']=='http.response.start')
    if start['status']==204: return True
    handler.send_response(start['status'])
    for key,value in start.get('headers',[]): handler.send_header(key.decode('latin1'),value.decode('latin1'))
    handler.end_headers()
    if handler.command!='HEAD':
        for message in messages:
            if message['type']=='http.response.body': handler.wfile.write(message.get('body',b''))
    return False


def main():
    import argparse
    import getpass
    parser=argparse.ArgumentParser(description='Create a local dashboard owner account with hidden input')
    parser.add_argument('--setup',action='store_true',required=True)
    parser.add_argument('--username',default='owner')
    args=parser.parse_args()
    if ACCOUNTS.exists(): parser.error('Account file already exists; refusing to overwrite it')
    if not args.username.isascii() or not args.username or not all(c.isalnum() or c in '._-' for c in args.username):
        parser.error('Use an ASCII account name containing letters, numbers, . _ -')
    password=getpass.getpass('New dashboard password (at least 14 characters): ')
    if password!=getpass.getpass('Confirm password: '): parser.error('Passwords do not match')
    record=shared.password_record(password,'admin')
    shared.atomic_private_write(ACCOUNTS,json.dumps({args.username:record}))
    print('Dashboard owner account configured. Password was not stored in plaintext.')


if __name__=='__main__': main()

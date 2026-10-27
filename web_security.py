"""Local-only web access, optional named accounts, and mutation auditing.

Configure accounts locally with: python -m src.web_security USER --role admin
Passwords are prompted, never accepted on the command line or stored in plaintext.
"""
import argparse
import base64
import binascii
import getpass
import hashlib
import hmac
import ipaddress
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import secrets
import tempfile
import time
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import JSONResponse

LOG = logging.getLogger('satun.security')
ROLES = {'viewer', 'operator', 'admin', 'bridge'}
ITERATIONS = 600_000
LOCAL_HOSTS = {'localhost', '127.0.0.1', '::1'}


def restrict_private_file(path):
    """Protect each newly created file, including replacements and rotated logs."""
    if os.name != 'nt':
        os.chmod(path, 0o600)
        return
    import ctypes
    from ctypes import wintypes
    # OWNER RIGHTS follows the file owner; retain SYSTEM and administrators.
    advapi = ctypes.WinDLL('advapi32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    descriptor = ctypes.c_void_p()
    convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    convert.restype = wintypes.BOOL
    apply = advapi.SetFileSecurityW
    apply.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
    apply.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    flags = 'OICI' if Path(path).is_dir() else ''
    sddl = 'D:P' + ''.join(f'(A;{flags};FA;;;{sid})' for sid in ('OW', 'SY', 'BA'))
    if not convert(sddl, 1, ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not apply(str(Path(path).resolve()), 0x80000004, descriptor):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.LocalFree(descriptor)


class PrivateAuditHandler(RotatingFileHandler):
    def emit(self, record):
        try:
            super().emit(record)
        finally:
            # Windows cannot rename/delete an open log; never retain its handle.
            self.close()

    def _open(self):
        fd = os.open(self.baseFilename, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            restrict_private_file(self.baseFilename)
        except Exception:
            os.close(fd)
            raise
        return os.fdopen(fd, 'a', encoding='utf-8')


def atomic_private_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            name = stream.name
            restrict_private_file(name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def password_record(password, role):
    if role not in ROLES or len(password) < 14:
        raise ValueError('ใช้รหัสผ่านอย่างน้อย 14 ตัวอักษรและสิทธิ์ที่รองรับ')
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), ITERATIONS).hex()
    return dict(role=role, salt=salt, hash=digest, iterations=ITERATIONS)


def loopback(address):
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def validate_tailscale_origin(value):
    if not value:
        return None
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or not parsed.hostname or not parsed.hostname.endswith('.ts.net')
            or parsed.username is not None or parsed.password is not None
            or parsed.path or parsed.query or parsed.fragment):
        raise ValueError('WEB_TAILSCALE_ORIGIN must be an exact HTTPS .ts.net origin')
    parsed.port  # Reject malformed ports before starting the server.
    return value.lower()


class LocalWebSecurity:
    def __init__(self, app, accounts_path, tailscale_origin=None):
        self.app = app
        self.tailscale_origin = validate_tailscale_origin(tailscale_origin)
        self.accounts_path = Path(accounts_path)
        self.cache = {}
        self.failures = {}
        self.cache_key = secrets.token_bytes(32)
        self.signature = None
        audit_path = self.accounts_path.with_name('security-audit.log')
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit = logging.getLogger('satun.audit.' + hashlib.sha256(str(audit_path).encode()).hexdigest()[:16])
        self.audit.setLevel(logging.INFO)
        if not self.audit.handlers:
            # Open with restrictive permissions before attaching the rotating handler.
            fd = os.open(audit_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            os.close(fd)
            restrict_private_file(audit_path)
            handler = PrivateAuditHandler(audit_path, maxBytes=1_000_000, backupCount=3, encoding='utf-8', delay=True)
            handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
            self.audit.addHandler(handler)
            self.audit.propagate = False

    async def __call__(self, scope, receive, send):
        if scope['type'] not in {'http', 'websocket'}:
            return await self.app(scope, receive, send)
        if scope['type'] == 'websocket':
            return await send({'type': 'websocket.close', 'code': 1008})
        request = Request(scope)
        async def reject(status, message, challenge=False):
            response = JSONResponse({'ok': False, 'error': message}, status_code=status,
                headers={'Cache-Control': 'no-store', **({'WWW-Authenticate': 'Basic realm="Satun AI Cam", charset="UTF-8"'} if challenge else {})})
            await response(scope, receive, send)

        # Do not trust forwarded headers or a Host header to establish a local peer.
        peer = request.client.host if request.client else ''
        host = request.url.hostname
        tailscale_access = bool(self.tailscale_origin and
                                request.url.netloc.lower() == urlsplit(self.tailscale_origin).netloc)
        if not loopback(peer) or (host not in LOCAL_HOSTS and not tailscale_access):
            return await reject(403, 'ระบบนี้อนุญาตให้ใช้จากเครื่องที่ติดตั้งเท่านั้น')
        if tailscale_access:
            # Only the explicit HTTPS Serve origin may arrive through loopback.
            # Forwarded headers never establish a trusted peer or grant a role.
            scope['scheme'] = 'https'
            request = Request(scope)
            if not self.accounts_path.exists() and self.signature is None:
                return await reject(503, 'ต้องตั้งบัญชีเว็บก่อนใช้ผ่าน Tailscale')
        # Block cross-site reads as well as writes (including DNS-rebinding hosts).
        if request.headers.get('sec-fetch-site') == 'cross-site':
            return await reject(403, 'ไม่อนุญาตคำขอจากเว็บไซต์อื่น')
        origin = request.headers.get('origin')
        if origin:
            parsed = urlsplit(origin)
            if (parsed.scheme, parsed.netloc) != (request.url.scheme, request.url.netloc):
                return await reject(403, 'ไม่อนุญาตคำขอจากเว็บไซต์อื่น')

        username, role = 'local-owner', 'admin'
        try:
            # Deleting/corrupting a previously configured account file fails closed.
            if self.accounts_path.exists() or self.signature is not None:
                stat = self.accounts_path.stat()
                signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
                if signature != self.signature:
                    self.cache.clear()
                    self.signature = signature
                users = json.loads(self.accounts_path.read_text(encoding='utf-8'))
                if not users or any(u.get('role') not in ROLES for u in users.values()):
                    raise ValueError('Invalid accounts')
                header = request.headers.get('authorization', '')
                now = time.monotonic()
                count, since = self.failures.get(peer, (0, now))
                if now - since >= 60:
                    count, since = 0, now
                if count >= 10:
                    return await reject(429, 'ลองเข้าสู่ระบบใหม่ในอีกหนึ่งนาที')
                cache_id = hmac.new(self.cache_key, header.encode(), hashlib.sha256).digest()
                cached = self.cache.get(cache_id)
                if cached and cached[2] > now:
                    username, role = cached[:2]
                else:
                    try:
                        if len(header) > 4096 or not header.startswith('Basic '):
                            raise ValueError('Missing credentials')
                        username, password = base64.b64decode(header[6:], validate=True).decode().split(':', 1)
                        record = users.get(username)
                        # A dummy record keeps unknown-user and wrong-password costs comparable.
                        check = record or dict(salt='00'*16, hash='00'*32, iterations=ITERATIONS)
                        iterations = int(check['iterations'])
                        if not 600_000 <= iterations <= 1_000_000:
                            raise ValueError('Invalid hash settings')
                        digest = hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(check['salt']), iterations).hex()
                        if not hmac.compare_digest(digest, check['hash']) or not record:
                            raise ValueError('Invalid credentials')
                        role = record['role']
                    except (ValueError, UnicodeError, binascii.Error, KeyError):
                        if header:
                            self.failures[peer] = (count + 1, since)
                            self.audit.warning('authentication_failed')
                        return await reject(401, 'กรุณาเข้าสู่ระบบด้วยบัญชีของคุณ', True)
                    if len(self.cache) >= 128:
                        self.cache.clear()
                    self.cache[cache_id] = (username, role, now + 60)
                    self.failures.pop(peer, None)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            LOG.error('Account configuration unavailable; access denied')
            return await reject(503, 'ไม่สามารถอ่านบัญชีผู้ใช้ กรุณาตรวจการตั้งค่าที่เครื่องเซิร์ฟเวอร์')

        path = request.url.path.rstrip('/')
        admin_only = path == '/settings' or path.startswith(('/api/settings', '/api/docs', '/openapi.json')) or path == '/api/cameras/update'
        read_post = path in {'/api/nvr/search-direct', '/api/nvr/panorama-preview'}
        mutation = request.method not in {'GET', 'HEAD', 'OPTIONS'}
        if role == 'bridge' and (request.method, path) not in {
            ('GET', '/api/nvr-cameras'), ('GET', '/api/live/stream'),
            ('GET', '/api/live/snapshot'), ('POST', '/api/live/reconnect'),
        }:
            return await reject(403, 'บัญชีเชื่อมต่อกล้องไม่มีสิทธิ์ดำเนินการนี้')
        if (admin_only and role != 'admin') or (mutation and not read_post and role == 'viewer'):
            return await reject(403, 'บัญชีนี้ไม่มีสิทธิ์ดำเนินการ')
        scope.setdefault('state', {}).update(web_user=username, web_role=role)
        async def secured_send(message):
            if message['type'] == 'http.response.start':
                headers = [(k, v) for k, v in message.get('headers', []) if k.lower() != b'cache-control']
                headers.extend([(b'cache-control', b'no-store'), (b'x-content-type-options', b'nosniff'),
                                (b'x-frame-options', b'SAMEORIGIN'), (b'referrer-policy', b'no-referrer')])
                message = {**message, 'headers': headers}
                if mutation:
                    # No bodies, query parameters, credentials, or image paths in audit records.
                    self.audit.info(json.dumps(dict(event='web_action', user=username, role=role,
                        method=request.method, path=scope['path'], status=message['status']), ensure_ascii=True))
            await send(message)
        await self.app(scope, receive, secured_send)


def main():
    parser = argparse.ArgumentParser(description='ตั้งบัญชีเข้าเว็บในเครื่อง โดยไม่ใช้รหัส NVR')
    parser.add_argument('username')
    parser.add_argument('--role', choices=sorted(ROLES), default='admin')
    args = parser.parse_args()
    if not args.username or ':' in args.username or len(args.username) > 80 or not args.username.isascii() or not all(c.isalnum() or c in '._-' for c in args.username):
        parser.error('ใช้ชื่อบัญชีภาษาอังกฤษ ตัวเลข . _ -')
    from .config import ROOT
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    path = Path(os.environ.get('WEB_AUTH_PATH', '').strip() or ROOT / 'data' / 'web-users.json')
    users = json.loads(path.read_text()) if path.exists() else {}
    password = getpass.getpass('รหัสผ่านเว็บใหม่ (อย่างน้อย 14 ตัวอักษร): ')
    if password != getpass.getpass('ยืนยันรหัสผ่าน: '):
        parser.error('รหัสผ่านไม่ตรงกัน')
    try:
        users[args.username] = password_record(password, args.role)
    except ValueError as error:
        parser.error(str(error))
    if not any(user['role'] == 'admin' for user in users.values()):
        parser.error('ต้องมีบัญชี admin อย่างน้อยหนึ่งบัญชี')
    atomic_private_write(path, json.dumps(users, ensure_ascii=False))
    print('บันทึกบัญชีแล้ว เว็บจะขอชื่อบัญชีและรหัสผ่านในการเข้าใช้ครั้งถัดไป')


if __name__ == '__main__':
    main()

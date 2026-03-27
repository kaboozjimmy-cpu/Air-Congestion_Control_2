"""
auth.py — Firebase Auth + Role-Based Access Control
Roles: user (passenger), controller (ATC), admin (full access)
Security: rate limiting, session timeout, input sanitization, brute-force protection
"""
import os
import re
import json
import time
import sqlite3
import hashlib
import requests
from datetime import datetime, timedelta
from functools import wraps

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "roles.db")
AUTH_BASE = "https://identitytoolkit.googleapis.com/v1/accounts"

ROLES = {
    'user':       {'level': 1, 'label': 'Passenger',        'icon': '👤', 'color': '#7bd0ff'},
    'controller': {'level': 2, 'label': 'ATC Controller',   'icon': '🎮', 'color': '#2ecc71'},
    'admin':      {'level': 3, 'label': 'Administrator',    'icon': '👑', 'color': '#d97722'},
}

SESSION_TIMEOUT_MINUTES = 60
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 15


# ═══════════════════════════════════════
#  LOCAL ROLE DB (maps Firebase UID → role)
# ═══════════════════════════════════════

def _get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS user_roles (
        uid TEXT PRIMARY KEY,
        email TEXT NOT NULL,
        full_name TEXT DEFAULT '',
        role TEXT DEFAULT 'user',
        approved INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        last_login TEXT,
        login_count INTEGER DEFAULT 0
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS login_attempts (
        email TEXT NOT NULL,
        attempt_time REAL NOT NULL,
        success INTEGER DEFAULT 0
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        uid TEXT, email TEXT, action TEXT,
        details TEXT, ip TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    return conn


def _log_audit(uid, email, action, details=''):
    try:
        conn = _get_db()
        conn.execute("INSERT INTO audit_log (uid, email, action, details) VALUES (?,?,?,?)",
                     (uid, email, action, details))
        conn.commit()
        conn.close()
    except Exception:
        pass


# ═══════════════════════════════════════
#  SECURITY: Rate Limiting + Brute Force
# ═══════════════════════════════════════

def _check_rate_limit(email):
    """Check if email is locked out from too many failed attempts."""
    conn = _get_db()
    cutoff = time.time() - (LOCKOUT_MINUTES * 60)
    rows = conn.execute(
        "SELECT COUNT(*) FROM login_attempts WHERE email=? AND attempt_time>? AND success=0",
        (email.lower(), cutoff)
    ).fetchone()
    conn.close()
    return rows[0] >= MAX_LOGIN_ATTEMPTS


def _record_attempt(email, success):
    conn = _get_db()
    conn.execute("INSERT INTO login_attempts (email, attempt_time, success) VALUES (?,?,?)",
                 (email.lower(), time.time(), 1 if success else 0))
    if success:
        conn.execute("DELETE FROM login_attempts WHERE email=? AND success=0",
                     (email.lower(),))
    conn.commit()
    conn.close()


# ═══════════════════════════════════════
#  SECURITY: Input Validation
# ═══════════════════════════════════════

def _sanitize(text, max_len=200):
    """Strip dangerous characters from user input."""
    if not text:
        return ''
    text = str(text)[:max_len].strip()
    text = re.sub(r'[<>{}|\\^`]', '', text)
    return text


def _validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def _validate_password(password):
    """Enforce password strength."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r'[A-Z]', password):
        return False, "Password needs at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password needs at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return False, "Password needs at least one number"
    return True, "OK"


# ═══════════════════════════════════════
#  FIREBASE REST API
# ═══════════════════════════════════════

def _get_api_key():
    return os.getenv("FIREBASE_API_KEY", "")


def _firebase_request(endpoint, payload):
    api_key = _get_api_key()
    if not api_key:
        return False, "Firebase API key not set. Add FIREBASE_API_KEY to .env"
    url = f"{AUTH_BASE}:{endpoint}?key={api_key}"
    try:
        resp = requests.post(url, json=payload, timeout=10)
        data = resp.json()
        if resp.status_code == 200:
            return True, data
        msg = data.get('error', {}).get('message', 'Unknown error')
        friendly = {
            'EMAIL_EXISTS': 'An account with this email already exists',
            'INVALID_EMAIL': 'Invalid email address',
            'WEAK_PASSWORD : Password should be at least 6 characters': 'Password too weak',
            'EMAIL_NOT_FOUND': 'No account found with this email',
            'INVALID_PASSWORD': 'Incorrect password',
            'INVALID_LOGIN_CREDENTIALS': 'Incorrect email or password',
            'USER_DISABLED': 'Account disabled by administrator',
            'TOO_MANY_ATTEMPTS_TRY_LATER': 'Too many attempts. Try again later',
            'OPERATION_NOT_ALLOWED': 'Email/Password auth not enabled in Firebase',
        }
        return False, friendly.get(msg, msg)
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Firebase. Check internet."
    except Exception as e:
        return False, f"Auth error: {str(e)}"


# ═══════════════════════════════════════
#  REGISTER / LOGIN / SESSION
# ═══════════════════════════════════════

def register_user(email, password, full_name='', requested_role='user'):
    """Register via Firebase + store role locally.
    First user auto-becomes admin. Controllers need admin approval."""
    email = _sanitize(email).lower()
    full_name = _sanitize(full_name)

    if not _validate_email(email):
        return False, "Please enter a valid email address"
    pw_ok, pw_msg = _validate_password(password)
    if not pw_ok:
        return False, pw_msg
    if not full_name:
        return False, "Please enter your full name"
    if requested_role not in ROLES:
        requested_role = 'user'

    ok, data = _firebase_request('signUp', {
        'email': email, 'password': password, 'returnSecureToken': True
    })
    if not ok:
        return False, data

    # Set display name
    if data.get('idToken'):
        _firebase_request('update', {
            'idToken': data['idToken'], 'displayName': full_name, 'returnSecureToken': False
        })

    uid = data.get('localId', '')

    # First user becomes admin automatically
    conn = _get_db()
    user_count = conn.execute("SELECT COUNT(*) FROM user_roles").fetchone()[0]
    if user_count == 0:
        role = 'admin'
        approved = 1
    elif requested_role == 'controller':
        role = 'controller'
        approved = 0  # needs admin approval
    else:
        role = requested_role
        approved = 1

    conn.execute(
        "INSERT OR REPLACE INTO user_roles (uid, email, full_name, role, approved, last_login) VALUES (?,?,?,?,?,?)",
        (uid, email, full_name, role, approved, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    _log_audit(uid, email, 'REGISTER', f'role={role}, approved={approved}')

    user = {
        'uid': uid, 'email': email, 'full_name': full_name,
        'role': role, 'approved': approved,
        'id_token': data.get('idToken', ''),
        'refresh_token': data.get('refreshToken', ''),
        'login_time': time.time(),
    }

    if role == 'controller' and not approved:
        return True, {**user, '_pending': True}
    return True, user


def login_user(email, password):
    """Login via Firebase + fetch local role."""
    email = _sanitize(email).lower()
    if not email or not password:
        return False, "Please fill in all fields"

    if _check_rate_limit(email):
        return False, f"Account locked for {LOCKOUT_MINUTES} minutes due to too many failed attempts"

    ok, data = _firebase_request('signInWithPassword', {
        'email': email, 'password': password, 'returnSecureToken': True
    })

    if not ok:
        _record_attempt(email, False)
        _log_audit('', email, 'LOGIN_FAILED', str(data))
        return False, data

    _record_attempt(email, True)
    uid = data.get('localId', '')

    # Get display name
    display_name = data.get('displayName', '')
    if not display_name:
        ok2, info = _firebase_request('lookup', {'idToken': data['idToken']})
        if ok2:
            users = info.get('users', [])
            if users:
                display_name = users[0].get('displayName', '')

    # Get role from local DB
    conn = _get_db()
    row = conn.execute("SELECT role, approved, full_name FROM user_roles WHERE uid=?", (uid,)).fetchone()
    if row:
        role, approved, stored_name = row
        if not display_name:
            display_name = stored_name
        conn.execute("UPDATE user_roles SET last_login=?, login_count=login_count+1 WHERE uid=?",
                     (datetime.now().isoformat(), uid))
    else:
        role, approved = 'user', 1
        conn.execute(
            "INSERT INTO user_roles (uid, email, full_name, role, approved, last_login) VALUES (?,?,?,?,?,?)",
            (uid, email, display_name or email.split('@')[0], role, approved, datetime.now().isoformat())
        )
    conn.commit()
    conn.close()

    _log_audit(uid, email, 'LOGIN', f'role={role}')

    user = {
        'uid': uid, 'email': email,
        'full_name': display_name or email.split('@')[0],
        'role': role, 'approved': approved,
        'id_token': data.get('idToken', ''),
        'refresh_token': data.get('refreshToken', ''),
        'login_time': time.time(),
    }

    if role == 'controller' and not approved:
        return True, {**user, '_pending': True}
    return True, user


def check_session_valid(user):
    """Check if session has timed out."""
    if not user:
        return False
    login_time = user.get('login_time', 0)
    if time.time() - login_time > SESSION_TIMEOUT_MINUTES * 60:
        return False
    return True


def send_password_reset(email):
    email = _sanitize(email).lower()
    if not _validate_email(email):
        return False, "Invalid email"
    ok, data = _firebase_request('sendOobCode', {
        'requestType': 'PASSWORD_RESET', 'email': email
    })
    if ok:
        _log_audit('', email, 'PASSWORD_RESET_SENT')
        return True, "Password reset email sent. Check your inbox."
    return False, data


# ═══════════════════════════════════════
#  ADMIN: User Management
# ═══════════════════════════════════════

def get_all_users():
    conn = _get_db()
    rows = conn.execute(
        "SELECT uid, email, full_name, role, approved, created_at, last_login, login_count "
        "FROM user_roles ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [{'uid': r[0], 'email': r[1], 'full_name': r[2], 'role': r[3],
             'approved': r[4], 'created_at': r[5], 'last_login': r[6],
             'login_count': r[7]} for r in rows]


def update_user_role(admin_uid, target_uid, new_role):
    if new_role not in ROLES:
        return False, "Invalid role"
    conn = _get_db()
    conn.execute("UPDATE user_roles SET role=? WHERE uid=?", (new_role, target_uid))
    conn.commit()
    conn.close()
    _log_audit(admin_uid, '', 'ROLE_CHANGE', f'target={target_uid} new_role={new_role}')
    return True, f"Role updated to {new_role}"


def approve_user(admin_uid, target_uid):
    conn = _get_db()
    conn.execute("UPDATE user_roles SET approved=1 WHERE uid=?", (target_uid,))
    conn.commit()
    conn.close()
    _log_audit(admin_uid, '', 'APPROVE_USER', f'target={target_uid}')
    return True, "User approved"


def deny_user(admin_uid, target_uid):
    conn = _get_db()
    conn.execute("UPDATE user_roles SET approved=0 WHERE uid=?", (target_uid,))
    conn.commit()
    conn.close()
    _log_audit(admin_uid, '', 'DENY_USER', f'target={target_uid}')
    return True, "User access denied"


def get_audit_log(limit=100):
    conn = _get_db()
    rows = conn.execute(
        "SELECT timestamp, email, action, details FROM audit_log ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [{'time': r[0], 'email': r[1], 'action': r[2], 'details': r[3]} for r in rows]


def get_user_count():
    try:
        conn = _get_db()
        count = conn.execute("SELECT COUNT(*) FROM user_roles").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def get_pending_count():
    try:
        conn = _get_db()
        count = conn.execute("SELECT COUNT(*) FROM user_roles WHERE approved=0").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0

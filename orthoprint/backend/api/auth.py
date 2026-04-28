"""
OrthoPrint — Authentication & RBAC
JWT-based login with role-based access control.

Roles per requirements document:
  clinician  — patient assessment, scan review, sign-off
  designer   — geometry design workflow
  reviewer   — review and approve designs
  fabricator — view fabrication queue, update dispatch status
  admin      — full access including user management
"""
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import jwt, JWTError

router = APIRouter()

# ─── Config ───────────────────────────────────────────────────────────────────

SECRET_KEY     = "CHANGE_THIS_IN_PRODUCTION_USE_ENV_VAR"
ALGORITHM      = "HS256"
TOKEN_EXPIRE_H = 8

# Try bcrypt first; fall back to sha256_crypt if the C extension is missing
try:
    import bcrypt as _bcrypt_check  # noqa: F401
    _schemes = ["bcrypt"]
except Exception:
    _schemes = ["sha256_crypt"]

pwd_ctx  = CryptContext(schemes=_schemes, deprecated="auto")
security = HTTPBearer()

ROLES = {"clinician", "designer", "reviewer", "fabricator", "admin"}

ROLE_PERMISSIONS: dict[str, set] = {
    "admin":      {"read", "write", "design", "review", "fabricate", "admin"},
    "clinician":  {"read", "write", "review"},
    "designer":   {"read", "design"},
    "reviewer":   {"read", "review"},
    "fabricator": {"read", "fabricate"},
}

# ─── Default dev accounts ─────────────────────────────────────────────────────

_DEFAULT_USERS = [
    {"email": "admin@pravartya.com",      "name": "System Admin",   "role": "admin",      "password": "Admin@123"},
    {"email": "clinician@pravartya.com",  "name": "Dr. Doddameti",  "role": "clinician",  "password": "Clinic@123"},
    {"email": "designer@pravartya.com",   "name": "Design Lead",    "role": "designer",   "password": "Design@123"},
    {"email": "fabricator@pravartya.com", "name": "Fab Technician", "role": "fabricator", "password": "Fabr@123"},
]

_users: dict[str, dict] = {}


def _seed_default_users():
    """Hash and store default dev accounts. Prints verification to stdout."""
    pw_map = {u["email"]: u["password"] for u in _DEFAULT_USERS}
    for u in _DEFAULT_USERS:
        hashed = pwd_ctx.hash(u["password"])
        _users[u["email"]] = {
            "user_id":    str(uuid.uuid4()),
            "email":      u["email"],
            "name":       u["name"],
            "role":       u["role"],
            "hashed_pw":  hashed,
            "is_active":  True,
            "created_at": datetime.utcnow().isoformat(),
        }

    # Verify each account immediately after seeding
    for email, user in _users.items():
        try:
            ok = pwd_ctx.verify(pw_map[email], user["hashed_pw"])
            print(f"[auth] seed {'OK' if ok else 'FAIL'}: {email}", flush=True)
        except Exception as exc:
            print(f"[auth] seed ERROR for {email}: {exc}", flush=True)


_seed_default_users()


# ─── JWT helpers ──────────────────────────────────────────────────────────────

def _create_token(user_id: str, email: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_H)
    return jwt.encode(
        {"sub": user_id, "email": email, "role": role, "exp": expire},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
        )


# ─── Dependencies ─────────────────────────────────────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    payload = _decode_token(credentials.credentials)
    email   = payload.get("email")
    user    = _users.get(email)
    if not user or not user["is_active"]:
        raise HTTPException(401, "User not found or inactive")
    return user


def require_role(*allowed_roles: str):
    """Dependency factory — restricts endpoint to the given roles."""
    def _dep(user: dict = Depends(get_current_user)):
        if user["role"] not in allowed_roles and user["role"] != "admin":
            raise HTTPException(
                status_code=403,
                detail=f"Role '{user['role']}' not permitted. Required: {list(allowed_roles)}",
            )
        return user
    return _dep


# ─── Pydantic models ──────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


class UserCreate(BaseModel):
    email: str
    name: str
    role: str
    password: str


class UserOut(BaseModel):
    user_id: str
    email: str
    name: str
    role: str
    is_active: bool
    created_at: str
    permissions: list


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/debug/users")
def debug_users():
    """Dev-only: confirm seeded users (no passwords returned)."""
    return [
        {
            "email":       u["email"],
            "role":        u["role"],
            "is_active":   u["is_active"],
            "hash_scheme": u["hashed_pw"].split("$")[1] if "$" in u["hashed_pw"] else "unknown",
        }
        for u in _users.values()
    ]


@router.post("/login")
def login(req: LoginRequest):
    """Authenticate and return a JWT access token."""
    user = _users.get(req.email)
    if not user:
        raise HTTPException(401, "Invalid email or password")
    try:
        valid = pwd_ctx.verify(req.password, user["hashed_pw"])
    except Exception as exc:
        print(f"[auth] verify error for {req.email}: {exc}", flush=True)
        raise HTTPException(500, "Password verification failed — check server logs")
    if not valid:
        raise HTTPException(401, "Invalid email or password")
    if not user["is_active"]:
        raise HTTPException(403, "Account is disabled")

    token = _create_token(user["user_id"], user["email"], user["role"])
    return {
        "access_token":    token,
        "token_type":      "bearer",
        "expires_in_hours": TOKEN_EXPIRE_H,
        "user": {
            "user_id":     user["user_id"],
            "name":        user["name"],
            "email":       user["email"],
            "role":        user["role"],
            "permissions": list(ROLE_PERMISSIONS.get(user["role"], set())),
        },
    }


@router.get("/me", response_model=UserOut)
def me(user: dict = Depends(get_current_user)):
    return {**user, "permissions": list(ROLE_PERMISSIONS.get(user["role"], set()))}


@router.post("/users", response_model=UserOut)
def create_user(data: UserCreate, _: dict = Depends(require_role("admin"))):
    if data.email in _users:
        raise HTTPException(409, "Email already registered")
    if data.role not in ROLES:
        raise HTTPException(400, f"Invalid role. Must be one of: {sorted(ROLES)}")
    user = {
        "user_id":    str(uuid.uuid4()),
        "email":      data.email,
        "name":       data.name,
        "role":       data.role,
        "hashed_pw":  pwd_ctx.hash(data.password),
        "is_active":  True,
        "created_at": datetime.utcnow().isoformat(),
    }
    _users[data.email] = user
    return {**user, "permissions": list(ROLE_PERMISSIONS.get(data.role, set()))}


@router.get("/users", response_model=list[UserOut])
def list_users(_: dict = Depends(require_role("admin"))):
    return [
        {**u, "permissions": list(ROLE_PERMISSIONS.get(u["role"], set()))}
        for u in _users.values()
    ]


@router.patch("/users/{user_id}/deactivate")
def deactivate_user(user_id: str, _: dict = Depends(require_role("admin"))):
    user = next((u for u in _users.values() if u["user_id"] == user_id), None)
    if not user:
        raise HTTPException(404, "User not found")
    user["is_active"] = False
    return {"ok": True}

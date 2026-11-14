import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

# passlib 1.7.4 reads bcrypt.__about__.__version__ but bcrypt 4.x moved it to bcrypt.__version__
import bcrypt as _bcrypt
if not hasattr(_bcrypt, "__about__"):
    class _About:
        __version__ = _bcrypt.__version__
    _bcrypt.__about__ = _About()

from dotenv import load_dotenv
from jose import JWTError, jwt
from passlib.context import CryptContext

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ALGORITHM       = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def _secret() -> str:
    secret = os.getenv("MILES_JWT_SECRET")
    if not secret:
        raise RuntimeError("MILES_JWT_SECRET not set. Run scripts/setup_auth.py first.")
    return secret

def get_password_hash() -> str:
    h = os.getenv("MILES_PASSWORD_HASH")
    if not h:
        raise RuntimeError("MILES_PASSWORD_HASH not set. Run scripts/setup_auth.py first.")
    return h


def verify_password(plain: str) -> bool:
    return _pwd_context.verify(plain, get_password_hash())


# What a device token may do. The phone writes its location and Health data while
# it sits in a pocket, long after the hour a login lasts, and renewing a login
# takes Face ID. So the app also holds this: long lived, and good only for those
# writes. server.py refuses it everywhere else, so a leaked one cannot read a
# memory or touch the calendar. Rotating MILES_JWT_SECRET revokes every token.
DEVICE_SCOPE = "device"
DEVICE_TOKEN_EXPIRE_DAYS = 180


def create_token(subject: str, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
                 scope: str | None = None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    payload = {"sub": subject, "exp": expire}
    if scope:
        payload["scope"] = scope
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    # raises JWTError (caught by the server) if signature is bad or token is expired
    return jwt.decode(token, _secret(), algorithms=[ALGORITHM])

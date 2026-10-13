"""
backend/security/auth.py - API Authentication & Google Authenticator (TOTP) Service
Handles password hashing, TOTP 2FA enrollment, QR code generation,
token issuance, and user session validation.
"""

import io
import base64
import hashlib
import os
import time
from typing import Optional, Dict, Any, Tuple
import pyotp
import qrcode
import jwt
from fastapi import Header, HTTPException, status, Depends
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database.connection import get_db


JWT_SECRET = settings.SECRET_KEY or "semantic-assistant-secure-jwt-key"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 72


def hash_password(password: str) -> str:
    """Hashes a password with a random 16-byte salt using PBKDF2-HMAC-SHA256."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"{salt.hex()}${key.hex()}"


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """Verifies a plain password against the stored salt$hash format."""
    try:
        parts = stored_hash.split("$")
        if len(parts) != 2:
            return False
        salt = bytes.fromhex(parts[0])
        expected_key = parts[1]
        actual_key = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, 100_000).hex()
        return actual_key == expected_key
    except Exception:
        return False


def generate_totp_secret() -> str:
    """Generates a base32 TOTP secret compatible with Google Authenticator."""
    return pyotp.random_base32()


def generate_totp_qr_data_url(email: str, secret: str) -> Tuple[str, str]:
    """
    Generates the otpauth URI and a base64 Data URL for the QR code
    so the user can scan it into Google Authenticator directly on screen.
    """
    totp = pyotp.TOTP(secret)
    uri = totp.provisioning_uri(
        name=email,
        issuer_name="Semantic Email Assistant"
    )

    qr = qrcode.QRCode(box_size=5, border=2)
    qr.add_data(uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    return uri, data_url


def verify_totp_code(secret: str, code: str) -> bool:
    """
    Verifies a 6-digit TOTP code against the secret key with a 30s window.
    """
    if not secret or not code:
        return False
    # Clean up whitespace
    clean_code = str(code).strip().replace(" ", "")
    totp = pyotp.TOTP(secret)
    return bool(totp.verify(clean_code, valid_window=1))


def create_access_token(user_id: str, email: str, name: str) -> str:
    """Issues a signed JWT access token for user sessions."""
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "exp": int(time.time()) + (JWT_EXPIRATION_HOURS * 3600),
        "iat": int(time.time())
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decodes and validates a JWT access token."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired. Please log in again."
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token."
        )


def get_current_user_optional(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """
    Dependency that extracts the logged-in User model if Authorization header is provided.
    Returns None if unauthenticated.
    """
    if not authorization:
        return None

    try:
        token = authorization.replace("Bearer ", "").strip()
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            return None
        from backend.models.db_models import User
        user = db.query(User).filter(User.id == user_id).first()
        return user
    except Exception:
        return None


def get_current_user(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """
    Enforces authentication for protected endpoints.
    Raises 401 Unauthorized if token is missing, expired, invalid, or user does not exist.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token."
        )

    token = authorization.replace("Bearer ", "").strip()
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token payload."
        )

    from backend.models.db_models import User
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found."
        )
    return user


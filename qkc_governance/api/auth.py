"""JWT authentication for the governance API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from qkc_governance.config import settings

_bearer = HTTPBearer(auto_error=True)


class TokenPayload(BaseModel):
    sub: str
    role: str = "operator"
    exp: int


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def create_token(subject: str, role: str = "operator") -> TokenResponse:
    secret = settings.jwt_secret.get_secret_value()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": subject, "role": role, "exp": expire}
    token = jwt.encode(payload, secret, algorithm=settings.jwt_algorithm)
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expire_minutes * 60,
    )


def decode_token(token: str) -> TokenPayload:
    secret = settings.jwt_secret.get_secret_value()
    try:
        data = jwt.decode(token, secret, algorithms=[settings.jwt_algorithm])
        return TokenPayload(**data)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_operator(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> TokenPayload:
    return decode_token(creds.credentials)


def require_admin(
    payload: TokenPayload = Depends(require_operator),
) -> TokenPayload:
    if payload.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return payload

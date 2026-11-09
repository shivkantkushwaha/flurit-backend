"""Supabase JWT Authentication Module.

Supports modern asymmetric ES256 (ECDSA P-256) JWT validation against Supabase's published
JWKS endpoint with bounded HTTP timeouts, in-memory key caching, and automatic key rotation
refresh. Supports legacy symmetric HS256 tokens only when explicitly enabled by configuration.
Enforces strict audience, issuer, expiry, and subject UUID validation. Fails closed on any
unverifiable token or missing configuration.
"""

import base64
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import (
    ExpiredSignatureError,
    InvalidAudienceError,
    InvalidIssuerError,
    InvalidSignatureError,
    PyJWKClientConnectionError,
    PyJWKClientError,
    PyJWKError,
    PyJWKSetError,
    PyJWTError,
)
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger("flurit.auth")

security_scheme = HTTPBearer(auto_error=False)

# In-memory singleton cache of PyJWKClient instances keyed by JWKS URL
_jwk_clients: dict[str, jwt.PyJWKClient] = {}


class AuthenticatedUser(BaseModel):
    """Normalized authenticated user identity from Supabase JWT claims."""

    user_id: str = Field(..., description="Unique Supabase user identifier (sub)")
    email: str | None = Field(default=None, description="User email address if present in claims")
    role: str = Field(default="authenticated", description="User role (e.g. authenticated)")
    claims: dict[str, Any] = Field(default_factory=dict, description="Raw decoded JWT claims")


def get_jwks_url(settings: Settings) -> str:
    """Derive the Supabase JWKS endpoint URL from configuration."""
    if settings.SUPABASE_JWKS_URL and settings.SUPABASE_JWKS_URL.strip():
        return settings.SUPABASE_JWKS_URL.strip()
    base_url = (settings.SUPABASE_URL or "").strip().rstrip("/")
    if not base_url:
        return ""
    return f"{base_url}/auth/v1/.well-known/jwks.json"


def get_jwks_client(settings: Settings) -> jwt.PyJWKClient:
    """Retrieve or construct a thread-safe PyJWKClient with bounded timeout and key caching."""
    url = get_jwks_url(settings)
    if not url:
        raise ValueError(
            "Cannot initialize JWKS client: SUPABASE_URL or SUPABASE_JWKS_URL is not configured."
        )
    if url not in _jwk_clients:
        _jwk_clients[url] = jwt.PyJWKClient(
            uri=url,
            cache_keys=True,
            max_cached_keys=16,
            cache_jwk_set=True,
            lifespan=86400,  # Cache keys for up to 24h
            timeout=10,  # Bounded 10s network timeout
            cooldown_duration=0,  # Allow immediate refresh on unknown key ID (rotation)
        )
    return _jwk_clients[url]


def clear_jwks_client_cache() -> None:
    """Clear cached PyJWKClient instances (useful in automated tests)."""
    _jwk_clients.clear()


def _b64_url_encode(data: bytes) -> str:
    """Encode bytes to unpadded URL-safe base64 string."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64_url_decode(data: str) -> bytes:
    """Decode unpadded URL-safe base64 string to bytes."""
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_jwt_token(
    payload: dict[str, Any],
    secret: str,
    algorithm: str = "HS256",
) -> str:
    """Generate a signed JWT token using HMAC-SHA256 (useful for tests and mock auth)."""
    if algorithm != "HS256":
        raise ValueError(f"Unsupported algorithm: {algorithm}. Only HS256 is supported.")

    token_payload = dict(payload)
    token_payload.setdefault("aud", "authenticated")
    token_payload.setdefault("iss", "https://jhankemazkagrybaoitu.supabase.co/auth/v1")

    header = {"alg": "HS256", "typ": "JWT"}
    h_b64 = _b64_url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    p_b64 = _b64_url_encode(json.dumps(token_payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{h_b64}.{p_b64}".encode()

    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64_url_encode(signature)

    return f"{h_b64}.{p_b64}.{sig_b64}"


def decode_jwt_token(
    token: str,
    secret: str,
    expected_algorithm: str = "HS256",
    expected_audience: str | None = "authenticated",
    expected_issuer: str | None = None,
) -> dict[str, Any]:
    """Verify and decode an HS256 Supabase JWT token against configured claims and signature."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Malformed authentication token: expected 3 dot-separated segments.")

    header_b64, payload_b64, signature_b64 = parts

    try:
        header_data = json.loads(_b64_url_decode(header_b64).decode("utf-8"))
    except Exception as exc:
        raise ValueError("Malformed authentication token: invalid header encoding.") from exc

    alg = header_data.get("alg")
    if alg != expected_algorithm:
        raise ValueError(
            f"Unsupported or mismatched token algorithm: '{alg}'. Expected '{expected_algorithm}'."
        )

    if expected_algorithm != "HS256":
        raise ValueError(
            f"Configured algorithm '{expected_algorithm}' is currently unsupported. Only HS256 is supported."
        )

    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected_sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    expected_sig_b64 = _b64_url_encode(expected_sig)

    if not hmac.compare_digest(signature_b64, expected_sig_b64):
        raise ValueError("Invalid authentication token signature.")

    try:
        payload_data: dict[str, Any] = json.loads(_b64_url_decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise ValueError("Malformed authentication token: invalid payload encoding.") from exc

    exp = payload_data.get("exp")
    if exp is None:
        raise ValueError("Invalid token: missing expiration claim ('exp').")
    if not isinstance(exp, (int, float)):
        raise ValueError("Invalid token: expiration claim ('exp') must be numeric.")
    if time.time() > float(exp):
        raise ValueError("Authentication token has expired.")

    nbf = payload_data.get("nbf")
    if nbf is not None:
        if not isinstance(nbf, (int, float)):
            raise ValueError("Invalid token: not-before claim ('nbf') must be numeric.")
        if time.time() < float(nbf):
            raise ValueError("Authentication token is not yet valid ('nbf' claim).")

    if expected_audience is not None:
        aud = payload_data.get("aud")
        if aud is None:
            raise ValueError(
                f"Invalid token: missing audience claim ('aud'). Expected '{expected_audience}'."
            )
        if isinstance(aud, list):
            if expected_audience not in aud:
                raise ValueError(
                    f"Invalid token audience claim: '{aud}' does not contain '{expected_audience}'."
                )
        elif str(aud) != expected_audience:
            raise ValueError(
                f"Invalid token audience claim: expected '{expected_audience}', got '{aud}'."
            )

    if expected_issuer is not None:
        iss = payload_data.get("iss")
        if iss is None or str(iss) != expected_issuer:
            raise ValueError(
                f"Invalid token issuer claim: expected '{expected_issuer}', got '{iss}'."
            )

    sub = payload_data.get("sub")
    if not sub or not str(sub).strip():
        raise ValueError("Invalid token: missing or empty subject claim ('sub').")

    return payload_data


def _decode_es256_token(token: str, settings: Settings) -> dict[str, Any]:
    """Verify and decode an ES256 JWT using Supabase's published JWKS endpoint."""
    jwks_url = get_jwks_url(settings)
    if not jwks_url:
        logger.error("Authentication failed closed: SUPABASE_URL / JWKS URL is unconfigured.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication system is misconfigured: SUPABASE_URL is required.",
        )

    try:
        client = get_jwks_client(settings)
        signing_key = client.get_signing_key_from_jwt(token)
    except (PyJWKClientConnectionError, TimeoutError, urllib.error.URLError) as exc:
        logger.error("JWKS connection error: unable to reach Supabase JWKS endpoint")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication service unavailable or unable to reach JWKS endpoint.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except (PyJWKClientError, PyJWKError, PyJWKSetError) as exc:
        logger.warning("JWKS key lookup failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unrecognized token signing key ID.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except Exception as exc:
        logger.warning("Unexpected error during JWKS key retrieval: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token verification failed: unable to resolve signing key.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256"],
            audience=settings.SUPABASE_JWT_AUDIENCE if settings.SUPABASE_JWT_AUDIENCE else None,
            issuer=settings.SUPABASE_JWT_ISSUER if settings.SUPABASE_JWT_ISSUER else None,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_aud": bool(settings.SUPABASE_JWT_AUDIENCE),
                "verify_iss": bool(settings.SUPABASE_JWT_ISSUER),
                "require": ["exp", "sub"]
                + (["iss"] if settings.SUPABASE_JWT_ISSUER else [])
                + (["aud"] if settings.SUPABASE_JWT_AUDIENCE else []),
            },
            leeway=10,
        )
        return payload
    except ExpiredSignatureError as exc:
        logger.warning("Token verification failed: Token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except InvalidAudienceError as exc:
        logger.warning("Token verification failed: Invalid audience claim")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token audience claim.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except InvalidIssuerError as exc:
        logger.warning("Token verification failed: Invalid issuer claim")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token issuer claim.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except (InvalidSignatureError, PyJWTError) as exc:
        logger.warning("Token verification failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token signature or claims: {type(exc).__name__}.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def _decode_hs256_token(token: str, settings: Settings) -> dict[str, Any]:
    """Verify and decode a legacy HS256 JWT using the configured shared secret."""
    secret = settings.SUPABASE_JWT_SECRET
    if not secret or not secret.strip():
        logger.error("Authentication failed closed: SUPABASE_JWT_SECRET is not configured.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication system is misconfigured: SUPABASE_JWT_SECRET is required.",
        )

    try:
        return decode_jwt_token(
            token=token,
            secret=secret,
            expected_algorithm="HS256",
            expected_audience=settings.SUPABASE_JWT_AUDIENCE,
            expected_issuer=settings.SUPABASE_JWT_ISSUER,
        )
    except ValueError as exc:
        err_msg = str(exc)
        logger.warning("HS256 token verification failed: %s", err_msg)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=err_msg,
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def verify_supabase_jwt(
    credentials: HTTPAuthorizationCredentials | None = Security(security_scheme),
    settings: Settings = Depends(get_settings),
) -> AuthenticatedUser:
    """Validate Supabase Bearer JWT token against configured algorithms and key sources."""
    # 1. Check presence of Authorization Bearer header
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Bearer token is missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials.strip()

    # 2. Extract unverified header to inspect algorithm and key ID
    try:
        unverified_header = jwt.get_unverified_header(token)
    except Exception as exc:
        logger.warning("Token verification failed: invalid token header encoding")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed authentication token: invalid header.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    alg = unverified_header.get("alg")

    # 3. Explicit algorithm allowlist check - reject unlisted or "none" algorithms
    primary_alg = settings.SUPABASE_JWT_ALGORITHM.upper()
    allowed_algs: set[str] = {primary_alg}
    if settings.SUPABASE_ALLOW_HS256:
        allowed_algs.add("HS256")

    if not alg or alg.upper() not in allowed_algs:
        logger.warning("Token rejected: disallowed algorithm '%s' (allowed: %s)", alg, allowed_algs)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unsupported or disallowed token signing algorithm.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    alg_upper = alg.upper()

    # 4. Dispatch verification based on algorithm
    if alg_upper == "ES256":
        payload = _decode_es256_token(token=token, settings=settings)
    elif alg_upper == "HS256":
        payload = _decode_hs256_token(token=token, settings=settings)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unsupported token signing algorithm: {alg_upper}.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 5. Validate subject claim (sub)
    sub = payload.get("sub")
    if not sub or not str(sub).strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing or empty subject claim ('sub').",
            headers={"WWW-Authenticate": "Bearer"},
        )

    sub_str = str(sub).strip()

    # Reject non-UUID subjects in production or when strict validation is enabled
    if settings.is_production or settings.SUPABASE_STRICT_SUB_UUID:
        try:
            val_uuid = uuid.UUID(sub_str)
            if str(val_uuid).lower() != sub_str.lower():
                raise ValueError("Non-canonical UUID representation")
        except (ValueError, AttributeError) as exc:
            logger.warning("Token rejected: subject claim 'sub' is not a valid UUID format")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: subject claim ('sub') must be a valid UUID in production.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

    return AuthenticatedUser(
        user_id=sub_str,
        email=payload.get("email"),
        role=str(payload.get("role", "authenticated")),
        claims=payload,
    )

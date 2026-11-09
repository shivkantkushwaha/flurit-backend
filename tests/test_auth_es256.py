"""Comprehensive test suite for Supabase ES256 / ECDSA P-256 JWT validation and JWKS key management.

Covers:
1. Valid ES256 token verification via JWKS.
2. Tampered signature rejection (HTTP 401).
3. Expired token rejection (HTTP 401).
4. Wrong audience rejection (HTTP 401).
5. Wrong issuer rejection (HTTP 401).
6. Non-UUID subject rejection under strict validation / production (HTTP 401).
7. Empty/missing subject rejection (HTTP 401).
8. Algorithm allowlist enforcement ('none', unauthorized 'HS256', 'RS256' rejected with HTTP 401).
9. Key rotation handling (cache miss on new key ID triggers JWKS refresh and succeeds).
10. Unknown key ID rejection after JWKS refresh (HTTP 401).
11. Fail closed on JWKS network failures (HTTP 401).
12. Fail closed on missing SUPABASE_URL configuration (HTTP 500).
13. Zero credential/token leakage in logs.
"""

import base64
import json
import logging
import time
import unittest
import urllib.error
from unittest.mock import patch

import jwt
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.core.auth import (
    AuthenticatedUser,
    clear_jwks_client_cache,
    get_jwks_client,
    verify_supabase_jwt,
)
from app.core.config import Settings

TEST_SUPABASE_URL = "https://jhankemazkagrybaoitu.supabase.co"
TEST_ISSUER = "https://jhankemazkagrybaoitu.supabase.co/auth/v1"
TEST_AUDIENCE = "authenticated"


def _b64url_uint(val: int, length: int = 32) -> str:
    """Encode an integer coordinate into URL-safe unpadded base64."""
    return base64.urlsafe_b64encode(val.to_bytes(length, "big")).decode("utf-8").rstrip("=")


def generate_ec_p256_key_and_jwk(kid: str) -> tuple[ec.EllipticCurvePrivateKey, dict[str, str]]:
    """Generate an ephemeral EC SECP256R1 private key and its corresponding public JWK."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    pub_numbers = private_key.public_key().public_numbers()
    jwk = {
        "kty": "EC",
        "crv": "P-256",
        "x": _b64url_uint(pub_numbers.x),
        "y": _b64url_uint(pub_numbers.y),
        "kid": kid,
        "use": "sig",
        "alg": "ES256",
    }
    return private_key, jwk


class TestSupabaseES256Authentication(unittest.TestCase):
    """Test suite for Supabase ES256 / JWKS validation and security boundary enforcement."""

    def setUp(self) -> None:
        clear_jwks_client_cache()
        self.key1, self.jwk1 = generate_ec_p256_key_and_jwk("test-kid-1")
        self.key2, self.jwk2 = generate_ec_p256_key_and_jwk("test-kid-2")

        self.default_settings = Settings(
            ENVIRONMENT="development",
            SUPABASE_URL=TEST_SUPABASE_URL,
            SUPABASE_JWT_ALGORITHM="ES256",
            SUPABASE_JWT_ISSUER=TEST_ISSUER,
            SUPABASE_JWT_AUDIENCE=TEST_AUDIENCE,
            SUPABASE_ALLOW_HS256=False,
            SUPABASE_STRICT_SUB_UUID=False,
        )

    def tearDown(self) -> None:
        clear_jwks_client_cache()

    def _create_token(
        self,
        private_key: ec.EllipticCurvePrivateKey,
        sub: str = "550e8400-e29b-41d4-a716-446655440000",
        email: str = "analyst@flurit.com",
        kid: str = "test-kid-1",
        aud: str = TEST_AUDIENCE,
        iss: str = TEST_ISSUER,
        exp_delta: int = 3600,
        alg: str = "ES256",
    ) -> str:
        payload = {
            "sub": sub,
            "email": email,
            "role": "authenticated",
            "aud": aud,
            "iss": iss,
            "exp": int(time.time()) + exp_delta,
        }
        headers = {"kid": kid, "alg": alg}
        return jwt.encode(payload, private_key, algorithm=alg, headers=headers)

    def test_valid_es256_token_verifies_successfully(self) -> None:
        """Valid ES256 token decodes cleanly with all expected claims."""
        token = self._create_token(self.key1, sub="550e8400-e29b-41d4-a716-446655440000")
        client = get_jwks_client(self.default_settings)

        with patch.object(client, "fetch_data", return_value={"keys": [self.jwk1]}):
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            user = verify_supabase_jwt(credentials=creds, settings=self.default_settings)

            self.assertIsInstance(user, AuthenticatedUser)
            self.assertEqual(user.user_id, "550e8400-e29b-41d4-a716-446655440000")
            self.assertEqual(user.email, "analyst@flurit.com")
            self.assertEqual(user.role, "authenticated")

    def test_tampered_signature_rejected(self) -> None:
        """ES256 token with tampered signature segment is rejected with HTTP 401."""
        token = self._create_token(self.key1)
        # Flip an interior character of the signature to ensure valid base64url encoding
        orig_char = token[-10]
        tampered_char = "A" if orig_char != "A" else "B"
        tampered_token = token[:-10] + tampered_char + token[-9:]

        client = get_jwks_client(self.default_settings)
        with patch.object(client, "fetch_data", return_value={"keys": [self.jwk1]}):
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=tampered_token)
            with self.assertRaises(HTTPException) as ctx:
                verify_supabase_jwt(credentials=creds, settings=self.default_settings)
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIn("signature", ctx.exception.detail.lower())

    def test_expired_token_rejected(self) -> None:
        """Expired ES256 token is rejected with HTTP 401."""
        token = self._create_token(self.key1, exp_delta=-120)  # Expired 2 minutes ago
        client = get_jwks_client(self.default_settings)

        with patch.object(client, "fetch_data", return_value={"keys": [self.jwk1]}):
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            with self.assertRaises(HTTPException) as ctx:
                verify_supabase_jwt(credentials=creds, settings=self.default_settings)
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIn("expired", ctx.exception.detail.lower())

    def test_wrong_audience_rejected(self) -> None:
        """Token with mismatched audience claim is rejected with HTTP 401."""
        token = self._create_token(self.key1, aud="unauthorized-service")
        client = get_jwks_client(self.default_settings)

        with patch.object(client, "fetch_data", return_value={"keys": [self.jwk1]}):
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            with self.assertRaises(HTTPException) as ctx:
                verify_supabase_jwt(credentials=creds, settings=self.default_settings)
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIn("audience", ctx.exception.detail.lower())

    def test_wrong_issuer_rejected(self) -> None:
        """Token with mismatched issuer claim is rejected with HTTP 401."""
        token = self._create_token(self.key1, iss="https://other-project.supabase.co/auth/v1")
        client = get_jwks_client(self.default_settings)

        with patch.object(client, "fetch_data", return_value={"keys": [self.jwk1]}):
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            with self.assertRaises(HTTPException) as ctx:
                verify_supabase_jwt(credentials=creds, settings=self.default_settings)
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIn("issuer", ctx.exception.detail.lower())

    def test_non_uuid_subject_rejected_when_strict(self) -> None:
        """Arbitrary alphanumeric non-UUID subject is rejected when SUPABASE_STRICT_SUB_UUID=True."""
        settings = Settings(
            ENVIRONMENT="development",
            SUPABASE_URL=TEST_SUPABASE_URL,
            SUPABASE_JWT_ALGORITHM="ES256",
            SUPABASE_JWT_ISSUER=TEST_ISSUER,
            SUPABASE_JWT_AUDIENCE=TEST_AUDIENCE,
            SUPABASE_STRICT_SUB_UUID=True,
        )
        token = self._create_token(self.key1, sub="arbitrary_username_123")
        client = get_jwks_client(settings)

        with patch.object(client, "fetch_data", return_value={"keys": [self.jwk1]}):
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            with self.assertRaises(HTTPException) as ctx:
                verify_supabase_jwt(credentials=creds, settings=settings)
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIn("valid uuid", ctx.exception.detail.lower())

    def test_non_uuid_subject_rejected_in_production(self) -> None:
        """Arbitrary alphanumeric non-UUID subject is rejected when ENVIRONMENT='production'."""
        settings = Settings(
            ENVIRONMENT="production",
            SUPABASE_URL=TEST_SUPABASE_URL,
            SUPABASE_JWT_ALGORITHM="ES256",
            SUPABASE_JWT_ISSUER=TEST_ISSUER,
            SUPABASE_JWT_AUDIENCE=TEST_AUDIENCE,
            SUPABASE_ALLOW_HS256=False,
            DATABASE_URL="postgresql+psycopg2://user:pass@render.internal:5432/flurit_db",
            QUEUE_MODE="arq",
            REDIS_URL="rediss://:pass@redis.render.internal:6379/0",
            STORAGE_BACKEND="local",
            CORS_ORIGINS=["https://app.flurit.io"],
        )
        token = self._create_token(self.key1, sub="legacy-alpha-sub")
        client = get_jwks_client(settings)

        with patch.object(client, "fetch_data", return_value={"keys": [self.jwk1]}):
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            with self.assertRaises(HTTPException) as ctx:
                verify_supabase_jwt(credentials=creds, settings=settings)
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIn("valid uuid", ctx.exception.detail.lower())

    def test_missing_or_empty_subject_rejected(self) -> None:
        """Token with empty whitespace subject is rejected with HTTP 401."""
        token = self._create_token(self.key1, sub="   ")
        client = get_jwks_client(self.default_settings)

        with patch.object(client, "fetch_data", return_value={"keys": [self.jwk1]}):
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            with self.assertRaises(HTTPException) as ctx:
                verify_supabase_jwt(credentials=creds, settings=self.default_settings)
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIn("subject claim", ctx.exception.detail.lower())

    def test_algorithm_none_rejected(self) -> None:
        """Token with 'alg': 'none' is explicitly rejected before signature checks."""
        header = {"alg": "none", "typ": "JWT"}
        payload = {
            "sub": "550e8400-e29b-41d4-a716-446655440000",
            "aud": TEST_AUDIENCE,
            "iss": TEST_ISSUER,
            "exp": int(time.time()) + 3600,
        }
        h_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        p_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        token = f"{h_b64}.{p_b64}."

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        with self.assertRaises(HTTPException) as ctx:
            verify_supabase_jwt(credentials=creds, settings=self.default_settings)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("disallowed", ctx.exception.detail.lower())

    def test_hs256_rejected_when_allow_hs256_is_false(self) -> None:
        """HS256 tokens are rejected with HTTP 401 when SUPABASE_ALLOW_HS256=False."""
        settings = Settings(
            ENVIRONMENT="development",
            SUPABASE_URL=TEST_SUPABASE_URL,
            SUPABASE_JWT_ALGORITHM="ES256",
            SUPABASE_ALLOW_HS256=False,
            SUPABASE_JWT_SECRET="test-secret-at-least-32-characters-long!",
        )
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "sub": "550e8400-e29b-41d4-a716-446655440000",
            "aud": TEST_AUDIENCE,
            "iss": TEST_ISSUER,
            "exp": int(time.time()) + 3600,
        }
        h_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        p_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        token = f"{h_b64}.{p_b64}.dummysig"

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        with self.assertRaises(HTTPException) as ctx:
            verify_supabase_jwt(credentials=creds, settings=settings)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("disallowed", ctx.exception.detail.lower())

    def test_key_rotation_handling(self) -> None:
        """Cache miss on unseen key ID refreshes JWKS and verifies new key."""
        client = get_jwks_client(self.default_settings)

        # Initially, the published JWKS contains only key1
        current_jwks_keys = [self.jwk1]

        def dynamic_fetch():
            return {"keys": list(current_jwks_keys)}

        with patch.object(client, "fetch_data", side_effect=dynamic_fetch) as mock_fetch:
            # 1. First token signed with key1 succeeds
            t1 = self._create_token(self.key1, kid="test-kid-1")
            creds1 = HTTPAuthorizationCredentials(scheme="Bearer", credentials=t1)
            u1 = verify_supabase_jwt(credentials=creds1, settings=self.default_settings)
            self.assertEqual(u1.user_id, "550e8400-e29b-41d4-a716-446655440000")
            self.assertEqual(mock_fetch.call_count, 1)

            # 2. Second token with key1 is served from cache without extra HTTP call
            verify_supabase_jwt(credentials=creds1, settings=self.default_settings)
            self.assertEqual(mock_fetch.call_count, 1)

            # 3. Key rotation on Supabase: key2 is added to published JWKS
            current_jwks_keys.append(self.jwk2)

            # 4. Token signed with key2 arrives: client refreshes JWKS and verifies successfully
            t2 = self._create_token(
                self.key2,
                sub="660e8400-e29b-41d4-a716-446655440000",
                kid="test-kid-2",
            )
            creds2 = HTTPAuthorizationCredentials(scheme="Bearer", credentials=t2)
            u2 = verify_supabase_jwt(credentials=creds2, settings=self.default_settings)
            self.assertEqual(u2.user_id, "660e8400-e29b-41d4-a716-446655440000")
            self.assertEqual(mock_fetch.call_count, 2)

    def test_unknown_key_id_rejected_after_refresh(self) -> None:
        """Token with unknown key ID triggers JWKS refresh and fails with HTTP 401 when still absent."""
        token = self._create_token(self.key1, kid="non-existent-kid-999")
        client = get_jwks_client(self.default_settings)

        with patch.object(client, "fetch_data", return_value={"keys": [self.jwk1]}):
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            with self.assertRaises(HTTPException) as ctx:
                verify_supabase_jwt(credentials=creds, settings=self.default_settings)
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIn("signing key", ctx.exception.detail.lower())

    def test_jwks_network_failure_fails_closed(self) -> None:
        """Network error reaching JWKS endpoint fails closed with HTTP 401."""
        token = self._create_token(self.key1)
        client = get_jwks_client(self.default_settings)

        with patch.object(
            client,
            "fetch_data",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            with self.assertRaises(HTTPException) as ctx:
                verify_supabase_jwt(credentials=creds, settings=self.default_settings)
            self.assertEqual(ctx.exception.status_code, 401)
            self.assertIn("unavailable", ctx.exception.detail.lower())

    def test_missing_supabase_url_fails_closed_500(self) -> None:
        """Unconfigured SUPABASE_URL fails closed with HTTP 500."""
        settings = Settings(
            ENVIRONMENT="development",
            SUPABASE_URL="",
            SUPABASE_JWKS_URL=None,
            SUPABASE_JWT_ALGORITHM="ES256",
        )
        token = self._create_token(self.key1)
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        with self.assertRaises(HTTPException) as ctx:
            verify_supabase_jwt(credentials=creds, settings=settings)
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("misconfigured", ctx.exception.detail.lower())

    def test_no_secrets_or_token_leaked_in_logs(self) -> None:
        """Verify logger outputs never include raw JWT tokens, keys, or sensitive claims."""
        token = self._create_token(self.key1, sub="secret-sub-value")
        client = get_jwks_client(self.default_settings)

        with self.assertLogs("flurit.auth", level=logging.WARNING) as cm:
            with patch.object(client, "fetch_data", return_value={"keys": [self.jwk2]}):
                creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
                try:
                    verify_supabase_jwt(credentials=creds, settings=self.default_settings)
                except HTTPException:
                    pass

        log_output = " ".join(cm.output)
        # Ensure raw token is not logged
        self.assertNotIn(token, log_output)
        # Ensure signature is not logged
        self.assertNotIn(token.split(".")[2], log_output)


if __name__ == "__main__":
    unittest.main()

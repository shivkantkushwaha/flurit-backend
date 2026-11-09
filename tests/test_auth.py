"""Unit tests for Supabase JWT authentication, claims verification, and security edge cases."""

import time
import unittest

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.core.auth import AuthenticatedUser, create_jwt_token, decode_jwt_token, verify_supabase_jwt
from app.core.config import Settings

TEST_SECRET = "flurit-secure-auth-test-key-32-chars-min!"


class TestSupabaseJWTAuthentication(unittest.TestCase):
    """Test suite for Supabase JWT verification, algorithm enforcement, and claims validation."""

    def test_valid_token_decodes_successfully(self) -> None:
        """Valid token with standard claims decodes without errors."""
        payload = {
            "sub": "user-uuid-12345",
            "email": "analyst@flurit.com",
            "role": "authenticated",
            "aud": "authenticated",
            "exp": int(time.time()) + 3600,
        }
        token = create_jwt_token(payload=payload, secret=TEST_SECRET)
        decoded = decode_jwt_token(
            token=token,
            secret=TEST_SECRET,
            expected_algorithm="HS256",
            expected_audience="authenticated",
        )
        self.assertEqual(decoded["sub"], "user-uuid-12345")
        self.assertEqual(decoded["email"], "analyst@flurit.com")
        self.assertEqual(decoded["aud"], "authenticated")

    def test_algorithm_mismatch_rejected(self) -> None:
        """Tokens with mismatched or unsupported algorithm must be rejected."""
        payload = {"sub": "user-1", "exp": int(time.time()) + 3600, "aud": "authenticated"}
        token = create_jwt_token(payload=payload, secret=TEST_SECRET)

        # Expecting RS256 but token has HS256
        with self.assertRaises(ValueError) as ctx:
            decode_jwt_token(
                token=token,
                secret=TEST_SECRET,
                expected_algorithm="RS256",
            )
        self.assertIn("mismatched", str(ctx.exception).lower())

    def test_wrong_audience_rejected(self) -> None:
        """Tokens with unexpected audience claim must be rejected."""
        payload = {
            "sub": "user-1",
            "exp": int(time.time()) + 3600,
            "aud": "external-service",
        }
        token = create_jwt_token(payload=payload, secret=TEST_SECRET)

        with self.assertRaises(ValueError) as ctx:
            decode_jwt_token(
                token=token,
                secret=TEST_SECRET,
                expected_audience="authenticated",
            )
        self.assertIn("audience", str(ctx.exception).lower())

    def test_missing_audience_rejected_when_expected(self) -> None:
        """Tokens missing required audience claim must be rejected."""
        payload = {
            "sub": "user-1",
            "exp": int(time.time()) + 3600,
        }
        # Create token without aud
        import base64
        import hashlib
        import hmac
        import json

        header = {"alg": "HS256", "typ": "JWT"}
        h_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        p_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        sig = (
            base64.urlsafe_b64encode(
                hmac.new(TEST_SECRET.encode(), f"{h_b64}.{p_b64}".encode(), hashlib.sha256).digest()
            )
            .decode()
            .rstrip("=")
        )
        token = f"{h_b64}.{p_b64}.{sig}"

        with self.assertRaises(ValueError) as ctx:
            decode_jwt_token(
                token=token,
                secret=TEST_SECRET,
                expected_audience="authenticated",
            )
        self.assertIn("missing audience", str(ctx.exception).lower())

    def test_audience_list_contains_expected(self) -> None:
        """Audience claim formatted as a list is accepted if it contains the expected audience."""
        payload = {
            "sub": "user-1",
            "exp": int(time.time()) + 3600,
            "aud": ["authenticated", "api-client"],
        }
        token = create_jwt_token(payload=payload, secret=TEST_SECRET)
        decoded = decode_jwt_token(
            token=token,
            secret=TEST_SECRET,
            expected_audience="authenticated",
        )
        self.assertEqual(decoded["sub"], "user-1")

    def test_issuer_claim_validation(self) -> None:
        """Issuer validation accepts matching issuer and rejects mismatch."""
        expected_iss = "https://flurit-project.supabase.co/auth/v1"
        payload = {
            "sub": "user-1",
            "exp": int(time.time()) + 3600,
            "aud": "authenticated",
            "iss": expected_iss,
        }
        token = create_jwt_token(payload=payload, secret=TEST_SECRET)

        # Matching issuer succeeds
        decoded = decode_jwt_token(
            token=token,
            secret=TEST_SECRET,
            expected_issuer=expected_iss,
        )
        self.assertEqual(decoded["iss"], expected_iss)

        # Mismatched issuer fails
        with self.assertRaises(ValueError) as ctx:
            decode_jwt_token(
                token=token,
                secret=TEST_SECRET,
                expected_issuer="https://other-project.supabase.co/auth/v1",
            )
        self.assertIn("issuer", str(ctx.exception).lower())

    def test_not_before_claim_future_rejected(self) -> None:
        """Tokens with 'nbf' claim in the future must be rejected."""
        payload = {
            "sub": "user-1",
            "exp": int(time.time()) + 3600,
            "nbf": int(time.time()) + 300,  # Valid 5 minutes in future
            "aud": "authenticated",
        }
        token = create_jwt_token(payload=payload, secret=TEST_SECRET)

        with self.assertRaises(ValueError) as ctx:
            decode_jwt_token(token=token, secret=TEST_SECRET)
        self.assertIn("not yet valid", str(ctx.exception).lower())

    def test_empty_sub_claim_rejected(self) -> None:
        """Tokens with empty or missing 'sub' claim must be rejected."""
        payload = {
            "sub": "   ",
            "exp": int(time.time()) + 3600,
            "aud": "authenticated",
        }
        token = create_jwt_token(payload=payload, secret=TEST_SECRET)

        with self.assertRaises(ValueError) as ctx:
            decode_jwt_token(token=token, secret=TEST_SECRET)
        self.assertIn("subject claim", str(ctx.exception).lower())

    def test_verify_supabase_jwt_dependency_fail_closed(self) -> None:
        """verify_supabase_jwt raises HTTP 500 when secret is unconfigured for HS256."""
        settings = Settings(
            SUPABASE_JWT_ALGORITHM="HS256",
            SUPABASE_ALLOW_HS256=True,
            SUPABASE_JWT_SECRET=None,
        )
        token = create_jwt_token(
            payload={"sub": "user-1", "exp": int(time.time()) + 3600},
            secret="some-secret",
        )
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        with self.assertRaises(HTTPException) as ctx:
            verify_supabase_jwt(credentials=creds, settings=settings)
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("misconfigured", ctx.exception.detail.lower())

    def test_verify_supabase_jwt_dependency_success(self) -> None:
        """verify_supabase_jwt returns AuthenticatedUser on valid credentials."""
        settings = Settings(
            SUPABASE_JWT_ALGORITHM="HS256",
            SUPABASE_ALLOW_HS256=True,
            SUPABASE_JWT_SECRET=TEST_SECRET,
        )
        token = create_jwt_token(
            payload={
                "sub": "user-999",
                "email": "test@flurit.com",
                "role": "authenticated",
                "exp": int(time.time()) + 3600,
            },
            secret=TEST_SECRET,
        )
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        user = verify_supabase_jwt(credentials=creds, settings=settings)
        self.assertIsInstance(user, AuthenticatedUser)
        self.assertEqual(user.user_id, "user-999")
        self.assertEqual(user.email, "test@flurit.com")


if __name__ == "__main__":
    unittest.main()

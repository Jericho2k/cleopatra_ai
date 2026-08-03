import hashlib
import hmac

from core.webhooks import valid_hmac_sha256_signature


def test_valid_hmac_signature_accepts_bare_hex_digest():
    body = b'{"event":"messages.received"}'
    secret = "webhook-signing-secret"
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    assert valid_hmac_sha256_signature(body, signature, secret) is True


def test_valid_hmac_signature_accepts_sha256_prefix():
    body = b'{"event":"messages.received"}'
    secret = "webhook-signing-secret"
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    assert valid_hmac_sha256_signature(body, f"sha256={signature}", secret) is True


def test_valid_hmac_signature_rejects_tampering_and_missing_values():
    body = b'{"event":"messages.received"}'
    secret = "webhook-signing-secret"
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    assert valid_hmac_sha256_signature(body + b" ", signature, secret) is False
    assert valid_hmac_sha256_signature(body, None, secret) is False
    assert valid_hmac_sha256_signature(body, signature, "") is False

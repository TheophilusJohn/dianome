"""Session tokens (Phase 5b).

The Worker mints `token = base64url(payload) + "." + base64url(HMAC-SHA256(SPLIT_SIGNING_KEY, payload))`
with `payload = {sid, model, exp, max_ctx, origin}` (JSON, exp = unix seconds). The server verifies the
signature and the expiry, binds the session to `sid`, and still accepts the static `SPLIT_TOKEN` bearer
for local runs and tests. `mint()` exists for the tests and for hand runs; production tokens come from
the Worker (`packages/worker/src/split.ts`), which uses the same layout.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import hmac
import json
import time
from typing import Any, Callable, Optional

REQUIRED = ("sid", "model", "exp", "max_ctx", "origin")


class AuthError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclasses.dataclass(frozen=True)
class SessionToken:
    sid: str
    model: str
    exp: int
    max_ctx: int
    origin: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _unb64u(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _sign(key: bytes, payload: bytes) -> bytes:
    return hmac.new(key, payload, hashlib.sha256).digest()


def mint(key: str | bytes, *, sid: str, model: str, exp: int, max_ctx: int, origin: str) -> str:
    """A token the way the Worker builds it (canonical JSON: sorted keys, no whitespace)."""
    k = key.encode() if isinstance(key, str) else key
    payload = json.dumps({"sid": sid, "model": model, "exp": int(exp), "max_ctx": int(max_ctx), "origin": origin},
                         separators=(",", ":"), sort_keys=True).encode()
    return f"{_b64u(payload)}.{_b64u(_sign(k, payload))}"


def verify(token: str, key: str | bytes, now: Optional[Callable[[], float]] = None) -> SessionToken:
    """Signature and expiry; raises AuthError(code) with code in {malformed, bad_signature, expired, bad_payload}."""
    k = key.encode() if isinstance(key, str) else key
    parts = token.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise AuthError("malformed", "token must be base64url(payload).base64url(signature)")
    try:
        payload, sig = _unb64u(parts[0]), _unb64u(parts[1])
    except (ValueError, TypeError) as e:
        raise AuthError("malformed", f"token is not base64url: {e}") from e
    if not hmac.compare_digest(sig, _sign(k, payload)):
        raise AuthError("bad_signature", "signature does not verify")
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as e:
        raise AuthError("bad_payload", f"payload is not JSON: {e}") from e
    if not isinstance(obj, dict) or any(f not in obj for f in REQUIRED):
        raise AuthError("bad_payload", f"payload must have {', '.join(REQUIRED)}")
    try:
        tok = SessionToken(sid=str(obj["sid"]), model=str(obj["model"]), exp=int(obj["exp"]),
                           max_ctx=int(obj["max_ctx"]), origin=str(obj["origin"]))
    except (TypeError, ValueError) as e:
        raise AuthError("bad_payload", f"bad field types: {e}") from e
    if not tok.sid:
        raise AuthError("bad_payload", "empty sid")
    t = (now or time.time)()
    if tok.exp <= t:
        raise AuthError("expired", f"token expired {t - tok.exp:.0f}s ago")
    return tok

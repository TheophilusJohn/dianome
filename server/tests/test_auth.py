"""Phase 5b: HMAC session tokens (valid, expired, bad signature, wrong origin), /plan fields, bearer still works."""

import asyncio
import http.client
import json
import time

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from dianome_server.auth import AuthError, mint, verify
from dianome_server.microbench import Microbench, run_microbench
from dianome_server.protocol import decode, encode
from dianome_server.ws import SplitServer
from tests.ws_client import SplitClient

TOKEN = "test-token"
KEY = "test-signing-key"
ORIGIN = "http://localhost:5177"
MB = Microbench(ms_per_block_decode=0.5, ms_per_block_prefill=0.2, lm_head_ms=1.5, prefill_T=32, runs=5, device="test")


# -- pure verify --------------------------------------------------------------------


def test_verify_valid_and_fields():
    t = mint(KEY, sid="abc", model="m", exp=int(time.time()) + 60, max_ctx=512, origin=ORIGIN)
    st = verify(t, KEY)
    assert (st.sid, st.model, st.max_ctx, st.origin) == ("abc", "m", 512, ORIGIN)


def test_verify_expired():
    t = mint(KEY, sid="abc", model="m", exp=int(time.time()) - 1, max_ctx=512, origin=ORIGIN)
    with pytest.raises(AuthError) as e:
        verify(t, KEY)
    assert e.value.code == "expired"
    # the clock is injectable: the same token is valid a minute earlier
    assert verify(t, KEY, now=lambda: time.time() - 60).sid == "abc"


def test_verify_bad_signature_and_malformed():
    t = mint(KEY, sid="abc", model="m", exp=int(time.time()) + 60, max_ctx=512, origin=ORIGIN)
    with pytest.raises(AuthError) as e:
        verify(t, "other-key")
    assert e.value.code == "bad_signature"
    payload, sig = t.split(".")
    tampered = mint(KEY, sid="abc", model="m", exp=int(time.time()) + 3600 * 24, max_ctx=512, origin=ORIGIN).split(".")[0]
    with pytest.raises(AuthError) as e:
        verify(f"{tampered}.{sig}", KEY)
    assert e.value.code == "bad_signature"
    for bad in ("", "abc", "a.b.c", "!!!.???"):
        with pytest.raises(AuthError) as e:
            verify(bad, KEY)
        assert e.value.code in ("malformed", "bad_signature")


# -- server ------------------------------------------------------------------------


@pytest.fixture
async def server(lm):
    s = SplitServer(lm, token=TOKEN, signing_key=KEY, reap_interval=0.1, microbench=MB)
    await s.start("127.0.0.1", 0)
    yield s
    await s.stop()


def url(server: SplitServer, token: str | None = None) -> str:
    return f"ws://127.0.0.1:{server.port}/" + (f"?token={token}" if token else "")


def _http_get(port: int, path: str, headers: dict | None = None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("GET", path, headers=headers or {})
    r = c.getresponse()
    return r.status, r.read(), dict(r.getheaders())


async def http_get(server, path, headers=None):
    return await asyncio.to_thread(_http_get, server.port, path, headers)


def good_token(lm, **over):
    kw = dict(sid="sid-1", model=lm.id, exp=int(time.time()) + 3600, max_ctx=256, origin=ORIGIN)
    kw.update(over)
    return mint(KEY, **kw)


async def test_session_token_opens_and_binds_sid(server, lm):
    t = good_token(lm)
    async with connect(url(server, t), additional_headers={"Origin": ORIGIN}, max_size=None) as ws:
        await ws.send(encode("ping", {"t": 1}))
        assert decode(await ws.recv()).header == {"type": "pong", "t": 1}
        await ws.send(encode("open", {"model": lm.id, "N": 24, "max_ctx": 4096, "sampling": {"temperature": 0}}))
        opened = decode(await ws.recv())
        assert opened.type == "opened" and opened["session"] == "sid-1"
        assert server.manager.sessions["sid-1"].max_ctx == 256  # capped by the token
        # the same sid cannot be open twice
        async with connect(url(server, t), additional_headers={"Origin": ORIGIN}, max_size=None) as ws2:
            await ws2.send(encode("open", {"model": lm.id, "N": 24, "max_ctx": 64}))
            err = decode(await ws2.recv())
            assert err.type == "error" and err["code"] == "sid_in_use"
    await asyncio.sleep(0.05)
    assert "sid-1" not in server.manager.sessions


async def test_bearer_still_works(server, lm):
    async with SplitClient(url(server), TOKEN) as c:  # Authorization header
        assert (await c.open(lm.id, 8)).type == "opened"
    async with connect(url(server, TOKEN), max_size=None) as ws:  # ?token=
        await ws.send(encode("open", {"model": lm.id, "N": 8, "max_ctx": 64}))
        assert decode(await ws.recv()).type == "opened"


@pytest.mark.parametrize("case", ["expired", "bad_signature", "wrong_origin", "wrong_model", "missing", "garbage"])
async def test_refused_tokens(server, lm, case):
    origin = ORIGIN
    if case == "expired":
        t = good_token(lm, exp=int(time.time()) - 5)
    elif case == "bad_signature":
        t = mint("other-key", sid="s", model=lm.id, exp=int(time.time()) + 60, max_ctx=64, origin=ORIGIN)
    elif case == "wrong_origin":
        t = good_token(lm)
        origin = "https://evil.example"
    elif case == "wrong_model":
        t = good_token(lm, model="not-this-model")
    elif case == "missing":
        t = None
    else:
        t = "not.a.token"
    with pytest.raises(InvalidStatus) as e:
        async with connect(url(server, t), additional_headers={"Origin": origin}):
            pass
    assert e.value.response.status_code == 401


async def test_expired_token_cannot_reopen(lm):
    now = [1000.0]
    s = SplitServer(lm, token=TOKEN, signing_key=KEY, reap_interval=0.1, clock=lambda: now[0])
    await s.start("127.0.0.1", 0)
    try:
        t = mint(KEY, sid="s2", model=lm.id, exp=1030, max_ctx=64, origin=ORIGIN)
        async with connect(url(s, t), additional_headers={"Origin": ORIGIN}, max_size=None) as ws:
            now[0] = 1031.0  # expires between upgrade and open
            await ws.send(encode("open", {"model": lm.id, "N": 24, "max_ctx": 64}))
            err = decode(await ws.recv())
            assert err.type == "error" and err["code"] == "token_expired"
        with pytest.raises(InvalidStatus):
            async with connect(url(s, t), additional_headers={"Origin": ORIGIN}):
                pass
    finally:
        await s.stop()


async def test_plan_is_public_and_has_microbench_fields(server, lm):
    status, body, headers = await http_get(server, "/plan")
    assert status == 200
    plan = json.loads(body)
    assert {"model", "L", "d_model", "active_sessions", "busy_fraction_60s",
            "ms_per_block_decode", "ms_per_block_prefill", "lm_head_ms", "microbench", "device"} <= set(plan)
    assert plan["ms_per_block_decode"] == 0.5 and plan["ms_per_block_prefill"] == 0.2 and plan["lm_head_ms"] == 1.5
    assert plan["microbench"]["runs"] == 5
    assert headers.get("Access-Control-Allow-Origin") == "*"
    # still works with the bearer header, and the old shape is a subset
    assert (await http_get(server, "/plan", {"Authorization": f"Bearer {TOKEN}"}))[0] == 200


async def test_plan_rate_limited(lm):
    s = SplitServer(lm, token=TOKEN, plan_rate_limit=3)
    await s.start("127.0.0.1", 0)
    try:
        codes = [(await http_get(s, "/plan"))[0] for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429]
    finally:
        await s.stop()


def test_run_microbench_shape(lm):
    mb = run_microbench(lm, runs=2, warmup=1)
    assert mb.ms_per_block_decode > 0 and mb.ms_per_block_prefill > 0 and mb.lm_head_ms > 0
    assert mb.runs == 2 and mb.prefill_T == 32

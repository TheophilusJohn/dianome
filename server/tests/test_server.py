"""Session limits, idle timeout, busy-time accounting, /plan, and the real-socket decode test."""

import asyncio
import http.client
import json
import time

import pytest
import torch
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from dianome_server.protocol import ProtocolError, decode, encode
from dianome_server.session import SessionManager
from dianome_server.ws import SplitServer
from tests.ws_client import SplitClient, generate

TOKEN = "test-token"


@pytest.fixture
async def server(lm):
    s = SplitServer(lm, token=TOKEN, reap_interval=0.1)
    await s.start("127.0.0.1", 0)
    yield s
    await s.stop()


def url(server: SplitServer) -> str:
    return f"ws://127.0.0.1:{server.port}"


def _http_get(port: int, path: str, token: str | None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    c.request("GET", path, headers=headers)
    r = c.getresponse()
    return r.status, r.read()


async def http_get(server: SplitServer, path: str, token: str | None):
    """Plain HTTP GET off the event loop thread (the server runs on this loop)."""
    return await asyncio.to_thread(_http_get, server.port, path, token)


# -- auth and /plan -----------------------------------------------------------------


async def test_refuses_without_token(server):
    with pytest.raises(InvalidStatus) as e:
        async with connect(url(server)):
            pass
    assert e.value.response.status_code == 401
    assert (await http_get(server, "/plan", None))[0] == 401
    assert (await http_get(server, "/plan", "wrong"))[0] == 401


def test_refuses_to_start_without_split_token(lm, monkeypatch):
    monkeypatch.delenv("SPLIT_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        SplitServer(lm)


async def test_plan_shape(server, lm):
    status, body = await http_get(server, "/plan", TOKEN)
    assert status == 200
    plan = json.loads(body)
    assert set(plan) == {"model", "L", "d_model", "active_sessions", "busy_fraction_60s"}
    assert plan["model"] == lm.id and plan["L"] == lm.L and plan["d_model"] == lm.d_model
    assert plan["active_sessions"] == 0 and 0.0 <= plan["busy_fraction_60s"] <= 1.0
    async with SplitClient(url(server), TOKEN) as c:
        await c.open(lm.id, 8)
        assert json.loads((await http_get(server, "/plan", TOKEN))[1])["active_sessions"] == 1
    await asyncio.sleep(0.05)
    assert json.loads((await http_get(server, "/plan", TOKEN))[1])["active_sessions"] == 0


# -- session limits ------------------------------------------------------------------


async def test_session_limit_four(server, lm):
    clients = [await SplitClient(url(server), TOKEN).__aenter__() for _ in range(4)]
    for c in clients:
        assert (await c.open(lm.id, 8)).type == "opened"
    fifth = await SplitClient(url(server), TOKEN).__aenter__()
    err = await fifth.open(lm.id, 8)
    assert err.type == "error" and err["code"] == "too_many_sessions"
    await clients[0].close()
    await asyncio.sleep(0.05)
    sixth = await SplitClient(url(server), TOKEN).__aenter__()
    assert (await sixth.open(lm.id, 8)).type == "opened"
    for c in clients[1:] + [sixth]:
        await c.__aexit__(None, None, None)


async def test_open_validation(server, lm):
    async with SplitClient(url(server), TOKEN) as c:
        assert (await c.send("prefill", {"T": 1, "positions": [0, 1]}, b"\0" * 4))["code"] == "not_open"
        assert (await c.open(lm.id, lm.L + 1))["code"] == "bad_N"
    async with SplitClient(url(server), TOKEN) as c:
        assert (await c.open(lm.id, 0, max_ctx=4097))["code"] == "bad_max_ctx"
    async with SplitClient(url(server), TOKEN) as c:
        assert (await c.open("other-model", 0))["code"] == "wrong_model"
    async with SplitClient(url(server), TOKEN) as c:
        assert (await c.open(lm.id, 8)).type == "opened"
        assert (await c.open(lm.id, 8))["code"] == "already_open"
        assert (await c.send("prefill", {"T": 2, "positions": [0, 3]}, b""))["code"] == "bad_positions"
        assert (await c.send("prefill", {"T": 2, "positions": [0, 2]}, b"\0" * 8))["code"] == "bad_payload"


def test_manager_limits_and_idle_reap(lm):
    now = [1000.0]
    m = SessionManager(lm, max_sessions=2, idle_timeout=120.0, clock=lambda: now[0])
    a = m.open(lm.id, 8, 64, None)
    b = m.open(lm.id, 8, 64, None)
    with pytest.raises(ProtocolError) as e:
        m.open(lm.id, 8, 64, None)
    assert e.value.code == "too_many_sessions"
    now[0] += 100
    b.touch()
    assert m.reap_idle() == []
    now[0] += 21  # a idle 121 s, b idle 21 s
    assert m.reap_idle() == [a]
    assert set(m.sessions) == {b.id}
    m.close(b)
    assert m.plan()["active_sessions"] == 0


async def test_idle_timeout_closes_socket(lm):
    s = SplitServer(lm, token=TOKEN, idle_timeout=0.3, reap_interval=0.05)
    await s.start("127.0.0.1", 0)
    try:
        async with SplitClient(url(s), TOKEN) as c:
            assert (await c.open(lm.id, 8)).type == "opened"
            msg = decode(await asyncio.wait_for(c.ws.recv(), timeout=3))
            assert msg.type == "error" and msg["code"] == "idle_timeout"
            with pytest.raises(ConnectionClosed):
                await asyncio.wait_for(c.ws.recv(), timeout=3)
        assert s.manager.plan()["active_sessions"] == 0
    finally:
        await s.stop()


# -- busy-time accounting -----------------------------------------------------------


async def test_busy_time_excludes_socket_sleep(lm):
    sleep = 0.25
    s = SplitServer(lm, token=TOKEN, debug_socket_sleep=sleep)
    await s.start("127.0.0.1", 0)
    try:
        ids = lm.tokenizer("Busy time is measured around the model only.", return_tensors="pt").input_ids[0]
        async with SplitClient(url(s), TOKEN) as c:
            await c.open(lm.id, 0)
            t0 = time.perf_counter()
            tok = await c.prefill(ids.to(torch.int32))
            pos = ids.shape[0]
            for _ in range(3):
                tok = await c.decode(torch.tensor([tok["id"]], dtype=torch.int32), pos)
                pos += 1
            wall = time.perf_counter() - t0
            st = await c.stats()
        assert st["tokens"] == 4
        assert wall >= 4 * sleep
        assert st["busy_seconds"] < wall - 4 * sleep, (st["busy_seconds"], wall)
        assert abs(st["gpu_seconds_per_token"] - st["busy_seconds"] / 4) < 1e-9
    finally:
        await s.stop()


# -- the definition-of-done client run ---------------------------------------------


async def test_ws_client_n8_matches_full_greedy(server, lm, prompt_ids):
    want = lm.full_greedy(prompt_ids, 16)
    got = await generate(url(server), TOKEN, lm, 8, prompt_ids, 16)
    assert got == want


async def test_ws_client_n0_and_topk(server, lm, prompt_ids):
    want = lm.full_greedy(prompt_ids, 4)
    async with SplitClient(url(server), TOKEN) as c:
        await c.open(lm.id, 0, logits_topk=5)
        tok = await c.prefill(prompt_ids.to(torch.int32))
        assert tok["topk_ids"][0] == tok["id"] == want[0]
        assert len(tok.payload) == 5 * 2
        got = [tok["id"]]
        pos = prompt_ids.shape[0]
        for _ in range(3):
            tok = await c.decode(torch.tensor([got[-1]], dtype=torch.int32), pos)
            got.append(tok["id"])
            pos += 1
        assert got == want
        assert tok["position"] == pos

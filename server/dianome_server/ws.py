"""websockets server: auth on the upgrade, one session per connection, GET /plan.

Auth (Phase 5b): a WebSocket upgrade carries either the static bearer `SPLIT_TOKEN`
(`Authorization: Bearer …` or `?token=`; local runs and tests) or a short-lived HMAC
session token minted by the Worker (`?token=`, verified with `SPLIT_SIGNING_KEY`, see
auth.py). A session opened with an HMAC token is bound to the token's `sid`, `model`
and `max_ctx`; the same `sid` cannot be open twice. `GET /plan` is public load
information (rate-limited per client address) and carries the startup microbench.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import http
import logging
import os
import time
import urllib.parse
from typing import Optional

import orjson
from websockets.asyncio.server import Server as WsServer, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from .auth import AuthError, SessionToken, verify
from .microbench import Microbench
from .model import LoadedModel
from .protocol import ProtocolError, decode, encode
from .session import IDLE_TIMEOUT_S, MAX_SESSIONS, Session, SessionManager

log = logging.getLogger("dianome_server")

MAX_FRAME = 64 * 1024 * 1024  # 4096 x 2048 (3B) fp16 is 16.8 MB
PLAN_RATE_LIMIT_PER_MINUTE = 120


class SplitServer:
    def __init__(
        self,
        lm: LoadedModel,
        token: Optional[str] = None,
        max_sessions: int = MAX_SESSIONS,
        idle_timeout: float = IDLE_TIMEOUT_S,
        reap_interval: float = 5.0,
        debug_socket_sleep: float = 0.0,
        signing_key: Optional[str] = None,
        microbench: Optional[Microbench] = None,
        clock=time.time,
        plan_rate_limit: int = PLAN_RATE_LIMIT_PER_MINUTE,
    ):
        tok = token if token is not None else os.environ.get("SPLIT_TOKEN")
        key = signing_key if signing_key is not None else os.environ.get("SPLIT_SIGNING_KEY")
        # Phase 6: a self-hosted image may run with only SPLIT_SIGNING_KEY (sessions come from the Worker); at least
        # one of the two credentials is required, or every upgrade would be refused and the server is pointless.
        if not tok and not key:
            raise RuntimeError("neither SPLIT_TOKEN nor SPLIT_SIGNING_KEY is set; refusing to start")
        self.token: Optional[str] = tok or None
        self.signing_key: Optional[bytes] = key.encode() if key else None
        self.lm = lm
        self.manager = SessionManager(lm, max_sessions=max_sessions, idle_timeout=idle_timeout)
        self.microbench = microbench
        self.clock = clock
        self.reap_interval = reap_interval
        self.debug_socket_sleep = debug_socket_sleep  # tests: a deliberate sleep on the socket path
        self.connections: dict[str, ServerConnection] = {}
        self.server: Optional[WsServer] = None
        self._reaper: Optional[asyncio.Task] = None
        self.plan_rate_limit = plan_rate_limit
        self._plan_hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        # auth outcome per connection, set in process_request and read in handler
        self._auth: dict[int, Optional[SessionToken]] = {}

    # -- HTTP layer ------------------------------------------------------------------

    def _query_token(self, request: Request) -> Optional[str]:
        query = urllib.parse.urlsplit(request.path).query
        toks = urllib.parse.parse_qs(query).get("token", [])
        return toks[0] if toks else None

    def _authenticate(self, request: Request) -> tuple[bool, Optional[SessionToken], str]:
        """(ok, session token or None for the static bearer, reason)."""
        auth = request.headers.get("Authorization", "")
        if self.token is not None and auth == f"Bearer {self.token}":
            return True, None, "bearer"
        # Browsers cannot set headers on a WebSocket upgrade: the token travels as `?token=`.
        qt = self._query_token(request)
        if qt is None:
            return False, None, "missing token"
        if self.token is not None and qt == self.token:
            return True, None, "bearer"
        if self.signing_key is None:
            return False, None, "bad bearer token (no SPLIT_SIGNING_KEY configured)"
        try:
            st = verify(qt, self.signing_key, now=self.clock)
        except AuthError as e:
            return False, None, f"{e.code}: {e.message}"
        origin = request.headers.get("Origin")
        if origin is not None and origin.rstrip("/") != st.origin.rstrip("/"):
            return False, None, f"wrong_origin: token is for {st.origin}"
        if st.model != self.lm.id:
            return False, None, f"wrong_model: token is for {st.model}"
        return True, st, "session token"

    def _plan_rate_limited(self, conn: ServerConnection) -> bool:
        try:
            addr = conn.remote_address[0] if conn.remote_address else "?"
        except Exception:
            addr = "?"
        now = self.clock()
        q = self._plan_hits[addr]
        while q and q[0] < now - 60.0:
            q.popleft()
        if len(q) >= self.plan_rate_limit:
            return True
        q.append(now)
        return False

    def _json_response(self, conn: ServerConnection, status: http.HTTPStatus, obj) -> Response:
        body = orjson.dumps(obj)
        resp = conn.respond(status, "")
        # websockets' Headers is a multi-dict: delete before setting or the
        # original Content-Length: 0 stays first and the client reads no body.
        for k in ("Content-Type", "Content-Length"):
            del resp.headers[k]
        resp.headers["Content-Type"] = "application/json"
        resp.headers["Content-Length"] = str(len(body))
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.body = body
        return resp

    def plan(self) -> dict:
        p = self.manager.plan()
        if self.microbench is not None:
            mb = self.microbench
            p.update({"ms_per_block_decode": mb.ms_per_block_decode, "ms_per_block_prefill": mb.ms_per_block_prefill,
                      "lm_head_ms": mb.lm_head_ms, "microbench": mb.as_dict()})
        else:
            p.update({"ms_per_block_decode": None, "ms_per_block_prefill": None, "lm_head_ms": None, "microbench": None})
        p["device"] = str(self.lm.device)
        return p

    def process_request(self, conn: ServerConnection, request: Request) -> Optional[Response]:
        path = urllib.parse.urlsplit(request.path).path
        if path == "/plan":
            # Public load information (no auth), rate-limited per client address.
            if self._plan_rate_limited(conn):
                return self._json_response(conn, http.HTTPStatus.TOO_MANY_REQUESTS, {"error": "rate_limited"})
            return self._json_response(conn, http.HTTPStatus.OK, self.plan())
        ok, st, reason = self._authenticate(request)
        if not ok:
            log.info("refused upgrade from %s: %s", conn.remote_address, reason)
            return conn.respond(http.HTTPStatus.UNAUTHORIZED, f"unauthorized: {reason}\n")
        self._auth[id(conn)] = st
        return None  # continue with the WebSocket handshake

    # -- WebSocket layer -------------------------------------------------------------

    async def _send(self, conn: ServerConnection, type: str, header: dict, payload: bytes = b"") -> None:
        await conn.send(encode(type, header, payload))

    async def handler(self, conn: ServerConnection) -> None:
        session: Optional[Session] = None
        st = self._auth.pop(id(conn), None)
        try:
            async for raw in conn:
                if isinstance(raw, str):
                    await self._send(conn, "error", {"code": "bad_frame", "message": "binary frames only"})
                    continue
                try:
                    msg = decode(raw)
                    if self.debug_socket_sleep:
                        await asyncio.sleep(self.debug_socket_sleep)
                    if msg.type == "ping":
                        await self._send(conn, "pong", {k: v for k, v in msg.header.items() if k != "type"})
                        continue
                    if msg.type == "open":
                        if session is not None:
                            raise ProtocolError("already_open", "session already open on this connection")
                        max_ctx = int(msg["max_ctx"])
                        if st is not None:
                            if st.exp <= self.clock():
                                raise ProtocolError("token_expired", "session token expired before open")
                            if msg["model"] != st.model:
                                raise ProtocolError("wrong_model", f"token is for {st.model!r}")
                            max_ctx = min(max_ctx, st.max_ctx)
                        session = self.manager.open(
                            msg["model"], msg["N"], max_ctx, msg.get("sampling"), msg.get("logits_topk", 0),
                            sid=st.sid if st is not None else None,
                        )
                        self.connections[session.id] = conn
                        await self._send(conn, "opened", {
                            "session": session.id, "L": self.lm.L, "d_model": self.lm.d_model,
                            "boundary": f"input_to_block_{session.N}",
                        })
                        continue
                    if session is None:
                        raise ProtocolError("not_open", "send 'open' first")
                    if msg.type == "prefill":
                        r = await asyncio.to_thread(session.prefill, msg.payload, int(msg["T"]), msg["positions"])
                        await self._send_token(conn, r)
                    elif msg.type == "decode":
                        r = await asyncio.to_thread(session.decode, msg.payload, int(msg["position"]))
                        await self._send_token(conn, r)
                    elif msg.type == "stats":
                        session.touch()
                        await self._send(conn, "stats", session.stats())
                    elif msg.type == "close":
                        break
                    else:
                        raise ProtocolError("bad_type", f"{msg.type} is server→client only")
                except ProtocolError as e:
                    await self._send(conn, "error", {"code": e.code, "message": e.message})
                    if e.code in ("too_many_sessions", "wrong_model", "bad_N", "bad_max_ctx", "sid_in_use", "token_expired"):
                        break
        except ConnectionClosed:
            pass
        finally:
            self._auth.pop(id(conn), None)
            if session is not None:
                self.connections.pop(session.id, None)
                self.manager.close(session)
            with contextlib.suppress(Exception):
                await conn.close()

    async def _send_token(self, conn: ServerConnection, r) -> None:
        header = {"id": r.id, "position": r.position, "busy_ms": r.busy_ms, "done": r.done}
        payload = b""
        if r.topk_ids is not None:
            header["topk_ids"] = r.topk_ids
            payload = r.topk_logits.astype("<f2").tobytes()
        await self._send(conn, "token", header, payload)

    async def _reap_loop(self) -> None:
        while True:
            await asyncio.sleep(self.reap_interval)
            for s in self.manager.reap_idle():
                conn = self.connections.pop(s.id, None)
                if conn is not None:
                    with contextlib.suppress(Exception):
                        await self._send(conn, "error", {"code": "idle_timeout", "message": "session idle > timeout"})
                        await conn.close(code=1000, reason="idle timeout")
                log.info("reaped idle session %s", s.id)

    # -- lifecycle -------------------------------------------------------------------

    async def start(self, host: str = "127.0.0.1", port: int = 8765) -> WsServer:
        self.server = await serve(
            self.handler, host, port, process_request=self.process_request,
            max_size=MAX_FRAME, compression=None, ping_interval=20,
        )
        self._reaper = asyncio.create_task(self._reap_loop())
        return self.server

    @property
    def port(self) -> int:
        assert self.server is not None
        return next(iter(self.server.sockets)).getsockname()[1]

    async def stop(self) -> None:
        if self._reaper:
            self._reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def serve_forever(self, host: str, port: int) -> None:
        await self.start(host, port)
        log.info("dianome-server %s on ws://%s:%d  (L=%d d_model=%d device=%s, session tokens %s)",
                 self.lm.id, host, self.port, self.lm.L, self.lm.d_model, self.lm.device,
                 "enabled" if self.signing_key else "disabled (no SPLIT_SIGNING_KEY)")
        try:
            await asyncio.Future()
        finally:
            await self.stop()

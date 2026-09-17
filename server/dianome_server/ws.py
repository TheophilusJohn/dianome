"""websockets server: bearer auth on the upgrade, one session per connection, GET /plan.

`SPLIT_TOKEN` must be set; every request (the WebSocket upgrade and the plain
HTTP `GET /plan`) must carry `Authorization: Bearer <SPLIT_TOKEN>`.
"""

from __future__ import annotations

import asyncio
import contextlib
import http
import logging
import os
from typing import Optional

import orjson
from websockets.asyncio.server import Server as WsServer, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from .model import LoadedModel
from .protocol import ProtocolError, decode, encode
from .session import IDLE_TIMEOUT_S, MAX_SESSIONS, Session, SessionManager

log = logging.getLogger("dianome_server")

MAX_FRAME = 64 * 1024 * 1024  # 4096 x 2048 (3B) fp16 is 16.8 MB


class SplitServer:
    def __init__(
        self,
        lm: LoadedModel,
        token: Optional[str] = None,
        max_sessions: int = MAX_SESSIONS,
        idle_timeout: float = IDLE_TIMEOUT_S,
        reap_interval: float = 5.0,
        debug_socket_sleep: float = 0.0,
    ):
        tok = token if token is not None else os.environ.get("SPLIT_TOKEN")
        if not tok:
            raise RuntimeError("SPLIT_TOKEN is not set; refusing to start")
        self.token = tok
        self.lm = lm
        self.manager = SessionManager(lm, max_sessions=max_sessions, idle_timeout=idle_timeout)
        self.reap_interval = reap_interval
        self.debug_socket_sleep = debug_socket_sleep  # tests: a deliberate sleep on the socket path
        self.connections: dict[str, ServerConnection] = {}
        self.server: Optional[WsServer] = None
        self._reaper: Optional[asyncio.Task] = None

    # -- HTTP layer ------------------------------------------------------------------

    def _authorized(self, request: Request) -> bool:
        auth = request.headers.get("Authorization", "")
        return auth == f"Bearer {self.token}"

    def process_request(self, conn: ServerConnection, request: Request) -> Optional[Response]:
        if not self._authorized(request):
            return conn.respond(http.HTTPStatus.UNAUTHORIZED, "missing or bad bearer token\n")
        if request.path == "/plan":
            body = orjson.dumps(self.manager.plan())
            resp = conn.respond(http.HTTPStatus.OK, "")
            # websockets' Headers is a multi-dict: delete before setting or the
            # original Content-Length: 0 stays first and the client reads no body.
            for k in ("Content-Type", "Content-Length"):
                del resp.headers[k]
            resp.headers["Content-Type"] = "application/json"
            resp.headers["Content-Length"] = str(len(body))
            resp.body = body
            return resp
        return None  # continue with the WebSocket handshake

    # -- WebSocket layer -------------------------------------------------------------

    async def _send(self, conn: ServerConnection, type: str, header: dict, payload: bytes = b"") -> None:
        await conn.send(encode(type, header, payload))

    async def handler(self, conn: ServerConnection) -> None:
        session: Optional[Session] = None
        try:
            async for raw in conn:
                if isinstance(raw, str):
                    await self._send(conn, "error", {"code": "bad_frame", "message": "binary frames only"})
                    continue
                try:
                    msg = decode(raw)
                    if self.debug_socket_sleep:
                        await asyncio.sleep(self.debug_socket_sleep)
                    if msg.type == "open":
                        if session is not None:
                            raise ProtocolError("already_open", "session already open on this connection")
                        session = self.manager.open(
                            msg["model"], msg["N"], msg["max_ctx"], msg.get("sampling"), msg.get("logits_topk", 0)
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
                    if e.code in ("too_many_sessions", "wrong_model", "bad_N", "bad_max_ctx"):
                        break
        except ConnectionClosed:
            pass
        finally:
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
        log.info("dianome-server %s on ws://%s:%d  (L=%d d_model=%d device=%s)",
                 self.lm.id, host, self.port, self.lm.L, self.lm.d_model, self.lm.device)
        try:
            await asyncio.Future()
        finally:
            await self.stop()

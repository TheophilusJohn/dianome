"""Session state: split point N, server-side KV cache, positions, busy-time accounting."""

from __future__ import annotations

import collections
import dataclasses
import secrets
import threading
import time
from typing import Callable, Optional

import numpy as np
import torch

from .model import LoadedModel, PartialDecoder, Sampler, Sampling, synchronize
from .protocol import ProtocolError, bytes_to_hidden, bytes_to_ids

MAX_CTX = 4096
MAX_SESSIONS = 4
IDLE_TIMEOUT_S = 120.0


@dataclasses.dataclass
class TokenResult:
    id: int
    position: int  # the position this token occupies when fed back
    busy_ms: float
    done: bool
    topk_ids: Optional[list[int]] = None
    topk_logits: Optional[np.ndarray] = None


class Session:
    def __init__(
        self,
        lm: LoadedModel,
        N: int,
        max_ctx: int,
        sampling: Sampling,
        logits_topk: int = 0,
        clock: Callable[[], float] = time.monotonic,
        exec_lock: Optional[threading.Lock] = None,
        sid: Optional[str] = None,
    ):
        if not (0 <= N <= lm.L):
            raise ProtocolError("bad_N", f"N must be in 0..{lm.L}, got {N}")
        if not (1 <= max_ctx <= MAX_CTX):
            raise ProtocolError("bad_max_ctx", f"max_ctx must be in 1..{MAX_CTX}, got {max_ctx}")
        self.id = sid if sid else secrets.token_hex(8)
        self.lm = lm
        self.N = N
        self.max_ctx = max_ctx
        self.dec = PartialDecoder(lm, N)
        self.sampler = Sampler(sampling)
        self.logits_topk = int(logits_topk or 0)
        self.clock = clock
        self.exec_lock = exec_lock or threading.Lock()
        self.created = clock()
        self.last_active = self.created
        self.busy_seconds = 0.0
        self.tokens = 0
        self.busy_log: collections.deque[tuple[float, float]] = collections.deque()  # (end_time, busy)
        self.closed = False

    @property
    def seen(self) -> int:
        return self.dec.seen

    def touch(self) -> None:
        self.last_active = self.clock()

    def idle_for(self) -> float:
        return self.clock() - self.last_active

    # -- payload decoding (outside the busy clock) ----------------------------------

    def _decode_payload(self, payload: bytes, T: int) -> torch.Tensor:
        if self.N == 0:
            ids = bytes_to_ids(payload, T)
            if ids.min(initial=0) < 0 or ids.max(initial=0) >= self.lm.vocab:
                raise ProtocolError("bad_payload", "token id out of range")
            return torch.from_numpy(ids.astype(np.int64))
        return torch.from_numpy(bytes_to_hidden(payload, T, self.lm.d_model).copy())

    # -- model execution (the only thing inside the busy clock) ----------------------

    def _execute(self, x: torch.Tensor, start: int) -> TokenResult:
        with self.exec_lock:
            t0 = time.perf_counter()
            logits = self.dec.prefill(x, start)[-1]
            tok = self.sampler(logits)
            topk_ids = topk_vals = None
            if self.logits_topk:
                v, i = torch.topk(logits.float(), self.logits_topk)
                topk_ids, topk_vals = i.cpu().tolist(), v.cpu().numpy().astype(np.float16)
            synchronize(self.lm.device)
            busy = time.perf_counter() - t0
        self.busy_seconds += busy
        self.tokens += 1
        self.busy_log.append((self.clock(), busy))
        self.touch()
        done = tok in self.eos_ids() or self.seen >= self.max_ctx
        return TokenResult(tok, self.seen, busy * 1e3, done, topk_ids, topk_vals)

    def eos_ids(self) -> set[int]:
        eos = self.lm.config.eos_token_id
        if eos is None:
            return set()
        return set(eos) if isinstance(eos, (list, tuple)) else {int(eos)}

    def prefill(self, payload: bytes, T: int, positions: list[int]) -> TokenResult:
        self.touch()
        if not (isinstance(positions, (list, tuple)) and len(positions) == 2):
            raise ProtocolError("bad_positions", "positions must be [start, end)")
        start, end = int(positions[0]), int(positions[1])
        if end - start != T or T <= 0:
            raise ProtocolError("bad_positions", f"positions [{start}, {end}) do not match T={T}")
        if start != self.seen:
            raise ProtocolError("bad_positions", f"expected start={self.seen}, got {start}")
        if end > self.max_ctx:
            raise ProtocolError("ctx_exceeded", f"end={end} exceeds max_ctx={self.max_ctx}")
        x = self._decode_payload(payload, T)
        return self._execute(x, start)

    def decode(self, payload: bytes, position: int) -> TokenResult:
        self.touch()
        position = int(position)
        if position != self.seen:
            raise ProtocolError("bad_positions", f"expected position={self.seen}, got {position}")
        if position + 1 > self.max_ctx:
            raise ProtocolError("ctx_exceeded", f"position={position} exceeds max_ctx={self.max_ctx}")
        x = self._decode_payload(payload, 1)
        return self._execute(x, position)

    def stats(self) -> dict:
        return {
            "tokens": self.tokens,
            "busy_seconds": self.busy_seconds,
            "gpu_seconds_per_token": (self.busy_seconds / self.tokens) if self.tokens else 0.0,
        }

    def close(self) -> None:
        self.closed = True
        self.dec.reset()


class SessionManager:
    """At most `max_sessions` live sessions; idle ones are reaped after `idle_timeout`."""

    def __init__(
        self,
        lm: LoadedModel,
        max_sessions: int = MAX_SESSIONS,
        idle_timeout: float = IDLE_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.lm = lm
        self.max_sessions = max_sessions
        self.idle_timeout = idle_timeout
        self.clock = clock
        self.sessions: dict[str, Session] = {}
        self.exec_lock = threading.Lock()
        self.busy_log: collections.deque[tuple[float, float]] = collections.deque()

    def open(self, model: str, N: int, max_ctx: int, sampling: Optional[dict], logits_topk: int = 0,
             sid: Optional[str] = None) -> Session:
        if model != self.lm.id:
            raise ProtocolError("wrong_model", f"server has {self.lm.id!r} loaded, not {model!r}")
        if len(self.sessions) >= self.max_sessions:
            raise ProtocolError("too_many_sessions", f"at most {self.max_sessions} concurrent sessions")
        if sid is not None and sid in self.sessions:
            raise ProtocolError("sid_in_use", "a session with this token's sid is already open")
        s = Session(
            self.lm, int(N), int(max_ctx), Sampling.from_dict(sampling), logits_topk,
            clock=self.clock, exec_lock=self.exec_lock, sid=sid,
        )
        s.busy_log = self.busy_log  # shared log for busy_fraction_60s
        self.sessions[s.id] = s
        return s

    def close(self, session: Session) -> None:
        session.close()
        self.sessions.pop(session.id, None)

    def reap_idle(self) -> list[Session]:
        dead = [s for s in self.sessions.values() if s.idle_for() > self.idle_timeout]
        for s in dead:
            self.close(s)
        return dead

    def busy_fraction_60s(self) -> float:
        now = self.clock()
        while self.busy_log and self.busy_log[0][0] < now - 60.0:
            self.busy_log.popleft()
        return min(1.0, sum(b for _, b in self.busy_log) / 60.0)

    def plan(self) -> dict:
        return {
            "model": self.lm.id,
            "L": self.lm.L,
            "d_model": self.lm.d_model,
            "active_sessions": len(self.sessions),
            "busy_fraction_60s": self.busy_fraction_60s(),
        }

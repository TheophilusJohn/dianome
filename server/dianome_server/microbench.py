"""Startup microbench for GET /plan: ms per block for decode (T = 1) and prefill (T = 32), and ms for
final norm + lm_head + argmax, each the median of 5 timed runs on the loaded model (busy-time clock,
device-synchronised, the same way the cost harness measures). The Phase 5b planner uses these to
estimate the server's share of a token."""

from __future__ import annotations

import dataclasses
import statistics
import time

import torch

from .model import LoadedModel, synchronize


@dataclasses.dataclass
class Microbench:
    ms_per_block_decode: float
    ms_per_block_prefill: float
    lm_head_ms: float
    prefill_T: int
    runs: int
    device: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@torch.no_grad()
def run_microbench(lm: LoadedModel, runs: int = 5, prefill_T: int = 32, warmup: int = 2) -> Microbench:
    L, d = lm.L, lm.d_model
    ids = torch.arange(1000, 1000 + prefill_T)
    h = lm.embed(ids)
    one = lm.embed(ids[:1])

    def timed(f) -> float:
        synchronize(lm.device)
        t0 = time.perf_counter()
        f()
        synchronize(lm.device)
        return (time.perf_counter() - t0) * 1e3

    def prefill():
        cache = lm.new_cache()
        cp = torch.arange(0, prefill_T, device=lm.device)
        lm.run_blocks(h, 0, L, cache, cp)

    # decode: one token at position prefill_T on top of a prefilled cache
    dec_cache = lm.new_cache()
    lm.run_blocks(h, 0, L, dec_cache, torch.arange(0, prefill_T, device=lm.device))
    dec_pos = [prefill_T]

    def decode():
        cp = torch.tensor([dec_pos[0]], device=lm.device)
        lm.run_blocks(one, 0, L, dec_cache, cp)
        dec_pos[0] += 1

    x = lm.run_blocks(h, 0, L, lm.new_cache(), torch.arange(0, prefill_T, device=lm.device))[-1:]

    def head():
        torch.argmax(lm.lm_head(lm.final_norm(x))[-1])

    for f in (prefill, decode, head):
        for _ in range(warmup):
            f()
    pre = statistics.median(timed(prefill) for _ in range(runs))
    dec = statistics.median(timed(decode) for _ in range(runs))
    hd = statistics.median(timed(head) for _ in range(runs))
    return Microbench(ms_per_block_decode=dec / L, ms_per_block_prefill=pre / L, lm_head_ms=hd,
                      prefill_T=prefill_T, runs=runs, device=str(lm.device))

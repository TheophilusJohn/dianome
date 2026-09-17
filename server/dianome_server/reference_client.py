"""The client half in PyTorch: embeddings + blocks 0..N-1.

Stands in for the browser in tests, the cost harness and the fixtures. For
N = 0 it forwards token ids unchanged (the server embeds). Keeps its own
DynamicCache so it can also produce the per-step hidden state during decode.
"""

from __future__ import annotations

import torch

from .model import LoadedModel, PartialDecoder


class ReferenceClient:
    def __init__(self, lm: LoadedModel, N: int):
        if not (0 <= N <= lm.L):
            raise ValueError(f"N must be in 0..{lm.L}, got {N}")
        self.lm = lm
        self.N = N
        self.dec = PartialDecoder(lm, 0, N)  # blocks 0..N-1, no head

    def reset(self) -> None:
        self.dec.reset()

    @property
    def seen(self) -> int:
        return self.dec.seen

    def cache_lengths(self) -> list[int]:
        return self.dec.cache_lengths()

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, start: int = 0) -> torch.Tensor:
        """ids [T] -> what goes on the wire: int32 ids [T] (N = 0) or fp16 [T, d_model] (N >= 1)."""
        ids = ids.to(torch.long)
        if self.N == 0:
            self.dec.seen += ids.shape[0]
            return ids.to(torch.int32)
        return self.dec.run(ids, start).to(torch.float16)

    @torch.no_grad()
    def decode(self, token_id: int, position: int) -> torch.Tensor:
        """One token -> int32 [1] (N = 0) or fp16 [1, d_model] (N >= 1)."""
        ids = torch.tensor([token_id], dtype=torch.long)
        if self.N == 0:
            self.dec.seen += 1
            return ids.to(torch.int32)
        return self.dec.run(ids, position).to(torch.float16)


def reference_client(lm: LoadedModel, ids: torch.Tensor, N: int) -> torch.Tensor:
    """One-shot: the boundary tensor for a whole prompt at split N (the brief's `reference_client(N)`)."""
    return ReferenceClient(lm, N).prefill(ids)

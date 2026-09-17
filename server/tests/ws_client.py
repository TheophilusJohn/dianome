"""A small Python client for the split protocol, used by the tests and by hand.

    python -m tests.ws_client --url ws://127.0.0.1:8765 --token $SPLIT_TOKEN --N 8 --prompt "..." --steps 16
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Optional

import numpy as np
import torch
from websockets.asyncio.client import connect

from dianome_server.model import LoadedModel
from dianome_server.protocol import Message, decode, encode, hidden_to_bytes, ids_to_bytes
from dianome_server.reference_client import ReferenceClient


class SplitClient:
    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token
        self.ws = None
        self.opened: Optional[Message] = None

    async def __aenter__(self):
        self.ws = await connect(self.url, additional_headers={"Authorization": f"Bearer {self.token}"}, max_size=None)
        return self

    async def __aexit__(self, *exc):
        await self.ws.close()

    async def send(self, type: str, header: dict, payload: bytes = b"") -> Message:
        await self.ws.send(encode(type, header, payload))
        return decode(await self.ws.recv())

    async def open(self, model: str, N: int, max_ctx: int = 4096, sampling: Optional[dict] = None, **extra) -> Message:
        self.opened = await self.send("open", {"model": model, "N": N, "max_ctx": max_ctx,
                                               "sampling": sampling or {"temperature": 0}, **extra})
        return self.opened

    async def prefill(self, boundary: torch.Tensor, start: int = 0) -> Message:
        T = boundary.shape[0]
        payload = ids_to_bytes(boundary) if boundary.dim() == 1 else hidden_to_bytes(boundary)
        return await self.send("prefill", {"T": T, "positions": [start, start + T]}, payload)

    async def decode(self, boundary: torch.Tensor, position: int) -> Message:
        payload = ids_to_bytes(boundary) if boundary.dim() == 1 else hidden_to_bytes(boundary)
        return await self.send("decode", {"position": position}, payload)

    async def stats(self) -> Message:
        return await self.send("stats", {})

    async def close(self) -> None:
        await self.ws.send(encode("close", {}))


async def generate(url: str, token: str, lm: LoadedModel, N: int, ids: torch.Tensor, steps: int) -> list[int]:
    """Prefill + `steps` decodes over the real socket; the client half runs in PyTorch."""
    client = ReferenceClient(lm, N)
    out: list[int] = []
    async with SplitClient(url, token) as c:
        opened = await c.open(lm.id, N)
        assert opened.type == "opened", opened
        tok = await c.prefill(client.prefill(ids))
        assert tok.type == "token", tok
        out.append(tok["id"])
        pos = ids.shape[0]
        for _ in range(steps - 1):
            if tok["done"]:
                break
            tok = await c.decode(client.decode(out[-1], pos), pos)
            assert tok.type == "token", tok
            out.append(tok["id"])
            pos += 1
        await c.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8765")
    ap.add_argument("--token", required=True)
    ap.add_argument("--model", default="qwen2.5-0.5b-instruct")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=16)
    a = ap.parse_args()
    lm = LoadedModel(a.model)
    ids = lm.tokenizer(a.prompt, return_tensors="pt").input_ids[0]
    toks = asyncio.run(generate(a.url, a.token, lm, a.N, ids, a.steps))
    print(toks)
    print(repr(lm.tokenizer.decode(toks)))


if __name__ == "__main__":
    main()

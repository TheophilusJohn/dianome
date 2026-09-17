"""Model loading, the explicit block loop (PartialDecoder), head and sampling.

Boundary definition (fixed by the Phase 4 brief): "split at N" means the client
runs the embedding lookup and blocks 0..N-1 and sends the *output of block N-1*
(= the input to block N) as fp16 [T, d_model]. The server runs blocks N..L-1,
the final norm, lm_head and sampling. N = 0: the client sends token ids. N = L:
the client sends the output of the last block and the server runs only
norm + lm_head + sampling.

The block loop below re-implements `Qwen2Model.forward` of the pinned
transformers version over `model.model.layers[a:b]` with a `DynamicCache`.
It depends on internal APIs (DynamicCache.update via the attention modules,
the `position_embeddings=(cos, sin)` argument of Qwen2DecoderLayer), hence the
exact pin in pyproject.toml.
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Optional

import torch
import transformers
from transformers import AutoTokenizer, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

# id -> (HF repo, pinned revision or None). The 0.5B revision matches Phase 1's ingest.
MODELS: dict[str, tuple[str, Optional[str]]] = {
    "qwen2.5-0.5b-instruct": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "qwen2.5-1.5b-instruct": ("Qwen/Qwen2.5-1.5B-Instruct", None),
    "qwen2.5-3b-instruct": ("Qwen/Qwen2.5-3B-Instruct", None),
}


def pick_device(explicit: Optional[str] = None) -> torch.device:
    """cuda -> mps -> cpu, never hardcoded. `DIANOME_DEVICE` or `explicit` overrides."""
    name = explicit or os.environ.get("DIANOME_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        import platform, subprocess

        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except Exception:  # pragma: no cover
            chip = platform.processor()
        return f"mps ({chip or platform.machine()})"
    return f"cpu ({os.cpu_count()} threads)"


def synchronize(device: torch.device) -> None:
    """Block until all queued kernels on `device` finish (used around busy-time clocks)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclasses.dataclass
class Sampling:
    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Sampling":
        d = d or {}
        return cls(
            temperature=float(d.get("temperature", 0.0)),
            top_p=float(d.get("top_p", 1.0)),
            seed=d.get("seed"),
        )


class LoadedModel:
    """A Qwen2ForCausalLM in fp16 on the selected device plus its tokenizer and shapes."""

    def __init__(self, model_id: str, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float16):
        repo, revision = MODELS.get(model_id, (model_id, None))
        self.id = model_id
        self.repo = repo
        self.revision = revision
        self.device = device or pick_device()
        self.dtype = dtype
        t0 = time.perf_counter()
        self.model: Qwen2ForCausalLM = Qwen2ForCausalLM.from_pretrained(
            repo, revision=revision, dtype=dtype, attn_implementation="sdpa"
        )
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        self.load_seconds = time.perf_counter() - t0
        cfg = self.model.config
        self.config = cfg
        self.L: int = cfg.num_hidden_layers
        self.d_model: int = cfg.hidden_size
        self.vocab: int = cfg.vocab_size
        if getattr(cfg, "layer_types", None) and any(t != "full_attention" for t in cfg.layer_types):
            raise RuntimeError("PartialDecoder only supports full-attention Qwen2 layers")

    # -- pieces of Qwen2Model.forward, exposed separately -------------------------------

    @property
    def inner(self):
        return self.model.model

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Token ids [T] -> fp16 [T, d_model] (the input to block 0)."""
        return self.inner.embed_tokens(ids.to(self.device))

    def rope(self, positions: torch.Tensor, like: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(cos, sin) each [1, T, head_dim] for absolute positions [T], exactly as Qwen2Model computes them."""
        pos = positions.to(self.device).unsqueeze(0)
        ref = like if like is not None else torch.empty(0, dtype=self.dtype, device=self.device)
        return self.inner.rotary_emb(ref, pos)

    def causal_mask(self, T: int, past: int) -> Optional[torch.Tensor]:
        """Attention mask for T new queries after `past` cached keys.

        None lets SDPA take the same path `Qwen2Model.forward` takes for a
        single unpadded sequence (`is_causal=True` for a fresh prefill, no mask
        for a single-token decode). A chunked prefill on top of a non-empty
        cache needs an explicit bool mask because SDPA's `is_causal` aligns the
        diagonal top-left, not to the cache offset.
        """
        if past == 0 or T == 1:
            return None
        q = torch.arange(past, past + T, device=self.device).unsqueeze(1)
        k = torch.arange(past + T, device=self.device).unsqueeze(0)
        return (k <= q)[None, None]  # [1, 1, T, past+T], True = attend

    def run_blocks(
        self, hidden: torch.Tensor, a: int, b: int, cache: DynamicCache, cache_position: torch.Tensor
    ) -> torch.Tensor:
        """Run blocks a..b-1 on hidden [T, d_model] (input to block a), returns input to block b."""
        if hidden.dim() != 2 or hidden.shape[1] != self.d_model:
            raise ValueError(f"hidden must be [T, {self.d_model}], got {tuple(hidden.shape)}")
        if not (0 <= a <= b <= self.L):
            raise ValueError(f"bad block range {a}..{b} for L={self.L}")
        T = hidden.shape[0]
        past = int(cache_position[0].item()) if T else 0
        x = hidden.to(self.device, self.dtype).unsqueeze(0)
        cache_position = cache_position.to(self.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.inner.rotary_emb(x, position_ids)
        mask = self.causal_mask(T, past)
        for layer in self.inner.layers[a:b]:
            x = layer(
                x,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        return x.squeeze(0)

    def final_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.inner.norm(hidden.to(self.device, self.dtype))

    def lm_head(self, normed: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(normed)

    def new_cache(self) -> DynamicCache:
        return DynamicCache(config=self.config)

    def full_forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Plain `model.forward` logits [T, vocab] for a prompt (the reference for the gates)."""
        with torch.no_grad():
            out = self.model(input_ids=ids.to(self.device).unsqueeze(0), use_cache=False)
        return out.logits.squeeze(0)

    def full_greedy(self, ids: torch.Tensor, steps: int) -> list[int]:
        """`steps` greedy tokens from `model.forward` + DynamicCache (the reference decode path)."""
        out: list[int] = []
        cache = self.new_cache()
        cur = ids.to(self.device).unsqueeze(0)
        with torch.no_grad():
            for _ in range(steps):
                logits = self.model(input_ids=cur, past_key_values=cache, use_cache=True).logits[0, -1]
                nxt = int(torch.argmax(logits).item())
                out.append(nxt)
                cur = torch.tensor([[nxt]], device=self.device)
        return out


def cache_lengths(cache: DynamicCache, L: int) -> list[int]:
    """Cached sequence length per layer index (0 for layers never touched)."""
    return [cache.get_seq_length(i) for i in range(L)]


class Sampler:
    """Greedy when temperature == 0, otherwise temperature + nucleus sampling on a seeded CPU generator."""

    def __init__(self, sampling: Sampling):
        self.s = sampling
        self.gen: Optional[torch.Generator] = None
        if sampling.temperature > 0:
            self.gen = torch.Generator(device="cpu")
            if sampling.seed is not None:
                self.gen.manual_seed(int(sampling.seed))

    def __call__(self, logits: torch.Tensor) -> int:
        if self.s.temperature <= 0:
            return int(torch.argmax(logits).item())
        probs = torch.softmax(logits.float().cpu() / self.s.temperature, dim=-1)
        if self.s.top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, dim=-1)
            keep = cum - sp < self.s.top_p
            sp = sp * keep
            probs = torch.zeros_like(probs).scatter_(0, si, sp)
            probs = probs / probs.sum()
        return int(torch.multinomial(probs, 1, generator=self.gen).item())


class PartialDecoder:
    """Blocks a..b-1 with their own DynamicCache; with b == L also final norm + lm_head + sampling.

    `PartialDecoder(lm, N)` is the server half for "split at N". Feed it the
    input to block N (or token ids when N == 0) via `prefill` / `decode`.
    """

    def __init__(self, lm: LoadedModel, a: int, b: Optional[int] = None):
        self.lm = lm
        self.a = a
        self.b = lm.L if b is None else b
        if not (0 <= self.a <= self.b <= lm.L):
            raise ValueError(f"bad block range {a}..{b} for L={lm.L}")
        self.cache = lm.new_cache()
        self.seen = 0  # positions already in the cache

    @property
    def has_head(self) -> bool:
        return self.b == self.lm.L

    def reset(self) -> None:
        self.cache = self.lm.new_cache()
        self.seen = 0

    def cache_lengths(self) -> list[int]:
        return cache_lengths(self.cache, self.lm.L)

    def _to_hidden(self, x: torch.Tensor) -> torch.Tensor:
        if self.a == 0:
            if x.dtype not in (torch.int32, torch.int64):
                raise ValueError("block range starting at 0 takes token ids")
            return self.lm.embed(x.to(torch.long))
        return x

    @torch.no_grad()
    def run(self, x: torch.Tensor, start: int) -> torch.Tensor:
        """Push T positions [start, start+T) through blocks a..b-1; returns the input to block b."""
        if start != self.seen:
            raise ValueError(f"positions must be contiguous: cache has {self.seen}, got start={start}")
        hidden = self._to_hidden(x)
        T = hidden.shape[0]
        cache_position = torch.arange(start, start + T, device=self.lm.device)
        out = self.lm.run_blocks(hidden, self.a, self.b, self.cache, cache_position)
        self.seen += T
        return out

    @torch.no_grad()
    def logits(self, hidden_last: torch.Tensor) -> torch.Tensor:
        """final norm + lm_head on [T, d_model] -> [T, vocab] (b must be L)."""
        if not self.has_head:
            raise ValueError("this PartialDecoder does not end at the last block")
        return self.lm.lm_head(self.lm.final_norm(hidden_last))

    def prefill(self, x: torch.Tensor, start: int = 0) -> torch.Tensor:
        """Returns logits [T, vocab] (with head) or the block-b input [T, d] (without)."""
        h = self.run(x, start)
        return self.logits(h) if self.has_head else h

    def decode(self, x: torch.Tensor, position: int) -> torch.Tensor:
        """One position; returns logits [vocab] (with head) or [1, d] hidden (without)."""
        h = self.run(x if x.dim() > 0 else x.unsqueeze(0), position)
        return self.logits(h)[-1] if self.has_head else h


def versions() -> dict[str, str]:
    return {"torch": torch.__version__, "transformers": transformers.__version__}

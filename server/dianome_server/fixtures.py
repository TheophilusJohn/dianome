"""Reference activations for Phase 5a's WGSL kernels.

One fixed 32-token prompt. Files (all little-endian, row-major, fp16 unless noted):
  prompt.json        text + token ids
  embed.npy          [T, d_model]  output of the embedding lookup = input to block 0
  block_NN.npy       [T, d_model]  output of block NN = input to block NN+1 (boundary NN+1)
  final_norm.npy     [T, d_model]  output of the final RMSNorm
  logits.npy         [T, vocab]    lm_head output
  rope_cos.npy       [T, head_dim] the cos table the attention kernels consume (fp16, as the model casts it)
  rope_sin.npy       [T, head_dim]
  fixtures.json      shapes, dtypes, sha256 per file, versions, the boundary definition
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import numpy as np
import torch

from .model import LoadedModel, PartialDecoder, versions

FIXTURE_PROMPT = (
    "The quick brown fox jumps over the lazy dog while the sun sets slowly behind "
    "the distant mountains, casting long shadows across the quiet valley below, where a small river"
)

BOUNDARY_DEFINITION = (
    '"Split at N" means the client runs the embedding lookup and transformer blocks 0..N-1 and sends '
    "the output of block N-1 (the input to block N) as fp16 [T, d_model]. The server runs blocks N..L-1, "
    "the final norm, lm_head, and sampling. N = 0: the client sends token ids and the server runs everything. "
    "N = L: the client sends the output of the last block and the server runs only final norm + lm_head + sampling. "
    "In this directory: embed.npy is the input to block 0 (boundary 0); block_NN.npy is the output of block NN, "
    "i.e. boundary NN+1; the input to block N for N >= 1 is block_{N-1:02d}.npy."
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save(out: str, name: str, arr: np.ndarray, files: dict) -> None:
    path = os.path.join(out, name)
    np.save(path, np.ascontiguousarray(arr))
    files[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": sha256_file(path), "bytes": os.path.getsize(path)}


@torch.no_grad()
def write_fixtures(lm: LoadedModel, out: str, prompt: str = FIXTURE_PROMPT, expect_tokens: Optional[int] = 32) -> dict:
    os.makedirs(out, exist_ok=True)
    ids = lm.tokenizer(prompt, return_tensors="pt").input_ids[0]
    if expect_tokens is not None and ids.shape[0] != expect_tokens:
        raise RuntimeError(f"fixture prompt tokenizes to {ids.shape[0]} tokens, expected {expect_tokens}")
    T = ids.shape[0]
    files: dict = {}

    with open(os.path.join(out, "prompt.json"), "w") as f:
        json.dump({"text": prompt, "token_ids": ids.tolist(), "T": T}, f, indent=2)
    files["prompt.json"] = {"sha256": sha256_file(os.path.join(out, "prompt.json"))}

    def f16(t: torch.Tensor) -> np.ndarray:
        return t.detach().to(torch.float16).cpu().numpy()

    h = lm.embed(ids)
    _save(out, "embed.npy", f16(h), files)
    dec = PartialDecoder(lm, 0, lm.L)
    # run one block at a time through the same cache so every block output is captured
    x = h
    cache = dec.cache
    cache_position = torch.arange(0, T, device=lm.device)
    for i in range(lm.L):
        x = lm.run_blocks(x, i, i + 1, cache, cache_position)
        _save(out, f"block_{i:02d}.npy", f16(x), files)
    normed = lm.final_norm(x)
    _save(out, "final_norm.npy", f16(normed), files)
    logits = lm.lm_head(normed)
    _save(out, "logits.npy", f16(logits), files)
    cos, sin = lm.rope(cache_position, like=h)
    _save(out, "rope_cos.npy", f16(cos[0]), files)
    _save(out, "rope_sin.npy", f16(sin[0]), files)

    # cross-check against the stock forward so a fixture can never disagree with the model
    ref = lm.full_forward(ids)
    max_diff = (ref.float() - logits.float()).abs().max().item()
    cfg = lm.config
    manifest = {
        "model": lm.id,
        "source": {"repo": lm.repo, "revision": lm.revision},
        "device": str(lm.device),
        "dtype": "float16",
        "T": T,
        "L": lm.L,
        "d_model": lm.d_model,
        "vocab": lm.vocab,
        "head_dim": cfg.hidden_size // cfg.num_attention_heads,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "layout": "row-major, little-endian .npy (numpy v1 header)",
        "boundary_definition": BOUNDARY_DEFINITION,
        "versions": versions() | {"numpy": np.__version__},
        "logits_max_abs_diff_vs_model_forward": max_diff,
        "files": dict(sorted(files.items())),
    }
    with open(os.path.join(out, "fixtures.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def hashes(manifest: dict) -> dict[str, str]:
    return {k: v["sha256"] for k, v in manifest["files"].items()}

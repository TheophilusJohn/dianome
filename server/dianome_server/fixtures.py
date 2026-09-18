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


# ---------------------------------------------------------------------------------------
# Phase 5a extensions: tokenizer cases (gate 1), block-0 intra stages (gate 3), decode
# steps (gate 5), and dequantised q8/q4 references (gate 7). Each adds files to the same
# `files` dict so fixtures.json carries their hashes too; the original files are untouched.

def _f16(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float16).cpu().numpy()


def write_tokenizer_cases(lm: LoadedModel, out: str, files: dict) -> list[dict]:
    """tokenizer_cases.json: 200 strings with HF token ids and HF decode of those ids."""
    from .fixtures_cases import cases

    rows = []
    for text in cases():
        ids = lm.tokenizer(text, add_special_tokens=False).input_ids
        rows.append({"text": text, "ids": ids, "decoded": lm.tokenizer.decode(ids)})
    path = os.path.join(out, "tokenizer_cases.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"count": len(rows), "add_special_tokens": False, "cases": rows}, f, ensure_ascii=False, indent=1)
    files["tokenizer_cases.json"] = {"sha256": sha256_file(path), "count": len(rows)}
    return rows


@torch.no_grad()
def write_intra(lm: LoadedModel, out: str, ids: torch.Tensor, files: dict, block: int = 0) -> None:
    """intra/*.npy (block 0) or intra_<block>/*.npy: every stage of one block on the fixture prompt,
    captured with forward hooks; blocks before it run normally through the same cache.

    q_rope/k_rope are recomputed from the hooked q/k with transformers' own
    apply_rotary_pos_emb and the model's cos/sin (the attention module does not
    expose them). Layout of the [T, heads*head_dim] tensors: head-major, as
    `.transpose(1, 2).reshape(T, -1)` of the [1, H, T, D] tensors.
    """
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    sub = "intra" if block == 0 else f"intra_{block}"
    os.makedirs(os.path.join(out, sub), exist_ok=True)
    layer = lm.inner.layers[block]
    cap: dict[str, torch.Tensor] = {}

    def out_hook(name):
        return lambda m, i, o: cap.__setitem__(name, o.detach().clone())

    def in_hook(name):
        return lambda m, i, o: cap.__setitem__(name, i[0].detach().clone())

    hooks = [
        layer.input_layernorm.register_forward_hook(out_hook("post_input_norm")),
        layer.self_attn.q_proj.register_forward_hook(out_hook("q_proj")),
        layer.self_attn.k_proj.register_forward_hook(out_hook("k_proj")),
        layer.self_attn.v_proj.register_forward_hook(out_hook("v_proj")),
        layer.self_attn.o_proj.register_forward_hook(in_hook("attn_out")),
        layer.self_attn.o_proj.register_forward_hook(out_hook("o_proj")),
        layer.post_attention_layernorm.register_forward_hook(in_hook("post_o_proj_residual")),
        layer.post_attention_layernorm.register_forward_hook(out_hook("post_post_attn_norm")),
        layer.mlp.register_forward_hook(out_hook("mlp_out")),
    ]
    try:
        h = lm.embed(ids)
        T = ids.shape[0]
        cp = torch.arange(0, T, device=lm.device)
        cache = lm.new_cache()
        x = lm.run_blocks(h, 0, block, cache, cp) if block > 0 else h
        cap["block_in"] = x.detach().clone()
        x = lm.run_blocks(x, block, block + 1, cache, cp)
    finally:
        for hk in hooks:
            hk.remove()
    cos, sin = lm.rope(cp, like=h)
    hd = lm.config.hidden_size // lm.config.num_attention_heads
    q = cap["q_proj"].view(1, T, -1, hd).transpose(1, 2)
    k = cap["k_proj"].view(1, T, -1, hd).transpose(1, 2)
    q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)
    cap["q_rope"] = q_r.transpose(1, 2).reshape(T, -1)
    cap["k_rope"] = k_r.transpose(1, 2).reshape(T, -1)
    cap["block_out"] = x
    order = ["block_in", "post_input_norm", "q_proj", "k_proj", "v_proj", "q_rope", "k_rope", "attn_out", "o_proj",
             "post_o_proj_residual", "post_post_attn_norm", "mlp_out", "block_out"]
    for name in order:
        t = cap[name]
        _save(out, f"{sub}/{name}.npy", _f16(t.reshape(T, -1)), files)


@torch.no_grad()
def write_decode_steps(lm: LoadedModel, out: str, ids: torch.Tensor, files: dict, steps: int = 16) -> dict:
    """greedy.json + decode_steps.npy [steps, L+1, d_model]: the full model's greedy tokens for the
    prompt, and for each of them fed one at a time after the prompt, the hidden state at every
    boundary 0..L (index 0 = embedding, index i = output of block i-1)."""
    greedy, margins, runners = greedy_with_margins(lm, ids, steps)
    assert greedy == lm.full_greedy(ids, steps)
    T = ids.shape[0]
    L, d = lm.L, lm.d_model
    cache = lm.new_cache()
    x = lm.embed(ids)
    cp = torch.arange(0, T, device=lm.device)
    for i in range(L):
        x = lm.run_blocks(x, i, i + 1, cache, cp)
    arr = np.zeros((steps, L + 1, d), dtype=np.float16)
    consistent = True
    for s, tok in enumerate(greedy):
        pos = T + s
        x = lm.embed(torch.tensor([tok]))
        cp = torch.tensor([pos], device=lm.device)
        arr[s, 0] = _f16(x)[0]
        for i in range(L):
            x = lm.run_blocks(x, i, i + 1, cache, cp)
            arr[s, i + 1] = _f16(x)[0]
        if s + 1 < steps:
            nxt = int(torch.argmax(lm.lm_head(lm.final_norm(x))[-1]).item())
            consistent = consistent and nxt == greedy[s + 1]
    _save(out, "decode_steps.npy", arr, files)
    info = {"steps": steps, "prompt_T": T, "tokens": greedy, "positions": [T + s for s in range(steps)],
            "margins": margins, "runners_up": runners,
            "block_loop_argmax_matches_full_greedy": consistent,
            "layout": "decode_steps.npy[s, b] = hidden state at boundary b (0 = embedding output, b = output of block b-1) "
                      "after feeding tokens[s] at positions[s]"}
    path = os.path.join(out, "greedy.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    files["greedy.json"] = {"sha256": sha256_file(path)}
    return info


@torch.no_grad()
def greedy_with_margins(lm: LoadedModel, ids: torch.Tensor, steps: int) -> tuple[list[int], list[float], list[int]]:
    """Greedy tokens plus, per step, the fp16 logit gap top1 - top2 and the runner-up id (tie evidence for gate 6/7)."""
    tokens: list[int] = []
    margins: list[float] = []
    runners: list[int] = []
    cache = lm.new_cache()
    cur = ids.to(lm.device).unsqueeze(0)
    for _ in range(steps):
        logits = lm.model(input_ids=cur, past_key_values=cache, use_cache=True).logits[0, -1].float()
        top = torch.topk(logits, 2)
        nxt = int(top.indices[0].item())
        tokens.append(nxt)
        margins.append(float(top.values[0] - top.values[1]))
        runners.append(int(top.indices[1].item()))
        cur = torch.tensor([[nxt]], device=lm.device)
    return tokens, margins, runners


# -- dequantised references (gate 7) --------------------------------------------------

def _read_entry(store: str, entry: dict) -> bytes:
    parts = []
    for seg in entry["segments"]:
        with open(os.path.join(store, "chunks", seg["chunk"]), "rb") as f:
            f.seek(seg["offset"])
            parts.append(f.read(seg["length"]))
    return b"".join(parts)


def dequantised_entry(store: str, entry: dict) -> np.ndarray:
    """An entry's weight as float32 [shape], through dianome_ingest's decoders (the exact bytes the browser gets)."""
    from dianome_ingest import quant  # installed into the server venv from ../ingest (see README)

    raw = _read_entry(store, entry)
    st = entry["storage"]
    shape = tuple(entry["shape"])
    if st["kind"] == "fp16":
        return np.frombuffer(raw, dtype="<f2").reshape(shape).astype(np.float32)
    out, inn = shape
    p = st["parts"]

    def part(name, dtype, count):
        o, n = p[name]["offset"], p[name]["length"]
        a = np.frombuffer(raw[o:o + n], dtype=dtype)
        assert a.size == count, (entry["name"], name, a.size, count)
        return a

    if st["kind"] == "q8":
        q = part("weights", np.int8, out * inn).reshape(out, inn)
        scale = part("scales", "<f2", out)
        return quant.q8_decode(q, scale)
    if st["kind"] == "q4":
        g = st["group_size"]
        packed = part("weights", np.uint8, out * inn // 2).reshape(out, inn // 2)
        scale = part("scales", "<f2", out * (inn // g)).reshape(out, inn // g)
        zero = part("zeros", np.uint8, out * (inn // g)).reshape(out, inn // g)
        return quant.q4_decode(packed, scale, zero)
    raise ValueError(st["kind"])


ROLE_TO_MODULE = {
    "attn.q_proj.weight": "self_attn.q_proj", "attn.k_proj.weight": "self_attn.k_proj",
    "attn.v_proj.weight": "self_attn.v_proj", "attn.o_proj.weight": "self_attn.o_proj",
    "mlp.gate_proj.weight": "mlp.gate_proj", "mlp.up_proj.weight": "mlp.up_proj", "mlp.down_proj.weight": "mlp.down_proj",
}


@torch.no_grad()
def write_variant_reference(lm: LoadedModel, out: str, ids: torch.Tensor, files: dict, variant: str, store: str,
                            splits=(1, 8, 16, 24), steps: int = 16) -> dict:
    """<out>/<variant>/: what PyTorch produces when the *client's* weights are the dequantised
    `variant` entries from the store (embed included; norms and biases are fp16 in every variant)
    and the server's blocks N..L-1 plus lm_head stay fp16.

    greedy.json: {"N": {"1": [...16 tokens], "8": ..., ...}} for the mixed model at each split
    (blocks 0..N-1 dequantised); embed.npy and block_NN.npy for the fully dequantised client
    (N = L). Dequantised weights are rounded to fp16 when written into the fp16 model
    (relative 2^-11, far below the quantisation noise). The model is restored afterwards.
    """
    import torch.nn as nn

    manifest_path = os.path.join(store, "manifests", lm.id, "latest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    groups = {g["name"]: g for g in manifest["variants"][variant]["groups"]}
    vdir = os.path.join(out, variant)
    os.makedirs(vdir, exist_ok=True)
    saved = {k: v.detach().to("cpu", copy=True) for k, v in lm.model.state_dict().items()}
    original_tied = lm.model.lm_head.weight is lm.inner.embed_tokens.weight
    try:
        # untie lm_head so replacing the embedding leaves the server's head fp16
        if original_tied:
            lm.model.lm_head.weight = nn.Parameter(lm.inner.embed_tokens.weight.detach().clone(), requires_grad=False)
        emb_entry = groups["embed"]["entries"][0]
        lm.inner.embed_tokens.weight.copy_(torch.from_numpy(dequantised_entry(store, emb_entry)).to(lm.device, torch.float16))
        done = 0
        greedy: dict[str, list[int]] = {}
        margins: dict[str, list[float]] = {}
        runners: dict[str, list[int]] = {}
        for N in sorted(splits):
            for i in range(done, N):
                layer = lm.inner.layers[i]
                for e in groups[f"layer.{i}"]["entries"]:
                    mod = ROLE_TO_MODULE.get(e["role"])
                    if mod is None:
                        continue  # norms and biases: fp16, identical in every variant
                    w = torch.from_numpy(dequantised_entry(store, e)).to(lm.device, torch.float16)
                    layer.get_submodule(mod).weight.copy_(w)
            done = N
            g, mg, ru = greedy_with_margins(lm, ids, steps)
            greedy[str(N)] = g
            margins[str(N)] = mg
            runners[str(N)] = ru
        for i in range(done, lm.L):  # the rest, for the per-block dumps
            layer = lm.inner.layers[i]
            for e in groups[f"layer.{i}"]["entries"]:
                mod = ROLE_TO_MODULE.get(e["role"])
                if mod is not None:
                    layer.get_submodule(mod).weight.copy_(torch.from_numpy(dequantised_entry(store, e)).to(lm.device, torch.float16))
        h = lm.embed(ids)
        _save(out, f"{variant}/embed.npy", _f16(h), files)
        x = h
        cache = lm.new_cache()
        cp = torch.arange(0, ids.shape[0], device=lm.device)
        for i in range(lm.L):
            x = lm.run_blocks(x, i, i + 1, cache, cp)
            _save(out, f"{variant}/block_{i:02d}.npy", _f16(x), files)
        info = {"variant": variant, "manifest": manifest_path, "steps": steps, "prompt_T": int(ids.shape[0]),
                "N": greedy, "margins": margins, "runners_up": runners, "definition": "blocks 0..N-1 and the embedding use the dequantised variant weights "
                "(rounded to fp16); blocks N..L-1, final norm and lm_head are the original fp16 weights"}
        path = os.path.join(vdir, "greedy.json")
        with open(path, "w") as f:
            json.dump(info, f, indent=2)
        files[f"{variant}/greedy.json"] = {"sha256": sha256_file(path)}
        return info
    finally:
        if original_tied:
            lm.model.lm_head.weight = lm.inner.embed_tokens.weight
        lm.model.load_state_dict(saved)


def write_fixtures_extended(lm: LoadedModel, out: str, *, intra: bool = False, variants=(), store: str = "store",
                            prompt: str = FIXTURE_PROMPT, intra_blocks=(0,)) -> dict:
    """write_fixtures + tokenizer cases + greedy/decode steps, optionally intra and variant references."""
    manifest = write_fixtures(lm, out, prompt)
    files = manifest["files"]
    ids = torch.tensor(json.load(open(os.path.join(out, "prompt.json")))["token_ids"])
    write_tokenizer_cases(lm, out, files)
    manifest["decode"] = write_decode_steps(lm, out, ids, files)
    if intra:
        for b in intra_blocks:
            write_intra(lm, out, ids, files, block=int(b))
    manifest["variants"] = {}
    for v in variants:
        manifest["variants"][v] = write_variant_reference(lm, out, ids, files, v, store)["N"]
    manifest["files"] = dict(sorted(files.items()))
    with open(os.path.join(out, "fixtures.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest

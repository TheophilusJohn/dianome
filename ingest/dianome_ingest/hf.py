"""Hugging Face download and one-tensor-at-a-time iteration via safetensors.safe_open."""
from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes  # noqa: F401  registers np.dtype("bfloat16") so safe_open(framework="np") can return bf16
import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "config.json"]
CONFIG_KEYS = [
    "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
    "intermediate_size", "vocab_size", "rms_norm_eps", "rope_theta",
    "tie_word_embeddings", "max_position_embeddings",
]


def resolve_snapshot(repo: str, revision: str | None = None, allow_patterns=("*.safetensors", "*.json")) -> tuple[Path, str]:
    """Download (or reuse the cache) and return (snapshot dir, commit sha)."""
    path = Path(snapshot_download(repo, revision=revision, allow_patterns=list(allow_patterns)))
    return path, path.name


def read_config(snapshot: Path) -> tuple[dict, dict]:
    """(full config.json, manifest config summary)."""
    cfg = json.loads((snapshot / "config.json").read_text())
    summary = {}
    for k in CONFIG_KEYS:
        if k == "num_key_value_heads":
            summary[k] = int(cfg.get(k) or cfg["num_attention_heads"])
        elif k == "tie_word_embeddings":
            summary[k] = bool(cfg.get(k, False))
        else:
            summary[k] = cfg[k]
    return cfg, summary


def tokenizer_files(snapshot: Path) -> list[Path]:
    return [snapshot / f for f in TOKENIZER_FILES if (snapshot / f).is_file()]


class SafetensorsSource:
    """TensorSource over one or many safetensors files; tensors are read one at a time."""

    def __init__(self, snapshot: Path):
        self.snapshot = Path(snapshot)
        index = self.snapshot / "model.safetensors.index.json"
        if index.is_file():
            wm = json.loads(index.read_text())["weight_map"]
            self._file_of = {k: self.snapshot / v for k, v in wm.items()}
        elif (self.snapshot / "model.safetensors").is_file():
            single = self.snapshot / "model.safetensors"
            with safe_open(single, framework="np") as f:
                self._file_of = {k: single for k in f.keys()}
        else:
            raise FileNotFoundError(f"no model.safetensors or index in {self.snapshot}")
        self._handles: dict[Path, object] = {}

    def _h(self, name: str):
        p = self._file_of[name]
        if p not in self._handles:
            self._handles[p] = safe_open(p, framework="np")
        return self._handles[p]

    def names(self) -> list[str]:
        return sorted(self._file_of)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._h(name).get_slice(name).get_shape())

    def dtype(self, name: str) -> str:
        return self._h(name).get_slice(name).get_dtype()

    def get(self, name: str) -> np.ndarray:
        return self._h(name).get_tensor(name)

    def get_rows(self, name: str, start: int, stop: int) -> np.ndarray:
        return self._h(name).get_slice(name)[start:stop]

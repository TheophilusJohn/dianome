"""Per-layer activations on held-out text, as memory-mapped .npy on disk.

acts.npy    fp16 [L+1, n_tokens, d_model]   boundary i = the input to block i (i = L: input to the final norm)
tokens.npy  int32 [n_tokens]
doc.npy     int32 [n_tokens]                 document index per token (for the by-document split)
pos.npy     int32 [n_tokens]                 position within its window
meta.json   dataset, counts, split, versions
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from dianome_server.model import LoadedModel, versions

from .data import DATASET, split_by_document


@torch.no_grad()
def collect(lm: LoadedModel, docs: list[list[int]], data_dir: str, max_seq: int = 1024, seed: int = 0) -> dict:
    os.makedirs(data_dir, exist_ok=True)
    n = sum(len(d) for d in docs)
    B = lm.L + 1
    acts = np.lib.format.open_memmap(os.path.join(data_dir, "acts.npy"), mode="w+", dtype=np.float16, shape=(B, n, lm.d_model))
    tokens = np.zeros(n, dtype=np.int32)
    doc = np.zeros(n, dtype=np.int32)
    pos = np.zeros(n, dtype=np.int32)
    t0 = time.perf_counter()
    off = 0
    for di, ids in enumerate(docs):
        for w0 in range(0, len(ids), max_seq):
            win = ids[w0 : w0 + max_seq]
            T = len(win)
            x = lm.embed(torch.tensor(win, dtype=torch.long))
            cache = lm.new_cache()
            cp = torch.arange(0, T, device=lm.device)
            acts[0, off : off + T] = x.to(torch.float16).cpu().numpy()
            for i in range(lm.L):
                x = lm.run_blocks(x, i, i + 1, cache, cp)
                acts[i + 1, off : off + T] = x.to(torch.float16).cpu().numpy()
            tokens[off : off + T] = win
            doc[off : off + T] = di
            pos[off : off + T] = np.arange(T)
            off += T
    assert off == n
    acts.flush()
    np.save(os.path.join(data_dir, "tokens.npy"), tokens)
    np.save(os.path.join(data_dir, "doc.npy"), doc)
    np.save(os.path.join(data_dir, "pos.npy"), pos)
    train, val = split_by_document(len(docs), seed=seed)
    meta = {
        "model": lm.id, "dataset": DATASET, "n_documents": len(docs), "n_tokens": n, "max_seq": max_seq,
        "boundaries": B, "d_model": lm.d_model, "vocab": lm.vocab,
        "split": {"by": "document", "seed": seed, "train_docs": train, "val_docs": val,
                  "train_tokens": int(sum(len(docs[i]) for i in train)), "val_tokens": int(sum(len(docs[i]) for i in val))},
        "collect_seconds": time.perf_counter() - t0, "versions": versions(), "device": str(lm.device),
    }
    with open(os.path.join(data_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


class ActivationStore:
    def __init__(self, data_dir: str):
        self.dir = data_dir
        with open(os.path.join(data_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.acts = np.load(os.path.join(data_dir, "acts.npy"), mmap_mode="r")
        self.tokens = np.load(os.path.join(data_dir, "tokens.npy"))
        self.doc = np.load(os.path.join(data_dir, "doc.npy"))
        self.pos = np.load(os.path.join(data_dir, "pos.npy"))
        train = set(self.meta["split"]["train_docs"])
        self.train_mask = np.array([d in train for d in self.doc])
        self.val_mask = ~self.train_mask

    @property
    def n_boundaries(self) -> int:
        return self.acts.shape[0]

    def boundary(self, i: int) -> np.ndarray:
        """fp16 [n_tokens, d_model] for boundary i, read into memory."""
        return np.ascontiguousarray(self.acts[i])

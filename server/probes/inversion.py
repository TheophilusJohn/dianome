"""Upper bound: a small attention decoder (2 layers, 4 heads, d = 512) that reads the
activation sequence of a window and predicts the token sequence."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import _autocast, topk_accuracy


class InversionDecoder(nn.Module):
    def __init__(self, d_in: int, vocab: int, d: int = 512, heads: int = 4, layers: int = 2, window: int = 128):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.pos = nn.Embedding(window, d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        h = self.inp(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        h = self.enc(h, src_key_padding_mask=pad)
        return self.out(self.norm(h))


def make_windows(acts: np.ndarray, y: np.ndarray, doc: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Cut each document into windows of `window` tokens, padded with y = -100."""
    xs, ys = [], []
    d = acts.shape[1]
    for di in np.unique(doc):
        idx = np.nonzero(doc == di)[0]
        for s in range(0, len(idx), window):
            sel = idx[s : s + window]
            x = np.zeros((window, d), dtype=np.float16)
            t = np.full(window, -100, dtype=np.int64)
            x[: len(sel)] = acts[sel]
            t[: len(sel)] = y[sel]
            xs.append(x)
            ys.append(t)
    return np.stack(xs), np.stack(ys)


def train_inversion(
    acts_tr: np.ndarray, y_tr: np.ndarray, doc_tr: np.ndarray,
    acts_va: np.ndarray, y_va: np.ndarray, doc_va: np.ndarray, vocab: int, device: torch.device,
    window: int = 128, batch: int = 8, max_epochs: int = 8, patience: int = 2, lr: float = 1e-3,
    d: int = 512, heads: int = 4, layers: int = 2, seed: int = 0,
) -> dict:
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    Xtr, Ytr = make_windows(acts_tr, y_tr, doc_tr, window)
    Xva, Yva = make_windows(acts_va, y_va, doc_va, window)
    Xtr, Ytr = torch.from_numpy(Xtr).to(device), torch.from_numpy(Ytr).to(device)
    Xva, Yva = torch.from_numpy(Xva).to(device), torch.from_numpy(Yva).to(device)
    mu = Xtr.float().flatten(0, 1).mean(0)
    sd = Xtr.float().flatten(0, 1).std(0) + 1e-5
    model = InversionDecoder(acts_tr.shape[1], vocab, d=d, heads=heads, layers=layers, window=window).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def run(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = y == -100
        with _autocast(device):
            logits = model((x.float() - mu) / sd, pad).float()
        return logits, pad

    def evaluate() -> tuple[float, float, float]:
        model.eval()
        loss = top1 = top5 = n = 0.0
        with torch.no_grad():
            for s in range(0, len(Yva), batch):
                x, y = Xva[s : s + batch], Yva[s : s + batch]
                logits, pad = run(x, y)
                keep = ~pad
                lk, yk = logits[keep], y[keep]
                del logits
                loss += F.cross_entropy(lk, yk, reduction="sum").item()
                a1, a5 = topk_accuracy(lk, yk)
                del lk
                top1 += a1
                top5 += a5
                n += keep.sum().item()
        return loss / n, top1 / n, top5 / n

    best = (float("inf"), 0.0, 0.0)
    bad = 0
    epochs = 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Ytr), generator=g).to(device)
        for s in range(0, len(Ytr), batch):
            idx = perm[s : s + batch]
            logits, pad = run(Xtr[idx], Ytr[idx])
            loss = F.cross_entropy(logits.flatten(0, 1), Ytr[idx].flatten(), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        epochs = epoch + 1
        val = evaluate()
        if val[0] < best[0] - 1e-4:
            best, bad = val, 0
        else:
            bad += 1
            if bad >= patience:
                break
    n_params = sum(p.numel() for p in model.parameters())
    del model, opt, Xtr, Xva
    return {"inversion_top1": best[1], "inversion_top5": best[2], "inversion_val_loss": best[0],
            "inversion_epochs": epochs, "inversion_params": n_params,
            "inversion_train_seconds": time.perf_counter() - t0}

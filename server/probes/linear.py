"""Lower bound: a linear map d_model -> vocab per boundary, plus the zero-training
nearest-neighbour baseline against the (tied) embedding matrix."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F


def _autocast(device: torch.device):
    if device.type in ("cuda", "mps"):
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    import contextlib

    return contextlib.nullcontext()


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, y: torch.Tensor, k: int = 5) -> tuple[int, int]:
    top = logits.topk(k, dim=-1).indices
    hit = top == y.unsqueeze(-1)
    return int(hit[:, 0].sum().item()), int(hit.any(-1).sum().item())


@torch.no_grad()
def nn_baseline(acts: np.ndarray, y: np.ndarray, embed: torch.Tensor, device: torch.device, batch: int = 512) -> dict:
    """Nearest embedding row, no training: what boundary 0 trivially inverts.

    `nn_top1`/`nn_top5` use cosine similarity (scale-free: the residual stream's
    norm grows with depth); `nn_l2_*` is plain L2 distance. Small batches on
    purpose: a [batch, vocab] fp32 block is 0.6 MB per row and MPS pages badly
    once a few of them are alive.
    """
    E = embed.to(device, torch.float32)
    e2 = (E * E).sum(-1)
    En = E / E.norm(dim=-1, keepdim=True)
    cos1 = cos5 = l21 = l25 = 0
    for s in range(0, len(y), batch):
        h = torch.from_numpy(acts[s : s + batch]).to(device, torch.float32)
        yy = torch.from_numpy(y[s : s + batch].astype(np.int64)).to(device)
        sim = (h / h.norm(dim=-1, keepdim=True)) @ En.T
        a1, a5 = topk_accuracy(sim, yy)
        cos1 += a1
        cos5 += a5
        del sim
        d = (h @ E.T).mul_(-2.0).add_(e2.unsqueeze(0))  # |h - E|^2 - |h|^2
        a1, a5 = topk_accuracy(d.neg_(), yy)
        l21 += a1
        l25 += a5
        del d
    n = len(y)
    return {"nn_top1": cos1 / n, "nn_top5": cos5 / n, "nn_l2_top1": l21 / n, "nn_l2_top5": l25 / n}


@torch.no_grad()
def _mean_std(X: torch.Tensor, chunk: int = 65536) -> tuple[torch.Tensor, torch.Tensor]:
    n, d = X.shape
    s1 = torch.zeros(d, dtype=torch.float64)  # accumulate on the CPU: MPS has no float64
    s2 = torch.zeros(d, dtype=torch.float64)
    for i in range(0, n, chunk):
        x = X[i : i + chunk].float()
        s1 += x.sum(0).cpu().double()
        s2 += (x * x).sum(0).cpu().double()
    mu = s1 / n
    var = (s2 / n - mu * mu).clamp_min(0)
    return mu.float().unsqueeze(0).to(X.device), (var.sqrt().float() + 1e-5).unsqueeze(0).to(X.device)


def train_linear(
    acts_tr: np.ndarray, y_tr: np.ndarray, acts_va: np.ndarray, y_va: np.ndarray, vocab: int,
    device: torch.device, batch: int = 1024, max_epochs: int = 5, patience: int = 1, lr: float = 2e-3, seed: int = 0,
) -> dict:
    """AdamW on cross-entropy, early stop on held-out loss; returns metrics only (weights are dropped)."""
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    d = acts_tr.shape[1]
    lin = torch.nn.Linear(d, vocab).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=0.0)
    Xtr = torch.from_numpy(acts_tr).to(device)
    Ytr = torch.from_numpy(y_tr.astype(np.int64)).to(device)
    Xva = torch.from_numpy(acts_va).to(device)
    Yva = torch.from_numpy(y_va.astype(np.int64)).to(device)
    mu, sd = _mean_std(Xtr)  # standardise inputs; deep boundaries have huge outlier dims

    def evaluate() -> tuple[float, float, float]:
        lin.eval()
        loss = top1 = top5 = 0.0
        with torch.no_grad(), _autocast(device):
            for s in range(0, len(Yva), batch):
                x = (Xva[s : s + batch].float() - mu) / sd
                logits = lin(x).float()
                loss += F.cross_entropy(logits, Yva[s : s + batch], reduction="sum").item()
                a1, a5 = topk_accuracy(logits, Yva[s : s + batch])
                del logits
                top1 += a1
                top5 += a5
        n = len(Yva)
        return loss / n, top1 / n, top5 / n

    best = (float("inf"), 0.0, 0.0)
    bad = 0
    epochs = 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(max_epochs):
        lin.train()
        perm = torch.randperm(len(Ytr), generator=g).to(device)
        for s in range(0, len(Ytr), batch):
            idx = perm[s : s + batch]
            x = (Xtr[idx].float() - mu) / sd
            with _autocast(device):
                loss = F.cross_entropy(lin(x).float(), Ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        epochs = epoch + 1
        val = evaluate()
        if val[0] < best[0] - 1e-4:
            best, bad = val, 0
        else:
            bad += 1
            if bad >= patience:
                break
    del lin, opt, Xtr, Xva
    return {"linear_top1": best[1], "linear_top5": best[2], "linear_val_loss": best[0],
            "linear_epochs": epochs, "linear_train_seconds": time.perf_counter() - t0}

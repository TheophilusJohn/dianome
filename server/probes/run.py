"""Orchestrates collect -> per-boundary nn / linear / inversion -> report. Resumable per boundary."""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
import torch

from dianome_server.model import LoadedModel, device_name, versions

from .collect import ActivationStore, collect
from .data import load_documents, tokenize_documents
from .inversion import train_inversion
from .linear import nn_baseline, train_linear
from .report import band_section, band_table, update_notes

HERE = os.path.dirname(os.path.abspath(__file__))


def probe_boundary(store: ActivationStore, i: int, embed: torch.Tensor, device: torch.device, **kw) -> dict:
    acts = store.boundary(i)
    y = store.tokens
    tr, va = store.train_mask, store.val_mask
    r: dict = {"boundary": i}
    r.update(nn_baseline(acts[va], y[va], embed, device))
    r.update(train_linear(acts[tr], y[tr], acts[va], y[va], store.meta["vocab"], device, **kw.get("linear", {})))
    r.update(train_inversion(acts[tr], y[tr], store.doc[tr], acts[va], y[va], store.doc[va],
                             store.meta["vocab"], device, **kw.get("inversion", {})))
    r["train_seconds"] = r["linear_train_seconds"] + r["inversion_train_seconds"]
    return r


def run_probes(lm: LoadedModel, n_tokens: int = 50_000, data_dir: Optional[str] = None,
               boundaries: Optional[list[int]] = None, skip_collect: bool = False,
               notes: Optional[str] = None, max_doc_tokens: int = 1024) -> dict:
    data_dir = data_dir or os.path.join(HERE, "data")
    results_path = os.path.join(HERE, "results", "band.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    wall0 = time.perf_counter()

    if not skip_collect or not os.path.exists(os.path.join(data_dir, "meta.json")):
        docs = tokenize_documents(lm.tokenizer, load_documents(), n_tokens, max_doc_tokens)
        print(f"collect: {len(docs)} documents, {sum(map(len, docs))} tokens -> {data_dir}")
        meta = collect(lm, docs, data_dir)
        print(f"collect done in {meta['collect_seconds']:.1f}s; acts.npy is "
              f"{os.path.getsize(os.path.join(data_dir, 'acts.npy')) / 1e9:.2f} GB")
    store = ActivationStore(data_dir)
    boundaries = boundaries if boundaries is not None else list(range(store.n_boundaries))

    results = {"meta": store.meta, "model": lm.id, "device": device_name(lm.device), "versions": versions(), "rows": []}
    if os.path.exists(results_path):
        with open(results_path) as f:
            prev = json.load(f)
        if prev.get("meta", {}).get("n_tokens") == store.meta["n_tokens"]:
            results["rows"] = [r for r in prev["rows"] if r["boundary"] in boundaries]
    done = {r["boundary"] for r in results["rows"]}
    embed = lm.inner.embed_tokens.weight.detach()
    print(f"probes on {device_name(lm.device)}: boundaries {boundaries} (already done: {sorted(done)})")
    for i in boundaries:
        if i in done:
            continue
        t0 = time.perf_counter()
        r = probe_boundary(store, i, embed, lm.device)
        results["rows"].append(r)
        print(f"  boundary {i:2d}: nn {r['nn_top1']:.4f}  linear {r['linear_top1']:.4f}/{r['linear_top5']:.4f} "
              f"({r['linear_epochs']} ep)  inversion {r['inversion_top1']:.4f}/{r['inversion_top5']:.4f} "
              f"({r['inversion_epochs']} ep)  {time.perf_counter() - t0:.0f}s")
        results["wall_seconds"] = time.perf_counter() - wall0
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        if lm.device.type == "mps":
            torch.mps.empty_cache()
    results["wall_seconds"] = time.perf_counter() - wall0
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(band_table(results["rows"]))
    if notes:
        update_notes(notes, band_section(results))
        print(f"updated {notes}")
    return results

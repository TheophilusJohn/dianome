"""Linear probe with a ~500k-token training set sampled from the WikiText-103 train split.

The held-out set is exactly Phase 4's: the val documents of `probes/data` (the
WikiText-103 test split as collected there). Training documents come from the
train split, sampled with a fixed seed, any title that also occurs in the test
split excluded, so train and test are disjoint by construction. Activations for
the training set go to a second memory-mapped store; only metrics are kept.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
import torch

from dianome_server.model import LoadedModel, device_name, versions

from .collect import ActivationStore, collect
from .data import load_documents, load_train_documents, sample_documents, title_of
from .linear import train_linear

HERE = os.path.dirname(os.path.abspath(__file__))
START = "<!-- linear500k:start -->"
END = "<!-- linear500k:end -->"


def build_train_store(lm: LoadedModel, data_dir: str, target_tokens: int, max_doc_tokens: int, seed: int) -> dict:
    test_titles = {title_of(d) for d in load_documents()}
    train_docs = load_train_documents()
    docs, chosen = sample_documents(lm.tokenizer, train_docs, test_titles, target_tokens, max_doc_tokens, seed)
    assert not ({title_of(train_docs[i]) for i in chosen} & test_titles)
    print(f"sampled {len(docs)} train documents, {sum(map(len, docs))} tokens (seed {seed}, "
          f"{len(test_titles)} test titles excluded) -> {data_dir}", flush=True)
    meta = collect(lm, docs, data_dir, all_train=True, extra_meta={
        "source_split": "train", "sample_seed": seed, "target_tokens": target_tokens,
        "max_doc_tokens": max_doc_tokens, "chosen_document_indices": chosen,
        "excluded_test_titles": len(test_titles),
    })
    print(f"collect done in {meta['collect_seconds']:.0f}s; acts.npy is "
          f"{os.path.getsize(os.path.join(data_dir, 'acts.npy')) / 1e9:.1f} GB", flush=True)
    return meta


def coverage(train_tokens: np.ndarray, val_tokens: np.ndarray) -> dict:
    seen_types = np.unique(train_tokens)
    seen = np.isin(val_tokens, seen_types)
    val_types = np.unique(val_tokens)
    return {
        "val_tokens": int(len(val_tokens)),
        "val_tokens_seen_in_train": int(seen.sum()),
        "coverage": float(seen.mean()),  # token-weighted: the ceiling for top-k accuracy
        "train_types": int(len(seen_types)),
        "val_types": int(len(val_types)),
        "val_types_seen_in_train": int(np.isin(val_types, seen_types).sum()),
    }


def run_linear500k(lm: LoadedModel, test_dir: Optional[str] = None, train_dir: Optional[str] = None,
                   target_tokens: int = 500_000, max_doc_tokens: int = 1024, seed: int = 1,
                   boundaries: Optional[list[int]] = None, notes: Optional[str] = None,
                   linear_kw: Optional[dict] = None) -> dict:
    test_dir = test_dir or os.path.join(HERE, "data")
    train_dir = train_dir or os.path.join(HERE, "data-train500k")
    results_path = os.path.join(HERE, "results", "linear-500k.json")
    wall0 = time.perf_counter()
    if not os.path.exists(os.path.join(train_dir, "meta.json")):
        build_train_store(lm, train_dir, target_tokens, max_doc_tokens, seed)
    test = ActivationStore(test_dir)
    train = ActivationStore(train_dir)
    assert train.n_boundaries == test.n_boundaries
    va = test.val_mask
    y_va = test.tokens[va]
    cov = coverage(train.tokens, y_va)
    print(f"coverage: {cov}", flush=True)
    boundaries = boundaries if boundaries is not None else list(range(test.n_boundaries))

    results = {"train_meta": {k: v for k, v in train.meta.items() if k != "chosen_document_indices"},
               "test_meta": test.meta, "coverage": cov, "model": lm.id, "device": device_name(lm.device),
               "versions": versions(), "linear_kw": linear_kw or {}, "rows": []}
    if os.path.exists(results_path):
        with open(results_path) as f:
            prev = json.load(f)
        if prev.get("train_meta", {}).get("n_tokens") == train.meta["n_tokens"]:
            results["rows"] = [r for r in prev["rows"] if r["boundary"] in boundaries]
    done = {r["boundary"] for r in results["rows"]}
    print(f"linear probe (500k) on {results['device']}: boundaries {boundaries} (done: {sorted(done)})", flush=True)
    for i in boundaries:
        if i in done:
            continue
        t0 = time.perf_counter()
        acts_tr = train.boundary(i)
        acts_va = test.boundary(i)[va]
        r = {"boundary": i, "coverage": cov["coverage"]}
        r.update(train_linear(acts_tr, train.tokens, acts_va, y_va, test.meta["vocab"], lm.device, **(linear_kw or {})))
        del acts_tr, acts_va
        r["norm_top1"] = r["linear_top1"] / cov["coverage"]
        r["norm_top5"] = r["linear_top5"] / cov["coverage"]
        results["rows"].append(r)
        results["wall_seconds"] = time.perf_counter() - wall0
        print(f"  boundary {i:2d}: linear {r['linear_top1']:.4f}/{r['linear_top5']:.4f} "
              f"norm {r['norm_top1']:.4f}/{r['norm_top5']:.4f} ({r['linear_epochs']} ep, "
              f"{r['linear_train_seconds']:.0f}s)  {time.perf_counter() - t0:.0f}s", flush=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        if lm.device.type == "mps":
            torch.mps.empty_cache()
    results["wall_seconds"] = time.perf_counter() - wall0
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(table(results["rows"]), flush=True)
    if notes:
        write_notes(notes, results)
        print(f"updated {notes}", flush=True)
    return results


COLUMNS = ["boundary", "coverage", "linear_top1", "linear_top5", "norm_top1", "norm_top5", "epochs", "train_seconds"]


def table(rows: list[dict]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    for r in sorted(rows, key=lambda r: r["boundary"]):
        lines.append(f"| {r['boundary']} | {r['coverage']:.4f} | {r['linear_top1']:.4f} | {r['linear_top5']:.4f} | "
                     f"{r['norm_top1']:.4f} | {r['norm_top5']:.4f} | {r['linear_epochs']} | {r['linear_train_seconds']:.1f} |")
    return "\n".join(lines) + "\n"


def section(results: dict) -> str:
    tm, cov, kw = results["train_meta"], results["coverage"], results["linear_kw"]
    head = (
        f"Training set: {tm['n_documents']} documents, {tm['n_tokens']} tokens sampled from `Salesforce/wikitext` "
        f"`wikitext-103-raw-v1` split `train` (seed {tm['sample_seed']}, each document capped at {tm['max_doc_tokens']} tokens, "
        f"{tm['excluded_test_titles']} test-split titles excluded, so train and test are disjoint by construction). "
        f"Held-out set: exactly the Phase 4 one above ({cov['val_tokens']} tokens from the {len(results['test_meta']['split']['val_docs'])} "
        f"held-out documents of the test split). `coverage` = fraction of held-out tokens whose type occurs in the training set "
        f"({cov['val_tokens_seen_in_train']}/{cov['val_tokens']}; {cov['val_types_seen_in_train']}/{cov['val_types']} of the held-out "
        f"token types; the training set has {cov['train_types']} types). It does not depend on the boundary, so the column is constant. "
        f"`norm_*` = `linear_*` / coverage. Linear probe: AdamW lr {kw.get('lr', 2e-3)}, batch {kw.get('batch', 1024)}, up to "
        f"{kw.get('max_epochs', 5)} epochs, patience {kw.get('patience', 1)} on held-out loss, inputs standardised per dimension, "
        f"one boundary at a time, weights discarded. Inversion decoder results are unchanged (table above). "
        f"Device {results['device']}, wall time {results.get('wall_seconds', 0):.0f} s.\n\n"
    )
    return head + table(results["rows"])


def write_notes(notes_path: str, results: dict) -> None:
    with open(notes_path) as f:
        text = f.read()
    block = f"{START}\n{section(results)}{END}"
    if START in text and END in text:
        a, b = text.index(START), text.index(END) + len(END)
        text = text[:a] + block + text[b:]
    else:
        marker = "## Fixtures for Phase 5a"
        sec = f"## Linear probe, 500k-token training set\n\n```\n$ server/.venv/bin/dianome-server linear-probe --notes docs/phase-4-notes.md\n```\n\n{block}\n\n"
        text = text.replace(marker, sec + marker, 1) if marker in text else text.rstrip("\n") + "\n\n" + sec
    with open(notes_path, "w") as f:
        f.write(text)

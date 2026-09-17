"""Held-out text for the privacy probes: WikiText-103 test split, split 80/20 by document."""

from __future__ import annotations

import random
import re
from typing import Optional

DATASET = {"repo": "Salesforce/wikitext", "config": "wikitext-103-raw-v1", "split": "test", "license": "CC BY-SA 3.0"}
HEADING = re.compile(r"^ = [^=].* = $")


def load_documents() -> list[str]:
    """One string per top-level article (a ' = Title = ' heading starts a document)."""
    import datasets

    ds = datasets.load_dataset(DATASET["repo"], DATASET["config"], split=DATASET["split"])
    docs: list[list[str]] = []
    for line in ds["text"]:
        if HEADING.match(line):
            docs.append([line])
        elif docs:
            docs[-1].append(line)
    return ["".join(d) for d in docs]


def tokenize_documents(tokenizer, docs: list[str], target_tokens: int, max_doc_tokens: int) -> list[list[int]]:
    """Tokenize every document, cap each at max_doc_tokens, stop once target_tokens is reached."""
    out: list[list[int]] = []
    total = 0
    for d in docs:
        ids = tokenizer(d, add_special_tokens=False).input_ids[:max_doc_tokens]
        if len(ids) < 8:
            continue
        if total + len(ids) > target_tokens:
            ids = ids[: target_tokens - total]
            if len(ids) < 8:
                break
        out.append(ids)
        total += len(ids)
        if total >= target_tokens:
            break
    return out


def split_by_document(n_docs: int, val_fraction: float = 0.2, seed: int = 0) -> tuple[list[int], list[int]]:
    """Deterministic 80/20 split of document indices."""
    idx = list(range(n_docs))
    random.Random(seed).shuffle(idx)
    n_val = max(1, round(n_docs * val_fraction))
    val = sorted(idx[:n_val])
    train = sorted(idx[n_val:])
    return train, val

"""Band table -> Markdown, written into docs/phase-4-notes.md between markers."""

from __future__ import annotations

import os

START = "<!-- probes:start -->"
END = "<!-- probes:end -->"
COLUMNS = ["boundary", "nn_top1", "linear_top1", "linear_top5", "inversion_top1", "inversion_top5", "train_seconds"]


def band_table(rows: list[dict]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    for r in sorted(rows, key=lambda r: r["boundary"]):
        vals = [str(r["boundary"])]
        for c in COLUMNS[1:-1]:
            vals.append(f"{r[c]:.4f}")
        vals.append(f"{r['train_seconds']:.1f}")
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def band_section(results: dict) -> str:
    meta = results["meta"]
    ds = meta["dataset"]
    sp = meta["split"]
    head = (
        f"Dataset: `{ds['repo']}` config `{ds['config']}` split `{ds['split']}` ({ds['license']}), "
        f"{meta['n_documents']} documents, {meta['n_tokens']} tokens, windows of {meta['max_seq']} tokens. "
        f"Split by document (seed {sp['seed']}): {len(sp['train_docs'])} train documents ({sp['train_tokens']} tokens), "
        f"{len(sp['val_docs'])} held-out documents ({sp['val_tokens']} tokens); every number below is on the held-out documents. "
        f"Boundary i = the input to block i (i = {meta['boundaries'] - 1}: the input to the final norm). "
        f"`nn` = nearest embedding row by cosine similarity, no training (L2 variant in band.json). `train_seconds` = linear + inversion training time on "
        f"{results.get('device', '?')}. Total probe wall time {results.get('wall_seconds', 0.0):.0f} s.\n\n"
    )
    return head + band_table(results["rows"])


def update_notes(notes_path: str, section: str) -> None:
    block = f"{START}\n{section}{END}"
    if os.path.exists(notes_path):
        with open(notes_path) as f:
            text = f.read()
    else:
        text = "# Phase 4 notes\n\n"
    if START in text and END in text:
        a, b = text.index(START), text.index(END) + len(END)
        text = text[:a] + block + text[b:]
    else:
        text = text.rstrip("\n") + "\n\n## Privacy band\n\n" + block + "\n"
    with open(notes_path, "w") as f:
        f.write(text)

"""Probe metrics on a tiny synthetic run: shapes and ranges only."""

import json

import numpy as np
import torch

from probes.inversion import make_windows, train_inversion
from probes.linear import nn_baseline, train_linear
from probes.report import band_section, band_table, update_notes


def _synthetic(n=300, d=16, vocab=40, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, vocab, n)
    E = rng.standard_normal((vocab, d)).astype(np.float32)
    acts = (E[y] + 0.1 * rng.standard_normal((n, d))).astype(np.float16)
    doc = np.repeat(np.arange(6), n // 6)
    return acts, y.astype(np.int32), doc.astype(np.int32), torch.from_numpy(E)


def test_nn_linear_inversion_shapes_and_ranges():
    acts, y, doc, E = _synthetic()
    dev = torch.device("cpu")
    tr, va = doc < 5, doc == 5
    nn = nn_baseline(acts[va], y[va], E, dev, batch=64)
    assert set(nn) == {"nn_top1", "nn_top5", "nn_l2_top1", "nn_l2_top5"} and 0 <= nn["nn_top1"] <= nn["nn_top5"] <= 1
    lin = train_linear(acts[tr], y[tr], acts[va], y[va], 40, dev, batch=64, max_epochs=2, patience=5)
    assert 0 <= lin["linear_top1"] <= lin["linear_top5"] <= 1 and lin["linear_epochs"] == 2
    assert lin["linear_train_seconds"] > 0
    inv = train_inversion(acts[tr], y[tr], doc[tr], acts[va], y[va], doc[va], 40, dev,
                          window=16, batch=4, max_epochs=1, d=32, heads=2, layers=1)
    assert 0 <= inv["inversion_top1"] <= inv["inversion_top5"] <= 1 and inv["inversion_epochs"] == 1
    assert inv["inversion_params"] > 0


def test_windows_pad_with_ignore_index():
    acts, y, doc, _ = _synthetic(n=60)
    X, Y = make_windows(acts, y, doc, window=8)
    assert X.shape == (12, 8, 16) and Y.shape == (12, 8)
    assert (Y == -100).sum() == 12 * 8 - 60


def test_report_table_and_notes(tmp_path):
    rows = [{"boundary": b, "nn_top1": 0.5, "linear_top1": 0.4, "linear_top5": 0.6,
             "inversion_top1": 0.7, "inversion_top5": 0.9, "train_seconds": 12.34} for b in (1, 0)]
    t = band_table(rows)
    assert t.splitlines()[2].startswith("| 0 |") and "12.3" in t
    results = {"rows": rows, "device": "cpu", "wall_seconds": 1.0,
               "meta": {"dataset": {"repo": "r", "config": "c", "split": "s", "license": "l"}, "n_documents": 2,
                        "n_tokens": 10, "max_seq": 8, "boundaries": 2,
                        "split": {"seed": 0, "train_docs": [0], "val_docs": [1], "train_tokens": 5, "val_tokens": 5}}}
    notes = tmp_path / "notes.md"
    update_notes(str(notes), band_section(results))
    update_notes(str(notes), band_section(results))  # idempotent
    text = notes.read_text()
    assert text.count("<!-- probes:start -->") == 1 and "| boundary |" in text


def test_split_grid_matches_the_brief():
    from bench.cost import split_grid

    assert split_grid(24) == [0, 4, 8, 12, 16, 20, 24]
    assert split_grid(36) == [0, 6, 12, 18, 24, 30, 36]
    assert split_grid(28) == [0, 5, 9, 14, 19, 23, 28]
    assert split_grid(1) == [0, 1]

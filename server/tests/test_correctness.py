"""The three correctness gates from the Phase 4 brief."""

import json
import os

import pytest
import torch

from dianome_server.model import PartialDecoder
from dianome_server.reference_client import ReferenceClient, reference_client

SPLITS = [0, 1, 8, 16, 23, 24]
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "bench", "results", "gate-max-abs-diff.json")


def _record(N: int, max_diff: float) -> None:
    """Persist the measured max_abs_diff so the notes can quote it (not typed by hand)."""
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    data = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            data = json.load(f)
    data[str(N)] = max_diff
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


@pytest.mark.parametrize("N", SPLITS)
def test_split_logits_match_full_forward(lm, prompt_ids, N):
    assert prompt_ids.shape[0] == 32
    ref = lm.full_forward(prompt_ids).float().cpu()
    boundary = reference_client(lm, prompt_ids, N)
    if N == 0:
        assert boundary.dtype == torch.int32 and boundary.shape == (32,)
    else:
        assert boundary.dtype == torch.float16 and boundary.shape == (32, lm.d_model)
    logits = PartialDecoder(lm, N).prefill(boundary).float().cpu()
    assert logits.shape == ref.shape
    max_diff = (logits - ref).abs().max().item()
    _record(N, max_diff)
    assert max_diff < 5e-2, f"N={N}: max_abs_diff={max_diff}"
    assert torch.equal(logits.argmax(-1), ref.argmax(-1)), f"N={N}: argmax differs"


@pytest.mark.parametrize("N", [0, 8, 24])
def test_decode_consistency_16_greedy(lm, prompt_ids, N):
    want = lm.full_greedy(prompt_ids, 16)
    client = ReferenceClient(lm, N)
    server = PartialDecoder(lm, N)
    logits = server.prefill(client.prefill(prompt_ids))
    tok = int(logits[-1].argmax().item())
    got = [tok]
    pos = prompt_ids.shape[0]
    for _ in range(15):
        tok = int(server.decode(client.decode(tok, pos), pos).argmax().item())
        got.append(tok)
        pos += 1
    assert got == want


@pytest.mark.parametrize("N", [1, 8, 23])
def test_kv_cache_lengths(lm, prompt_ids, N):
    client = ReferenceClient(lm, N)
    server = PartialDecoder(lm, N)
    T = prompt_ids.shape[0]
    logits = server.prefill(client.prefill(prompt_ids))
    tok = int(logits[-1].argmax().item())
    for i in range(3):
        tok = int(server.decode(client.decode(tok, T + i), T + i).argmax().item())
    lengths = server.cache_lengths()
    assert lengths[:N] == [0] * N, lengths
    assert lengths[N:] == [T + 3] * (lm.L - N), lengths
    # and the client cache is the mirror image
    assert client.cache_lengths()[:N] == [T + 3] * N
    assert client.cache_lengths()[N:] == [0] * (lm.L - N)

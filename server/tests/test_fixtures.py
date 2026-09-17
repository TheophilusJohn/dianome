import json
import os

import numpy as np

from dianome_server.fixtures import hashes, write_fixtures


def test_fixtures_reproducible_and_consistent(lm, tmp_path):
    a = write_fixtures(lm, str(tmp_path / "a"))
    b = write_fixtures(lm, str(tmp_path / "b"))
    assert hashes(a) == hashes(b)
    assert a["files"]["embed.npy"]["shape"] == [32, lm.d_model]
    assert a["files"]["logits.npy"]["shape"] == [32, lm.vocab]
    assert a["files"]["rope_cos.npy"]["shape"] == [32, a["head_dim"]]
    assert len([k for k in a["files"] if k.startswith("block_")]) == lm.L
    assert a["logits_max_abs_diff_vs_model_forward"] < 5e-2
    # block_23 output is boundary 24 = the input to the final norm
    last = np.load(tmp_path / "a" / f"block_{lm.L - 1:02d}.npy")
    assert last.dtype == np.float16 and last.shape == (32, lm.d_model)
    # the manifest hashes match the files on disk
    from dianome_server.fixtures import sha256_file
    for name, info in a["files"].items():
        assert sha256_file(str(tmp_path / "a" / name)) == info["sha256"]
    with open(tmp_path / "a" / "fixtures.json") as f:
        assert json.load(f)["files"] == a["files"]

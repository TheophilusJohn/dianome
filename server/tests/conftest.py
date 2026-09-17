import os

import pytest
import torch

from dianome_server.model import LoadedModel, pick_device

MODEL_ID = os.environ.get("DIANOME_TEST_MODEL", "qwen2.5-0.5b-instruct")

# Tokenizes to exactly 32 tokens with the Qwen2.5 tokenizer (checked in the fixture).
PROMPT_32 = (
    "The quick brown fox jumps over the lazy dog while the sun sets slowly behind "
    "the distant mountains, casting long shadows across the quiet valley below, where a small river"
)


@pytest.fixture(scope="session")
def lm() -> LoadedModel:
    return LoadedModel(MODEL_ID, device=pick_device())


@pytest.fixture(scope="session")
def prompt_ids(lm: LoadedModel) -> torch.Tensor:
    ids = lm.tokenizer(PROMPT_32, return_tensors="pt").input_ids[0]
    assert ids.shape[0] == 32, ids.shape
    return ids

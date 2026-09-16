"""R2 upload logic against a fake S3 client: skip-if-exists, headers, counts, env check."""
import pytest

from dianome_ingest import manifest as mf, r2
from dianome_ingest.chunker import LocalStore
from dianome_ingest.layout import pack_model


class _ClientError(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class FakeS3:
    def __init__(self):
        self.objects: dict[str, dict] = {}
        self.heads = 0

        class _Exc:
            ClientError = _ClientError

        self.exceptions = _Exc()

    def head_object(self, Bucket, Key):
        self.heads += 1
        if Key not in self.objects:
            raise _ClientError("404")
        return {}

    def put_object(self, Bucket, Key, Body, ContentType, CacheControl):
        self.objects[Key] = {"body": bytes(Body), "ct": ContentType, "cc": CacheControl}


@pytest.fixture
def packed_store(tmp_path, tiny_model):
    source, cfg = tiny_model
    store = LocalStore(tmp_path)
    packed = pack_model(source, cfg["num_hidden_layers"], cfg["tie_word_embeddings"], ("fp16", "q4"), store.sink())
    m = mf.build_model_manifest(id="tiny", repo="t/t", revision="r", family="qwen2", config=cfg, packed=packed, tokenizer_files=[])
    mf.save(store, m)
    return store, m


def test_upload_then_skip(packed_store):
    store, m = packed_store
    s3 = FakeS3()
    logs = []
    first = r2.upload_manifest(store, m, client=s3, bucket="b", log=logs.append)
    n = len(m["chunks"])
    assert first.uploaded == n + 2 and first.skipped == 0
    assert first.bytes == sum(c["bytes"] for c in m["chunks"].values()) + 2 * len(mf.canonical_json(m))
    h = mf.manifest_hash(m)
    for sha in m["chunks"]:
        o = s3.objects[f"chunks/{sha}"]
        assert o["ct"] == "application/octet-stream" and o["cc"] == "public, max-age=31536000, immutable"
        assert o["body"] == store.read_chunk(sha)
    assert s3.objects[f"manifests/tiny/{h}.json"]["cc"] == "public, max-age=31536000, immutable"
    assert s3.objects[f"manifests/tiny/{h}.json"]["ct"] == "application/json"
    assert s3.objects["manifests/tiny/latest.json"]["cc"] == "public, max-age=60"
    assert s3.objects["manifests/tiny/latest.json"]["body"] == mf.canonical_json(m)
    # Second run: every chunk and the hashed manifest are skipped; only latest.json is rewritten.
    second = r2.upload_manifest(store, m, client=s3, bucket="b", log=logs.append)
    assert second.skipped == n + 1 and second.uploaded == 1
    assert second.bytes == len(mf.canonical_json(m))
    assert "uploaded=1 skipped=%d" % (n + 1) in logs[-1]


def test_require_env_names_missing(monkeypatch):
    for k in r2.ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("R2_BUCKET", "x")
    with pytest.raises(r2.R2ConfigError) as ei:
        r2.require_env()
    msg = str(ei.value)
    assert "R2_ACCOUNT_ID" in msg and "R2_ACCESS_KEY_ID" in msg and "R2_SECRET_ACCESS_KEY" in msg
    assert "R2_BUCKET," not in msg  # the set one is not reported missing

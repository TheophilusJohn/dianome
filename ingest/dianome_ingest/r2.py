"""Cloudflare R2 upload over the S3 API: skip-if-exists, immutable cache headers, concurrency 8."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .chunker import LocalStore
from .manifest import canonical_json, manifest_hash

ENV_VARS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
CACHE_IMMUTABLE = "public, max-age=31536000, immutable"
CACHE_LATEST = "public, max-age=60"
CONCURRENCY = 8


class R2ConfigError(RuntimeError):
    pass


def require_env() -> dict[str, str]:
    """Fail early with a clear message if any R2 variable is unset."""
    missing = [k for k in ENV_VARS if not os.environ.get(k)]
    if missing:
        raise R2ConfigError(
            "R2 upload requested but these environment variables are unset: "
            + ", ".join(missing)
            + ". Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY and R2_BUCKET."
        )
    return {k: os.environ[k] for k in ENV_VARS}


def make_client(env: dict[str, str]):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=env["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(max_pool_connections=CONCURRENCY * 2, retries={"max_attempts": 5, "mode": "standard"}),
    )


@dataclass
class UploadStats:
    uploaded: int = 0
    skipped: int = 0
    bytes: int = 0


def _exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except client.exceptions.ClientError as e:  # type: ignore[attr-defined]
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def upload_manifest(store: LocalStore, manifest: dict, client=None, bucket: str | None = None, log=print) -> UploadStats:
    """Upload every chunk the manifest references plus both manifest keys."""
    if client is None:
        env = require_env()
        client, bucket = make_client(env), env["R2_BUCKET"]
    assert bucket
    stats = UploadStats()
    mid = manifest["id"]

    def put(key: str, body: bytes, content_type: str, cache: str, skip_if_exists: bool) -> None:
        if skip_if_exists and _exists(client, bucket, key):
            stats.skipped += 1
            return
        client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type, CacheControl=cache)
        stats.uploaded += 1
        stats.bytes += len(body)

    def put_chunk(sha: str) -> None:
        put(f"chunks/{sha}", store.read_chunk(sha), "application/octet-stream", CACHE_IMMUTABLE, True)

    shas = sorted(manifest["chunks"])
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for i, _ in enumerate(ex.map(put_chunk, shas), 1):
            if i % 50 == 0 or i == len(shas):
                log(f"  chunks {i}/{len(shas)}  uploaded={stats.uploaded} skipped={stats.skipped}")
    data = canonical_json(manifest)
    h = manifest_hash(manifest)
    put(f"manifests/{mid}/{h}.json", data, "application/json", CACHE_IMMUTABLE, True)
    put(f"manifests/{mid}/latest.json", data, "application/json", CACHE_LATEST, False)
    log(f"upload done: uploaded={stats.uploaded} skipped={stats.skipped} bytes={stats.bytes}")
    return stats

"""`dianome-ingest` command line: pack-model, pack-dir, verify, inspect, upload."""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import click

from . import hf, manifest as mf, r2
from .chunker import CHUNK_SIZE, LocalStore
from .layout import VARIANTS, pack_model
from .quant import to_fp16


def _mib(n: int) -> str:
    return f"{n / (1024 * 1024):,.1f} MiB"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _variant_order(m: dict) -> list[str]:
    return [v for v in VARIANTS if v in m["variants"]]


def _err(msg: str) -> None:
    click.echo(msg, err=True)


def _require_upload_env(upload: bool) -> None:
    if upload:
        try:
            r2.require_env()
        except r2.R2ConfigError as e:
            raise click.ClickException(str(e))


@click.group()
def main() -> None:
    """Dianome Phase 1 ingest: safetensors -> fp16/q8/q4 layer groups -> 8 MB chunks -> manifest -> R2."""


@main.command("pack-model")
@click.option("--repo", required=True, help="Hugging Face repo, e.g. Qwen/Qwen2.5-0.5B-Instruct")
@click.option("--id", "artifact_id", required=True)
@click.option("--variants", default="fp16,q8,q4", show_default=True)
@click.option("--out", default="./store", show_default=True, type=click.Path(file_okay=False))
@click.option("--revision", default=None, help="HF commit sha (default: main)")
@click.option("--upload", is_flag=True, help="Upload to R2 after packing (needs R2_* env vars)")
def pack_model_cmd(repo: str, artifact_id: str, variants: str, out: str, revision: str | None, upload: bool) -> None:
    """Pack a safetensors model into per-layer fp16/q8/q4 streams and write the manifest."""
    _require_upload_env(upload)
    vs = tuple(v.strip() for v in variants.split(",") if v.strip())
    if not vs or any(v not in VARIANTS for v in vs):
        raise click.ClickException(f"--variants must be a non-empty subset of {VARIANTS}; got {vs}")
    t0 = time.time()
    snapshot, sha = hf.resolve_snapshot(repo, revision)
    cfg, summary = hf.read_config(snapshot)
    family = cfg.get("model_type", "unknown")
    num_layers, tied = summary["num_hidden_layers"], summary["tie_word_embeddings"]
    _err(f"source {repo}@{sha}  family={family} layers={num_layers} tied={tied}  variants={','.join(vs)}")
    store = LocalStore(out)
    source = hf.SafetensorsSource(snapshot)
    n_total = len(source.names())
    count = [0]

    def progress(group: str, name: str) -> None:
        count[0] += 1
        if count[0] % 20 == 0 or count[0] == n_total:
            _err(f"  [{count[0]}/{n_total}] {group} {name}")

    try:
        packed = pack_model(source, num_layers, tied, vs, store.sink(), progress)
    except ValueError as e:
        raise click.ClickException(str(e))
    tok = [mf.pack_file_stream(p, store.sink()) for p in hf.tokenizer_files(snapshot)]
    m = mf.build_model_manifest(id=artifact_id, repo=repo, revision=sha, family=family, config=summary, packed=packed, tokenizer_files=tok)
    latest, hashed, h = mf.save(store, m)
    wall = time.time() - t0
    click.echo(f"pack-model {artifact_id}")
    click.echo(f"  source      {repo} @ {sha}")
    click.echo(f"  git commit  {_git_commit()}")
    click.echo(f"  manifest    {latest}  sha256={h}")
    click.echo(f"  chunk table {len(m['chunks'])} unique chunks, {_mib(sum(c['bytes'] for c in m['chunks'].values()))}")
    for v, variant in m["variants"].items():
        n_chunks = sum(len(g["chunks"]) for g in variant["groups"])
        click.echo(f"  variant {v:<5} bytes={variant['bytes']:>13,} ({_mib(variant['bytes'])})  chunks={n_chunks}")
    click.echo(f"  tokenizer   {len(tok)} files, {sum(f['bytes'] for f in tok):,} bytes")
    click.echo(f"  wall time   {wall:.1f} s")
    if upload:
        r2.upload_manifest(store, m, log=click.echo)


@main.command("pack-dir")
@click.option("--dir", "directory", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--id", "artifact_id", required=True)
@click.option("--runtime", required=True, type=click.Choice(["webllm", "transformersjs", "other"]))
@click.option("--out", default="./store", show_default=True, type=click.Path(file_okay=False))
@click.option("--repo", default=None, help="Source repo to record (default: local:<dir name>)")
@click.option("--revision", default="", help="Source revision to record")
@click.option("--upload", is_flag=True)
def pack_dir_cmd(directory: str, artifact_id: str, runtime: str, out: str, repo: str | None, revision: str, upload: bool) -> None:
    """Pack every file under a directory as one stream each (ONNX graphs, MLC shards)."""
    _require_upload_env(upload)
    t0 = time.time()
    root = Path(directory)
    store = LocalStore(out)
    seen: set[str] = set()
    files = []
    paths = sorted(p for p in root.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts))
    for p in paths:
        rel = p.relative_to(root).as_posix()
        files.append(mf.pack_file_stream(p, store.sink(), seen, name=rel))
        _err(f"  {rel}  {files[-1]['bytes']:,} bytes  {len(files[-1]['chunks'])} chunks")
    m = mf.build_files_manifest(id=artifact_id, repo=repo or f"local:{root.name}", revision=revision, runtime=runtime, files=files)
    latest, hashed, h = mf.save(store, m)
    click.echo(f"pack-dir {artifact_id}  runtime={runtime}")
    click.echo(f"  source      {m['source']['repo']} @ {revision or '(none)'}")
    click.echo(f"  manifest    {latest}  sha256={h}")
    click.echo(f"  files       {len(files)}, {sum(f['bytes'] for f in files):,} bytes, {sum(len(f['chunks']) for f in files)} chunks listed, {len(m['chunks'])} unique")
    click.echo(f"  wall time   {time.time() - t0:.1f} s")
    if upload:
        r2.upload_manifest(store, m, log=click.echo)


@main.command("verify")
@click.option("--store", "store_dir", default="./store", show_default=True, type=click.Path(exists=True, file_okay=False))
@click.option("--id", "artifact_id", required=True)
@click.option("--against-hf", is_flag=True, help="Reconstruct every tensor and compare with the HF source")
def verify_cmd(store_dir: str, artifact_id: str, against_hf: bool) -> None:
    """Check hashes, byte sums, schema, layout; optionally reconstruct against the source."""
    store = LocalStore(store_dir)
    try:
        m = mf.load(store, artifact_id)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    report = mf.Report()
    mf.verify_store(store, m, report)
    for name, ok, detail in report.checks:
        click.echo(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
    if not report.ok:
        raise click.ClickException("verify failed")
    if not against_hf:
        click.echo("verify: ok")
        return
    if "variants" not in m:
        raise click.ClickException("--against-hf applies to model manifests only")

    snapshot, sha = hf.resolve_snapshot(m["source"]["repo"], m["source"]["revision"])
    if sha != m["source"]["revision"]:
        raise click.ClickException(f"resolved revision {sha} != manifest revision {m['source']['revision']}")
    source = hf.SafetensorsSource(snapshot)
    reader = mf.ChunkReader(store)
    checks: list[mf.EntryCheck] = []
    fails = 0
    click.echo(f"against-hf: {m['source']['repo']} @ {sha}")
    click.echo(f"  {'variant':<7} {'entry':<48} {'kind':<5} {'result':<12} {'max_abs':>10} {'rel_rms':>10}")
    for vname in _variant_order(m):
        variant = m["variants"][vname]
        for g in variant["groups"]:
            for e in g["entries"]:
                src_name = "model.embed_tokens.weight" if e.get("tied") else e["name"]
                w16 = to_fp16(source.get(src_name))
                data = reader.entry_bytes(e)
                c = mf.check_entry_against_reference(vname, g["name"], e, data, w16)
                checks.append(c)
                if c.kind == "fp16":
                    res = "exact" if c.exact else "MISMATCH"
                    ok = bool(c.exact)
                    nums = f"{'':>10} {'':>10}"
                else:
                    res = "consistent" if c.consistent else "INCONSISTENT"
                    ok = bool(c.consistent)
                    nums = f"{c.max_abs:>10.4g} {c.rel_rms:>10.4g}"
                fails += 0 if ok else 1
                click.echo(f"  {vname:<7} {e['name']:<48} {c.kind:<5} {res:<12} {nums}")
    click.echo("")
    click.echo("per-role quantization error (aggregated over layers: min / median / max)")
    click.echo(f"  {'variant':<7} {'role':<34} {'n':>3}  {'max_abs min':>11} {'median':>10} {'max':>10}   {'rel_rms min':>11} {'median':>10} {'max':>10}")
    for r in mf.aggregate_by_role(checks):
        a, b = r["max_abs"], r["rel_rms"]
        click.echo(f"  {r['variant']:<7} {r['role']:<34} {r['n']:>3}  {a[0]:>11.4g} {a[1]:>10.4g} {a[2]:>10.4g}   {b[0]:>11.4g} {b[1]:>10.4g} {b[2]:>10.4g}")
    n_fp16 = sum(1 for c in checks if c.kind == "fp16")
    n_exact = sum(1 for c in checks if c.exact)
    n_q = sum(1 for c in checks if c.kind != "fp16")
    n_cons = sum(1 for c in checks if c.consistent)
    click.echo("")
    click.echo(f"fp16 entries bit-equal to source: {n_exact}/{n_fp16}; quantized entries encoder/decoder-consistent: {n_cons}/{n_q}; entries checked: {len(checks)}")
    if fails:
        raise click.ClickException(f"verify --against-hf failed: {fails} entries")
    click.echo("verify --against-hf: ok")


def inspect_summary(m: dict) -> dict:
    """Machine-readable sizes (what `inspect --json` writes): per variant bytes and chunk counts, dedup, tokenizer."""
    table = m["chunks"]
    out: dict = {
        "id": m["id"], "source": m["source"], "manifest_sha256": mf.manifest_hash(m), "chunk_size": m["chunk_size"],
        "unique_chunks": len(table), "unique_bytes": sum(c["bytes"] for c in table.values()),
    }
    if "files" in m:
        out["runtime"] = m["runtime"]
        out["files"] = [{"name": f["name"], "bytes": f["bytes"], "chunks": len(f["chunks"])} for f in m["files"]]
        return out
    out["family"], out["layers"], out["tied"] = m["family"], m["config"]["num_hidden_layers"], m["config"]["tie_word_embeddings"]
    out["variants"] = {}
    for vname in _variant_order(m):
        variant = m["variants"][vname]
        listed = [c for g in variant["groups"] for c in g["chunks"]]
        out["variants"][vname] = {"bytes": variant["bytes"], "chunks_listed": len(listed), "chunks_unique": len(set(listed)),
                                  "groups": len(variant["groups"])}
    d = mf.dedup_stats(m)
    out["dedup"] = {k: d[k] for k in ("listed_chunks", "listed_bytes", "unique_chunks", "unique_bytes", "shared_chunks", "shared_bytes", "bytes_saved")}
    out["tokenizer"] = {"files": len(m["tokenizer"]["files"]), "bytes": sum(f["bytes"] for f in m["tokenizer"]["files"])}
    return out


@main.command("inspect")
@click.option("--store", "store_dir", default="./store", show_default=True, type=click.Path(exists=True, file_okay=False))
@click.option("--id", "artifact_id", required=True)
@click.option("--json", "json_out", default=None, type=click.Path(dir_okay=False), help="also write the sizes as JSON to this path")
def inspect_cmd(store_dir: str, artifact_id: str, json_out: str | None) -> None:
    """Sizes per variant/group, dedup stats, chunk counts."""
    store = LocalStore(store_dir)
    try:
        m = mf.load(store, artifact_id)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps({**inspect_summary(m), "git_commit": _git_commit(), "date": dt.date.today().isoformat()}, indent=2) + "\n")
        _err(f"wrote {json_out}")
    table = m["chunks"]
    click.echo(f"inspect {m['id']}  source={m['source']['repo']}@{m['source']['revision']}  manifest sha256={mf.manifest_hash(m)}")
    click.echo(f"  chunk table: {len(table)} unique chunks, {sum(c['bytes'] for c in table.values()):,} bytes ({_mib(sum(c['bytes'] for c in table.values()))}), chunk_size={m['chunk_size']}")
    if "files" in m:
        click.echo(f"  runtime={m['runtime']}  files={len(m['files'])}")
        click.echo(f"  {'file':<60} {'bytes':>14} {'chunks':>7}")
        for f in m["files"]:
            click.echo(f"  {f['name']:<60} {f['bytes']:>14,} {len(f['chunks']):>7}")
        return
    click.echo(f"  family={m['family']}  layers={m['config']['num_hidden_layers']}  tied={m['config']['tie_word_embeddings']}")
    for vname in _variant_order(m):
        variant = m["variants"][vname]
        listed = [c for g in variant["groups"] for c in g["chunks"]]
        click.echo("")
        click.echo(f"variant {vname}: bytes={variant['bytes']:,} ({_mib(variant['bytes'])})  chunks={len(listed)} listed, {len(set(listed))} unique")
        click.echo(f"  {'group':<12} {'bytes':>13} {'chunks':>7} {'entries':>8}")
        for g in variant["groups"]:
            tied = "  (tied -> embed)" if g.get("tied") else ""
            click.echo(f"  {g['name']:<12} {g['bytes']:>13,} {len(g['chunks']):>7} {len(g['entries']):>8}{tied}")
    d = mf.dedup_stats(m)
    click.echo("")
    click.echo("dedup across variants")
    click.echo(f"  chunks listed across variants: {d['listed_chunks']} ({d['listed_bytes']:,} bytes)")
    click.echo(f"  unique chunks:                 {d['unique_chunks']} ({d['unique_bytes']:,} bytes)")
    click.echo(f"  chunks shared by >1 variant:   {d['shared_chunks']} ({d['shared_bytes']:,} bytes)")
    click.echo(f"  bytes saved by dedup:          {d['bytes_saved']:,} ({_mib(d['bytes_saved'])})")
    for pair, (n, b) in d["pairs"].items():
        click.echo(f"  {pair:<12} shared chunks={n} bytes={b:,}")
    click.echo("  groups whose chunk lists are identical across variants:")
    for name, vs, n, b in d["shared_groups"]:
        click.echo(f"    {name:<12} {' = '.join(vs)}  ({n} chunks, {b:,} bytes)")
    if not d["shared_groups"]:
        click.echo("    (none)")
    embed = {v: m["variants"][v]["groups"][0]["chunks"] for v in _variant_order(m)}
    click.echo("  embed group chunk ids (first 3 of each variant):")
    for v, cs in embed.items():
        click.echo(f"    {v:<5} n={len(cs)}  " + " ".join(c[:12] for c in cs[:3]) + (" …" if len(cs) > 3 else ""))
    tok = m["tokenizer"]["files"]
    click.echo(f"  tokenizer files: " + ", ".join(f"{f['name']} ({f['bytes']:,} B, {len(f['chunks'])} chunks)" for f in tok))


@main.command("upload")
@click.option("--store", "store_dir", default="./store", show_default=True, type=click.Path(exists=True, file_okay=False))
@click.option("--id", "artifact_id", required=True)
@click.option("--json", "json_out", default=None, type=click.Path(dir_okay=False), help="write the upload stats as JSON to this path (written only on success)")
def upload_cmd(store_dir: str, artifact_id: str, json_out: str | None) -> None:
    """Upload an artifact's chunks and manifest to R2 (skips chunks already present)."""
    _require_upload_env(True)
    store = LocalStore(store_dir)
    try:
        m = mf.load(store, artifact_id)
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    t0 = time.time()
    stats = r2.upload_manifest(store, m, log=click.echo)
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps({
            "id": m["id"], "manifest_sha256": mf.manifest_hash(m), "bucket": r2.require_env()["R2_BUCKET"],
            "variants": _variant_order(m) if "variants" in m else None,
            "uploaded": stats.uploaded, "skipped": stats.skipped, "bytes_uploaded": stats.bytes,
            "referenced_chunks": len(m["chunks"]), "referenced_bytes": sum(c["bytes"] for c in m["chunks"].values()),
            "wall_seconds": round(time.time() - t0, 1), "git_commit": _git_commit(), "date": dt.date.today().isoformat(),
        }, indent=2) + "\n")
        _err(f"wrote {json_out}")


if __name__ == "__main__":
    main()

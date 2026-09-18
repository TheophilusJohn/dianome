"""dianome-server serve|bench|probes|fixtures"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import click

from .model import LoadedModel, device_name, pick_device, versions


@click.group()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _load(model: str, device: str | None) -> LoadedModel:
    lm = LoadedModel(model, device=pick_device(device))
    click.echo(
        f"loaded {lm.id} ({lm.repo}@{lm.revision}) L={lm.L} d_model={lm.d_model} "
        f"device={device_name(lm.device)} in {lm.load_seconds:.1f}s  {versions()}",
        err=True,
    )
    return lm


@main.command()
@click.option("--model", default="qwen2.5-0.5b-instruct", show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, show_default=True)
@click.option("--device", default=None, help="override cuda/mps/cpu autodetection")
@click.option("--no-microbench", is_flag=True, help="skip the startup microbench (/plan reports null timings)")
def serve(model: str, host: str, port: int, device: str | None, no_microbench: bool) -> None:
    """Run the split-inference WebSocket server (needs SPLIT_TOKEN; SPLIT_SIGNING_KEY enables session tokens)."""
    from .microbench import run_microbench
    from .ws import SplitServer

    if not os.environ.get("SPLIT_TOKEN"):
        click.echo("SPLIT_TOKEN is not set; refusing to start", err=True)
        sys.exit(2)
    lm = _load(model, device)
    mb = None
    if not no_microbench:
        mb = run_microbench(lm)
        click.echo(f"microbench (median of {mb.runs}): decode {mb.ms_per_block_decode:.3f} ms/block, "
                   f"prefill {mb.ms_per_block_prefill:.3f} ms/block at T={mb.prefill_T}, lm_head {mb.lm_head_ms:.2f} ms", err=True)
    asyncio.run(SplitServer(lm, microbench=mb).serve_forever(host, port))


@main.command()
@click.option("--model", default="qwen2.5-0.5b-instruct", show_default=True)
@click.option("--out", default="fixtures/qwen2.5-0.5b-instruct/", show_default=True)
@click.option("--device", default=None)
@click.option("--intra", is_flag=True, help="also dump every stage of block 0 (intra/*.npy) for gate 3")
@click.option("--variant", "variants", multiple=True, type=click.Choice(["q8", "q4"]),
              help="also dump dequantised references for this variant (repeatable) for gate 7")
@click.option("--store", default="store", show_default=True, help="chunk store the --variant entries are read from")
@click.option("--intra-blocks", default="0", show_default=True, help="comma list of blocks to dump with --intra (block 0 -> intra/, others -> intra_<i>/)")
def fixtures(model: str, out: str, device: str | None, intra: bool, variants: tuple[str, ...], store: str, intra_blocks: str) -> None:
    """Write the Phase 5a reference activations + fixtures.json with hashes.

    Always: embed/block/final_norm/logits/rope .npy, tokenizer_cases.json (200 strings), greedy.json +
    decode_steps.npy (16 greedy decode steps at every boundary). --intra and --variant add more.
    """
    from .fixtures import write_fixtures_extended

    lm = _load(model, device)
    m = write_fixtures_extended(lm, out, intra=intra, variants=variants, store=store, intra_blocks=[int(b) for b in intra_blocks.split(",")])
    for name, info in m["files"].items():
        click.echo(f"  {info['sha256']}  {name}  {info.get('shape', '')}")
    click.echo(f"fixtures {model} -> {out}  T={m['T']} L={m['L']} d_model={m['d_model']}  "
               f"logits max_abs_diff vs model.forward = {m['logits_max_abs_diff_vs_model_forward']}")


@main.command()
@click.option("--model", default="qwen2.5-0.5b-instruct", show_default=True)
@click.option("--splits", default="0,4,8,12,16,20,24", show_default=True)
@click.option("--gen-tokens", default=128, show_default=True)
@click.option("--runs", default=3, show_default=True)
@click.option("--prompts", default=None, help="prompts.json (default: bench/prompts.json next to the package)")
@click.option("--rates", default=None, help="rates.json (default: bench/rates.json)")
@click.option("--out-dir", default=None, help="results dir (default: bench/results)")
@click.option("--device", default=None)
def bench(model, splits, gen_tokens, runs, prompts, rates, out_dir, device) -> None:
    """Cost harness: GPU-seconds per token as a function of the split point."""
    from bench.cost import run_bench

    lm = _load(model, device)
    run_bench(lm, [int(s) for s in splits.split(",")], gen_tokens=gen_tokens, runs=runs,
              prompts_path=prompts, rates_path=rates, out_dir=out_dir)


@main.command()
@click.option("--model", default="qwen2.5-0.5b-instruct", show_default=True)
@click.option("--tokens", default=50_000, show_default=True, help="held-out tokens to collect")
@click.option("--data-dir", default=None, help="activation store (default: probes/data)")
@click.option("--boundaries", default=None, help="comma list; default all 0..L")
@click.option("--skip-collect", is_flag=True, help="reuse activations already on disk")
@click.option("--notes", default=None, help="docs/phase-4-notes.md to update (default: none)")
@click.option("--device", default=None)
def probes(model, tokens, data_dir, boundaries, skip_collect, notes, device) -> None:
    """Privacy band: nearest-neighbour, linear probe and inversion decoder per boundary."""
    from probes.run import run_probes

    lm = _load(model, device)
    bl = [int(b) for b in boundaries.split(",")] if boundaries else None
    run_probes(lm, n_tokens=tokens, data_dir=data_dir, boundaries=bl, skip_collect=skip_collect, notes=notes)


if __name__ == "__main__":
    main()


@main.command("linear-probe")
@click.option("--model", default="qwen2.5-0.5b-instruct", show_default=True)
@click.option("--train-tokens", default=500_000, show_default=True)
@click.option("--seed", default=1, show_default=True)
@click.option("--boundaries", default=None, help="comma list; default all 0..L")
@click.option("--max-epochs", default=5, show_default=True)
@click.option("--batch", default=1024, show_default=True)
@click.option("--notes", default=None, help="docs/phase-4-notes.md to update (default: none)")
@click.option("--device", default=None)
def linear_probe(model, train_tokens, seed, boundaries, max_epochs, batch, notes, device) -> None:
    """Linear probe with a large training set from the WikiText-103 train split; held-out set unchanged."""
    from probes.linear500k import run_linear500k

    lm = _load(model, device)
    bl = [int(b) for b in boundaries.split(",")] if boundaries else None
    run_linear500k(lm, target_tokens=train_tokens, seed=seed, boundaries=bl, notes=notes,
                   linear_kw={"max_epochs": max_epochs, "batch": batch})

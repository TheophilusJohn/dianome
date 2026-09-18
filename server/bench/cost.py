"""Cost harness: gpu_seconds_per_token(N) = server_busy_seconds(N) / tokens_generated.

For each split N and each prompt: run the reference client (untimed; it stands in
for the browser), then prefill + `gen_tokens` greedy decode steps through the
server half in-process, timing only server execution with a device sync before
the clock stops. Three runs, median per N. Batch size 1.

cost_per_1M_tokens(N) = gpu_seconds_per_token(N) * 1e6 * hourly_rate / 3600,
with hourly_rate from bench/rates.json (never hardcoded). When rates.json has
no rate yet the cost column is left empty; the ratio cost(N)/cost(0) does not
depend on the rate and is always reported.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import statistics
import time
from typing import Optional

import torch

from dianome_server.model import LoadedModel, PartialDecoder, device_name, synchronize, versions
from dianome_server.reference_client import ReferenceClient

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS = ["N", "prompt_tokens", "gen_tokens", "busy_seconds", "gpu_seconds_per_token",
           "cost_per_1M_tokens", "hourly_rate", "gpu",
           # extra, after the brief's columns: where the busy time went
           "prefill_seconds", "decode_seconds_per_token", "runs", "device"]


def split_grid(L: int, steps: int = 6) -> list[int]:
    """Phase 4's N grid scaled to L: N in {0, L/6, L/3, L/2, 2L/3, 5L/6, L}, rounded half up, deduplicated.

    L=24 -> [0, 4, 8, 12, 16, 20, 24]; L=36 -> [0, 6, 12, 18, 24, 30, 36]; L=28 -> [0, 5, 9, 14, 19, 23, 28].
    """
    return sorted({int(L * k / steps + 0.5) for k in range(steps + 1)})


def load_rates(path: Optional[str]) -> dict:
    with open(path or os.path.join(HERE, "rates.json")) as f:
        r = json.load(f)
    if r.get("usd_per_hour") is None:
        print("WARNING: bench/rates.json has no usd_per_hour; cost_per_1M_tokens will be empty (fill it for the GPU run)")
    return r


def load_prompts(path: Optional[str]) -> list[dict]:
    with open(path or os.path.join(HERE, "prompts.json")) as f:
        return json.load(f)["prompts"]


@torch.no_grad()
def time_split(lm: LoadedModel, N: int, ids: torch.Tensor, gen_tokens: int) -> dict:
    """One prompt through split N: returns busy seconds (prefill + decode) and token counts."""
    client = ReferenceClient(lm, N)
    server = PartialDecoder(lm, N)
    T = ids.shape[0]
    boundary = client.prefill(ids)  # untimed: the browser's half
    synchronize(lm.device)

    t0 = time.perf_counter()
    logits = server.prefill(boundary)
    tok = int(torch.argmax(logits[-1]).item())
    synchronize(lm.device)
    prefill_s = time.perf_counter() - t0

    decode_s = 0.0
    pos = T
    for _ in range(gen_tokens):
        x = client.decode(tok, pos)  # untimed
        synchronize(lm.device)
        t0 = time.perf_counter()
        tok = int(torch.argmax(server.decode(x, pos)).item())
        synchronize(lm.device)
        decode_s += time.perf_counter() - t0
        pos += 1
    generated = 1 + gen_tokens  # prefill yields the first token, then gen_tokens decode steps
    return {"prompt_tokens": T, "gen_tokens": generated, "prefill_seconds": prefill_s,
            "decode_seconds": decode_s, "busy_seconds": prefill_s + decode_s}


def run_bench(lm: LoadedModel, splits: list[int], gen_tokens: int = 128, runs: int = 3,
              prompts_path: Optional[str] = None, rates_path: Optional[str] = None,
              out_dir: Optional[str] = None) -> dict:
    rates = load_rates(rates_path)
    prompts = load_prompts(prompts_path)
    out_dir = out_dir or os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    dev = device_name(lm.device)
    rate = rates.get("usd_per_hour")
    gpu = rates.get("gpu", "")
    if lm.device.type != "cuda":
        gpu = f"{dev} [NOT A GPU RATE; rates.json gpu={gpu!r}]"
    print(f"cost harness: model={lm.id} device={dev} splits={splits} gen_tokens={gen_tokens} runs={runs} "
          f"prompts={len(prompts)} {versions()}")
    if lm.device.type != "cuda":
        print(f"NOTE: absolute seconds below are for {dev}; they are not GPU numbers. Phase 7 re-runs this on a rented GPU.")

    ids_list = [lm.tokenizer(p["text"], return_tensors="pt").input_ids[0] for p in prompts]
    for p, ids in zip(prompts, ids_list):
        assert ids.shape[0] == p["tokens"], (p["id"], ids.shape[0], p["tokens"])

    # warm-up: MPS/CUDA compile kernels lazily per shape, so touch every prompt length
    # and every decode position once before the clock matters
    for ids in ids_list:
        time_split(lm, splits[0], ids, gen_tokens)

    date = dt.date.today().isoformat()
    detail_rows: list[dict] = []
    per_n: dict[int, dict] = {}
    for N in splits:
        run_gspt: list[float] = []
        run_busy: list[float] = []
        run_prefill: list[float] = []
        run_decode_pt: list[float] = []
        total_prompt = total_gen = 0
        for r in range(runs):
            busy = prefill = decode = 0.0
            gen = 0
            total_prompt = 0
            for p, ids in zip(prompts, ids_list):
                m = time_split(lm, N, ids, gen_tokens)
                detail_rows.append({"N": N, "run": r, "prompt": p["id"], **m})
                busy += m["busy_seconds"]; prefill += m["prefill_seconds"]; decode += m["decode_seconds"]
                gen += m["gen_tokens"]; total_prompt += m["prompt_tokens"]
            total_gen = gen
            run_busy.append(busy); run_gspt.append(busy / gen); run_prefill.append(prefill)
            run_decode_pt.append(decode / (gen - len(prompts)))
            print(f"  N={N:2d} run {r}: busy={busy:.3f}s gen={gen} -> {busy / gen * 1e3:.3f} ms/token "
                  f"(prefill {prefill:.3f}s, decode {decode / (gen - len(prompts)) * 1e3:.3f} ms/token)")
        gspt = statistics.median(run_gspt)
        cost = gspt * 1e6 * rate / 3600 if rate is not None else None
        per_n[N] = {
            "N": N, "prompt_tokens": total_prompt, "gen_tokens": total_gen,
            "busy_seconds": statistics.median(run_busy), "gpu_seconds_per_token": gspt,
            "cost_per_1M_tokens": cost, "hourly_rate": rate, "gpu": gpu,
            "prefill_seconds": statistics.median(run_prefill),
            "decode_seconds_per_token": statistics.median(run_decode_pt), "runs": runs, "device": dev,
        }

    base = per_n[splits[0]]["gpu_seconds_per_token"]
    for row in per_n.values():
        row["ratio_vs_N0"] = row["gpu_seconds_per_token"] / base if base else None

    stem = f"{lm.device.type}-{lm.id}-{date}"
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS + ["ratio_vs_N0"])
        w.writeheader()
        for row in per_n.values():
            w.writerow({k: ("" if row[k] is None else row[k]) for k in w.fieldnames})
    with open(os.path.join(out_dir, f"{stem}-detail.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader(); w.writerows(detail_rows)

    md = markdown_table(per_n, dev, rate, gpu, gen_tokens, runs, len(prompts), total_prompt)
    md_path = os.path.join(out_dir, f"{stem}.md")
    with open(md_path, "w") as f:
        f.write(md)
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
        json.dump({"device": dev, "model": lm.id, "date": date, "versions": versions(), "rates": rates,
                   "gen_tokens": gen_tokens, "runs": runs, "rows": list(per_n.values())}, f, indent=2)
    print(md)
    print(f"wrote {csv_path} and {md_path}")
    return per_n


def markdown_table(per_n: dict[int, dict], dev: str, rate, gpu: str, gen_tokens: int, runs: int,
                   n_prompts: int, prompt_tokens: int) -> str:
    lines = [
        f"Device: **{dev}**. Batch size 1, {n_prompts} prompts "
        f"({prompt_tokens} prompt tokens total), prefill + {gen_tokens} greedy decode steps each, "
        f"{runs} runs, median. hourly_rate = {rate!r} ({gpu}).",
        "",
        "| N | prompt_tokens | gen_tokens | busy_seconds | gpu_seconds_per_token | ms/token | prefill_s | decode ms/token | cost_per_1M_tokens | cost(N)/cost(0) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in per_n.values():
        cost = f"{r['cost_per_1M_tokens']:.4f}" if r["cost_per_1M_tokens"] is not None else "n/a (no rate)"
        lines.append(
            f"| {r['N']} | {r['prompt_tokens']} | {r['gen_tokens']} | {r['busy_seconds']:.3f} | "
            f"{r['gpu_seconds_per_token']:.6f} | {r['gpu_seconds_per_token'] * 1e3:.3f} | {r['prefill_seconds']:.3f} | "
            f"{r['decode_seconds_per_token'] * 1e3:.3f} | {cost} | {r['ratio_vs_N0']:.3f} |"
        )
    return "\n".join(lines) + "\n"

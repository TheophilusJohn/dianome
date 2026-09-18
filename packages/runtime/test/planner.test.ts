import { describe, expect, it } from "vitest";
import { cost0Per1M, feasibleN, lmHeadBlockEquivalents, plan, serverShare, type PlanInput, type PrivacyBand } from "../src/planner";

const L = 24;
const band: PrivacyBand = {
  model: "m", coverage500k: 0.9728,
  rows: Array.from({ length: L + 1 }, (_, b) => ({ boundary: b, linear500k_top1: 1 - b * 0.008, linear500k_top5: 1, linear_top1: 0.8 - b * 0.01, linear_top5: 0.82, inversion_top1: 0.82 - b * 0.01, inversion_top5: 0.82, nn_top1: b === 0 ? 1 : 0 })),
};
const MiB = 2 ** 20;

interface Over { model?: PlanInput["model"]; device?: Partial<PlanInput["device"]>; network?: Partial<PlanInput["network"]>; policy?: Partial<PlanInput["policy"]>; server?: PlanInput["server"]; rate?: PlanInput["rate"]; prompt?: PlanInput["prompt"] }
function input(over: Over = {}): PlanInput {
  const base: PlanInput = {
    model: { id: "m", variant: "q4", L, dModel: 896, blockBytes: Array(L).fill(7.5 * MiB), embedBytes: 130 * MiB, lmHeadGpuBytes: 130 * MiB, lmHeadDownloadBytes: 0, kvBytesPerBlock: 0.5 * MiB, workspaceBytes: 40 * MiB, maxEntryBytes: 130 * MiB, serverCostFloor: 0.224, privacy: band },
    device: { webgpu: true, maxBufferSize: 1024 * MiB, gpuBudgetBytes: 2048 * MiB, msPerBlockT1: 0.7, msPerBlockT32: 1.9, lmHeadMs: 3.5, exportMs: 0.4, cachedBlocks: 0, embedCached: false },
    network: { bytesPerSecond: 50e6, rttMs: 4, serverReachable: true },
    server: { L, busyFraction60s: 0.1, msPerBlockDecode: 0.65, msPerBlockPrefill: 0.9, lmHeadMs: 3.0 },
    policy: { prefer: "cost" },
    prompt: { promptTokens: 32, maxNewTokens: 64 },
    rate: { gpu: "test", usdPerHour: 3.6, source: "x", retrieved: "2026-09-17" },
  };
  return {
    model: over.model ?? base.model, prompt: over.prompt ?? base.prompt, rate: ("rate" in over ? over.rate : base.rate) ?? null,
    device: { ...base.device, ...(over.device ?? {}) }, network: { ...base.network, ...(over.network ?? {}) }, policy: { ...base.policy, ...(over.policy ?? {}) },
    server: over.server === undefined ? base.server : over.server,
  };
}

describe("planner", () => {
  it("lists every candidate 0..L plus local, in order, in the output", () => {
    const p = plan(input());
    expect(p.candidates.map((c) => `${c.mode}${c.N}`)).toEqual(["server0", ...Array.from({ length: L }, (_, i) => `split${i + 1}`), `local${L}`]);
    expect(p.inputs).toBeDefined();
    for (const c of p.candidates) expect(typeof c.serverShare).toBe("number");
  });

  it("server share uses the measured lm_head floor and is 0 for local", () => {
    const h = lmHeadBlockEquivalents(L, 0.224);
    expect(h / (L + h)).toBeCloseTo(0.224, 6);
    expect(serverShare(0, L, 0.224, false)).toBe(1);
    expect(serverShare(L, L, 0.224, false)).toBeCloseTo(0.224, 6);
    expect(serverShare(L, L, 0.224, true)).toBe(0);
    const c = plan(input()).candidates;
    expect(c[12]!.serverShare).toBeGreaterThan(c[24]!.serverShare);
  });

  it("cost per 1M scales the N = 0 cost by the share; no rate → null, shares only", () => {
    const i = input();
    const cost0 = cost0Per1M(i)!;
    expect(cost0).toBeCloseTo(((0.65 * L + 3.0) / 1000) * 1e6 * 3.6 / 3600, 9);
    const c = plan(i).candidates;
    expect(c[0]!.costPer1M).toBeCloseTo(cost0, 9);
    expect(c[8]!.costPer1M).toBeCloseTo(cost0 * c[8]!.serverShare, 9);
    expect(c[25]!.costPer1M).toBe(0);
    const n = plan(input({ rate: null })).candidates;
    expect(n[0]!.costPer1M).toBeNull();
    expect(n[0]!.serverShare).toBe(1);
  });

  it("feasibility by GPU memory budget", () => {
    const c = feasibleN(input({ device: { gpuBudgetBytes: 100 * MiB } }));
    // 40 MiB workspace + N × 8 MiB: N ≤ 7 fits, local (needs +130 MiB) does not
    expect(c.filter((x) => x.feasible).map((x) => x.N)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(c[25]!.feasible).toBe(false);
    expect(c[25]!.reasons.join()).toMatch(/GPU .* > budget/);
  });

  it("feasibility by load budget at the measured bandwidth (cached bytes excluded)", () => {
    const c = feasibleN(input({ policy: { maxLoadSeconds: 4 }, network: { bytesPerSecond: 50e6 } }));
    // embed 130 MiB = 2.73 s, plus 7.5 MiB per block: N ≤ 8 fits in 4 s
    const ok = c.filter((x) => x.feasible).map((x) => x.N);
    expect(ok[ok.length - 1]).toBe(8);
    expect(c[25]!.reasons.join()).toMatch(/load .* s > 4 s/);
    const cached = feasibleN(input({ policy: { maxLoadSeconds: 4 }, device: { embedCached: true, cachedBlocks: 20 } }));
    expect(cached[25]!.feasible).toBe(true);
    expect(cached[25]!.downloadBytes).toBe(4 * 7.5 * MiB);
    // unknown bandwidth: not checked, noted
    const unk = feasibleN(input({ policy: { maxLoadSeconds: 4 }, network: { bytesPerSecond: null } }));
    expect(unk[25]!.feasible).toBe(true);
    expect(unk[25]!.reasons.join()).toMatch(/bandwidth unknown/);
    expect(unk[25]!.loadSeconds).toBeNull();
  });

  it("local requires lm_head to fit too", () => {
    // budget fits every block (40 + 24 × 8 = 232 MiB) but not the 130 MiB head
    const c = feasibleN(input({ device: { gpuBudgetBytes: 300 * MiB } }));
    expect(c[24]!.feasible).toBe(true);
    expect(c[25]!.feasible).toBe(false);
    const p = plan(input({ device: { gpuBudgetBytes: 300 * MiB }, policy: { prefer: "local" } }));
    expect(p.mode).toBe("split"); expect(p.N).toBe(24);
    expect(p.reasons.join()).toMatch(/local infeasible/);
  });

  it("policies: cost → max feasible N (local if feasible); latency → argmin ms/token; local; server", () => {
    expect(plan(input({ policy: { prefer: "cost" } }))).toMatchObject({ mode: "local", N: L });
    expect(plan(input({ policy: { prefer: "cost" }, device: { gpuBudgetBytes: 120 * MiB } }))).toMatchObject({ mode: "split", N: 10 });
    expect(plan(input({ policy: { prefer: "server" } }))).toMatchObject({ mode: "server", N: 0 });
    expect(plan(input({ policy: { prefer: "local" } }))).toMatchObject({ mode: "local", N: L });
    // latency: client 0.7/block vs server 0.65/block + 4 ms rtt + export: local = 24×0.7 + 3.5 + 0.3 = 20.6; server = 4 + 15.6 + 3 = 22.6 → local
    const lat = plan(input({ policy: { prefer: "latency" } }));
    expect(lat.mode).toBe("local");
    // a slow client makes the server win
    const slow = plan(input({ policy: { prefer: "latency" }, device: { msPerBlockT1: 5 } }));
    expect(slow).toMatchObject({ mode: "server", N: 0 });
    // a client faster per block than the server, with local infeasible (150 MiB budget) → split at the largest feasible N (13)
    const far = plan(input({ policy: { prefer: "latency" }, network: { rttMs: 80 }, device: { gpuBudgetBytes: 150 * MiB, msPerBlockT1: 0.5 } }));
    expect(far.mode).toBe("split");
    expect(far.N).toBe(13);
    expect(far.estimate.msPerToken).toBeCloseTo(0.5 * 13 + 0.4 + 80 + 0.65 * 11 + 3.0, 6);
    // the same distant server with a client slower per block than the server → server wins the argmin
    expect(plan(input({ policy: { prefer: "latency" }, network: { rttMs: 80 }, device: { gpuBudgetBytes: 150 * MiB } }))).toMatchObject({ mode: "server", N: 0 });
  });

  it("degraded inputs → server: no WebGPU, no microbench", () => {
    const p = plan(input({ device: { webgpu: false, msPerBlockT1: null, msPerBlockT32: null }, policy: { prefer: "cost" } }));
    expect(p).toMatchObject({ mode: "server", N: 0 });
    expect(p.candidates[1]!.reasons).toContain("no WebGPU");
    expect(p.candidates.filter((c) => c.feasible).length).toBe(1);
    const q = plan(input({ device: { msPerBlockT1: null }, policy: { prefer: "local" } }));
    expect(q).toMatchObject({ mode: "server", N: 0 });
    expect(q.candidates[25]!.reasons).toContain("no client microbench");
  });

  it("server unreachable → only local candidates; prefer server falls back with a reason", () => {
    const p = plan(input({ network: { serverReachable: false, rttMs: null }, policy: { prefer: "server" } }));
    expect(p).toMatchObject({ mode: "local", N: L });
    expect(p.reasons.join()).toMatch(/server infeasible|server unreachable/);
    const none = plan(input({ network: { serverReachable: false, rttMs: null }, device: { webgpu: false, msPerBlockT1: null } }));
    expect(none.mode).toBe("server");
    expect(none.reasons.join()).toMatch(/no feasible candidate/);
  });

  it("maxServerShare and requireLocal", () => {
    const p = plan(input({ policy: { prefer: "latency", maxServerShare: 0.5 } }));
    expect(p.candidates.filter((c) => c.feasible).every((c) => c.serverShare <= 0.5)).toBe(true);
    const r = plan(input({ policy: { prefer: "cost", requireLocal: true }, device: { gpuBudgetBytes: 100 * MiB } }));
    expect(r.reasons.join()).toMatch(/no feasible candidate/);
    expect(plan(input({ policy: { prefer: "server", requireLocal: true } }))).toMatchObject({ mode: "local" });
  });

  it("privacy is the band row at boundary N; nothing for local", () => {
    const c = plan(input()).candidates;
    expect(c[0]!.privacy).toEqual({ boundary: 0, linear500kTop1: 1, inversionTop1: 0.82, linearTop1: 0.8 });
    expect(c[12]!.privacy!.boundary).toBe(12);
    expect(c[25]!.privacy).toBeNull();
    expect(plan(input({ model: { ...input().model, privacy: null } })).candidates[3]!.privacy).toBeNull();
  });

  it("estimates: breakdown terms add up; server microbench missing → no timing", () => {
    const c = plan(input()).candidates;
    const s = c[8]!;
    expect(s.breakdown).toEqual({ clientMs: 0.7 * 8, exportMs: 0.4, rttMs: 4, serverMs: 0.65 * 16, lmHeadMs: 3.0, samplingMs: 0 });
    expect(s.msPerToken).toBeCloseTo(0.7 * 8 + 0.4 + 4 + 0.65 * 16 + 3.0, 9);
    expect(c[0]!.msPerToken).toBeCloseTo(4 + 0.65 * 24 + 3.0, 9);
    expect(c[25]!.msPerToken).toBeCloseTo(0.7 * 24 + 3.5 + 0.3, 9);
    expect(c[8]!.prefillMs).toBeGreaterThan(0);
    const nomb = plan(input({ server: { L, busyFraction60s: 0, msPerBlockDecode: null, msPerBlockPrefill: null, lmHeadMs: null } })).candidates;
    expect(nomb[8]!.msPerToken).toBeNull();
    expect(nomb[25]!.msPerToken).not.toBeNull();
  });
});

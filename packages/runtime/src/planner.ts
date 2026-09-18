// The planner (Phase 5b): a pure function from measured inputs to { mode, N, estimate, reasons } plus every
// candidate's estimate, so a page can show why. No browser-name checks anywhere; everything is a measurement or a
// manifest fact. Definitions (docs/briefs/05b-split-brief.md):
//   ms_per_token(N) = client_ms_per_block(T=1) × N + export_ms + rtt + server_ms_per_block × (L − N) + lm_head_ms + sampling
//   local: no rtt or server terms (client lm_head instead); server: no client terms.
//   server_share(N) = (L − N + h) / (L + h) with h the lm_head's block-equivalents from the Phase 4 measured floor
//   cost(L)/cost(0) = h / (L + h); local: 0.
//   cost_per_1M(N) = server_share(N) × cost_per_1M(0) at the stated rate; without a rate, the share only.
//   privacy(N) = band lookup at boundary N (what a server could recover from what it receives); local: nothing sent.

export type Mode = "local" | "split" | "server";

export interface PrivacyRow {
  boundary: number;
  /** Linear probe, 500k-token training set (docs/phase-4-notes.md "Linear probe, 500k-token training set"). */
  linear500k_top1: number;
  linear500k_top5: number;
  /** Original 40k-token linear probe (Phase 4 "Privacy band"). */
  linear_top1: number;
  linear_top5: number;
  inversion_top1: number;
  inversion_top5: number;
  nn_top1: number;
}

export interface PrivacyBand {
  model: string;
  /** Held-out-token coverage of the 500k training set (norm_* = linear500k_* / coverage). */
  coverage500k: number;
  rows: PrivacyRow[];
}

export interface PlanModel {
  id: string;
  variant: string;
  L: number;
  dModel: number;
  /** Stored (= GPU) bytes of each block group, index i = block i. */
  blockBytes: number[];
  /** Embed group bytes: the CPU-side table (downloaded, not on the GPU). */
  embedBytes: number;
  /** GPU bytes of the head weights when local (the tied embed bytes again + final norm). */
  lmHeadGpuBytes: number;
  /** Download bytes for final_norm + lm_head beyond the embed group (0 when tied). */
  lmHeadDownloadBytes: number;
  /** KV cache bytes per block at the session's maxCtx. */
  kvBytesPerBlock: number;
  /** Activation workspace + export buffers at maxCtx (independent of N). */
  workspaceBytes: number;
  /** Largest single GPU buffer any candidate needs (must fit maxBufferSize). */
  maxEntryBytes: number;
  /** Phase 4 measured cost(L)/cost(0): the lm_head floor (0.224 on the Mac). */
  serverCostFloor: number;
  privacy: PrivacyBand | null;
}

export interface PlanDevice {
  webgpu: boolean;
  maxBufferSize: number | null;
  /** min(quota-derived, configured cap); the planner never reads storage itself. */
  gpuBudgetBytes: number;
  /** navigator.storage.estimate().quota, recorded for the record (the budget above is derived from it). */
  quotaBytes?: number | null;
  /** Microbench on this device for this variant (null = not measured → no client candidates). */
  msPerBlockT1: number | null;
  msPerBlockT32: number | null;
  /** Client lm_head + readback ms (local mode); null = estimated from bytes. */
  lmHeadMs: number | null;
  exportMs: number | null;
  /** What is already cached: blocks 0..cachedBlocks-1 and the embed group. */
  cachedBlocks: number;
  embedCached: boolean;
}

export interface PlanNetwork {
  /** Measured download bandwidth (bytes/s); null = unknown (load budget cannot be checked). */
  bytesPerSecond: number | null;
  /** WebSocket ping median (ms); null when the server is unreachable. */
  rttMs: number | null;
  serverReachable: boolean;
}

export interface PlanServer {
  L: number;
  busyFraction60s: number;
  msPerBlockDecode: number | null;
  msPerBlockPrefill: number | null;
  lmHeadMs: number | null;
  device?: string;
}

export interface PlanRate { gpu: string; usdPerHour: number | null; source: string; retrieved: string }

export interface PlanPolicy {
  prefer: "cost" | "latency" | "local" | "server";
  maxLoadSeconds?: number;
  /** Exclude candidates whose server share exceeds this (0..1). */
  maxServerShare?: number;
  requireLocal?: boolean;
}

export interface PlanPrompt { promptTokens: number; maxNewTokens: number }

export interface PlanInput {
  model: PlanModel;
  device: PlanDevice;
  network: PlanNetwork;
  server: PlanServer | null;
  policy: PlanPolicy;
  prompt: PlanPrompt;
  rate?: PlanRate | null;
  /** CPU sampling ms per token (default 0.3). */
  samplingMs?: number;
}

export interface PlanBreakdown { clientMs: number; exportMs: number; rttMs: number; serverMs: number; lmHeadMs: number; samplingMs: number }

export interface PlanCandidate {
  mode: Mode;
  N: number;
  feasible: boolean;
  reasons: string[];
  gpuBytes: number;
  downloadBytes: number;
  /** Download seconds at the measured bandwidth; null when unknown. */
  loadSeconds: number | null;
  /** Estimated decode ms per token; null when an input it needs is missing. */
  msPerToken: number | null;
  breakdown: PlanBreakdown | null;
  /** Estimated prefill ms for prompt.promptTokens. */
  prefillMs: number | null;
  serverShare: number;
  /** USD per 1M generated tokens at the stated rate; null without a rate. */
  costPer1M: number | null;
  privacy: { boundary: number; linear500kTop1: number; inversionTop1: number; linearTop1: number } | null;
}

export interface Plan {
  mode: Mode;
  N: number;
  estimate: PlanCandidate;
  reasons: string[];
  candidates: PlanCandidate[];
  /** cost_per_1M at N = 0 (the reference the shares scale), null without a rate. */
  cost0Per1M: number | null;
  lmHeadBlockEquivalents: number;
  inputs: PlanInput;
}

export const DEFAULT_GPU_BUDGET = 2 * 1024 ** 3;

/** lm_head as block-equivalents h from the measured floor f = cost(L)/cost(0) = h / (L + h). */
export function lmHeadBlockEquivalents(L: number, floor: number): number {
  const f = Math.min(Math.max(floor, 0), 0.999);
  return (f * L) / (1 - f);
}

export function serverShare(N: number, L: number, floor: number, local: boolean): number {
  if (local) return 0;
  const h = lmHeadBlockEquivalents(L, floor);
  return (L - N + h) / (L + h);
}

function sum(a: number[], n: number): number { let s = 0; for (let i = 0; i < n; i++) s += a[i] ?? 0; return s; }

function gpuBytesFor(m: PlanModel, N: number, local: boolean): number {
  return sum(m.blockBytes, N) + N * m.kvBytesPerBlock + m.workspaceBytes + (local ? m.lmHeadGpuBytes : 0);
}

function downloadBytesFor(m: PlanModel, d: PlanDevice, N: number, local: boolean): number {
  let b = d.embedCached ? 0 : m.embedBytes;
  for (let i = d.cachedBlocks; i < N; i++) b += m.blockBytes[i] ?? 0;
  if (local) b += m.lmHeadDownloadBytes;
  return b;
}

function privacyAt(m: PlanModel, boundary: number): PlanCandidate["privacy"] {
  const r = m.privacy?.rows.find((x) => x.boundary === boundary);
  return r ? { boundary, linear500kTop1: r.linear500k_top1, inversionTop1: r.inversion_top1, linearTop1: r.linear_top1 } : null;
}

/** Every candidate 0..L plus local, each with feasibility, reasons and estimates. */
export function feasibleN(input: PlanInput): PlanCandidate[] {
  const { model: m, device: d, network: n, server: s, policy, prompt } = input;
  const L = m.L;
  const samplingMs = input.samplingMs ?? 0.3;
  const floor = m.serverCostFloor;
  const cost0 = cost0Per1M(input);
  const out: PlanCandidate[] = [];
  const serverMsPerBlock = s?.msPerBlockDecode ?? null;
  const serverPrefillPerBlock = s?.msPerBlockPrefill ?? null;
  const serverHead = s?.lmHeadMs ?? null;
  const clientHead = d.lmHeadMs ?? (d.msPerBlockT1 !== null && m.blockBytes[0] ? d.msPerBlockT1 * (m.lmHeadGpuBytes / m.blockBytes[0]) : null);
  const exportMs = d.exportMs ?? 0.5;

  for (let k = 0; k <= L + 1; k++) {
    const local = k === L + 1;
    const N = local ? L : k;
    const mode: Mode = local ? "local" : N === 0 ? "server" : "split";
    const reasons: string[] = [];
    let feasible = true;
    const gpuBytes = N === 0 ? 0 : gpuBytesFor(m, N, local);
    const downloadBytes = N === 0 ? 0 : downloadBytesFor(m, d, N, local);
    const loadSeconds = N === 0 ? 0 : n.bytesPerSecond ? downloadBytes / n.bytesPerSecond : null;

    if (!local && !n.serverReachable) { feasible = false; reasons.push("server unreachable"); }
    if (!local && s && s.L !== L) { feasible = false; reasons.push(`server L=${s.L} != model L=${L}`); }
    if (N >= 1) {
      if (!d.webgpu) { feasible = false; reasons.push("no WebGPU"); }
      if (d.msPerBlockT1 === null) { feasible = false; reasons.push("no client microbench"); }
      if (gpuBytes > d.gpuBudgetBytes) { feasible = false; reasons.push(`GPU ${(gpuBytes / 2 ** 20).toFixed(0)} MiB > budget ${(d.gpuBudgetBytes / 2 ** 20).toFixed(0)} MiB`); }
      if (d.maxBufferSize !== null && m.maxEntryBytes > d.maxBufferSize) { feasible = false; reasons.push(`entry ${(m.maxEntryBytes / 2 ** 20).toFixed(0)} MiB > maxBufferSize`); }
      if (policy.maxLoadSeconds !== undefined) {
        if (loadSeconds === null) reasons.push("bandwidth unknown: load budget not checked");
        else if (loadSeconds > policy.maxLoadSeconds) { feasible = false; reasons.push(`load ${loadSeconds.toFixed(1)} s > ${policy.maxLoadSeconds} s`); }
      }
    }
    if (!local && N < L + 1 && s && (serverMsPerBlock === null || serverHead === null)) reasons.push("server microbench missing: server ms estimated as unknown");
    const share = serverShare(N, L, floor, local);
    if (policy.maxServerShare !== undefined && share > policy.maxServerShare + 1e-9) { feasible = false; reasons.push(`server share ${share.toFixed(2)} > ${policy.maxServerShare}`); }
    if (policy.requireLocal && !local) { feasible = false; reasons.push("policy requires local"); }

    // estimates
    let msPerToken: number | null = null, breakdown: PlanBreakdown | null = null, prefillMs: number | null = null;
    const clientMs = N >= 1 && d.msPerBlockT1 !== null ? d.msPerBlockT1 * N : N >= 1 ? null : 0;
    if (local) {
      if (clientMs !== null && clientHead !== null) {
        breakdown = { clientMs, exportMs: 0, rttMs: 0, serverMs: 0, lmHeadMs: clientHead, samplingMs };
        msPerToken = clientMs + clientHead + samplingMs;
      }
      if (d.msPerBlockT32 !== null) prefillMs = d.msPerBlockT32 * (prompt.promptTokens / 32) * N + (clientHead ?? 0);
    } else {
      const serverMs = serverMsPerBlock !== null && serverHead !== null ? serverMsPerBlock * (L - N) + serverHead : null;
      const rtt = n.rttMs;
      if (clientMs !== null && serverMs !== null && rtt !== null) {
        breakdown = { clientMs, exportMs: N >= 1 ? exportMs : 0, rttMs: rtt, serverMs: serverMsPerBlock! * (L - N), lmHeadMs: serverHead!, samplingMs: 0 };
        msPerToken = clientMs + (N >= 1 ? exportMs : 0) + rtt + serverMs;
      }
      if (rtt !== null && serverPrefillPerBlock !== null && (N === 0 || d.msPerBlockT32 !== null)) {
        prefillMs = (N >= 1 ? d.msPerBlockT32! * (prompt.promptTokens / 32) * N + exportMs : 0) + rtt + serverPrefillPerBlock * (prompt.promptTokens / 32) * (L - N) + (serverHead ?? 0);
      }
    }
    const privacy = local ? null : privacyAt(m, N);
    out.push({ mode, N, feasible, reasons, gpuBytes, downloadBytes, loadSeconds, msPerToken, breakdown, prefillMs, serverShare: share, costPer1M: cost0 === null ? null : cost0 * share, privacy });
  }
  return out;
}

/** USD per 1M tokens at N = 0: gpu_seconds_per_token(0) × 1e6 × rate / 3600, from the server microbench. */
export function cost0Per1M(input: PlanInput): number | null {
  const s = input.server, r = input.rate;
  if (!s || !r || r.usdPerHour === null || s.msPerBlockDecode === null || s.lmHeadMs === null) return null;
  const gpuSecondsPerToken = (s.msPerBlockDecode * s.L + s.lmHeadMs) / 1000;
  return (gpuSecondsPerToken * 1e6 * r.usdPerHour) / 3600;
}

export function plan(input: PlanInput): Plan {
  const candidates = feasibleN(input);
  const feasible = candidates.filter((c) => c.feasible);
  const { prefer } = input.policy;
  const reasons: string[] = [];
  const local = feasible.find((c) => c.mode === "local");
  const server = feasible.find((c) => c.mode === "server");
  const byN = (a: PlanCandidate, b: PlanCandidate) => b.N - a.N;
  const largestSplit = [...feasible].filter((c) => c.mode !== "local").sort(byN)[0];
  let pick: PlanCandidate | undefined;
  if (!input.device.webgpu) reasons.push("no WebGPU on this device: only the server can run the model");
  if (!input.network.serverReachable) reasons.push("server unreachable: only local candidates");
  if (prefer === "server") {
    pick = server;
    if (pick) reasons.push("policy prefers the server: N = 0");
    else { pick = local ?? largestSplit; if (pick) reasons.push("server infeasible; falling back to the largest feasible client N"); }
  } else if (prefer === "local") {
    pick = local;
    if (pick) reasons.push("policy prefers local and local is feasible");
    else { pick = largestSplit; if (pick) reasons.push(`local infeasible (${candidates[candidates.length - 1]!.reasons.join("; ") || "?"}); largest feasible N instead`); }
  } else if (prefer === "cost") {
    pick = local ?? largestSplit;
    if (pick) reasons.push(pick.mode === "local" ? "policy prefers cost: local (server share 0)" : `policy prefers cost: max feasible N = ${pick.N} (server share ${pick.serverShare.toFixed(2)})`);
  } else {
    const timed = feasible.filter((c) => c.msPerToken !== null).sort((a, b) => a.msPerToken! - b.msPerToken! || byN(a, b));
    pick = timed[0] ?? local ?? largestSplit ?? server;
    if (pick) reasons.push(pick.msPerToken !== null ? `policy prefers latency: argmin ms/token = ${pick.mode} N=${pick.N} (${pick.msPerToken.toFixed(1)} ms)` : "policy prefers latency but no candidate has a timing estimate; largest feasible N");
  }
  if (!pick) {
    const why = candidates.map((c) => `${c.mode}/N=${c.N}: ${c.reasons.join("; ")}`).join(" | ");
    return { mode: "server", N: 0, estimate: candidates[0]!, reasons: [...reasons, `no feasible candidate: ${why}`], candidates, cost0Per1M: cost0Per1M(input), lmHeadBlockEquivalents: lmHeadBlockEquivalents(input.model.L, input.model.serverCostFloor), inputs: input };
  }
  return { mode: pick.mode, N: pick.N, estimate: pick, reasons, candidates, cost0Per1M: cost0Per1M(input), lmHeadBlockEquivalents: lmHeadBlockEquivalents(input.model.L, input.model.serverCostFloor), inputs: input };
}

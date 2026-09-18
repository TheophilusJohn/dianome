// Browser side of the split e2e: `Dianome.run()` / `planRun()` from the SDK source (the run entry and the runtime
// package's built entries are bundled in), everything against the local static server + wrangler dev + dianome-server.
// Each function returns plain data; the specs assert.
import { Dianome, type Plan, type RunOptions, type RunResult } from "../../../src/index";
import { PRIVACY_BAND } from "../../../src/privacyBand.gen";

const origin = location.origin;
const MODEL = "qwen2.5-0.5b-instruct";
const d = new Dianome({ api: origin, cdn: origin, cache: "none", telemetry: true });

export interface HarnessRunOptions extends Omit<RunOptions, "onToken" | "onPlan" | "onProgress" | "signal"> { split?: { url: string; token: string } | null }

/** run(); the result minus functions, plus the token deltas seen. */
export async function run(opts: HarnessRunOptions): Promise<Omit<RunResult, "plan"> & { plan: Plan; deltas: string[]; loadMs: number | null }> {
  const deltas: string[] = [];
  const r = await d.run(MODEL, { ...opts, onToken: (t) => deltas.push(t) });
  return { ...r, plan: r.plan, deltas, loadMs: r.load?.ms ?? null };
}

export async function plan(opts: HarnessRunOptions): Promise<Plan> {
  return d.planRun(MODEL, opts);
}

export async function prompt(): Promise<{ text: string; token_ids: number[]; T: number }> {
  return (await fetch("/fixtures/prompt.json")).json();
}

export async function greedy(variant: "fp16" | "q8" | "q4"): Promise<{ tokens: number[]; local: number[]; byN: Record<string, number[]> }> {
  if (variant === "fp16") { const g = await (await fetch("/fixtures/greedy.json")).json(); return { tokens: g.tokens, local: g.tokens, byN: {} }; }
  const g = await (await fetch(`/fixtures/${variant}/greedy.json`)).json();
  return { tokens: g.N["24"], local: g.N.local, byN: g.N };
}

export function band(): typeof PRIVACY_BAND { return PRIVACY_BAND; }
export function info(): { ua: string; webgpu: boolean } { return { ua: navigator.userAgent, webgpu: Boolean((navigator as Navigator & { gpu?: unknown }).gpu) }; }

declare global { interface Window { harness: typeof harness } }
const harness = { run, plan, prompt, greedy, band, info };
window.harness = harness;

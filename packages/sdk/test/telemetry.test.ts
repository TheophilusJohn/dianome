import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import Ajv2020 from "ajv/dist/2020";
import { describe, expect, it } from "vitest";
import type { DeviceInfo } from "../src/device";
import { detectBrowser } from "../src/device";
import { buildReport, classifySource, countCacheHits, median, postReport } from "../src/telemetry";

const device: DeviceInfo = { browser: "chrome", webgpu: true, maxBufferSize: 4294967292, maxStorageBufferBindingSize: 4294967292, maxComputeWorkgroupStorageSize: 32768, quotaBytes: 10_000_000_000, usageBytes: 0 };

describe("source classification", () => {
  it("all one source → that source; otherwise mixed; none → network", () => {
    expect(classifySource(["network", "network"])).toBe("network");
    expect(classifySource(["per-site-cache"])).toBe("per-site-cache");
    expect(classifySource(["cross-site-cache", "cross-site-cache"])).toBe("cross-site-cache");
    expect(classifySource(["per-site-cache", "network"])).toBe("mixed");
    expect(classifySource(["per-site-cache", "cross-site-cache"])).toBe("mixed");
    expect(classifySource([])).toBe("network");
  });
  it("cache_hits counts chunks not fetched from the network", () => {
    expect(countCacheHits(["network", "per-site-cache", "cross-site-cache", "network"])).toBe(2);
  });
});

describe("buildReport", () => {
  it("produces a schema-2 report that validates against schemas/telemetry.v2.json", () => {
    const schema = JSON.parse(readFileSync(resolve(__dirname, "../../../schemas/telemetry.v2.json"), "utf8"));
    const validate = new Ajv2020({ strict: false }).compile(schema);
    const r = buildReport({ model: "qwen2.5-0.5b-instruct", variant: "q4", bytes: 323893760, sources: ["network", "per-site-cache", "per-site-cache"], ms: 1234.6, verifyMs: 88.2, transferSamples: [], cacheMode: "per-site", device });
    expect(validate(r), JSON.stringify(validate.errors)).toBe(true);
    expect(r).toMatchObject({ schema: 2, source: "mixed", cache_hits: 2, chunks: 3, ms: 1235, verify_ms: 88, transfer_ms: 0, cache_mode: "per-site", quota_bytes: 10_000_000_000, max_buffer_size: 4294967292, browser: "chrome", webgpu: true });
    expect(r.bytes_per_second).toBe(Math.round(323893760 / 1.2346));
  });
  it("omits quota and max_buffer_size when the device does not report them", () => {
    const r = buildReport({ model: "m", variant: "fp16", bytes: 0, sources: [], ms: 0, verifyMs: 0, transferSamples: [], cacheMode: "none", device: { ...device, quotaBytes: null, maxBufferSize: null } });
    expect("quota_bytes" in r).toBe(false);
    expect("max_buffer_size" in r).toBe(false);
    expect(r.bytes_per_second).toBe(0);
  });
});

describe("transfer_ms", () => {
  it("median(): empty → 0, odd and even counts, unsorted input", () => {
    expect(median([])).toBe(0);
    expect(median([7])).toBe(7);
    expect(median([5, 1, 3])).toBe(3);
    expect(median([1, 5, 3, 100])).toBe(4);
  });
  it("is the per-chunk median of the postMessage hops, never their sum", () => {
    const samples = [1.4, 1.9, 1.6, 250, 1.7, 1.5]; // one slow hop must not dominate
    const r = buildReport({ model: "m", variant: "q4", bytes: 6 * 8388608, sources: Array(6).fill("cross-site-cache"), ms: 15_000, verifyMs: 20, transferSamples: samples, cacheMode: "cross-site", device });
    expect(r.transfer_ms).toBe(2); // round(median = 1.8)
    expect(r.transfer_ms).toBeLessThan(samples.reduce((a, b) => a + b, 0));
  });
});

describe("postReport", () => {
  it("POSTs JSON with keepalive and returns the status; never throws", async () => {
    const calls: { url: string; init: RequestInit | undefined }[] = [];
    const status = await postReport("https://api.test/", { schema: 2, model: "m", variant: "q4", bytes: 1, chunks: 1, ms: 1, source: "network", cache_hits: 0, browser: "other", webgpu: false }, async (url, init) => { calls.push({ url, init }); return new Response(null, { status: 202 }); });
    expect(status).toBe(202);
    expect(calls[0]!.url).toBe("https://api.test/v1/telemetry/load");
    expect(calls[0]!.init).toMatchObject({ method: "POST", keepalive: true, headers: { "Content-Type": "application/json" } });
    expect(await postReport("https://api.test", { schema: 2 } as never, async () => { throw new TypeError("offline"); })).toBeNull();
  });
});

describe("detectBrowser", () => {
  it("maps user agents to the four families", () => {
    expect(detectBrowser("Mozilla/5.0 (Macintosh) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36")).toBe("chrome");
    expect(detectBrowser("Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 Chrome/152.0.0.0 Safari/537.36 Edg/152.0")).toBe("chrome");
    expect(detectBrowser("Mozilla/5.0 (Macintosh; rv:152.0) Gecko/20100101 Firefox/152.0")).toBe("firefox");
    expect(detectBrowser("Mozilla/5.0 (Macintosh) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.5 Safari/605.1.15")).toBe("safari");
    expect(detectBrowser("curl/8.0")).toBe("other");
  });
});

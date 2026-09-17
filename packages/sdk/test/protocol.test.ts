import { describe, expect, it } from "vitest";
import { epochNow, frameUrlFor, hopSince, markerUrlFor, optinUrlFor, parseProgress, parseRequest, parseResponse, sameOrigin, stampSent, transferablesOf, versionMismatchId, type FrameRequest } from "../src/cache/protocol";

const sha = "a".repeat(64);

describe("frame protocol", () => {
  it("parses every request op and rejects malformed ones", () => {
    expect(parseRequest({ v: 1, id: 1, op: "hello" })).toEqual({ v: 1, id: 1, op: "hello" });
    expect(parseRequest({ v: 1, id: 2, op: "grant", mode: "silent" })).toEqual({ v: 1, id: 2, op: "grant", mode: "silent" });
    expect(parseRequest({ v: 1, id: 2, op: "grant", mode: "await-click" })).toMatchObject({ mode: "await-click" });
    expect(parseRequest({ v: 1, id: 2, op: "grant", mode: "now" })).toBeNull();
    expect(parseRequest({ v: 1, id: 3, op: "get", sha, modelId: "m" })).toEqual({ v: 1, id: 3, op: "get", sha, modelId: "m" });
    expect(parseRequest({ v: 1, id: 3, op: "get", sha })).toEqual({ v: 1, id: 3, op: "get", sha });
    expect(parseRequest({ v: 1, id: 3, op: "get", sha: "abc" })).toBeNull();
    const buf = new ArrayBuffer(4);
    expect(parseRequest({ v: 1, id: 4, op: "put", sha, buf })).toEqual({ v: 1, id: 4, op: "put", sha, buf });
    expect(parseRequest({ v: 1, id: 4, op: "put", sha, buf: new Uint8Array(4) })).toBeNull();
    expect(parseRequest({ v: 1, id: 5, op: "has", sha })).toMatchObject({ op: "has" });
    expect(parseRequest({ v: 1, id: 6, op: "status" })).toMatchObject({ op: "status" });
    expect(parseRequest({ v: 1, id: 7, op: "evict", shas: [sha, "b".repeat(64)] })).toMatchObject({ op: "evict" });
    expect(parseRequest({ v: 1, id: 7, op: "evict", shas: [sha, "nope"] })).toBeNull();
    expect(parseRequest({ v: 1, id: 8, op: "evictModel", modelId: "m" })).toMatchObject({ op: "evictModel", modelId: "m" });
    expect(parseRequest({ v: 1, id: 9, op: "clear" })).toMatchObject({ op: "clear" });
    expect(parseRequest({ v: 1, id: 10, op: "fetch", sha, bytes: 10, modelId: "m" })).toMatchObject({ op: "fetch", bytes: 10 });
    expect(parseRequest({ v: 1, id: 10, op: "fetch", sha, bytes: 0, modelId: "m" })).toBeNull();
    expect(parseRequest({ v: 1, id: 10, op: "fetch", sha, bytes: 1.5, modelId: "m" })).toBeNull();
    expect(parseRequest({ v: 1, id: 11, op: "format" })).toBeNull();
    expect(parseRequest({ v: 2, id: 1, op: "hello" })).toBeNull();
    expect(parseRequest({ v: 1, id: "1", op: "hello" })).toBeNull();
    expect(parseRequest(null)).toBeNull();
    expect(parseRequest("hello")).toBeNull();
  });

  it("parses responses and detects version mismatches", () => {
    expect(parseResponse({ v: 1, id: 1, ok: true, result: 42, ms: 1.5 })).toEqual({ v: 1, id: 1, ok: true, result: 42, ms: 1.5 });
    expect(parseResponse({ v: 1, id: 1, ok: false, error: "x", code: "no_grant", ms: 0 })).toMatchObject({ ok: false, code: "no_grant" });
    expect(parseResponse({ v: 1, id: 1, ok: false, ms: 0 })).toBeNull();
    expect(parseResponse({ v: 1, id: 1, ok: true })).toBeNull();
    expect(parseResponse({ v: 0, id: 1, ok: true, result: 1, ms: 0 })).toBeNull();
    expect(versionMismatchId({ v: 2, id: 7, op: "hello" })).toBe(7);
    expect(versionMismatchId({ v: 1, id: 7, op: "hello" })).toBeNull();
    expect(versionMismatchId({ v: 2, op: "hello" })).toBeNull();
  });

  it("carries sentAt / hopInMs stamps through parsing and measures hops with a clock comparable across documents", () => {
    expect(parseRequest({ v: 1, id: 1, op: "hello", sentAt: 1700000000000.5 })).toEqual({ v: 1, id: 1, op: "hello", sentAt: 1700000000000.5 });
    expect(parseRequest({ v: 1, id: 1, op: "hello", sentAt: "now" })).toEqual({ v: 1, id: 1, op: "hello" });
    expect(parseResponse({ v: 1, id: 1, ok: true, result: null, ms: 0, sentAt: 5, hopInMs: 0.4 })).toEqual({ v: 1, id: 1, ok: true, result: null, ms: 0, sentAt: 5, hopInMs: 0.4 });
    expect(parseResponse({ v: 1, id: 1, ok: false, error: "e", code: "internal", ms: 0, sentAt: 5 })).toMatchObject({ ok: false, sentAt: 5 });
    expect("hopInMs" in parseResponse({ v: 1, id: 1, ok: true, result: null, ms: 0, sentAt: "x" })!).toBe(false);
    const msg = stampSent<FrameRequest>({ v: 1, id: 1, op: "hello" });
    expect(Math.abs(msg.sentAt! - (performance.timeOrigin + performance.now()))).toBeLessThan(50);
    expect(hopSince(undefined)).toBeNull();
    expect(hopSince(100, 103.5)).toBe(3.5);
    expect(hopSince(100, 90)).toBe(0); // skew clamps at 0
    expect(hopSince(epochNow() - 2)).toBeGreaterThanOrEqual(1.5);
  });

  it("parses progress notes and rejects anything else", () => {
    expect(parseProgress({ v: 1, id: 3, progress: "clicked" })).toEqual({ v: 1, id: 3, progress: "clicked" });
    expect(parseProgress({ v: 1, id: 3, progress: "done" })).toBeNull();
    expect(parseProgress({ v: 1, id: 3, ok: true, result: 1, ms: 0 })).toBeNull();
    expect(parseResponse({ v: 1, id: 3, progress: "clicked" })).toBeNull();
  });

  it("lists the buffers to transfer", () => {
    const buf = new ArrayBuffer(8);
    expect(transferablesOf({ v: 1, id: 1, op: "put", sha, buf })).toEqual([buf]);
    expect(transferablesOf({ v: 1, id: 1, op: "get", sha })).toEqual([]);
    expect(transferablesOf({ v: 1, id: 1, ok: true, result: buf, ms: 0 })).toEqual([buf]);
    expect(transferablesOf({ v: 1, id: 1, ok: true, result: { buf, fromCache: true }, ms: 0 })).toEqual([buf]);
    expect(transferablesOf({ v: 1, id: 1, ok: true, result: null, ms: 0 })).toEqual([]);
    expect(transferablesOf({ v: 1, id: 1, ok: false, error: "e", code: "internal", ms: 0 })).toEqual([]);
  });

  it("origin checks are exact and never accept null or *", () => {
    expect(sameOrigin("https://cdn.dianome.dev", "https://cdn.dianome.dev")).toBe(true);
    expect(sameOrigin("https://cdn.dianome.dev", "https://cdn.dianome.dev.evil.com")).toBe(false);
    expect(sameOrigin("http://cdn.dianome.dev", "https://cdn.dianome.dev")).toBe(false);
    expect(sameOrigin("null", "null")).toBe(false);
    expect(sameOrigin("*", "*")).toBe(false);
    expect(sameOrigin("", "")).toBe(false);
  });

  it("builds the frame, opt-in and marker URLs", () => {
    expect(frameUrlFor("https://cdn.dianome.dev/")).toBe("https://cdn.dianome.dev/frame/v1/index.html");
    expect(optinUrlFor("https://cdn.dianome.dev", "https://site-a.pages.dev/?x=1")).toBe("https://cdn.dianome.dev/frame/v1/optin.html?return=https%3A%2F%2Fsite-a.pages.dev%2F%3Fx%3D1");
    expect(markerUrlFor("https://cdn.dianome.dev")).toBe("https://cdn.dianome.dev/frame/v1/marker");
  });
});

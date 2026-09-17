// A scriptable FramePlatform: separate "handle" and "global" storages so a test can prove which one the frame used.
import { IDBFactory } from "fake-indexeddb";
import type { FramePlatform, StorageAccessHandle } from "../../src/cache/frameServer";
import { CACHE_NAME, chunkUrl } from "../../src/cache/types";
import { markerUrlFor } from "../../src/cache/protocol";
import { FakeCacheStorage } from "./fakeCaches";

export type Mode = "chrome" | "firefox" | "no-api";

export interface FakePlatformOptions {
  mode?: Mode;
  /** Browser family handed to the platform (default: "chrome" for mode chrome, "firefox" for mode firefox). */
  browser?: "chrome" | "firefox" | "safari" | "other";
  origin?: string;
  /** What requestStorageAccess does when called without a gesture (silent). Default: reject NotAllowedError. */
  silent?: "grant" | "reject";
  /** What it does inside a gesture. Default: grant. */
  gesture?: "grant" | "reject";
  permission?: "granted" | "prompt" | "denied" | "throws";
  markerInHandle?: boolean;
  markerInGlobals?: boolean;
  hasStorageAccess?: boolean;
  chunks?: Map<string, Uint8Array>;
  capacity?: number;
}

export interface FakePlatform {
  platform: FramePlatform;
  handleCaches: FakeCacheStorage;
  globalCaches: FakeCacheStorage;
  handleIdb: IDBFactory;
  globalIdb: IDBFactory;
  globalsTouched: number;
  rsaCalls: { all?: true; inGesture: boolean }[];
  fetches: string[];
  /** Simulates the user clicking the frame's button. */
  click(): void;
  faults: Map<string, ("network" | number)[]>;
  /** Lines the frame server logged. */
  logs: string[];
}

const notAllowed = () => new DOMException("requestStorageAccess not allowed", "NotAllowedError");

export async function fakePlatform(o: FakePlatformOptions = {}): Promise<FakePlatform> {
  const origin = o.origin ?? "https://cdn.test";
  const mode = o.mode ?? "chrome";
  const handleCaches = new FakeCacheStorage(o.capacity ?? Number.POSITIVE_INFINITY), globalCaches = new FakeCacheStorage(o.capacity ?? Number.POSITIVE_INFINITY);
  const handleIdb = new IDBFactory(), globalIdb = new IDBFactory();
  const marker = new Response(new Uint8Array(1024));
  if (o.markerInHandle ?? true) await (await handleCaches.open(CACHE_NAME)).put(markerUrlFor(origin), marker.clone());
  if (o.markerInGlobals ?? true) await (await globalCaches.open(CACHE_NAME)).put(markerUrlFor(origin), marker.clone());
  let granted = false;
  let inGesture = false;
  let pendingGesture: (() => void) | null = null;
  const self: FakePlatform = {
    handleCaches, globalCaches, handleIdb, globalIdb, globalsTouched: 0, rsaCalls: [], fetches: [], faults: new Map(), logs: [],
    // Like the real button (disabled until a grant is waiting): a click before inGesture() is registered waits for it.
    click() {
      const fire = (): void => {
        const p = pendingGesture;
        if (p) { pendingGesture = null; p(); } else setTimeout(fire, 1);
      };
      fire();
    },
    platform: {
      origin,
      browser: o.browser ?? (mode === "firefox" ? "firefox" : "chrome"),
      log: (m) => { self.logs.push(m); },
      hasStorageAccess: async () => granted || (o.hasStorageAccess ?? false),
      permissionState: async () => { if (o.permission === "throws") throw new TypeError("no permissions API"); return o.permission ?? "prompt"; },
      globals: () => { self.globalsTouched++; return { caches: globalCaches.asCacheStorage(), indexedDB: globalIdb, storage: { estimate: async () => ({ quota: 2000, usage: globalCaches.used }) } }; },
      fetch: async (url) => {
        self.fetches.push(url);
        const sha = url.replace("/chunks/", "");
        const f = self.faults.get(sha)?.shift();
        if (f === "network") throw new TypeError("Failed to fetch");
        if (typeof f === "number") return new Response("err", { status: f });
        const body = o.chunks?.get(sha);
        return body ? new Response(body.slice()) : new Response("missing", { status: 404 });
      },
      inGesture: (run) => new Promise((resolve, reject) => {
        pendingGesture = () => { inGesture = true; try { run().then(resolve, reject); } finally { inGesture = false; } };
      }),
    },
  };
  if (mode !== "no-api") {
    self.platform.requestStorageAccess = async (opts?: { all: true }): Promise<StorageAccessHandle | undefined> => {
      self.rsaCalls.push({ ...(opts?.all ? { all: true as const } : {}), inGesture });
      const behaviour = granted ? "grant" : inGesture ? (o.gesture ?? "grant") : (o.silent ?? "reject");
      if (behaviour === "reject") throw notAllowed();
      granted = true;
      if (mode === "chrome" && opts?.all) return { caches: handleCaches.asCacheStorage(), indexedDB: handleIdb, estimate: async () => ({ quota: 1000, usage: handleCaches.used }) };
      return undefined; // Firefox: resolves without a handle
    };
  }
  void chunkUrl;
  return self;
}

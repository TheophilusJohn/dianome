// Frame protocol v1 between the SDK (parent) and the cache frame at `${cdn}/frame/v1/index.html`. Every message
// carries `v: 1`; both sides post with an explicit targetOrigin and validate `event.origin` and `event.source`.
//
//   parent → frame: { v, id, op, ...params }
//   frame → parent: { v, id, ok: true, result, ms } | { v, id, ok: false, error, code, ms }
//
// `get` and `fetch` transfer the buffer back; `put` transfers it in. Bumping v1 in the path is how a breaking
// frame change ships; a v mismatch is answered with code "bad_version".
//
// Transfer timing: both sides stamp `sentAt` (epoch ms from performance.timeOrigin + performance.now(), comparable
// across documents on one machine) immediately before postMessage. The receiver's `epochNow() - sentAt` is the
// postMessage hop alone: no network, no cache work. The frame reports the request hop back as `hopInMs`.

import { SHA_RE } from "../sha";
import type { ChunkStoreStatus } from "./types";

export const PROTOCOL_VERSION = 1 as const;
export const FRAME_PATH = "/frame/v1/index.html";
export const OPTIN_PATH = "/frame/v1/optin.html";
export const MARKER_PATH = "/frame/v1/marker";
export const MARKER_BYTES = 1024;

export type GrantMode = "silent" | "await-click";
/** How the frame reached storage: a storage-access handle (`requestStorageAccess({all: true})` returned one) or the plain grant with the document's own globals. Which one works is detected, never assumed from the browser name. */
export type GrantPath = "chrome-handle" | "plain-globals";
export const VISITED_KEY = "dianome:visited";
/** Query parameter the opt-in page appends to the return URL, so the SDK knows a visit was completed. */
export const VISITED_RETURN_PARAM = "dianome-visited";
export type GrantState = "granted" | "unsupported" | "denied" | "needs-visit" | "needs-click";

export interface GrantResult {
  state: GrantState;
  path?: GrantPath;
  reason?: string;
}

export type FrameRequest = FrameRequestBody & { /** Epoch ms stamped by the parent right before posting. */ sentAt?: number };

export type FrameRequestBody =
  | { v: 1; id: number; op: "hello" }
  | { v: 1; id: number; op: "grant"; mode: GrantMode }
  | { v: 1; id: number; op: "get"; sha: string; modelId?: string }
  | { v: 1; id: number; op: "put"; sha: string; modelId?: string; buf: ArrayBuffer }
  | { v: 1; id: number; op: "has"; sha: string }
  | { v: 1; id: number; op: "status" }
  | { v: 1; id: number; op: "evict"; shas: string[] }
  | { v: 1; id: number; op: "evictModel"; modelId: string }
  | { v: 1; id: number; op: "clear" }
  | { v: 1; id: number; op: "fetch"; sha: string; bytes: number; modelId: string };

export type FrameOp = FrameRequestBody["op"];

export interface HelloResult { v: 1; hasStorageAccess: boolean | null; granted: boolean }
export interface FetchResult { buf: ArrayBuffer; fromCache: boolean; /** The put after a network fetch hit quota with nothing left to evict. */ quota?: true }

export type FrameResultOf<O extends FrameOp> =
  O extends "hello" ? HelloResult :
  O extends "grant" ? GrantResult :
  O extends "get" ? ArrayBuffer | null :
  O extends "put" ? null :
  O extends "has" ? boolean :
  O extends "status" ? ChunkStoreStatus :
  O extends "evict" | "evictModel" | "clear" ? null :
  O extends "fetch" ? FetchResult : never;

/** Stages of an await-click grant, posted by the frame before the final reply. */
export type GrantProgress = "waiting-click" | "clicked" | "requesting";

/** An unsolicited progress note for a pending request (same id); never terminates the request. */
export interface FrameProgress { v: 1; id: number; progress: GrantProgress }

export function parseProgress(data: unknown): FrameProgress | null {
  if (!isObj(data) || data.v !== PROTOCOL_VERSION || typeof data.id !== "number" || typeof data.progress !== "string") return null;
  if (data.progress !== "waiting-click" && data.progress !== "clicked" && data.progress !== "requesting") return null;
  return { v: 1, id: data.id, progress: data.progress };
}

export type FrameErrorCode = "bad_request" | "bad_version" | "unknown_op" | "no_grant" | "quota" | "network" | `http_${number}` | "internal";

export type FrameResponse = FrameResponseBody & {
  /** Epoch ms stamped by the frame right before posting (see stampSent). */
  sentAt?: number;
  /** Parent → frame hop of the request this answers, measured by the frame when the request carried sentAt. */
  hopInMs?: number;
};

export type FrameResponseBody =
  | { v: 1; id: number; ok: true; result: unknown; ms: number }
  | { v: 1; id: number; ok: false; error: string; code: FrameErrorCode; ms: number };

/** Wall-clock ms comparable between the parent and the frame documents (sub-ms where the browser allows). */
export function epochNow(): number {
  if (typeof performance !== "undefined" && typeof performance.timeOrigin === "number") return performance.timeOrigin + performance.now();
  return Date.now();
}

/** Stamps `sentAt`; call it as the last thing before postMessage so the hop excludes any local work. */
export function stampSent<T extends { sentAt?: number }>(msg: T): T {
  msg.sentAt = epochNow();
  return msg;
}

/** The hop from a stamped message to now; null when the message was not stamped. Clock skew is clamped at 0. */
export function hopSince(sentAt: number | undefined, receivedAt = epochNow()): number | null {
  return typeof sentAt === "number" ? Math.max(0, receivedAt - sentAt) : null;
}

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null;
const isSha = (v: unknown): v is string => typeof v === "string" && SHA_RE.test(v);
const optStr = (v: unknown): v is string | undefined => v === undefined || typeof v === "string";

/** Validates an incoming parent → frame message. Returns null when it is not a well-formed v1 request. */
export function parseRequest(data: unknown): FrameRequest | null {
  const body = parseRequestBody(data);
  if (!body) return null;
  const sentAt = (data as { sentAt?: unknown }).sentAt;
  return typeof sentAt === "number" ? { ...body, sentAt } : body;
}

function parseRequestBody(data: unknown): FrameRequestBody | null {
  if (!isObj(data) || data.v !== PROTOCOL_VERSION || typeof data.id !== "number" || typeof data.op !== "string") return null;
  const { id } = data as { id: number };
  switch (data.op) {
    case "hello": case "status": case "clear": return { v: 1, id, op: data.op };
    case "grant": return data.mode === "silent" || data.mode === "await-click" ? { v: 1, id, op: "grant", mode: data.mode } : null;
    case "get": return isSha(data.sha) && optStr(data.modelId) ? { v: 1, id, op: "get", sha: data.sha, ...(data.modelId !== undefined ? { modelId: data.modelId } : {}) } : null;
    case "put": return isSha(data.sha) && optStr(data.modelId) && data.buf instanceof ArrayBuffer ? { v: 1, id, op: "put", sha: data.sha, buf: data.buf, ...(data.modelId !== undefined ? { modelId: data.modelId } : {}) } : null;
    case "has": return isSha(data.sha) ? { v: 1, id, op: "has", sha: data.sha } : null;
    case "evict": return Array.isArray(data.shas) && data.shas.every(isSha) ? { v: 1, id, op: "evict", shas: data.shas as string[] } : null;
    case "evictModel": return typeof data.modelId === "string" ? { v: 1, id, op: "evictModel", modelId: data.modelId } : null;
    case "fetch": return isSha(data.sha) && typeof data.bytes === "number" && Number.isInteger(data.bytes) && data.bytes > 0 && typeof data.modelId === "string"
      ? { v: 1, id, op: "fetch", sha: data.sha, bytes: data.bytes, modelId: data.modelId } : null;
    default: return null;
  }
}

/** Validates an incoming frame → parent message. */
export function parseResponse(data: unknown): FrameResponse | null {
  if (!isObj(data) || data.v !== PROTOCOL_VERSION || typeof data.id !== "number" || typeof data.ok !== "boolean" || typeof data.ms !== "number") return null;
  const extra: { sentAt?: number; hopInMs?: number } = {};
  if (typeof data.sentAt === "number") extra.sentAt = data.sentAt;
  if (typeof data.hopInMs === "number") extra.hopInMs = data.hopInMs;
  if (data.ok) return { v: 1, id: data.id, ok: true, result: data.result, ms: data.ms, ...extra };
  if (typeof data.error !== "string" || typeof data.code !== "string") return null;
  return { v: 1, id: data.id, ok: false, error: data.error, code: data.code as FrameErrorCode, ms: data.ms, ...extra };
}

/** A message from a bad version still gets a reply so the parent can diagnose it: returns the id when present. */
export function versionMismatchId(data: unknown): number | null {
  return isObj(data) && data.v !== PROTOCOL_VERSION && typeof data.id === "number" && typeof data.op === "string" ? data.id : null;
}

/** Buffers to transfer with a request or response, if any. */
export function transferablesOf(msg: FrameRequest | FrameResponse): ArrayBuffer[] {
  if ("op" in msg) return msg.op === "put" ? [msg.buf] : [];
  if (!msg.ok) return [];
  const r = msg.result;
  if (r instanceof ArrayBuffer) return [r];
  if (isObj(r) && r.buf instanceof ArrayBuffer) return [r.buf];
  return [];
}

export function frameUrlFor(cdn: string): string { return `${cdn.replace(/\/+$/, "")}${FRAME_PATH}`; }
export function optinUrlFor(cdn: string, returnTo: string): string {
  return `${cdn.replace(/\/+$/, "")}${OPTIN_PATH}?return=${encodeURIComponent(returnTo)}`;
}
export function markerUrlFor(origin: string): string { return `${origin.replace(/\/+$/, "")}${MARKER_PATH}`; }

/** Origins must match exactly; "null" (opaque) and "*" are never accepted. */
export function sameOrigin(a: string, b: string): boolean {
  return a === b && a !== "null" && a !== "*" && a.length > 0;
}

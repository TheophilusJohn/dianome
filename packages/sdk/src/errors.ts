// Error types thrown by the SDK. Every error is a plain subclass with a stable `code` for programmatic handling.

export type DianomeErrorCode =
  | "manifest_invalid"
  | "manifest_http"
  | "chunk_http"
  | "chunk_network"
  | "chunk_verify"
  | "chunk_missing"
  | "aborted"
  | "frame_error"
  | "frame_timeout";

export class DianomeError extends Error {
  readonly code: DianomeErrorCode;
  constructor(code: DianomeErrorCode, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = "DianomeError";
    this.code = code;
  }
}

export class ManifestError extends DianomeError {
  constructor(code: "manifest_invalid" | "manifest_http", message: string, options?: { cause?: unknown }) {
    super(code, message, options);
    this.name = "ManifestError";
  }
}

export class ChunkError extends DianomeError {
  readonly sha: string;
  /** HTTP status when the failure came from a response; undefined for network errors. */
  readonly status: number | undefined;
  constructor(code: "chunk_http" | "chunk_network" | "chunk_verify" | "chunk_missing", sha: string, message: string, status?: number, options?: { cause?: unknown }) {
    super(code, `${sha.slice(0, 12)}: ${message}`, options);
    this.name = "ChunkError";
    this.sha = sha;
    this.status = status;
  }
}

export class AbortedError extends DianomeError {
  constructor() {
    super("aborted", "load aborted");
    this.name = "AbortError";
  }
}

/** True for the DOMException the Cache API and IndexedDB throw when storage is full (and for our own re-throw of it). */
export function isQuotaError(e: unknown): boolean {
  return typeof e === "object" && e !== null && (e as { name?: unknown }).name === "QuotaExceededError";
}

export function quotaError(detail = "storage quota exceeded"): Error {
  if (typeof DOMException !== "undefined") return new DOMException(detail, "QuotaExceededError");
  const e = new Error(detail);
  e.name = "QuotaExceededError";
  return e;
}

export function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw new AbortedError();
}

export function isAbort(e: unknown): boolean {
  return e instanceof AbortedError || (typeof e === "object" && e !== null && (e as { name?: unknown }).name === "AbortError");
}

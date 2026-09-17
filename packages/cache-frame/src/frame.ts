// The cache frame at cdn.dianome.dev/frame/v1/index.html. All logic lives in the SDK's FrameServer (imported from
// source and bundled here); this file wires it to the real document: origin locking, message dispatch, the
// opt-in button, and the browser surface (requestStorageAccess, permissions, globals, same-origin fetch).
//
// Rules enforced here (Phase 0): never post to "*"; accept messages only from window.parent and only from one
// origin; touch `caches`/`indexedDB` only after the grant (FrameServer constructs the store inside grant()).

import { FrameServer, type FramePlatform, type StorageAccessHandle } from "../../sdk/src/cache/frameServer";
import { epochNow, parseRequest, sameOrigin, stampSent, transferablesOf, versionMismatchId, PROTOCOL_VERSION, type FrameProgress, type FrameResponse, type GrantResult } from "../../sdk/src/cache/protocol";

const button = document.getElementById("enable") as HTMLButtonElement;
const note = document.getElementById("note") as HTMLParagraphElement;
const embedded = window.parent !== window;

if (!embedded) {
  button.hidden = true;
  note.hidden = false;
  note.textContent = "This page is the Dianome cache frame; sites embed it to share downloaded model files. To enable the shared cache on this device, use optin.html.";
}

// The parent's origin: from the referrer (the SDK sets referrerpolicy=strict-origin-when-cross-origin) or
// ancestorOrigins when the referrer was stripped; else locked to the first message that arrives from window.parent.
function referrerOrigin(): string | null {
  try { if (document.referrer) return new URL(document.referrer).origin; } catch { /* ignore */ }
  const anc = (location as Location & { ancestorOrigins?: DOMStringList }).ancestorOrigins;
  if (anc && anc.length > 0 && anc[0] && anc[0] !== "null") return anc[0];
  return null;
}
let parentOrigin: string | null = referrerOrigin();

// One click → one gesture-bound continuation. The SDK asks for `await-click` grants; the button stays disabled
// until one is waiting, and the click handler runs the request synchronously so requestStorageAccess is the first
// statement inside the gesture. The label is the user's only feedback inside a 48 px frame, so every outcome
// lands there (a browser permission prompt can keep "Requesting…" up until the user answers it).
const IDLE_LABEL = "Enable shared model cache";
let gestureWaiter: (() => void) | null = null;
button.addEventListener("click", () => {
  const w = gestureWaiter;
  gestureWaiter = null;
  if (!w) { button.disabled = true; button.textContent = IDLE_LABEL; return; }
  button.disabled = true;
  button.textContent = "Requesting permission…";
  w(); // runs the pending grant synchronously: requestStorageAccess is its first statement
});

function showOutcome(g: GrantResult): void {
  const labels: Record<GrantResult["state"], string> = {
    granted: "Shared model cache enabled",
    "needs-visit": "Visit the cache page first (see the site)",
    "needs-click": IDLE_LABEL,
    denied: "Permission denied by the browser",
    unsupported: "Not supported in this browser",
  };
  button.textContent = labels[g.state];
  button.disabled = true; // needs-click: inGesture() re-enables it when the SDK is actually waiting for the click
}

const platform: FramePlatform = {
  origin: location.origin,
  log: (m) => console.log(m),
  ...(typeof document.requestStorageAccess === "function"
    ? { requestStorageAccess: (opts?: { all: true }) => (document.requestStorageAccess as (o?: unknown) => Promise<StorageAccessHandle | undefined>)(opts) }
    : {}),
  ...(typeof document.hasStorageAccess === "function" ? { hasStorageAccess: () => document.hasStorageAccess() } : {}),
  permissionState: async () => (await navigator.permissions.query({ name: "storage-access" as PermissionName })).state,
  globals: () => ({ caches, indexedDB, storage: navigator.storage }),
  fetch: (url, init) => fetch(url, init),
  inGesture: <T>(run: () => Promise<T>) => new Promise<T>((resolve, reject) => {
    button.disabled = false;
    gestureWaiter = () => { run().then(resolve, reject); };
  }),
};
const server = new FrameServer({ platform });

function reply(res: FrameResponse): void {
  if (!parentOrigin) return;
  const transfer = transferablesOf(res);
  window.parent.postMessage(stampSent(res), parentOrigin, transfer); // stamped last: the hop starts at the post
}

window.addEventListener("message", (ev: MessageEvent) => {
  const receivedAt = epochNow(); // the request hop ends here, before any validation work
  if (!embedded || ev.source !== window.parent) return;
  if (parentOrigin === null) { if (!ev.origin || ev.origin === "null") return; parentOrigin = ev.origin; }
  if (!sameOrigin(ev.origin, parentOrigin)) return;
  const badVersion = versionMismatchId(ev.data);
  if (badVersion !== null) return reply({ v: PROTOCOL_VERSION, id: badVersion, ok: false, error: "unsupported protocol version", code: "bad_version", ms: 0 });
  const req = parseRequest(ev.data);
  if (!req) {
    const id = (ev.data as { id?: unknown } | null)?.id;
    if (typeof id === "number") reply({ v: PROTOCOL_VERSION, id, ok: false, error: "malformed request", code: "bad_request", ms: 0 });
    return;
  }
  const progress = req.op === "grant" ? (stage: FrameProgress["progress"]) => { if (parentOrigin) window.parent.postMessage({ v: PROTOCOL_VERSION, id: req.id, progress: stage } satisfies FrameProgress, parentOrigin); } : undefined;
  void server.handle(req, receivedAt, progress).then((res) => {
    if (req.op === "grant" && res.ok) showOutcome(res.result as GrantResult);
    reply(res);
  });
});

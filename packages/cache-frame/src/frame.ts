// The cache frame at cdn.dianome.dev/frame/v1/index.html. All logic lives in the SDK's FrameServer (imported from
// source and bundled here); this file wires it to the real document: origin locking, message dispatch, the
// opt-in button, and the browser surface (requestStorageAccess, permissions, globals, same-origin fetch).
//
// Rules enforced here (Phase 0): never post to "*"; accept messages only from window.parent and only from one
// origin; touch `caches`/`indexedDB` only after the grant (FrameServer constructs the store inside grant()).

import { FrameServer, type FramePlatform, type StorageAccessHandle } from "../../sdk/src/cache/frameServer";
import { parseRequest, sameOrigin, transferablesOf, versionMismatchId, PROTOCOL_VERSION, type FrameResponse } from "../../sdk/src/cache/protocol";

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

// One click → one gesture-bound continuation. The SDK asks for `await-click` grants; the click handler runs the
// request synchronously so requestStorageAccess is the first statement inside the gesture.
let gestureWaiter: (() => void) | null = null;
button.addEventListener("click", () => {
  const w = gestureWaiter;
  gestureWaiter = null;
  if (!w) { note.hidden = false; note.textContent = "Nothing is waiting for this click; reload the page that embeds this frame."; return; }
  button.disabled = true;
  w(); // runs the pending grant synchronously: requestStorageAccess is its first statement
});

const platform: FramePlatform = {
  origin: location.origin,
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
  window.parent.postMessage(res, parentOrigin, transferablesOf(res));
}

window.addEventListener("message", (ev: MessageEvent) => {
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
  void server.handle(req).then((res) => {
    if (req.op === "grant" && res.ok && (res.result as { state?: string }).state === "granted") { button.disabled = true; button.textContent = "Shared model cache enabled"; }
    reply(res);
  });
});

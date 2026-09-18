// Static server for the gate suite (port 8798): the harness page and bundle, the fixtures directory, and the
// local chunk store with the api/cdn URL shapes (manifest + chunks) on one origin, so the SDK's stream() works
// unchanged with api = cdn = this origin. Telemetry POSTs are accepted and dropped.
//
//   node test/e2e/run-server.mjs        (builds the bundle first)
import http from "node:http";
import { createReadStream, existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";

const port = Number(process.env.RUNTIME_E2E_PORT ?? 8798);
const here = resolve("test/e2e");
const repo = resolve(here, "../../../..");
const store = process.env.DIANOME_STORE ? resolve(process.env.DIANOME_STORE) : join(repo, "store");
const fixtures = join(repo, "fixtures/qwen2.5-0.5b-instruct");
const bundle = resolve("test-results/harness-bundle.js");
const page = readFileSync(join(here, "page/index.html"), "utf8");

const cors = { "Access-Control-Allow-Origin": "*", "Access-Control-Expose-Headers": "ETag, X-Dianome-Manifest-Sha, Content-Length" };
const IMMUTABLE = "public, max-age=31536000, immutable";

function send(res: http.ServerResponse, status: number, body: string | Uint8Array, headers: Record<string, string> = {}): void {
  res.writeHead(status, { ...cors, ...headers });
  res.end(body);
}
function file(res: http.ServerResponse, path: string, type: string, cacheControl: string, extra: Record<string, string> = {}): void {
  let st;
  try { st = statSync(path); } catch { return send(res, 404, `not found: ${path}`); }
  if (!st.isFile()) return send(res, 404, "not found");
  res.writeHead(200, { ...cors, "Content-Type": type, "Content-Length": String(st.size), "Cache-Control": cacheControl, ...extra });
  createReadStream(path).pipe(res);
}

http.createServer((req, res) => {
  const url = new URL(req.url ?? "/", "http://x");
  const p = url.pathname;
  if (req.method === "OPTIONS") return send(res, 204, "", { "Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS", "Access-Control-Allow-Headers": "Content-Type, If-None-Match" });
  if (p === "/healthz") return send(res, 200, "ok");
  if (p === "/" || p === "/index.html") return send(res, 200, page, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" });
  if (p === "/bundle.js") return file(res, bundle, "text/javascript; charset=utf-8", "no-store");
  if (p === "/bundle.js.map") return file(res, bundle + ".map", "application/json", "no-store");
  let m: RegExpExecArray | null;
  if ((m = /^\/fixtures\/(.+)$/.exec(p))) {
    const rel = decodeURIComponent(m[1]!);
    if (rel.includes("..")) return send(res, 400, "bad path");
    return file(res, join(fixtures, rel), rel.endsWith(".json") ? "application/json" : "application/octet-stream", "no-store");
  }
  if ((m = /^\/v1\/models\/([a-z0-9][a-z0-9._-]*)\/manifest$/.exec(p))) {
    const dir = join(store, "manifests", m[1]!);
    if (!existsSync(dir)) return send(res, 404, JSON.stringify({ error: "not_found" }), { "Content-Type": "application/json" });
    const sha = readdirSync(dir).map((f) => /^([0-9a-f]{64})\.json$/.exec(f)?.[1]).find(Boolean) ?? "";
    return file(res, join(dir, "latest.json"), "application/json", "no-store", { ETag: `"${sha}"`, "X-Dianome-Manifest-Sha": sha });
  }
  if ((m = /^\/chunks\/([0-9a-f]{64})$/.exec(p))) return file(res, join(store, "chunks", m[1]!), "application/octet-stream", IMMUTABLE);
  if (p === "/v1/telemetry/load" && req.method === "POST") { req.resume(); req.on("end", () => send(res, 202, "")); return; }
  return send(res, 404, "not found");
}).listen(port, "127.0.0.1", () => console.log(`runtime e2e server on http://127.0.0.1:${port}/ (store ${store}, fixtures ${fixtures})`));

// Static server for the site e2e (port 8797): dist/ (built first), the local chunk store with the api/cdn URL shapes
// (/v1/models, /v1/models/:id/manifest, /chunks/:sha), and a proxy of the Worker routes (/v1/split/*, /v1/stats/*,
// /v1/telemetry/load, and Phase 6's /v1/keys + /v1/me/* with Authorization forwarded) to wrangler dev on 8787 with this
// origin, so the page has one `api` origin. Telemetry bodies
// and the Worker's status codes are recorded and readable at /__telemetry.
import http from "node:http";
import { createReadStream, existsSync, readdirSync, statSync } from "node:fs";
import { extname, join, resolve } from "node:path";

const port = Number(process.env.SITE_E2E_PORT ?? 8797);
const worker = process.env.SITE_E2E_WORKER ?? "http://127.0.0.1:8787";
const here = resolve(import.meta.dirname);
const repo = resolve(here, "../../../..");
const dist = resolve(here, "../../dist");
const store = process.env.DIANOME_STORE ? resolve(process.env.DIANOME_STORE) : join(repo, "store");
if (!existsSync(join(dist, "index.html"))) { console.error(`no build at ${dist}: run \`pnpm --filter site build\` first`); process.exit(1); }
const telemetry = [];
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".json": "application/json", ".map": "application/json", ".txt": "text/plain; charset=utf-8" };
const cors = { "Access-Control-Allow-Origin": "*", "Access-Control-Expose-Headers": "ETag, X-Dianome-Manifest-Sha, Content-Length" };
const IMMUTABLE = "public, max-age=31536000, immutable";
function send(res, status, body, headers = {}) { res.writeHead(status, { ...cors, ...headers }); res.end(body); }
function file(res, path, type, cacheControl, extra = {}) {
  let st;
  try { st = statSync(path); } catch { return send(res, 404, `not found: ${path}`); }
  if (!st.isFile()) return send(res, 404, "not found");
  res.writeHead(200, { ...cors, "Content-Type": type, "Content-Length": String(st.size), "Cache-Control": cacheControl, ...extra });
  createReadStream(path).pipe(res);
}
function readBody(req) { return new Promise((resolve) => { const chunks = []; req.on("data", (c) => chunks.push(c)); req.on("end", () => resolve(Buffer.concat(chunks))); }); }

http.createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", "http://x");
  const p = url.pathname;
  if (req.method === "OPTIONS") return send(res, 204, "", { "Access-Control-Allow-Methods": "GET, HEAD, POST, DELETE, OPTIONS", "Access-Control-Allow-Headers": "Content-Type, If-None-Match, Authorization" });
  if (p === "/healthz") return send(res, 200, "ok");
  if (p === "/__telemetry") return send(res, 200, JSON.stringify(telemetry), { "Content-Type": "application/json", "Cache-Control": "no-store" });
  let m;
  if (p === "/v1/models") {
    const ids = existsSync(join(store, "manifests")) ? readdirSync(join(store, "manifests")).filter((d) => existsSync(join(store, "manifests", d, "latest.json"))).sort() : [];
    return send(res, 200, JSON.stringify({ models: ids }), { "Content-Type": "application/json", "Cache-Control": "no-store" });
  }
  if ((m = /^\/v1\/models\/([a-z0-9][a-z0-9._-]*)\/manifest$/.exec(p))) {
    const dir = join(store, "manifests", m[1]);
    if (!existsSync(dir)) return send(res, 404, JSON.stringify({ error: "not_found" }), { "Content-Type": "application/json" });
    const sha = readdirSync(dir).map((f) => /^([0-9a-f]{64})\.json$/.exec(f)?.[1]).find(Boolean) ?? "";
    return file(res, join(dir, "latest.json"), "application/json", "no-store", { ETag: `"${sha}"`, "X-Dianome-Manifest-Sha": sha });
  }
  if ((m = /^\/chunks\/([0-9a-f]{64})$/.exec(p))) return file(res, join(store, "chunks", m[1]), "application/octet-stream", IMMUTABLE);
  if (p.startsWith("/v1/split/") || p.startsWith("/v1/stats/") || p === "/v1/telemetry/load" || p === "/v1/keys" || p.startsWith("/v1/me/")) {
    const body = req.method === "POST" ? await readBody(req) : undefined;
    try {
      const auth = req.headers["authorization"] ? { Authorization: req.headers["authorization"] } : {};
      const r = await fetch(worker + p + url.search, { method: req.method, headers: { "Content-Type": req.headers["content-type"] ?? "application/json", Origin: `http://127.0.0.1:${port}`, ...auth }, body });
      const text = await r.text();
      if (p === "/v1/telemetry/load" && body) telemetry.push({ at: new Date().toISOString(), status: r.status, body: JSON.parse(body.toString("utf8")), response: text });
      return send(res, r.status, text, { "Content-Type": r.headers.get("content-type") ?? "application/json", "Cache-Control": "no-store" });
    } catch (e) {
      return send(res, 502, JSON.stringify({ error: "worker_unreachable", detail: String(e) }), { "Content-Type": "application/json" });
    }
  }
  // the built site: /x/ -> /x/index.html
  const rel = decodeURIComponent(p);
  if (rel.includes("..")) return send(res, 400, "bad path");
  let path = join(dist, rel);
  if (existsSync(path) && statSync(path).isDirectory()) path = join(path, "index.html");
  return file(res, path, types[extname(path)] ?? "application/octet-stream", "no-store");
}).listen(port, "127.0.0.1", () => console.log(`site e2e server on http://127.0.0.1:${port}/ (dist ${dist}, store ${store}, worker ${worker})`));

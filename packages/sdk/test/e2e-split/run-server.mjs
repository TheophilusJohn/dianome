// Static server for the split e2e (port 8799): the harness page and bundle (SDK src + runtime dist, .wgsl inlined),
// the fixtures directory, the local chunk store with the api/cdn URL shapes, and a proxy of the Worker routes
// (/v1/split/*, /v1/telemetry/load) to wrangler dev on 8787 so the page has one `api` origin. Telemetry bodies and
// the Worker's status codes are recorded and readable at /__telemetry.
//
//   node test/e2e-split/run-server.mjs        (builds the bundle first)
import { build } from "esbuild";
import http from "node:http";
import { createReadStream, existsSync, mkdirSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";

const port = Number(process.env.SPLIT_E2E_PORT ?? 8799);
const worker = process.env.SPLIT_E2E_WORKER ?? "http://127.0.0.1:8787";
const here = resolve("test/e2e-split");
const repo = resolve(here, "../../../..");
const store = process.env.DIANOME_STORE ? resolve(process.env.DIANOME_STORE) : join(repo, "store");
const fixtures = join(repo, "fixtures/qwen2.5-0.5b-instruct");
const bundle = resolve("test-results/split-harness.js");
mkdirSync("test-results", { recursive: true });
await build({
  entryPoints: [join(here, "page/harness.ts")], bundle: true, format: "esm", platform: "browser", target: ["es2022"],
  loader: { ".wgsl": "text" }, sourcemap: true, outfile: bundle, logLevel: "warning",
});
const page = readFileSync(join(here, "page/index.html"), "utf8");
const telemetry = [];

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
  if (req.method === "OPTIONS") return send(res, 204, "", { "Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS", "Access-Control-Allow-Headers": "Content-Type, If-None-Match" });
  if (p === "/healthz") return send(res, 200, "ok");
  if (p === "/" || p === "/index.html") return send(res, 200, page, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" });
  if (p === "/bundle.js") return file(res, bundle, "text/javascript; charset=utf-8", "no-store");
  if (p === "/bundle.js.map") return file(res, bundle + ".map", "application/json", "no-store");
  if (p === "/__telemetry") return send(res, 200, JSON.stringify(telemetry), { "Content-Type": "application/json", "Cache-Control": "no-store" });
  let m;
  if ((m = /^\/fixtures\/(.+)$/.exec(p))) {
    const rel = decodeURIComponent(m[1]);
    if (rel.includes("..")) return send(res, 400, "bad path");
    return file(res, join(fixtures, rel), rel.endsWith(".json") ? "application/json" : "application/octet-stream", "no-store");
  }
  if ((m = /^\/v1\/models\/([a-z0-9][a-z0-9._-]*)\/manifest$/.exec(p))) {
    const dir = join(store, "manifests", m[1]);
    if (!existsSync(dir)) return send(res, 404, JSON.stringify({ error: "not_found" }), { "Content-Type": "application/json" });
    const sha = readdirSync(dir).map((f) => /^([0-9a-f]{64})\.json$/.exec(f)?.[1]).find(Boolean) ?? "";
    return file(res, join(dir, "latest.json"), "application/json", "no-store", { ETag: `"${sha}"`, "X-Dianome-Manifest-Sha": sha });
  }
  if ((m = /^\/chunks\/([0-9a-f]{64})$/.exec(p))) return file(res, join(store, "chunks", m[1]), "application/octet-stream", IMMUTABLE);
  if (p.startsWith("/v1/split/") || p === "/v1/telemetry/load") {
    // proxy to wrangler dev with the page's origin
    const body = req.method === "POST" ? await readBody(req) : undefined;
    try {
      const r = await fetch(worker + p + url.search, { method: req.method, headers: { "Content-Type": req.headers["content-type"] ?? "application/json", Origin: `http://127.0.0.1:${port}` }, body });
      const text = await r.text();
      if (p === "/v1/telemetry/load" && body) telemetry.push({ at: new Date().toISOString(), status: r.status, body: JSON.parse(body.toString("utf8")), response: text });
      return send(res, r.status, text, { "Content-Type": r.headers.get("content-type") ?? "application/json", "Cache-Control": "no-store" });
    } catch (e) {
      return send(res, 502, JSON.stringify({ error: "worker_unreachable", detail: String(e) }), { "Content-Type": "application/json" });
    }
  }
  return send(res, 404, "not found");
}).listen(port, "127.0.0.1", () => console.log(`split e2e server on http://127.0.0.1:${port}/ (store ${store}, worker ${worker})`));

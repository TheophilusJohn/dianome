// Local stand-in for api.dianome.dev + cdn.dianome.dev over ./store, for the demo site (`?api=&cdn=` overrides)
// and the Playwright e2e run. CORS *, Range not needed.
//
//   node scripts/serve-store.mjs [--store ./store] [--port 8788] [--frame packages/cache-frame/dist]
//
//   GET /v1/models                       -> ids under <store>/manifests
//   GET /v1/models/<id>/manifest         -> <store>/manifests/<id>/latest.json (+ X-Dianome-Manifest-Sha, ETag)
//   GET /v1/models/<id>/manifest/<sha>   -> the immutable manifest
//   POST /v1/telemetry/load              -> 202, body logged to stdout as one JSON line
//   GET /chunks/<sha>                    -> <store>/chunks/<sha>
//   GET /frame/v1/<file>                 -> <frame dir>/<file>   (index.html, frame.js, optin.html)
import http from "node:http";
import { createReadStream, readdirSync, readFileSync, statSync } from "node:fs";
import { resolve, join } from "node:path";

const args = process.argv.slice(2);
const opt = (name, dflt) => { const i = args.indexOf(`--${name}`); return i >= 0 ? args[i + 1] : dflt; };
const root = resolve(opt("store", "store"));
const frameDir = resolve(opt("frame", "packages/cache-frame/dist"));
const port = Number(opt("port", 8788));
const quiet = args.includes("--quiet");

const IMMUTABLE = "public, max-age=31536000, immutable";
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".json": "application/json", ".map": "application/json" };

function send(res, status, body, headers = {}) {
  res.writeHead(status, { "Access-Control-Allow-Origin": "*", "Access-Control-Expose-Headers": "ETag, X-Dianome-Manifest-Sha, Content-Length", ...headers });
  res.end(body);
}
function file(res, path, type, cacheControl, extra = {}) {
  let st;
  try { st = statSync(path); } catch { return send(res, 404, "not found"); }
  if (!st.isFile()) return send(res, 404, "not found");
  res.writeHead(200, { "Access-Control-Allow-Origin": "*", "Access-Control-Expose-Headers": "ETag, X-Dianome-Manifest-Sha, Content-Length", "Content-Type": type, "Content-Length": st.size, "Cache-Control": cacheControl, ...extra });
  createReadStream(path).pipe(res);
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url ?? "/", "http://x");
  const p = url.pathname;
  if (req.method === "OPTIONS") return send(res, 204, "", { "Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS", "Access-Control-Allow-Headers": "Content-Type, If-None-Match" });
  let m;
  if (p === "/v1/models") {
    let ids = [];
    try { ids = readdirSync(join(root, "manifests")).filter((d) => !d.startsWith(".")); } catch { /* empty */ }
    return send(res, 200, JSON.stringify({ models: ids }), { "Content-Type": "application/json" });
  }
  if ((m = /^\/v1\/models\/([a-z0-9][a-z0-9._-]*)\/manifest$/.exec(p))) {
    const dir = join(root, "manifests", m[1]);
    let sha = "";
    try { sha = readdirSync(dir).map((f) => /^([0-9a-f]{64})\.json$/.exec(f)?.[1]).find(Boolean) ?? ""; } catch { return send(res, 404, JSON.stringify({ error: "not_found" }), { "Content-Type": "application/json" }); }
    const etag = `"${sha || "latest"}"`;
    if (req.headers["if-none-match"] === etag) return send(res, 304, "", { ETag: etag, "X-Dianome-Manifest-Sha": sha });
    return file(res, join(dir, "latest.json"), "application/json", "public, max-age=60", { ETag: etag, "X-Dianome-Manifest-Sha": sha });
  }
  if ((m = /^\/v1\/models\/([a-z0-9][a-z0-9._-]*)\/manifest\/([0-9a-f]{64})$/.exec(p))) return file(res, join(root, "manifests", m[1], `${m[2]}.json`), "application/json", IMMUTABLE);
  if (p === "/v1/telemetry/load" && req.method === "POST") {
    let body = "";
    req.on("data", (c) => { body += c; });
    req.on("end", () => { if (!quiet) console.log(`telemetry ${body}`); send(res, 202, ""); });
    return;
  }
  if ((m = /^\/chunks\/([0-9a-f]{64})$/.exec(p))) return file(res, join(root, "chunks", m[1]), "application/octet-stream", IMMUTABLE);
  // The frame carries COEP + CORP so that host pages that set Cross-Origin-Embedder-Policy can still embed it.
  if ((m = /^\/frame\/v1\/([a-z]+\.(?:html|js|map))$/.exec(p))) return file(res, join(frameDir, m[1]), TYPES[m[1].slice(m[1].lastIndexOf("."))] ?? "application/octet-stream", "public, max-age=300", { "Cross-Origin-Embedder-Policy": "credentialless", "Cross-Origin-Resource-Policy": "cross-origin" });
  if ((m = /^\/manifests\/([a-z0-9][a-z0-9._-]*)\/(latest|[0-9a-f]{64})\.json$/.exec(p))) return file(res, join(root, "manifests", m[1], `${m[2]}.json`), "application/json", m[2] === "latest" ? "public, max-age=60" : IMMUTABLE);
  return send(res, 404, "not found");
});
server.listen(port, () => { if (!quiet) console.log(`serving ${root} (frame: ${frameDir}) on http://localhost:${port}/`); });

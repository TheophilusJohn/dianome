// E2E server: builds a synthetic model store in memory (real SHA-256 chunks, chunkSize shrunk so groups span
// chunks), serves it with the API + CDN URL shapes on one origin, serves the built SDK (dist/index.js) and the test
// page, and logs telemetry POSTs so the spec can assert on them.
//
//   node --experimental-strip-types test/e2e/server.ts        (port 8797)
import http from "node:http";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { synthModel } from "../fixtures/synth";

const port = 8797;
// Paths are relative to the package directory (Playwright runs the webServer command there).
const here = resolve("test/e2e");
// ~6 MB q4 model: 8 layers of 1024x1024 q4 q_proj (~544 KB each) + q8 embed; 256 KB chunks so every group spans
// several chunks and entries straddle chunk boundaries.
const model = await synthModel({ id: "e2e-model", layers: 8, hidden: 1024, vocab: 512, chunkSize: 1 << 18, variants: ["fp16", "q4"] });
const manifestJson = JSON.stringify(model.manifest);
const telemetry: unknown[] = [];
let chunkHits = 0;
const dist = resolve(here, "../../dist");
const page = readFileSync(resolve(here, "page/index.html"), "utf8");

const cors = { "Access-Control-Allow-Origin": "*", "Access-Control-Expose-Headers": "ETag, X-Dianome-Manifest-Sha, Content-Length" };
http.createServer((req, res) => {
  const url = new URL(req.url ?? "/", "http://x");
  const p = url.pathname;
  const send = (status: number, body: string | Uint8Array, headers: Record<string, string> = {}) => { res.writeHead(status, { ...cors, ...headers }); res.end(body); };
  if (req.method === "OPTIONS") return send(204, "", { "Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS", "Access-Control-Allow-Headers": "Content-Type, If-None-Match" });
  if (p === "/healthz") return send(200, "ok");
  if (p === "/" || p === "/index.html") return send(200, page, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" });
  if (p.startsWith("/sdk/")) {
    try { return send(200, readFileSync(resolve(dist, p.slice(5))), { "Content-Type": p.endsWith(".map") ? "application/json" : "text/javascript", "Cache-Control": "no-store" }); }
    catch { return send(404, "build the sdk first: pnpm --filter dianome build"); }
  }
  if (p === "/v1/models/e2e-model/manifest") return send(200, manifestJson, { "Content-Type": "application/json", "X-Dianome-Manifest-Sha": "e".repeat(64), ETag: '"e2e"', "Cache-Control": "no-store" });
  const c = /^\/chunks\/([0-9a-f]{64})$/.exec(p);
  if (c) {
    const body = model.chunks.get(c[1]!);
    if (!body) return send(404, "missing");
    chunkHits++;
    return send(200, body, { "Content-Type": "application/octet-stream", "Content-Length": String(body.length), "Cache-Control": "public, max-age=31536000, immutable" });
  }
  if (p === "/v1/telemetry/load" && req.method === "POST") {
    let body = "";
    req.on("data", (d) => { body += d; });
    req.on("end", () => { telemetry.push(JSON.parse(body)); send(202, ""); });
    return;
  }
  // Test hooks.
  if (p === "/__telemetry") return send(200, JSON.stringify(telemetry), { "Content-Type": "application/json", "Cache-Control": "no-store" });
  if (p === "/__chunk-hits") return send(200, JSON.stringify({ chunkHits }), { "Content-Type": "application/json", "Cache-Control": "no-store" });
  if (p === "/__reset") { telemetry.length = 0; chunkHits = 0; return send(200, "ok"); }
  if (p === "/__manifest") return send(200, manifestJson, { "Content-Type": "application/json" });
  return send(404, "not found");
}).listen(port, "127.0.0.1", () => console.log(`e2e server on http://127.0.0.1:${port}/ (${model.chunks.size} chunks)`));

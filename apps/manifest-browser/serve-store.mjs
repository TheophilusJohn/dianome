// Static server for ../../store with CORS, mirroring the R2 key layout.
// Usage: node serve-store.mjs [dir] [port]   (defaults: ../../store, 8788)
import http from "node:http";
import { createReadStream, statSync } from "node:fs";
import { resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));
const root = resolve(process.argv[2] ?? resolve(here, "../../store"));
const port = Number(process.argv[3] ?? 8788);

http
  .createServer((req, res) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS");
    if (req.method === "OPTIONS") return void res.writeHead(204).end();
    const url = new URL(req.url ?? "/", "http://x");
    const path = resolve(root, "." + decodeURIComponent(url.pathname));
    if (!path.startsWith(root + sep) && path !== root) return void res.writeHead(403).end();
    let st;
    try {
      st = statSync(path);
    } catch {
      return void res.writeHead(404).end("not found");
    }
    if (!st.isFile()) return void res.writeHead(404).end("not found");
    const isJson = path.endsWith(".json");
    res.writeHead(200, {
      "Content-Type": isJson ? "application/json" : "application/octet-stream",
      "Content-Length": st.size,
      "Cache-Control": path.includes(`${sep}chunks${sep}`) || /[0-9a-f]{64}\.json$/.test(path) ? "public, max-age=31536000, immutable" : "public, max-age=60",
    });
    if (req.method === "HEAD") return void res.end();
    createReadStream(path).pipe(res);
  })
  .listen(port, () => console.log(`serving ${root} on http://localhost:${port}/ (CORS *)`));

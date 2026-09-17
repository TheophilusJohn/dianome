// Asserts the gzipped size of the main entry (dist/index.js plus the shared chunks it imports) is under budget.
// Run after `pnpm build`; used by `pnpm ci` and `prepack`.
import { gzipSync } from "node:zlib";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const BUDGET = 25 * 1024;
const dist = "dist";

function graph(entry, seen = new Set()) {
  if (seen.has(entry)) return seen;
  seen.add(entry);
  const src = readFileSync(join(dist, entry), "utf8");
  for (const m of src.matchAll(/from\s*"(\.\/[^"]+\.js)"|import\s*"(\.\/[^"]+\.js)"/g)) graph((m[1] ?? m[2]).slice(2), seen);
  return seen;
}

let failed = false;
for (const entry of readdirSync(dist).filter((f) => /^(index|transformersjs|webllm)\.js$/.test(f))) {
  const files = [...graph(entry)];
  const raw = files.reduce((n, f) => n + readFileSync(join(dist, f)).length, 0);
  const gz = files.reduce((n, f) => n + gzipSync(readFileSync(join(dist, f)), { level: 9 }).length, 0);
  const line = `${entry.padEnd(18)} ${String(raw).padStart(7)} bytes  ${String(gz).padStart(6)} gzipped  (${files.join(", ")})`;
  if (entry === "index.js" && gz > BUDGET) { failed = true; console.error(`FAIL ${line} > budget ${BUDGET}`); }
  else console.log(`ok   ${line}`);
}
if (failed) process.exit(1);

// Builds the browser harness bundle (runtime + SDK + test page code, .wgsl as text) and the server, then runs it.
import { build } from "esbuild";
import { mkdirSync } from "node:fs";
import { pathToFileURL } from "node:url";

mkdirSync("test-results", { recursive: true });
await build({
  entryPoints: ["test/e2e/page/harness.ts"], bundle: true, format: "esm", platform: "browser", target: ["es2022"],
  loader: { ".wgsl": "text" }, sourcemap: true, outfile: "test-results/harness-bundle.js", logLevel: "warning",
});
await build({ entryPoints: ["test/e2e/server.ts"], bundle: true, platform: "node", format: "esm", target: "node22", outfile: "test-results/e2e-server.mjs", logLevel: "warning" });
await import(pathToFileURL("test-results/e2e-server.mjs").href);

// Bundles test/e2e/server.ts (it imports the TS fixtures, which node cannot resolve extensionless) and runs it.
import { build } from "esbuild";
import { mkdirSync } from "node:fs";
import { pathToFileURL } from "node:url";

mkdirSync("test-results", { recursive: true });
await build({ entryPoints: ["test/e2e/server.ts"], bundle: true, platform: "node", format: "esm", target: "node22", outfile: "test-results/e2e-server.mjs", logLevel: "warning" });
await import(pathToFileURL("test-results/e2e-server.mjs").href);

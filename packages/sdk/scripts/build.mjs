// Bundles the three entry points with esbuild (ESM, code-split so adapters share the core), no minification of
// identifiers beyond whitespace so stack traces stay readable. Declarations come from `tsc -p tsconfig.build.json`.
import { build } from "esbuild";
import { rmSync } from "node:fs";

rmSync("dist", { recursive: true, force: true });
const result = await build({
  entryPoints: { index: "src/index.ts", transformersjs: "src/adapters/transformersjs.ts", webllm: "src/adapters/webllm.ts" },
  outdir: "dist",
  bundle: true,
  format: "esm",
  platform: "browser",
  target: ["es2022"],
  splitting: true,
  // `Dianome.run()` reaches the run entry through a literal `import("./run.js")` that stays as written, so the main
  // graph (index + its chunks) does not change shape; run.js is bundled separately below.
  external: ["./run.js", "dianome-runtime", "dianome-runtime/*"],
  sourcemap: true,
  minifySyntax: true,
  minifyWhitespace: true,
  treeShaking: true,
  legalComments: "none",
  metafile: true,
  logLevel: "info",
});
const outputs = Object.entries(result.metafile.outputs).filter(([f]) => f.endsWith(".js"));
for (const [f, o] of outputs) console.log(`${f}\t${o.bytes} bytes`);

// The run entry (`dianome/run`, Phase 5b): its own bundle with private copies of the small shared modules, so the
// main entry's bytes do not move. The runtime package stays an import of the consumer's own copy.
const runResult = await build({
  entryPoints: { run: "src/run.ts" },
  outdir: "dist",
  bundle: true,
  format: "esm",
  platform: "browser",
  target: ["es2022"],
  external: ["dianome-runtime", "dianome-runtime/*"],
  sourcemap: true,
  minifySyntax: true,
  minifyWhitespace: true,
  treeShaking: true,
  legalComments: "none",
  metafile: true,
  logLevel: "info",
});
for (const [f, o] of Object.entries(runResult.metafile.outputs).filter(([f]) => f.endsWith(".js"))) console.log(`${f}\t${o.bytes} bytes`);

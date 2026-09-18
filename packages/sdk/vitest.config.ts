import { defineConfig } from "vitest/config";
import { resolve } from "node:path";

// Unit tests run in node by default; a file that needs a DOM opts in with `// @vitest-environment happy-dom`.
export default defineConfig({
  resolve: {
    alias: [
      { find: "dianome-runtime/tokenizer", replacement: resolve(import.meta.dirname, "../runtime/src/entries/tokenizer.ts") },
      { find: "dianome-runtime/planner", replacement: resolve(import.meta.dirname, "../runtime/src/entries/planner.ts") },
      { find: "dianome-runtime/protocol", replacement: resolve(import.meta.dirname, "../runtime/src/entries/protocol.ts") },
      { find: "dianome-runtime/session", replacement: resolve(import.meta.dirname, "../runtime/src/entries/session.ts") },
      // The main entry (WebGPU kernels) must never be imported in node: the stub throws on import.
      { find: /^dianome-runtime$/, replacement: resolve(import.meta.dirname, "test/helpers/runtimeMainStub.ts") },
    ],
  },
  test: {
    include: ["test/**/*.test.ts"],
    exclude: ["test/e2e/**"],
  },
});

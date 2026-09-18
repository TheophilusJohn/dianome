import { defineConfig, type Plugin } from "vitest/config";

// WGSL kernels are imported as strings (esbuild `text` loader in the build); this plugin does the same for vitest.
const wgsl: Plugin = { name: "wgsl-text", transform(code, id) { if (id.endsWith(".wgsl")) return { code: `export default ${JSON.stringify(code)};`, map: null }; return null; } };

export default defineConfig({
  plugins: [wgsl],
  test: { include: ["test/**/*.test.ts"], exclude: ["test/e2e/**"], testTimeout: 60_000 },
});

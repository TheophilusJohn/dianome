import { defineConfig } from "vite";

// Runtime bench page. Local: `pnpm --filter runtime-bench dev` on 5176 with `?api=http://localhost:8788&cdn=http://localhost:8788`
// pointing at `node scripts/serve-store.mjs` (repo root). The runtime package is consumed from its build
// (`pnpm --filter dianome-runtime build`), like the SDK.
export default defineConfig({
  server: { port: 5176 },
  preview: { port: 5176 },
  build: { target: "es2022" },
});

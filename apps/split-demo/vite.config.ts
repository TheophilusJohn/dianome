import { defineConfig } from "vite";

// Split demo (Phase 5b). Deploy target: dianome-demo-split.pages.dev (Theo). Local:
//   pnpm --filter split-demo dev   → http://localhost:5177/?api=http://127.0.0.1:8787&cdn=http://localhost:8788&split=ws://127.0.0.1:8765&token=$SPLIT_TOKEN
// with `node scripts/serve-store.mjs` (8788), `wrangler dev` in packages/worker (8787) and `dianome-server serve` (8765).
// `split=` + `token=` bypass the Worker's session endpoint (static bearer); without them the page mints a session token
// from `${api}/v1/split/session`, which only works from an allowed origin.
export default defineConfig({
  server: { port: 5177 },
  preview: { port: 5177 },
  build: { target: "es2022" },
});

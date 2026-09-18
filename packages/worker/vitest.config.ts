import { cloudflareTest } from "@cloudflare/vitest-pool-workers";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [
    cloudflareTest({
      wrangler: { configPath: "./wrangler.toml" },
      // Never let tests touch the real bucket: STORE has `remote = true` for `wrangler dev` only.
      remoteBindings: false,
      miniflare: { r2Buckets: ["STORE"], kvNamespaces: ["STATS_CACHE"], bindings: { GIT_SHA: "test", SPLIT_SIGNING_KEY: "test-signing-key", SPLIT_WS_URL: "ws://127.0.0.1:8765", SPLIT_PLAN_URL: "http://127.0.0.1:8765/plan", SPLIT_ALLOWED_ORIGINS: "http://localhost:5177,https://dianome-demo-split.pages.dev" } },
    }),
  ],
});

import { cloudflareTest } from "@cloudflare/vitest-pool-workers";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [
    cloudflareTest({
      wrangler: { configPath: "./wrangler.toml" },
      // Never let tests touch the real bucket: STORE has `remote = true` for `wrangler dev` only.
      remoteBindings: false,
      miniflare: { r2Buckets: ["STORE"], kvNamespaces: ["STATS_CACHE"], bindings: { GIT_SHA: "test" } },
    }),
  ],
});

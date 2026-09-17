import { defineConfig } from "vitest/config";

// Unit tests run in node by default; a file that needs a DOM opts in with `// @vitest-environment happy-dom`.
export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    exclude: ["test/e2e/**"],
  },
});

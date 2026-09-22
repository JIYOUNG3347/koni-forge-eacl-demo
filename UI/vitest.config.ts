import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react-swc";
import path from "path";

import viteConfig from "./vite.config";

// Reuse the app's versioned→unversioned aliases (e.g. "@radix-ui/react-slot@1.1.2"
// → "@radix-ui/react-slot") so component modules that import the pinned
// specifiers resolve under vitest exactly as they do in the build. Spread
// (not duplicate) to avoid drift with vite.config.ts.
const viteAlias = (viteConfig as { resolve?: { alias?: Record<string, string> } }).resolve?.alias ?? {};

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    passWithNoTests: true,
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
  },
  resolve: {
    alias: {
      ...viteAlias,
      "@": path.resolve(__dirname, "./src"),
    },
  },
});

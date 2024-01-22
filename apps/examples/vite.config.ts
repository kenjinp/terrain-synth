import react from "@vitejs/plugin-react-swc"
import { defineConfig } from "vite"
import glsl from "vite-plugin-glsl"
import svgr from "vite-plugin-svgr"
import { getLastCommit } from "./commit-info"
const commitInfo = getLastCommit({})

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), svgr(), glsl()],
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  optimizeDeps: {
    exclude: ["@hello-worlds/planets", "@hello-worlds/react"],
  },
  build: {
    outDir: "../../_dist",
  },
  define: {
    __COMMIT_INFO__: JSON.stringify(commitInfo),
    __BUILD_INFO__: JSON.stringify({
      buildTime: Date.now(),
    }),
  },
})

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Vite serves the SPA on :5173 in dev and proxies API calls + MJPEG streams to Flask.
// In prod, `pnpm build` outputs to webapp/backend/static and Flask serves the whole app.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: path.resolve(__dirname, "../backend/static"),
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api":    { target: "http://127.0.0.1:5000", changeOrigin: true },
      "/camera": { target: "http://127.0.0.1:5000", changeOrigin: true, ws: false },
    },
  },
});

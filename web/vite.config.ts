import { defineConfig, type Plugin, type Connect } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";

// Cross-origin isolation: gives the worker ~5 µs performance.now() (vs ~100 µs)
// and, later, SharedArrayBuffer for threaded wasm.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "same-origin",
};

const root = path.resolve(__dirname, "..");
const WEIGHTS_DIR = process.env.NANOINFER_WEIGHTS ?? path.join(root, "weights");
const BENCH_FILE = path.join(root, "bench", "results.jsonl");

/** Local-dev only: serve ../weights at /weights and ../bench/results.jsonl. */
function localData(): Plugin {
  const mw: Connect.NextHandleFunction = (req, res, next) => {
    const url = decodeURIComponent((req.url ?? "").split("?")[0]);
    for (const [k, v] of Object.entries(isolation)) res.setHeader(k, v);
    if (url === "/bench/results.jsonl") {
      res.setHeader("Content-Type", "application/x-ndjson");
      res.setHeader("Cache-Control", "no-store");
      res.end(fs.existsSync(BENCH_FILE) ? fs.readFileSync(BENCH_FILE) : "");
      return;
    }
    if (url.startsWith("/weights/")) {
      const file = path.join(WEIGHTS_DIR, url.slice("/weights/".length));
      if (!file.startsWith(WEIGHTS_DIR) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
        res.statusCode = 404;
        res.end(`not found: ${url} (run scripts/fetch_weights.sh)`);
        return;
      }
      const st = fs.statSync(file);
      res.setHeader("Content-Length", String(st.size));
      res.setHeader("Content-Type", "application/octet-stream");
      res.setHeader("Last-Modified", st.mtime.toUTCString());
      res.setHeader("ETag", `"${st.size}-${st.mtimeMs}"`);
      fs.createReadStream(file).pipe(res);
      return;
    }
    next();
  };
  return {
    name: "nanoinfer-local-data",
    configureServer: (s) => void s.middlewares.use(mw),
    configurePreviewServer: (s) => void s.middlewares.use(mw),
  };
}

export default defineConfig({
  plugins: [react(), localData()],
  server: { headers: isolation, port: 5199 },
  preview: { headers: isolation, port: 5199 },
  worker: { format: "es" },
  build: { target: "es2022" },
});

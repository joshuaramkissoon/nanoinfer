import { defineConfig, loadEnv, type Plugin, type Connect } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { askEnvFrom, handleAsk } from "./server/ask";

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

/** Select-to-ask in dev: /api/ask runs server/ask.ts here, so the key (from the shell
 *  env or web/.env.local, never VITE_-prefixed) stays on the server. */
function askApi(mode: string): Plugin {
  const env = { ...loadEnv(mode, __dirname, ""), ...process.env } as Record<string, string | undefined>;
  const mw: Connect.NextHandleFunction = async (req, res, next) => {
    if (!(req.url ?? "").startsWith("/api/ask")) return next();
    try {
      const chunks: Buffer[] = [];
      for await (const c of req) chunks.push(c as Buffer);
      const request = new Request(`http://localhost${req.url}`, {
        method: req.method,
        headers: { "content-type": String(req.headers["content-type"] ?? "application/json") },
        body: req.method === "POST" ? Buffer.concat(chunks) : undefined,
      });
      const r = await handleAsk(request, askEnvFrom(env), req.socket.remoteAddress ?? "?");
      res.statusCode = r.status;
      r.headers.forEach((v, k) => res.setHeader(k, v));
      for (const [k, v] of Object.entries(isolation)) res.setHeader(k, v);
      if (r.body) Readable.fromWeb(r.body as import("node:stream/web").ReadableStream).pipe(res);
      else res.end();
    } catch (e) {
      res.statusCode = 500;
      res.end(JSON.stringify({ error: (e as Error).message }));
    }
  };
  return {
    name: "nanoinfer-ask",
    configureServer: (s) => void s.middlewares.use(mw),
    configurePreviewServer: (s) => void s.middlewares.use(mw),
  };
}

export default defineConfig(({ mode }) => ({
  plugins: [react(), localData(), askApi(mode)],
  server: { headers: isolation, port: 5199 },
  preview: { headers: isolation, port: 5199 },
  worker: { format: "es" },
  build: { target: "es2022" },
}));

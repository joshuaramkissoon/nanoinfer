// Select-to-ask: a small proxy from the page to an OpenAI-compatible chat API
// (Together, like Tangent). The API key lives only here, on the server: in dev
// this runs inside the Vite dev server (vite.config.ts), in production as a
// Vercel function (api/ask.ts). Web-standard Request/Response so both can share it.
//
// Env:
//   TOGETHER_API_KEY        the key (or NANOINFER_ASK_KEY for another provider)
//   NANOINFER_ASK_MODEL     comma-separated preference list; the first one the
//                           provider serves wins (default below)
//   NANOINFER_ASK_BASE_URL  default https://api.together.xyz/v1
//   NANOINFER_ASK_RATE_LIMIT=off  disables the per-IP limit

export interface AskEnv {
  key: string | undefined;
  keyEnv: string;
  baseUrl: string;
  models: string[];
  rateLimit: boolean;
}

export const DEFAULT_MODELS = ["deepseek-ai/DeepSeek-V4.1-Flash-1017", "deepseek-ai/DeepSeek-V4.1-Flash"];

export function askEnvFrom(env: Record<string, string | undefined>): AskEnv {
  const models = (env.NANOINFER_ASK_MODEL ?? "").split(",").map((s) => s.trim()).filter(Boolean);
  return {
    key: env.NANOINFER_ASK_KEY || env.TOGETHER_API_KEY || undefined,
    keyEnv: env.NANOINFER_ASK_KEY ? "NANOINFER_ASK_KEY" : "TOGETHER_API_KEY",
    baseUrl: (env.NANOINFER_ASK_BASE_URL || "https://api.together.xyz/v1").replace(/\/$/, ""),
    models: models.length ? models : DEFAULT_MODELS,
    rateLimit: env.NANOINFER_ASK_RATE_LIMIT !== "off",
  };
}

// ---- rate limit: per-IP sliding windows, in memory (a speed bump, not a wall: on a
// serverless host each instance counts separately, so also set a spend limit with the provider)

const WINDOWS: [number, number][] = [
  [12, 60_000], // 12 a minute
  [60, 3_600_000], // 60 an hour
];
const hits = new Map<string, number[]>();

function limited(ip: string): boolean {
  const now = Date.now();
  const q = (hits.get(ip) ?? []).filter((t) => t > now - WINDOWS[WINDOWS.length - 1][1]);
  for (const [n, w] of WINDOWS) if (q.filter((t) => t > now - w).length >= n) return true;
  q.push(now);
  hits.set(ip, q);
  if (hits.size > 20_000) for (const [k, v] of hits) if (!v.length || v[v.length - 1] < now - 3_600_000) hits.delete(k);
  return false;
}

// ---- prompt ------------------------------------------------------------------------

const SYSTEM = `You are a patient tutor beside "nanoinfer", an interactive course on how large language models run
(inference): tokens, embeddings, matrix multiplies, attention, sampling, memory bandwidth, KV caches, quantization,
CPUs vs GPUs, batching. The reader is learning from scratch and has selected some text on the page.

Rules (unless the request below asks for a different shape):
- Start with one plain sentence that directly answers. Then at most 3 short sentences of useful context.
- Everyday words, short sentences. Define any technical term you use. No headings.
- Prefer the page's own numbers and analogies (they come from real measurements on an Apple M4 Mac mini) and refer to
  them. If you need knowledge from outside the page, say "Outside this page, ...".
- Markdown is rendered: \`inline code\`, **bold** for the 1-2 key phrases, short lists only if they truly help.
- Never invent measurements.

PAGE: {title}

SECTION THE READER IS LOOKING AT:
{context}`;

// The selection toolbar's actions, worded like Tangent's (server.py KIND_PROMPTS), adapted
// from "¶n of an article" to "this part of the course".
const KINDS: Record<string, string> = {
  ask: 'About "{sel}": {q}',
  explain: 'Explain "{sel}" as used on this page. What does it mean here, and why does the page say it?',
  define: 'Define "{sel}" in plain words, as used on this page. One sentence definition, then one sentence on how it is used here.',
  simplify:
    "Rewrite this paragraph in plain, simple English for a reader who finds dense text hard. Keep every idea; shorter sentences. Output only the rewrite.\n\nPARAGRAPH:\n{sel}",
  deeper:
    'Go deeper on "{sel}": a short reading for a curious learner, like a good magazine explainer. 3 to 5 short paragraphs (about 300 words): what it is, how it actually works, and why it matters for how a model runs or how fast. Concrete facts and numbers; prefer the page\'s own. No headings, no bullet lists.',
  followup: "{q}",
};
/** answer length per action: the reading and the rewrite need more room */
const MAX_TOKENS: Record<string, number> = { deeper: 1100, simplify: 900 };

interface Body {
  kind?: string;
  selection?: string;
  question?: string;
  title?: string;
  context?: string;
  mode?: string;
  history?: { role: string; content: string }[];
}

const cap = (s: unknown, n: number) => (typeof s === "string" ? s.slice(0, n) : "");

let resolved: { model: string; at: number } | null = null;

function json(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });
}

export async function handleAsk(req: Request, env: AskEnv, ip: string): Promise<Response> {
  if (req.method === "GET") {
    return json(200, { available: !!env.key, keyEnv: env.keyEnv, model: resolved?.model ?? env.models[0], models: env.models });
  }
  if (req.method !== "POST") return json(405, { error: "method not allowed" });
  if (!env.key) return json(503, { error: `Ask isn't configured: set ${env.keyEnv} on the server.`, keyEnv: env.keyEnv });
  if (env.rateLimit && limited(ip)) return json(429, { error: "That's a lot of questions! The hourly limit for this demo is reached; try again in a bit." });

  let b: Body;
  try {
    b = (await req.json()) as Body;
  } catch {
    return json(400, { error: "bad request" });
  }
  const kind = KINDS[b.kind ?? ""] ? (b.kind as string) : "ask";
  const sel = cap(b.selection, 1500).trim();
  const q = cap(b.question, 1000).trim();
  if (!sel && !q) return json(400, { error: "nothing to ask about" });
  const user = KINDS[kind].replace("{sel}", sel).replace("{q}", q || "What does this mean?");
  const history = (Array.isArray(b.history) ? b.history : [])
    .filter((t) => (t.role === "user" || t.role === "assistant") && typeof t.content === "string")
    .slice(-8)
    .map((t) => ({ role: t.role, content: t.content.slice(0, 4000) }));
  const messages = [
    { role: "system", content: SYSTEM.replace("{title}", cap(b.title, 200)).replace("{context}", cap(b.context, 6000)) },
    ...history,
    { role: "user", content: user },
  ];
  const think = b.mode === "think";

  // try the preferred models in order; remember the first one the provider serves (for 10 minutes)
  const order = resolved && Date.now() - resolved.at < 600_000 ? [resolved.model, ...env.models.filter((m) => m !== resolved!.model)] : env.models;
  let upstream: Response | null = null, model = "", lastErr = "";
  for (const m of order) {
    const r = await fetch(`${env.baseUrl}/chat/completions`, {
      method: "POST",
      headers: { Authorization: `Bearer ${env.key}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model: m, messages, stream: true, max_tokens: (think ? 1300 : 0) + (MAX_TOKENS[kind] ?? 700), temperature: 0.4, reasoning: { enabled: think } }),
    }).catch((e: Error) => {
      lastErr = e.message;
      return null;
    });
    if (r?.ok && r.body) {
      upstream = r;
      model = m;
      resolved = { model: m, at: Date.now() };
      break;
    }
    const text = r ? await r.text().catch(() => "") : "";
    lastErr = text.slice(0, 300) || lastErr;
    // only fall through to the next model when this one isn't available; other errors are final
    if (r && !/model_not_available|not found|does not exist|Unable to access model/i.test(text)) break;
  }
  if (!upstream) return json(502, { error: "The model provider didn't answer. Try again in a moment.", detail: lastErr });

  const reader = upstream.body!.getReader();
  const dec = new TextDecoder();
  const enc = new TextEncoder();
  let buf = "";
  const stream = new ReadableStream<Uint8Array>({
    async pull(ctrl) {
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) {
            ctrl.close();
            return;
          }
          buf += dec.decode(value, { stream: true });
          let out = "";
          let nl: number;
          while ((nl = buf.indexOf("\n")) >= 0) {
            const line = buf.slice(0, nl).trim();
            buf = buf.slice(nl + 1);
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (data === "[DONE]") continue;
            try {
              const j = JSON.parse(data);
              const c = j.choices?.[0]?.delta?.content;
              if (c) out += c;
              if (j.error) out += `\u0000${j.error.message ?? "model error"}`;
            } catch {
              /* partial line: ignore */
            }
          }
          if (out) {
            ctrl.enqueue(enc.encode(out));
            return;
          }
        }
      } catch (e) {
        ctrl.enqueue(enc.encode(`\u0000${(e as Error).message}`));
        ctrl.close();
      }
    },
    cancel() {
      void reader.cancel();
    },
  });
  return new Response(stream, { headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-cache", "X-Model": model } });
}

// Vercel serverless function for select-to-ask (same handler as the Vite dev
// middleware). Set TOGETHER_API_KEY in the Vercel project's environment.

import { askEnvFrom, handleAsk } from "../server/ask";

declare const process: { env: Record<string, string | undefined> };

export const config = { runtime: "edge" };

export default function handler(req: Request): Promise<Response> {
  const ip = (req.headers.get("x-forwarded-for") ?? "").split(",")[0].trim() || "?";
  return handleAsk(req, askEnvFrom(process.env), ip);
}

#!/usr/bin/env node
// Readability stats for the site's reader-facing copy (STYLE.md). Parses every .ts/.tsx file under src/ with the
// TypeScript compiler, keeps prose: JSX text in <p>, <li>, <h2>, <h3>, <Idea>, and the caption/tryit/title/q props,
// plus string properties that hold sentences (checks.ts q/options/why, field guide plain/why/where, glossary).
// Interpolated values count as one word ("N"). Code, SVG and controls are dropped.
//
//   node scripts/copystats.mjs            per-file table
//   node scripts/copystats.mjs --long     also list sentences over 30 words, captions over 60, tries over 35
//   node scripts/copystats.mjs --json     machine-readable

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const ts = require("typescript");
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../src");
const args = new Set(process.argv.slice(2));

const BLOCK_TAGS = new Set(["p", "li", "h2", "h3", "Idea", "dd", "td"]);
const SKIP_TAGS = new Set(["svg", "code", "pre", "button", "select", "option", "input", "textarea", "style", "script", "text", "tspan", "kbd"]);
const PROSE_ATTRS = new Set(["caption", "tryit", "title", "q", "alt"]);
const PROSE_KEYS = new Set(["q", "why", "plain", "where", "def", "blurb", "caption", "tryit", "body", "desc", "note", "breaks"]);

function tagName(node) {
  const t = node.tagName ?? node.openingElement?.tagName;
  return t ? t.getText() : "";
}

function attr(el, name) {
  const attrs = (el.attributes ?? el.openingElement?.attributes)?.properties ?? [];
  for (const a of attrs) {
    if (a.name?.getText() === name && a.initializer) {
      if (ts.isStringLiteral(a.initializer)) return a.initializer.text;
      if (ts.isJsxExpression(a.initializer) && a.initializer.expression && ts.isStringLiteralLike(a.initializer.expression))
        return a.initializer.expression.text;
    }
  }
  return null;
}

// Flatten a JSX subtree to plain text.
function text(node) {
  if (!node) return "";
  if (ts.isJsxText(node)) return node.text.replace(/\s+/g, " ");
  if (ts.isJsxExpression(node)) {
    const e = node.expression;
    if (!e) return "";
    if (ts.isStringLiteralLike(e)) return e.text;
    if (ts.isJsxElement(e) || ts.isJsxSelfClosingElement(e) || ts.isJsxFragment(e)) return text(e);
    if (ts.isParenthesizedExpression(e)) return text({ ...node, expression: e.expression, kind: node.kind });
    return " N ";
  }
  if (ts.isJsxSelfClosingElement(node)) {
    const t = tagName(node);
    if (t === "G") return attr(node, "show") ?? attr(node, "k") ?? "";
    if (t === "Tk") return `“${attr(node, "s") ?? ""}”`;
    if (t === "br") return " ";
    return "";
  }
  if (ts.isJsxElement(node) || ts.isJsxFragment(node)) {
    const t = ts.isJsxElement(node) ? tagName(node) : "";
    if (SKIP_TAGS.has(t)) return "";
    return node.children.map(text).join("");
  }
  return "";
}

function clean(s) {
  return s.replace(/\s+/g, " ").replace(/\s+([.,;:!?)])/g, "$1").trim();
}

function sentences(s) {
  return clean(s)
    .split(/(?<=[.!?…])\s+(?=[A-Z“"(0-9‘])/)
    .map((x) => x.trim())
    .filter((x) => /[A-Za-z]/.test(x));
}

const words = (s) => s.split(/\s+/).filter((w) => /[A-Za-z0-9]/.test(w)).length;
const PASSIVE = /\b(is|are|was|were|be|been|being|gets?|got)\s+(\w+ly\s+)?\w+(ed|en)\b/i;

function collect(file) {
  const src = fs.readFileSync(file, "utf8");
  const sf = ts.createSourceFile(file, src, ts.ScriptTarget.Latest, true, file.endsWith("x") ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
  const out = []; // {kind: prose|caption|tryit|data, s}
  function visit(n, inBlock) {
    if (ts.isJsxElement(n) || ts.isJsxSelfClosingElement(n)) {
      const t = tagName(n);
      if (SKIP_TAGS.has(t)) return;
      const attrs = (n.attributes ?? n.openingElement?.attributes)?.properties ?? [];
      for (const a of attrs) {
        const nm = a.name?.getText();
        if (!PROSE_ATTRS.has(nm) || !a.initializer) continue;
        let s = "";
        if (ts.isStringLiteral(a.initializer)) s = a.initializer.text;
        else if (ts.isJsxExpression(a.initializer)) s = text(a.initializer);
        s = clean(s);
        if (words(s) < 2 || s === "N") continue;
        out.push({ kind: nm === "caption" ? "caption" : nm === "tryit" ? "tryit" : "prose", s });
      }
      if (!inBlock && ts.isJsxElement(n) && BLOCK_TAGS.has(t)) {
        const s = clean(text(n));
        if (words(s) >= 2) out.push({ kind: "prose", s });
        return; // don't double count nested blocks
      }
    }
    if (ts.isPropertyAssignment(n) && PROSE_KEYS.has(n.name?.getText?.() ?? "")) {
      const v = n.initializer;
      if (ts.isStringLiteralLike(v) && words(v.text) >= 3) out.push({ kind: "data", s: v.text });
      if (ts.isArrayLiteralExpression(v))
        for (const e of v.elements) if (ts.isStringLiteralLike(e) && words(e.text) >= 2) out.push({ kind: "data", s: e.text });
    }
    if (ts.isPropertyAssignment(n) && n.name?.getText?.() === "options" && ts.isArrayLiteralExpression(n.initializer)) {
      for (const e of n.initializer.elements) if (ts.isStringLiteralLike(e)) out.push({ kind: "data", s: e.text });
      return;
    }
    ts.forEachChild(n, (c) => visit(c, inBlock));
  }
  visit(sf, false);
  return out;
}

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((d) => {
    const p = path.join(dir, d.name);
    if (d.isDirectory()) return d.name.startsWith("pkg") ? [] : walk(p);
    return /\.(tsx?|ts)$/.test(d.name) && !d.name.endsWith(".d.ts") ? [p] : [];
  });
}

const TICS = {
  ", not": /, not\b/g,
  "colon chain (2+ colons)": null,
  "That's it": /That's it\./g,
  "Hold on to": /Hold on to/g,
  "Here is/Here's": /\bHere(?: is|'s)\b/g,
  ceiling: /\bceilings?\b/gi,
  "speed of light": /speed of light/gi,
  rung: /\brungs?\b/gi,
  "per word / a second": /(?:per word|tokens a second|words a second)/gi,
  "just/really/actually": /\b(just|really|actually)\b/gi,
  hover: /\bhover\b/gi,
};

const rows = [];
const longS = [];
const longCap = [];
const longTry = [];
const tics = Object.fromEntries(Object.keys(TICS).map((k) => [k, 0]));
for (const f of walk(ROOT)) {
  const items = collect(f);
  if (!items.length) continue;
  const rel = path.relative(ROOT, f);
  const ss = [];
  for (const it of items) {
    if (it.kind === "caption") {
      const w = words(it.s);
      rows.cap ??= [];
      rows.cap.push(w);
      if (w > 60) longCap.push([w, rel, it.s]);
    }
    if (it.kind === "tryit") {
      const w = words(it.s);
      rows.try ??= [];
      rows.try.push(w);
      if (w > 35) longTry.push([w, rel, it.s]);
    }
    for (const s of sentences(it.s)) {
      ss.push(s);
      if (words(s) > 30) longS.push([words(s), rel, s]);
      for (const [k, re] of Object.entries(TICS)) {
        if (re) tics[k] += (s.match(re) ?? []).length;
        else if ((s.match(/:/g) ?? []).length >= 2) tics[k]++;
      }
    }
  }
  if (!ss.length) continue;
  const w = ss.map(words).sort((a, b) => a - b);
  rows.push({
    file: rel,
    n: ss.length,
    avg: +(w.reduce((a, b) => a + b, 0) / w.length).toFixed(1),
    median: w[Math.floor(w.length / 2)],
    over25: Math.round((100 * w.filter((x) => x > 25).length) / w.length),
    over30: w.filter((x) => x > 30).length,
    longest: w[w.length - 1],
    passive: Math.round((100 * ss.filter((s) => PASSIVE.test(s)).length) / ss.length),
    colon: Math.round((100 * ss.filter((s) => s.includes(":")).length) / ss.length),
  });
}

const all = rows.reduce((a, r) => a + r.n, 0);
const cap = rows.cap ?? [];
const tr = rows.try ?? [];
const mean = (a) => (a.length ? +(a.reduce((x, y) => x + y, 0) / a.length).toFixed(1) : 0);
const summary = {
  sentences: all,
  over30: longS.length,
  colonPct: Math.round(rows.reduce((a, r) => a + (r.colon * r.n) / 100, 0) / all * 100),
  captions: { n: cap.length, avg: mean(cap), over60: cap.filter((x) => x > 60).length, max: Math.max(0, ...cap) },
  tries: { n: tr.length, avg: mean(tr), over35: tr.filter((x) => x > 35).length, max: Math.max(0, ...tr) },
  tics,
};

if (args.has("--json")) {
  console.log(JSON.stringify({ rows, summary }, null, 1));
} else {
  const pad = (s, n) => String(s).padEnd(n);
  console.log(pad("file", 34), "sent  avg  med  >25%  >30  max  pass%  colon%");
  for (const r of rows.sort((a, b) => a.file.localeCompare(b.file)))
    console.log(pad(r.file, 34), pad(r.n, 5), pad(r.avg, 4), pad(r.median, 4), pad(r.over25, 5), pad(r.over30, 4), pad(r.longest, 4), pad(r.passive, 6), r.colon);
  console.log("\nsummary", JSON.stringify(summary, null, 1));
  if (args.has("--long")) {
    console.log("\n-- sentences over 30 words");
    for (const [w, f, s] of longS.sort((a, b) => b[0] - a[0])) console.log(`${w}\t${f}\t${s}`);
    console.log("\n-- captions over 60 words");
    for (const [w, f, s] of longCap.sort((a, b) => b[0] - a[0])) console.log(`${w}\t${f}\t${s.slice(0, 120)}…`);
    console.log("\n-- tries over 35 words");
    for (const [w, f, s] of longTry.sort((a, b) => b[0] - a[0])) console.log(`${w}\t${f}\t${s}`);
  }
}

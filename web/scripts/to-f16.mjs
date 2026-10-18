#!/usr/bin/env node
// Convert HF GPT-2 model.safetensors (F32) into an F16 safetensors file: half
// the download (274 MB vs 548 MB). The browser widens it back to f32 while it
// streams in (crates/wasm `Bytes::from_f16`), so the engine computes in f32 on
// weights rounded to f16 once.
//
//   node web/scripts/to-f16.mjs [in=weights/gpt2/model.safetensors] [out=weights/gpt2/model-f16.safetensors]
//
// Needs Node 24+ (Float16Array). Layout is unchanged (HF's [in, out] Conv1D
// order); core still transposes on load.
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "../..");
const src = process.argv[2] ?? path.join(root, "weights/gpt2/model.safetensors");
const dst = process.argv[3] ?? path.join(root, "weights/gpt2/model-f16.safetensors");

const buf = fs.readFileSync(src);
const hlen = Number(buf.readBigUInt64LE(0));
const header = JSON.parse(buf.subarray(8, 8 + hlen).toString("utf8"));
const data0 = 8 + hlen;

const names = Object.keys(header).filter((k) => k !== "__metadata__");
names.sort((a, b) => header[a].data_offsets[0] - header[b].data_offsets[0]);
const out = {};
if (header.__metadata__) out.__metadata__ = header.__metadata__;
let off = 0;
const parts = [];
let maxErr = 0;
for (const n of names) {
  const t = header[n];
  if (t.dtype !== "F32") throw new Error(`${n}: expected F32, got ${t.dtype}`);
  const [a, b] = t.data_offsets;
  const f32 = new Float32Array(buf.buffer.slice(buf.byteOffset + data0 + a, buf.byteOffset + data0 + b));
  const f16 = new Float16Array(f32);
  for (let i = 0; i < f32.length; i += 97) maxErr = Math.max(maxErr, Math.abs(f32[i] - f16[i]));
  const bytes = new Uint8Array(f16.buffer);
  out[n] = { dtype: "F16", shape: t.shape, data_offsets: [off, off + bytes.length] };
  parts.push(bytes);
  off += bytes.length;
}
let hjson = Buffer.from(JSON.stringify(out), "utf8");
const pad = (8 - (hjson.length % 8)) % 8;
hjson = Buffer.concat([hjson, Buffer.alloc(pad, 0x20)]);
const len = Buffer.alloc(8);
len.writeBigUInt64LE(BigInt(hjson.length));
const fd = fs.openSync(dst, "w");
fs.writeSync(fd, len);
fs.writeSync(fd, hjson);
for (const p of parts) fs.writeSync(fd, p);
fs.closeSync(fd);
console.log(`${dst}: ${(fs.statSync(dst).size / 1e6).toFixed(1)} MB (from ${(buf.length / 1e6).toFixed(1)} MB), max |Δw| sampled ${maxErr.toExponential(2)}`);

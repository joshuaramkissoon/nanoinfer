import { useState } from "react";
import { useRoleColor } from "../../ConstraintMap";
import type { Role } from "../../concepts";
import { Fig, Idea, Seg } from "../../kit";
import { linkTo } from "../../router";
import { chapterBySlug, chapterNum } from "../../chapters";

interface Row {
  group: "state" | "sched" | "scale" | "speed" | "metrics";
  role: Role;
  term: string;
  analog: string;
  holds: string;
  breaks: string;
  /** chapter slug or #field-guide-id */
  to: string;
}

export const ROSETTA: Row[] = [
  { group: "state", role: "memory", term: "Weights", analog: "A read-only dataset every request scans", holds: "Immutable, shared by all requests, loaded once at start-up.", breaks: "Every request reads all of it, every step. There is no index and no partial read.", to: "what-is-inference" },
  { group: "state", role: "memory", term: "KV cache", analog: "Per-session state / a session cache", holds: "Grows with the conversation, belongs to one request, is reused on every step and freed at the end.", breaks: "It's read in full on every step, so it costs bandwidth as well as space; at long contexts it's bigger than the weights.", to: "kv-cache" },
  { group: "state", role: "memory", term: "Paged attention", analog: "Virtual memory paging", holds: "Fixed-size pages, a per-request page table, allocate on demand, no contiguous reservation, near-zero fragmentation.", breaks: "No swapping to disk in the fast path; when pages run out a request is preempted and recomputed or offloaded.", to: "#paged-attention" },
  { group: "state", role: "memory", term: "Prefix caching", analog: "Memoisation / a content-addressed cache", holds: "Hash the token prefix; if a previous request had the same system prompt, reuse its KV pages and skip that prefill.", breaks: "Only exact prefixes match: one different token early on invalidates everything after it.", to: "#prefix-caching" },
  { group: "state", role: "memory", term: "Quantisation", analog: "Lossy compression", holds: "Fewer bits per number, a per-block scale, decompress just before use; the trade is size against fidelity.", breaks: "The “decompression” runs inside the matrix multiply, millions of times per token, so formats are chosen for fast decoding rather than the best ratio.", to: "smaller-numbers" },
  { group: "sched", role: "time", term: "Batching", analog: "Amortising a fixed cost: group commit, Nagle", holds: "One expensive operation (reading the weights) is shared by many requests.", breaks: "The batch runs in lockstep: every user waits for the step, so per-user latency rises with batch size.", to: "batching" },
  { group: "sched", role: "time", term: "Continuous batching", analog: "A request scheduler admitting work every step", holds: "Re-form the batch at every step: finished requests leave, queued ones join. No head-of-line blocking on the longest request.", breaks: "Admission is limited by KV cache memory as well as queue depth; long prompts joining can stall everyone's next token (hence chunked prefill).", to: "batching" },
  { group: "sched", role: "time", term: "Prefill vs decode", analog: "Two phases with different bottlenecks: bulk load vs point reads", holds: "Prefill reads the prompt in one compute-bound pass; decode writes one token per memory-bound pass.", breaks: "Both run on the same weights and the same device, so they interfere unless you separate them.", to: "the-loop" },
  { group: "sched", role: "time", term: "Disaggregated serving", analog: "Separating read and write paths (CQRS-style)", holds: "Run prefill on one pool of GPUs and decode on another, each tuned for its bottleneck; ship the KV cache between them.", breaks: "Shipping the KV cache (hundreds of MB per long request) needs fast links; it only pays at scale.", to: "#disaggregated" },
  { group: "scale", role: "comm", term: "Data parallel (replicas)", analog: "Stateless replicas behind a load balancer", holds: "Each GPU (or group) holds a full copy and serves its own requests; scale by adding copies.", breaks: "Requests are sticky to the replica holding their KV cache; routing is cache-aware, not round-robin.", to: "#data-parallel" },
  { group: "scale", role: "comm", term: "Tensor parallel", analog: "Sharding one request across nodes, then an all-reduce", holds: "Every matrix split across GPUs; each computes a partial result; combine with an all-reduce.", breaks: "A synchronous barrier on every layer of every token, so it only works on very fast links (NVLink).", to: "#tensor-parallel" },
  { group: "scale", role: "comm", term: "Pipeline parallel", analog: "A staged pipeline", holds: "Layers 1–40 on one GPU, 41–80 on the next; activations flow stage to stage; keep stages busy with micro-batches.", breaks: "One token still visits every stage in turn, so it adds latency for a single user; it helps throughput and capacity.", to: "#pipeline-parallel" },
  { group: "scale", role: "comm", term: "Expert parallel (MoE)", analog: "Key-based routing to shards, with hot shards", holds: "A router sends each token to a few experts living on different GPUs; all-to-all traffic like a shuffle.", breaks: "Routing is learned, not hashed, so load is skewed and changes with the input; capacity limits drop or reroute overflow.", to: "#moe" },
  { group: "speed", role: "compute", term: "Speculative decoding", analog: "Optimistic execution, then validation and rollback", holds: "A cheap drafter guesses k tokens; the big model checks all k in one pass; wrong guesses are discarded.", breaks: "The check is exact, so output never changes; the payoff depends entirely on acceptance rate (text type matters).", to: "speculative-decoding" },
  { group: "speed", role: "compute", term: "Kernel launch / sync overhead", analog: "An RPC round-trip per call", holds: "Each GPU operation has a fixed dispatch cost; waiting for each to finish before sending the next serialises everything.", breaks: "Unlike an RPC, the cost is mostly the CPU waiting for each result. Batching the calls without removing the wait helps little (chapter 16).", to: "the-gpu" },
  { group: "speed", role: "compute", term: "Kernel fusion", analog: "Fusing pipeline stages to skip materialising intermediates", holds: "Do several operations in one pass so intermediates stay in registers instead of round-tripping through memory.", breaks: "Fused kernels are bespoke per shape and hardware; it's compiler work, not configuration.", to: "#fusion" },
  { group: "speed", role: "memory", term: "Memory-bound decode", analog: "An I/O-bound service with idle CPUs", holds: "Throughput is set by how fast data arrives, not by how fast it's processed. Faster CPUs don't help.", breaks: "The “I/O” is local DRAM or HBM rather than disk or network: nanoseconds away, but terabytes per second of it.", to: "memory-or-maths" },
  { group: "metrics", role: "time", term: "TTFT / TPOT", analog: "Latency SLOs: time to first byte / per-chunk streaming latency", holds: "Two user-facing latencies: how long until the answer starts, and how smoothly it streams.", breaks: "They pull in opposite directions under load: admitting a new request's prefill delays everyone else's next token.", to: "#ttft" },
  { group: "metrics", role: "time", term: "Goodput", analog: "Throughput of requests that met their SLO", holds: "Raw tok/s can rise while users suffer; count only requests within latency targets.", breaks: "Targets differ per product (chat vs batch jobs), so the same server has different goodput for different traffic.", to: "#goodput" },
];

const GROUPS = [
  { v: "all", label: "all" },
  { v: "state", label: "state and memory" },
  { v: "sched", label: "scheduling" },
  { v: "scale", label: "scale-out" },
  { v: "speed", label: "speed tricks" },
  { v: "metrics", label: "metrics" },
];

function linkLabel(to: string) {
  if (to.startsWith("#")) return "field guide";
  const c = chapterBySlug(to);
  return c ? `chapter ${chapterNum(c).replace(/^0(?=\d$)/, "")}` : "";
}

function Stone() {
  const color = useRoleColor();
  const [g, setG] = useState("all");
  const [open, setOpen] = useState<string | null>("KV cache");
  const rows = ROSETTA.filter((r) => g === "all" || r.group === g);
  return (
    <Fig
      n="0.4.1"
      title="Inference terms, translated"
      prov="diagram"
      caption={
        <>
          Left: the term you'll hear. Right: the thing you already know. The dot's colour is the resource it's about (blue
          memory, ember compute, olive communication, grey time). Each row opens to where the analogy holds, where it breaks,
          and the chapter that explains it.
        </>
      }
      tryit={<>Filter to scale-out and open each row. Notice what the “breaks” lines have in common.</>}
    >
      <div className="lr-controls">
        <Seg options={GROUPS} value={g} onChange={setG} label="group" />
      </div>
      <div className="p0-stone">
        {rows.map((r) => {
          const on = open === r.term;
          const href = r.to.startsWith("#") ? `/learn/field-guide${r.to}` : `/learn/${r.to}`;
          return (
            <div key={r.term} className={`p0-stone-row${on ? " on" : ""}`} style={{ ["--rc" as string]: color(r.role) }}>
              <button className="p0-stone-h" onClick={() => setOpen(on ? null : r.term)} aria-expanded={on}>
                <span className="t">
                  <i className="dot" />
                  {r.term}
                </span>
                <span className="eq" aria-hidden>
                  ≈
                </span>
                <span className="a">{r.analog}</span>
              </button>
              {on && (
                <div className="p0-stone-b">
                  <p>
                    <span className="lr-cap-k">holds</span>
                    {r.holds}
                  </p>
                  <p>
                    <span className="lr-cap-k">breaks</span>
                    {r.breaks}
                  </p>
                  <a href={href} onClick={linkTo(href)} className="p0-tchip">
                    explained in {linkLabel(r.to)} →
                  </a>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </Fig>
  );
}

export default function Rosetta() {
  return (
    <>
      <p>
        Most inference terms name ideas you already use. The difference is the workload: every step reads gigabytes. The
        table maps each term to its systems equivalent.
      </p>
      <Stone />
      <p>
        You can reason with these analogies. They usually break in one place. In a normal service, caching or batching
        spreads out a cost. In inference, every step reads <i>all</i> the weights and the whole KV cache. That cost comes
        back on every token, so the same patterns matter much more here.
      </p>
      <Idea>
        Inference uses patterns you know: caching, paging, batching, sharding, optimistic execution and SLOs. They matter
        more because every step reads gigabytes.
      </Idea>
      <p>
        Part 1 opens the model and shows what those weights do, starting with its one job: guessing the next token.
      </p>
    </>
  );
}

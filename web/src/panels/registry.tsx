// The page is this list. To add a panel: write a component in this folder that
// reads the shared run with useEngine(...) (or drives the engine itself via
// claim()/engine.*), then add an entry here. See web/README.md.

import type { ComponentType, ReactNode } from "react";
import RunIt from "./RunIt";
import SpeedOfLight from "./SpeedOfLight";
import ByteBudget from "./ByteBudget";
import TimeStrip from "./TimeStrip";
import Roofline from "./Roofline";
import KvRace from "./KvRace";
import Attention from "./Attention";
import Ladder from "./Ladder";
import InsideToken from "./InsideToken";
import LogitLens from "./LogitLens";
import StageRace from "./StageRace";

export interface PanelDef {
  id: string;
  title: string;
  Component: ComponentType;
  /** 1–3 sentence learner caption under the figure */
  caption: ReactNode;
  /** optional prose shown before the figure */
  intro?: ReactNode;
  narrow?: boolean;
}

export const PANELS: PanelDef[] = [
  {
    id: "run",
    title: "Run it",
    Component: RunIt,
    caption: (
      <>
        Each token is picked from a probability distribution over all 50,257 tokens; the underline shows how likely the
        model thought the chosen one was. <b>Click a token</b> to see its rivals, then click a rival to branch the story from
        that point.
      </>
    ),
  },
  {
    id: "inside",
    title: "Inside one token",
    Component: InsideToken,
    intro: (
      <>
        <h2>Follow one token</h2>
        <p>
          Every word above came out of the same loop: take the newest token, push it through 12 layers, score the whole
          vocabulary, pick one. Scroll, and one real step of that loop plays out op by op, with the time, bytes and FLOPs the
          engine measured for each.
        </p>
      </>
    ),
    caption: (
      <>
        One decode step from the run in fig. 1: 135 ops. Each segment's width is its measured time, and the barcode is
        the actual residual stream after each layer. The final distribution is the real one; re-sampling it costs nothing
        because the expensive part, the forward pass, is already done.
      </>
    ),
  },
  {
    id: "lens",
    title: "What is it thinking at each layer?",
    Component: LogitLens,
    intro: (
      <>
        <h2>Reading its mind, one layer at a time</h2>
        <p>
          The last two ops of every step (a LayerNorm and the LM head) turn the residual stream into a guess. Nothing stops us
          applying them early. Decode the stream after layer 3 and you get what the model "would say" if it had only three
          layers. This trick is called the <i>logit lens</i>.
        </p>
      </>
    ),
    caption: (
      <>
        Each column is a position in the text, each row a depth; the cell is the top guess for the <i>next</i> token.
        Shallow layers mostly guess the previous word or something common like “ the”. The answer often only appears in the
        last few layers, and for easy tokens it locks in early. <b>Click a column</b> to see its guesses climb.
      </>
    ),
  },
  {
    id: "speed-of-light",
    title: "The speed limit",
    Component: SpeedOfLight,
    narrow: true,
    intro: (
      <>
        <h2>A speed limit you can compute before writing any code</h2>
        <p>
          To produce one token, the model multiplies a vector by every weight matrix it has. Each weight gets used once, for
          one multiply and one add, and then it isn't needed again until the next token. So the work is cheap and the reading
          is expensive. Decode speed can't beat <b>memory bandwidth ÷ bytes per token</b>, however clever the arithmetic is.
        </p>
      </>
    ),
    caption: (
      <>
        This tab measured how fast it can stream memory. Divide that by the bytes one token has to read and you get a ceiling
        that no kernel can beat. Switch the weight type: halving the bytes doubles the ceiling, which is why quantisation is
        the biggest lever for decode.
      </>
    ),
  },
  {
    id: "bytes",
    title: "The byte budget of one token",
    Component: ByteBudget,
    caption: (
      <>
        Area is bytes read for the last token you generated, taken from the engine's own per-op accounting. The four matmuls
        per layer are the bulk. The single biggest tile is the <b>LM head</b>: GPT-2 reuses its 154 MB embedding table to
        score every word in the vocabulary, every token.
      </>
    ),
  },
  {
    id: "time",
    title: "Where a token's time goes",
    Component: TimeStrip,
    intro: (
      <>
        <h2>Following the time</h2>
        <p>
          The engine reports every op it runs (about 130 per token) with its start time, duration, FLOPs and bytes. Each
          row below is one real token, drawn to scale.
        </p>
      </>
    ),
    caption: (
      <>
        One row per decoded token and one block per op, coloured by kind. Matmuls dominate, and the long orange block at the
        end of each row is the LM head. Switch to <b>naive</b> in fig. 1 and the rows get longer as the context grows.
      </>
    ),
  },
  {
    id: "roofline",
    title: "Roofline: truck or factory?",
    Component: Roofline,
    caption: (
      <>
        Every kernel waits on either the truck (memory, the slanted roof) or the factory (compute, the flat roof); both
        roofs were measured in this tab. A dot on the slope needs fewer bytes, a dot on the flat roof needs faster math, and
        a dot below both is held back by the kernel itself. Switch stages in fig. 1 and watch the matmuls climb.
      </>
    ),
  },
  {
    id: "kv",
    title: "KV cache: the race",
    Component: KvRace,
    intro: (
      <>
        <h2>Notes in the margin</h2>
        <p>
          A transformer's output at position <code>t</code> depends only on tokens up to <code>t</code>. So when a token is
          appended, the naive engine recomputes everything it already knew. The KV cache keeps each layer's keys and values,
          so each new token pushes one row through the network instead of the whole context.
        </p>
      </>
    ),
    caption: (
      <>
        Both lanes produce the same words, because greedy decoding makes them identical by construction. They just pay very
        different amounts for them. Naive's cost per token grows with context, so its total is O(n²). The cache keeps the
        per-token cost nearly flat, at the price of memory: 72 KB for every token it remembers.
      </>
    ),
  },
  {
    id: "attention",
    title: "What the cache is for: attention",
    Component: Attention,
    caption: (
      <>
        These are the 144 attention heads (12 layers × 12 heads) on the text from fig. 1, captured by an analysis pass
        through the same kernels. Each row shows where one token looks back. Some heads look at the previous token, some at
        the first token, and some follow the syntax. Hover the big matrix or the text to move the query.
      </>
    ),
  },
  {
    id: "race",
    title: "Every rung, racing",
    Component: StageRace,
    intro: (
      <>
        <h2>Feel the climb</h2>
        <p>
          Numbers on a chart are one thing. Here every stage writes the same sentence, each at its own measured speed.
        </p>
      </>
    ),
    caption: (
      <>
        Same prompt, same greedy words (until a quantised stage rounds the weights enough to change its mind), very different
        waits. Naive recomputes the whole sentence for every token; the KV
        cache stops that; the unrolled and SIMD kernels make each remaining matmul fast. The browser-only rungs (orange) are
        kernels written for wasm itself.
      </>
    ),
  },
  {
    id: "ladder",
    title: "The ladder",
    Component: Ladder,
    intro: (
      <>
        <h2>The climb</h2>
        <p>
          Every optimisation in nanoinfer is added as a new stage next to the old ones and never written over them, so every
          rung of the hillclimb still runs. This page reads the stage list from the engine, which means new rungs show up
          here as soon as they land.
        </p>
      </>
    ),
    caption: (
      <>
        The dark bars come from the native hillclimb on a Mac mini M4. The orange bars are the same Rust compiled to wasm,
        measured in a browser: dashed is the recording, solid is your tab once you benchmark it. Switch to <b>prefill</b> to
        see where the browser-only SIMD rungs pay off. The gap between a bar and its line is the work left to do.
      </>
    ),
  },
];

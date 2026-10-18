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
        The black bars come from the native hillclimb on a Mac mini M4. The orange bars were measured by your browser just
        now, running the same Rust code compiled to wasm. Both are drawn under their speed-of-light lines. The gap between
        a bar and its line is the work left to do.
      </>
    ),
  },
];

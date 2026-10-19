// The lab (/lab) is this list. To add a panel: write a component in this folder that
// reads the shared run with useEngine(...) (or drives the engine itself via
// claim()/engine.*), then add an entry here. See web/README.md.
// Each entry: a short intro linking to the chapter that explains the idea, and a
// caption that says what you're looking at and what to try. ByteBudget, TimeStrip
// and StageRace still exist but aren't on the page (chapters 4 and 10 and the
// ladder cover them).

import type { ComponentType, ReactNode } from "react";
import { Swatch } from "../learn/kit";
import RunIt from "./RunIt";
import SpeedOfLight from "./SpeedOfLight";
import Roofline from "./Roofline";
import KvRace from "./KvRace";
import Attention from "./Attention";
import Ladder from "./Ladder";
import InsideToken from "./InsideToken";
import LogitLens from "./LogitLens";

export interface PanelDef {
  id: string;
  title: string;
  Component: ComponentType;
  /** 1–3 sentence learner caption under the figure: what you're looking at */
  caption: ReactNode;
  /** one concrete thing to do with the figure (shown as "→ Try …") */
  tryit?: ReactNode;
  /** optional prose shown before the figure */
  intro?: ReactNode;
  narrow?: boolean;
}

export const PANELS: PanelDef[] = [
  {
    id: "run",
    title: "Run it",
    Component: RunIt,
    intro: (
      <>
        <h2>Generate on any stage of the climb</h2>
        <p>
          The generation loop from <a href="/learn/the-loop">chapter 8</a>, with every stage of the engine selectable, from
          the first correct one to the GPU stages. Pick a stage and press Generate. That stage writes the text, you see how
          fast it ran in this tab, and the figures below redraw from that run.
        </p>
      </>
    ),
    caption: (
      <>
        The prompt (grey) and what GPT-2 wrote, each token underlined by how confident the model was. On the right: the
        probabilities behind the selected token. Below: the speed, and the time each token took.
      </>
    ),
    tryit: (
      <>
        Generate on the starting point, then on the KV cache stage, and compare the ms-per-token bars. Select any token, then
        a rival, to branch the text.
      </>
    ),
  },
  {
    id: "inside",
    title: "Inside one token",
    Component: InsideToken,
    intro: (
      <>
        <h2>One step, operation by operation</h2>
        <p>
          <a href="/learn/layers">Chapter 6</a> followed one token down the 12 layers. This is the same walk with the engine's
          stopwatch running: each of the ~135 operations, with the time, bytes and arithmetic it measured.
        </p>
      </>
    ),
    caption: (
      <>
        One real decode step. Each row of the strip is a layer, and each segment is an operation whose width is its measured
        time. Colours mark kinds of operation (legend below the strip).
      </>
    ),
    tryit: <>Scroll and watch the counters. Notice which row takes the most time.</>,
  },
  {
    id: "lens",
    title: "What is it thinking at each layer?",
    Component: LogitLens,
    intro: (
      <>
        <h2>The logit lens, for every position</h2>
        <p>
          The question from <a href="/learn/layers">chapter 6</a>, “what would it say if it stopped here?”, for a whole
          stretch of text at once.
        </p>
      </>
    ),
    caption: (
      <>
        Across: positions in the text. Down: depth, from just after the embedding (top) to after layer 12 (bottom). Each cell
        shows the top guess for the next token at that depth.
      </>
    ),
    tryit: <>Select a column. Notice at which depth its guess settles.</>,
  },
  {
    id: "attention",
    title: "All 144 attention heads",
    Component: Attention,
    intro: (
      <>
        <h2>Every attention head at once</h2>
        <p>
          <a href="/learn/attention">Chapter 5</a> showed one attention head at a time. GPT-2 has 12 layers × 12 heads, and
          here they all are, on the text from fig 1.
        </p>
      </>
    ),
    caption: (
      <>
        Each small square is one attention head's pattern. Row: the token doing the looking. Column: the earlier token it
        looks at. The further a cell's colour is from the background, the more attention.
      </>
    ),
    tryit: <>Select a square to enlarge it. Look for heads drawn as a diagonal line, and heads that light up the left edge.</>,
  },
  {
    id: "speed-of-light",
    title: "The speed limit, in this tab",
    Component: SpeedOfLight,
    narrow: true,
    intro: (
      <>
        <h2>Your machine's speed limit</h2>
        <p>
          The division from <a href="/learn/speed-limit">chapter 10</a>, done with the memory bandwidth this tab measures. The
          probe follows the backend that ran fig 1: the CPU probe for a CPU stage, the WebGPU probe for a GPU stage.
        </p>
      </>
    ),
    caption: (
      <>
        The speed limit for the run in fig 1 (its backend's memory bandwidth ÷ the bytes its number format reads per token),
        and how close the engine got.
      </>
    ),
    tryit: <>Pick another number format. Notice how the limit moves with the bytes per token.</>,
  },
  {
    id: "roofline",
    title: "Roofline: truck or factory?",
    Component: Roofline,
    intro: (
      <>
        <h2>Every operation on the roofline</h2>
        <p>
          The chart from <a href="/learn/memory-or-maths">chapter 9</a>, with each of the engine's operations from the run in
          fig 1, against roofs measured in this tab for the backend it ran on.
        </p>
      </>
    ),
    caption: (
      <>
        Across: arithmetic per byte (log scale). Up: arithmetic per second achieved. Dots are operations from the last run.
        Dots near the slanted roof are memory-bound; dots near the flat roof are compute-bound. A GPU stage shows as one dot,
        because it runs the whole token as one GPU job.
      </>
    ),
    tryit: <>Switch between CPU stages in fig 1 and run again. Notice where the matrix-multiply dots move.</>,
  },
  {
    id: "kv",
    title: "KV cache: the race",
    Component: KvRace,
    intro: (
      <>
        <h2>With and without the KV cache</h2>
        <p>
          The idea from <a href="/learn/kv-cache">chapter 12</a>, run live: two copies of the engine write the same tokens,
          one recomputing everything and one with the KV cache.
        </p>
      </>
    ),
    caption: (
      <>
        Two lanes writing identical text, one without the KV cache and one with it. Each lane plots the time each token takes
        as the text grows.
      </>
    ),
    tryit: <>Let it run for a long text. Notice how each lane's time per token changes as the text grows.</>,
  },
  {
    id: "ladder",
    title: "The ladder",
    Component: Ladder,
    intro: (
      <>
        <h2>Every stage of the climb</h2>
        <p>
          Part 3 of the course groups the optimisations into six milestones. This is every individual stage, native and in
          the browser (WebGPU stages included), each still runnable.
        </p>
      </>
    ),
    caption: (
      <>
        Tokens per second for each stage, log scale. <Swatch c="var(--m-fg)">Solid</Swatch> bars: the native engine on the M4
        Mac mini. <Swatch c="var(--m-accent)">Ember</Swatch> bars: the same Rust in the browser (dashed outline: recorded;
        filled: measured in your tab). In the browser, 8-bit and 4-bit stages that keep a 16-bit LM head are held back by
        that head.
      </>
    ),
    tryit: <>Switch to prefill. Notice which optimisations only help reading the prompt.</>,
  },
];

// The Lab (/lab) is this list. To add a panel: write a component in this folder that
// reads the shared run with useEngine(...) (or drives the engine itself via
// claim()/engine.*), then add an entry here. See web/README.md.
// Each entry: a short intro linking to the chapter that explains the idea, and a
// caption that says what you're looking at and what to try. ByteBudget, TimeStrip
// and StageRace still exist but aren't on the page (chapters 4 and 10 and the
// ladder cover them).

import type { ComponentType, ReactNode } from "react";
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
    intro: (
      <>
        <h2>Generate, on any stage of the climb</h2>
        <p>
          The generation loop from <a href="/learn/the-loop">Chapter 8</a>, with every version of the engine selectable,
          from the first correct one to the fastest GPU stage. Pick a version, press Generate: that engine writes the text,
          you see how fast it went in this tab, and the figures below redraw from that run.
        </p>
      </>
    ),
    caption: (
      <>
        <b>What you're looking at:</b> the prompt (grey) and what GPT-2 wrote, each token underlined by how confident the
        model was. On the right, the probabilities behind the selected token. Below, speed and the time each token took.{" "}
        <b>Try:</b> generate with <i>Starting point</i>, then <i>KV cache</i>, and compare the ms-per-token bars; click any token, then a rival,
        to branch the story.
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
          stopwatch running: every one of the ~135 operations, with the time, bytes and arithmetic it measured.
        </p>
      </>
    ),
    caption: (
      <>
        <b>What you're looking at:</b> one real decode step. Each row of the strip is a layer and each segment an operation,
        its width the measured time; colours are operation kinds (legend below it). <b>Try:</b> scroll and watch the counters;
        notice the last row, the LM head, is the single biggest block.
      </>
    ),
  },
  {
    id: "lens",
    title: "What is it thinking at each layer?",
    Component: LogitLens,
    intro: (
      <>
        <h2>The logit lens, for every position</h2>
        <p>
          <a href="/learn/layers">Chapter 6</a>'s “what would it say if it stopped here?”, for a whole stretch of text at once.
        </p>
      </>
    ),
    caption: (
      <>
        <b>What you're looking at:</b> columns are positions in the text, rows are depths (top = after the embedding, bottom =
        after layer 12). Each cell shows the top guess for the next token at that depth.{" "}
        <b>Try:</b> click a column to see its guesses sharpen with depth.
      </>
    ),
  },
  {
    id: "attention",
    title: "All 144 attention heads",
    Component: Attention,
    intro: (
      <>
        <h2>Every head at once</h2>
        <p>
          <a href="/learn/attention">Chapter 5</a> showed one head at a time. GPT-2 has 12 layers × 12 heads; here they all are,
          on the text from fig 1.
        </p>
      </>
    ),
    caption: (
      <>
        <b>What you're looking at:</b> each small square is one head's attention pattern: row = the token doing the looking,
        column = the earlier token it looks at; the further a cell's colour is from the background, the more attention. <b>Try:</b> click a square to enlarge it; the
        diagonal-line heads look at the previous token, the left-edge heads at the first token.
      </>
    ),
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
          <a href="/learn/speed-limit">Chapter 10</a>'s division, done with the memory bandwidth this tab measures for the
          backend that ran fig 1: the CPU probe for a CPU stage, the WebGPU probe for a GPU stage.
        </p>
      </>
    ),
    caption: (
      <>
        <b>What you're looking at:</b> the fastest decode this tab could manage for the run in fig 1 (its backend's memory
        bandwidth ÷ the bytes its number format reads per token), and how close the engine got. <b>Try:</b> pick another
        number format: halving the bytes doubles the speed limit.
      </>
    ),
  },
  {
    id: "roofline",
    title: "Roofline: truck or factory?",
    Component: Roofline,
    intro: (
      <>
        <h2>Every operation on the roofline</h2>
        <p>
          <a href="/learn/memory-or-maths">Chapter 9</a>'s chart, with each of the engine's operations plotted from the run in
          fig 1, against roofs measured in this tab for the backend it ran on.
        </p>
      </>
    ),
    caption: (
      <>
        <b>What you're looking at:</b> across, work per byte; up, arithmetic per second achieved. Dots are operations from the
        last run. Dots near the slanted roof are memory-bound, near the flat roof compute-bound. A GPU stage runs the whole
        token as one GPU job, so it shows as a single dot under the GPU's memory roof. <b>Try:</b> switch CPU stages in fig 1
        and rerun: the matmul dots climb toward the roof.
      </>
    ),
  },
  {
    id: "kv",
    title: "KV cache: the race",
    Component: KvRace,
    intro: (
      <>
        <h2>With and without the cache, racing</h2>
        <p>
          <a href="/learn/kv-cache">Chapter 12</a>'s idea, run live: two copies of the engine write the same words, one
          recomputing everything, one with the cache.
        </p>
      </>
    ),
    caption: (
      <>
        <b>What you're looking at:</b> two lanes writing identical text, one without the cache and one with it. Watch the
        time each token takes as the text grows: without the cache it climbs; with it, it stays flat. <b>Try:</b> let it run long and watch the gap widen.
      </>
    ),
  },
  {
    id: "ladder",
    title: "The ladder",
    Component: Ladder,
    intro: (
      <>
        <h2>Every stage of the climb</h2>
        <p>
          Part three of the course grouped the optimisations into six milestones. This is every individual stage, native and
          in the browser (including the WebGPU stages), each still runnable.
        </p>
      </>
    ),
    caption: (
      <>
        <b>What you're looking at:</b> tokens per second for each stage. Dark bars are the native engine on our M4; orange are
        the same Rust compiled for the browser (dashed = recorded, solid = measured in your tab). <b>Try:</b> switch to{" "}
        <b>prefill</b> to see which optimisations only help reading the prompt.
      </>
    ),
  },
];

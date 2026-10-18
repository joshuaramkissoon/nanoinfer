// The learner path is this list (see web/IA.md). Each chapter is one idea and one
// primary figure, and leans only on the chapters before it.

import type { ComponentType } from "react";
import NextToken from "./ch/NextToken";
import Tokens from "./ch/Tokens";
import Embeddings from "./ch/Embeddings";
import DotProduct from "./ch/DotProduct";
import Attention from "./ch/Attention";
import Block from "./ch/Block";
import Sampling from "./ch/Sampling";
import Loop from "./ch/Loop";
import BytesFlops from "./ch/BytesFlops";
import SpeedOfLight from "./ch/SpeedOfLight";
import StartingLine from "./ch/StartingLine";
import KvCache from "./ch/KvCache";
import Arithmetic from "./ch/Arithmetic";
import Threads from "./ch/Threads";
import Quantization from "./ch/Quantization";
import Gpu from "./ch/Gpu";
import Batching from "./ch/Batching";
import SpecDecode from "./ch/SpecDecode";
import ManyGpus from "./ch/ManyGpus";

export interface Chapter {
  slug: string;
  title: string;
  /** the question the chapter answers (shown under the title and in the map) */
  q: string;
  part: number;
  Component: ComponentType;
}

export const PARTS = ["How a model writes text", "Why it's slow", "Making it fast"];
export const PART_BLURB = [
  "From a prompt to the next word: tokens, vectors, attention, and the loop that writes one token at a time.",
  "Where the time actually goes, and the speed limit you can work out before writing any code.",
  "The engine's climb from 0.37 to thousands of tokens per second, one bottleneck at a time.",
];

export const CHAPTERS: Chapter[] = [
  { slug: "next-token", part: 0, title: "Guess the next word", q: "What does a language model actually do?", Component: NextToken },
  { slug: "tokens", part: 0, title: "Tokens", q: "What does the model read, if not letters or words?", Component: Tokens },
  { slug: "embeddings", part: 0, title: "Embeddings", q: "How does a number like 3290 come to mean “dog”?", Component: Embeddings },
  { slug: "multiply-add", part: 0, title: "Multiply and add", q: "What is the one operation that does almost all the work?", Component: DotProduct },
  { slug: "attention", part: 0, title: "Attention", q: "How does a word use the words before it?", Component: Attention },
  { slug: "layers", part: 0, title: "Layers and the residual stream", q: "What happens across the model's twelve layers?", Component: Block },
  { slug: "sampling", part: 0, title: "Sampling", q: "Given the probabilities, which word do we actually say?", Component: Sampling },
  { slug: "the-loop", part: 0, title: "The generation loop", q: "How does one guess become a paragraph?", Component: Loop },
  { slug: "memory-or-maths", part: 1, title: "Memory or maths", q: "Is the computer waiting on arithmetic, or on memory?", Component: BytesFlops },
  { slug: "speed-limit", part: 1, title: "The speed limit", q: "How fast could this possibly go, before writing any code?", Component: SpeedOfLight },
  { slug: "starting-line", part: 1, title: "The starting line", q: "Why was the first working engine 650 times slower than that limit?", Component: StartingLine },
  { slug: "kv-cache", part: 2, title: "The KV cache", q: "Why re-read the whole text for every new word?", Component: KvCache },
  { slug: "faster-arithmetic", part: 2, title: "Faster arithmetic", q: "Why would sixteen running totals beat one?", Component: Arithmetic },
  { slug: "many-cores", part: 2, title: "Many cores", q: "Ten cores: ten times faster?", Component: Threads },
  { slug: "smaller-numbers", part: 2, title: "Smaller numbers", q: "What if every weight took fewer bytes?", Component: Quantization },
  { slug: "the-gpu", part: 2, title: "The GPU", q: "Why was the first GPU version no faster than one CPU core?", Component: Gpu },
  { slug: "batching", part: 2, title: "Batching", q: "How do you serve many people with one copy of the model?", Component: Batching },
  { slug: "speculative-decoding", part: 2, title: "Guess ahead", q: "Can a small model do some of a big model's writing, without changing a word?", Component: SpecDecode },
  { slug: "many-gpus", part: 2, title: "Many GPUs (a teaser)", q: "What happens when the model doesn't fit on one chip?", Component: ManyGpus },
];

export const chapterHref = (c: Chapter) => `/learn/${c.slug}`;

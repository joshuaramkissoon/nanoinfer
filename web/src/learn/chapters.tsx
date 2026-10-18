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
import WhatIsInference from "./ch/p0/WhatIsInference";
import WhyExpensive from "./ch/p0/WhyExpensive";
import Hardware from "./ch/p0/Hardware";
import Rosetta from "./ch/p0/Rosetta";

export interface Chapter {
  slug: string;
  title: string;
  /** the question the chapter answers (shown under the title and in the map) */
  q: string;
  part: number;
  Component: ComponentType;
}

/** Part 0 is the framing for people who know systems but not ML; parts 1–3 are the original course. */
export const PARTS = ["Inference for systems engineers", "How a model writes text", "Why it's slow", "Making it fast"];
export const PART_BLURB = [
  "The recommended start. What a model physically is, what an inference engine does, why every token is expensive, the hardware, and the field's jargon mapped onto distributed systems you already know.",
  "From a prompt to the next word: tokens, vectors, attention, and the loop that writes one token at a time.",
  "Where the time actually goes, and the speed limit you can work out before writing any code.",
  "The engine's climb from 0.37 to thousands of tokens per second, one bottleneck at a time.",
];

export const CHAPTERS: Chapter[] = [
  { slug: "what-is-inference", part: 0, title: "What inference is", q: "What is a model, physically, and what does it mean to run one?", Component: WhatIsInference },
  { slug: "why-its-expensive", part: 0, title: "Every token reads every weight", q: "Why is each token a model writes so expensive?", Component: WhyExpensive },
  { slug: "the-hardware", part: 0, title: "The hardware in one picture", q: "CPU or GPU, and why is memory the wall?", Component: Hardware },
  { slug: "rosetta-stone", part: 0, title: "A Rosetta stone", q: "Which of these weird terms do you already know under another name?", Component: Rosetta },
  { slug: "next-token", part: 1, title: "Guess the next word", q: "What does a language model actually do?", Component: NextToken },
  { slug: "tokens", part: 1, title: "Tokens", q: "What does the model read, if not letters or words?", Component: Tokens },
  { slug: "embeddings", part: 1, title: "Embeddings", q: "How does a number like 3290 come to mean “dog”?", Component: Embeddings },
  { slug: "multiply-add", part: 1, title: "Multiply and add", q: "What is the one operation that does almost all the work?", Component: DotProduct },
  { slug: "attention", part: 1, title: "Attention", q: "How does a word use the words before it?", Component: Attention },
  { slug: "layers", part: 1, title: "Layers and the residual stream", q: "What happens across the model's twelve layers?", Component: Block },
  { slug: "sampling", part: 1, title: "Sampling", q: "Given the probabilities, which word do we actually say?", Component: Sampling },
  { slug: "the-loop", part: 1, title: "The generation loop", q: "How does one guess become a paragraph?", Component: Loop },
  { slug: "memory-or-maths", part: 2, title: "Memory or maths", q: "Is the computer waiting on arithmetic, or on memory?", Component: BytesFlops },
  { slug: "speed-limit", part: 2, title: "The speed limit", q: "How fast could this possibly go, before writing any code?", Component: SpeedOfLight },
  { slug: "starting-line", part: 2, title: "The starting line", q: "Why was the first working engine 650 times slower than that limit?", Component: StartingLine },
  { slug: "kv-cache", part: 3, title: "The KV cache", q: "Why re-read the whole text for every new word?", Component: KvCache },
  { slug: "faster-arithmetic", part: 3, title: "Faster arithmetic", q: "Why would sixteen running totals beat one?", Component: Arithmetic },
  { slug: "many-cores", part: 3, title: "Many cores", q: "Ten cores: ten times faster?", Component: Threads },
  { slug: "smaller-numbers", part: 3, title: "Smaller numbers", q: "What if every weight took fewer bytes?", Component: Quantization },
  { slug: "the-gpu", part: 3, title: "The GPU", q: "Why was the first GPU version no faster than one CPU core?", Component: Gpu },
  { slug: "batching", part: 3, title: "Batching", q: "How do you serve many people with one copy of the model?", Component: Batching },
  { slug: "speculative-decoding", part: 3, title: "Guess ahead", q: "Can a small model do some of a big model's writing, without changing a word?", Component: SpecDecode },
  { slug: "many-gpus", part: 3, title: "Many GPUs (a teaser)", q: "What happens when the model doesn't fit on one chip?", Component: ManyGpus },
];

export const chapterHref = (c: Chapter) => `/learn/${c.slug}`;

/** Chapter numbers: part 0 is "0.1"–"0.4"; parts 1–3 keep 01–19, so "chapter 9" in the prose still means memory-or-maths. */
const NUMS = new Map<string, string>();
{
  let p0 = 0, rest = 0;
  for (const c of CHAPTERS) NUMS.set(c.slug, c.part === 0 ? `0.${++p0}` : String(++rest).padStart(2, "0"));
}
export const chapterNum = (c: Chapter) => NUMS.get(c.slug) ?? "";
export const chapterBySlug = (slug: string) => CHAPTERS.find((c) => c.slug === slug);

/** The field guide (a reference page, not a chapter). */
export const GUIDE_HREF = "/learn/field-guide";

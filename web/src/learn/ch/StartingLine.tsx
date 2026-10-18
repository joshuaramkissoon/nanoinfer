import { Climb, WorkGrid } from "../Climb";
import { Deeper, Idea } from "../kit";

export default function StartingLine() {
  return (
    <>
      <p>
        Time to meet the engine. The first version of nanoinfer was the simplest correct program possible: plain Rust, one
        core, every number 32-bit, checked against the reference implementation until its answers matched to the fourth
        decimal place.
      </p>
      <p>
        It wrote <span className="k">0.37 tokens per second</span>. Nearly three seconds per word. The speed limit from the last
        chapter is 243. The first engine ran at <b>0.15%</b> of it.
      </p>
      <h2>The problem</h2>
      <p>
        The clue was in the timings. Reading the 64-token prompt went at 26 tokens per second; writing went at 0.37. And
        26 ÷ 72 (the length of the text by then) ≈ 0.36. Every new token cost as much as reading the <i>whole text</i> again,
        because that's exactly what the program did: it had no memory of earlier passes, so to write token 73 it pushed all
        72 previous tokens back through all 12 layers.
      </p>
      <WorkGrid n="11.1" mode="naive" />
      <p>
        That makes the work grow with every token: the 1,000th token costs 1,000 times the first. A correct program can be
        absurdly wasteful, and this one was. That's the starting line for the climb:
      </p>
      <Climb upto={0} n="11.2" />
      <Idea>The first working engine was correct and 650× too slow, because it redid all the previous work for every new token. Part three fixes the biggest waste first, then the next, measuring each step.</Idea>
      <Deeper title="Why start so slow on purpose?">
        <p>
          A simple version is easy to check. Every later, faster version is tested against it: same prompt, same answer, token
          for token. When an optimisation quietly breaks the maths (and several did), the simple version catches it.
        </p>
        <p>
          Each step of the climb is kept as a separate, runnable stage in the engine, so the whole history can be raced side by
          side in the Lab.
        </p>
      </Deeper>
    </>
  );
}

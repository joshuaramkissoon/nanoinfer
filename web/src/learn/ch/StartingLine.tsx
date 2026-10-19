import { Climb, WorkGrid } from "../Climb";
import { Deeper, Idea } from "../kit";

export default function StartingLine() {
  return (
    <>
      <p>
        The first stage of nanoinfer was the simplest correct program possible. It was plain Rust on one core, with every
        number 32-bit. It was checked against the reference implementation until its answers matched to the fourth decimal
        place.
      </p>
      <p>
        It wrote <span className="k">0.37 tokens per second</span>, nearly 3 seconds per token. The speed limit from chapter
        10 is 243 tokens per second, so the first stage ran at <b>0.15%</b> of it.
      </p>
      <h2>The problem</h2>
      <p>
        The clue was in the timings. Reading the 64-token prompt went at 26 tokens per second, and writing went at 0.37.
        The text was 72 tokens long by then, and 26 ÷ 72 ≈ 0.36. Every new token cost as much as reading the <i>whole
        text</i> again. That is what the program did. It kept nothing from earlier passes, so to write token 73 it pushed all
        72 previous tokens back through all 12 layers.
      </p>
      <WorkGrid n="11.1" mode="naive" />
      <p>
        So the work grows with every token, and the 1,000th token costs 1,000 times the first. The program was correct and
        very wasteful. That waste is where the climb starts:
      </p>
      <Climb upto={0} n="11.2" />
      <Idea>The first working engine was correct and 650× too slow, because it redid all the previous work for every new token. Part 3 fixes the biggest waste first, then the next, and measures each stage.</Idea>
      <Deeper title="Why start so slow on purpose?">
        <p>
          A simple stage is easy to check. Every later, faster stage is tested against it: same prompt, same answer, token
          for token. When an optimisation breaks the arithmetic without any error (and several did), the simple stage
          catches it.
        </p>
        <p>
          Every stage of the climb stays in the engine and still runs, so you can race the whole history side by side in
          the lab.
        </p>
      </Deeper>
    </>
  );
}

import { useState } from "react";
import { usePalette } from "../../ui/theme";
import { Climb, WorkGrid } from "../Climb";
import { chartColors, Deeper, G, Idea, Slider } from "../kit";

function CacheSize() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [n, setN] = useState(512);
  const per = 2 * 12 * 768 * 4; // keys + values, 12 layers, 768 numbers, 4 bytes
  const kv = n * per;
  const rows: [string, number, string][] = [
    ["KV cache, 32-bit", kv, cc.c2],
    ["KV cache, 16-bit", kv / 2, cc.c2],
    ["weights, 4-bit", 97e6, pal.muted],
  ];
  const mx = Math.max(...rows.map((r) => r[1]), 1);
  return (
    <div>
      <Slider label="tokens of text" value={n} min={16} max={1024} step={16} onChange={setN} />
      <div className="lr-bars" style={{ marginTop: "0.6rem" }}>
        {rows.map(([k, v, c]) => (
          <div key={k} className="lr-bar" style={{ gridTemplateColumns: "8.5rem 1fr 4.5rem" }}>
            <span className="tk">{k}</span>
            <span className="b">
              <i style={{ width: `${(v / mx) * 100}%`, background: c }} />
            </span>
            <span className="pc">{(v / 1e6).toFixed(1)} MB</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function KvCache() {
  return (
    <>
      <h2 style={{ marginTop: "2.2rem" }}>The bottleneck</h2>
      <p>
        The first engine recomputed every earlier token for every new one. But look at what attention (chapter 5) actually
        needs from an earlier token: only its <b>key</b> and its <b>value</b>. And those never change. Once “cat” has been
        through layer 3, its key and value in layer 3 are fixed forever.
      </p>
      <h2>The idea</h2>
      <p>
        Save them. Each time a token goes through a layer, write its key and value into a table, the <G k="KV cache" />. A new
        token computes only itself, and reads the earlier tokens' keys and values from the table instead of recomputing
        them.
      </p>
      <WorkGrid n="12.1" />
      <h2>The result</h2>
      <p>
        Same maths, same answers, decimal for decimal. Decode went from 0.37 to <span className="k">18.95 tokens per second</span>,
        51× faster. The big matrix multiplies now cost the same for every token, however long the text. Attention still
        grows: each new token reads the key and value of every earlier token from the cache, so the cost per token rises
        slowly with the text instead of steeply.
      </p>
      <Climb upto={1} n="12.2" />
      <p>
        But 19 is still only 8% of the speed limit. If decode is meant to be waiting on memory, it should be reading 120 GB/s.
        It was reading 10. Something else is slow, and it's the next chapter.
      </p>
      <Idea>Old tokens' keys and values never change, so store them. Each new token then costs one token's work, plus a read of the cache, instead of the whole text's.</Idea>
      <Deeper title="The cache isn't free: it's memory">
        <p>
          For every token of text the cache holds a key and a value (768 numbers each) for each of the 12 layers: 73.7 KB at
          32 bits. That grows with the text, and it has to be read on every step. At 1,000 tokens the cache is bigger than the
          4-bit weights themselves:
        </p>
        <CacheSize />
        <p>
          That's why long conversations get slower, why later stages store the cache in 16 bits (+60% decode speed at ~900
          tokens), and why serving many users is limited by cache memory (chapter 17).
        </p>
      </Deeper>
    </>
  );
}

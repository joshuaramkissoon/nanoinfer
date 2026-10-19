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
      <h2 style={{ marginTop: "2.2rem" }}>The bottleneck: redoing every earlier token</h2>
      <p>
        The first stage recomputed every earlier token for every new one. But attention (chapter 5) needs only two things
        from an earlier token: its <b>key</b> and its <b>value</b>. Those never change. Once “cat” has been through layer 3,
        its key and value in layer 3 are fixed.
      </p>
      <h2>The idea: store keys and values</h2>
      <p>
        Save them. Each time a token goes through a layer, the engine writes its key and value into a table, the{" "}
        <G k="KV cache" />. A new token computes only itself. It reads the earlier tokens' keys and values from the table
        instead of recomputing them.
      </p>
      <WorkGrid n="12.1" />
      <h2>The result: 51× faster</h2>
      <p>
        Same arithmetic, same answers, decimal for decimal. Decode went from 0.37 to{" "}
        <span className="k">18.95 tokens per second</span>, 51× faster. The big matrix multiplies now cost the same for every
        token, however long the text. Attention still grows, because each new token reads the key and value of every earlier
        token from the KV cache. So the cost per token rises slowly with the text instead of steeply.
      </p>
      <Climb upto={1} n="12.2" />
      <p>
        But 19 tokens per second is still only 8% of the speed limit. If decode were waiting on memory, it would read close
        to 120 GB/s. It was reading 10 GB/s, so something other than memory is slow. Chapter 13 finds it.
      </p>
      <Idea>Old tokens' keys and values never change, so store them. Each new token then costs one token's work, plus a read of the cache, instead of the whole text's.</Idea>
      <Deeper title="The KV cache costs memory">
        <p>
          For every token of text, the KV cache holds a key and a value (768 numbers each) for each of the 12 layers. That is
          73.7 KB at 32 bits. It grows with the text, and every step reads it. At 1,000 tokens the KV cache is bigger than
          the 4-bit weights:
        </p>
        <CacheSize />
        <p>
          So long conversations get slower. Later stages store the KV cache in 16 bits (+60% decode speed at about 900
          tokens). And KV cache memory limits how many users one server can hold (chapter 17).
        </p>
      </Deeper>
    </>
  );
}

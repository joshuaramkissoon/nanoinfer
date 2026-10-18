// End-of-chapter questions. Each tests an intuition the chapter built ("why is decode memory-bound?"), not a number to
// memorise. Wrong options are the misconceptions a systems engineer would plausibly hold.

export interface Q {
  q: string;
  options: string[];
  answer: number;
  why: string;
}

export const CHECKS: Record<string, Q[]> = {
  // ---- part 0 ----
  "what-is-inference": [
    {
      q: "A team retrains its model once a quarter and serves it to millions of users daily. Where does most of the lifetime compute bill go?",
      options: ["Training: it's the expensive part", "Inference: the frozen function runs billions of times", "Storage of the weights"],
      answer: 1,
      why: "Training is a huge one-off cost; inference is a smaller cost paid on every request, forever. At scale the running cost dominates, which is why a whole field exists to make serving cheaper.",
    },
    {
      q: "Two users chat with the same model at the same time. What do they share, and what is theirs alone?",
      options: [
        "They share the weights (read-only); each has their own KV cache (per-session state)",
        "Each needs their own copy of the weights",
        "They share everything, including the conversation state",
      ],
      answer: 0,
      why: "Weights are an immutable, shared asset, like a read-only dataset every request reads. The KV cache is per-request state that grows with the conversation and must be stored, scheduled and evicted, like session state in any stateful service.",
    },
    {
      q: "Is a model's forward pass deterministic?",
      options: [
        "Yes: same weights + same tokens give the same scores; randomness only enters when sampling the next token",
        "No: the network itself is random each time",
      ],
      answer: 0,
      why: "The forward pass is a pure function of (weights, tokens). The variety you see in chat answers comes from sampling (chapter 07), plus tiny floating-point differences when batch shapes change.",
    },
  ],
  "why-its-expensive": [
    {
      q: "Why is generating one token for one user memory-bound on almost any chip?",
      options: [
        "Because each weight read from memory is used for just one multiply-add: ~1 FLOP per byte, while chips can do hundreds of FLOPs per byte read",
        "Because the arithmetic is very complicated",
        "Because the tokens themselves are large",
      ],
      answer: 0,
      why: "Every token touches every weight once. At bf16 that's 2 bytes moved per multiply-add. An H100 can do ~295 FLOPs in the time it reads one byte, so the maths units sit idle waiting for memory.",
    },
    {
      q: "A 70B model in bf16 on one H100: what's the first problem?",
      options: ["It's too slow", "It doesn't fit: 140 GB of weights vs 80 GB of memory", "The tokenizer can't handle it"],
      answer: 1,
      why: "Capacity comes before bandwidth. 70 billion × 2 bytes = 140 GB. It must be quantised (int4 ≈ 35 GB) or split across GPUs, and splitting brings the interconnect into every token.",
    },
    {
      q: "Why does batching 64 users barely slow each one down (at first)?",
      options: [
        "The weights are read once per step and used 64 times, and the idle maths units absorb the extra work",
        "The GPU gets 64× more memory bandwidth",
        "Each user gets a smaller model",
      ],
      answer: 0,
      why: "The step time is set by reading the weights. Until the extra arithmetic catches up with that read (the ridge point), more users ride along almost free, so cost per token falls nearly 64×.",
    },
  ],
  "the-hardware": [
    {
      q: "Why do GPUs win at inference, beyond having more cores?",
      options: [
        "Mainly memory bandwidth: an H100 reads ~3.35 TB/s, ~30× an M4, and decode is bandwidth-bound",
        "GPUs have smarter branch predictors",
        "GPUs run at a higher clock speed",
      ],
      answer: 0,
      why: "Decode waits on bytes, so the memory system decides speed. GPUs pair thousands of simple lanes with HBM stacked right next to the die. The lanes matter most for prefill and batching, where work per byte is high.",
    },
    {
      q: "You split a model across 8 GPUs. Inside each GPU memory moves at ~3,350 GB/s; between them NVLink moves ~450 GB/s each way. What does that ratio tell you?",
      options: [
        "Only send data between GPUs when it's much smaller than what each GPU reads locally",
        "Multi-GPU is always 8× faster",
        "The interconnect doesn't matter for inference",
      ],
      answer: 0,
      why: "Every byte crossing a link is ~7× slower than a local one (and ~67× over the network). Good splits send small activations, never weights, and overlap the talking with the work. Same instinct as minimising cross-shard traffic.",
    },
  ],
  "rosetta-stone": [
    {
      q: "Paged attention borrows its key idea from…",
      options: ["Virtual memory: fixed-size pages + a per-request page table, so memory needn't be contiguous", "TCP congestion control", "Consistent hashing"],
      answer: 0,
      why: "Reserving one contiguous KV region per request for its maximum length wastes most of it. Pages allocated on demand cut fragmentation to the last half-page, exactly as an OS does for processes.",
    },
    {
      q: "Speculative decoding is like optimistic concurrency because…",
      options: [
        "A cheap drafter guesses ahead, the big model validates all guesses in one pass, and wrong guesses are rolled back with no change to the output",
        "It takes locks on the KV cache",
        "It retries failed requests on another GPU",
      ],
      answer: 0,
      why: "Execute speculatively, validate in bulk, discard on conflict. It pays off when guesses are usually right (high acceptance), just like optimistic transactions pay off when conflicts are rare.",
    },
    {
      q: "In a mixture-of-experts model, what's the analogue of a hot shard?",
      options: ["An expert the router picks for a large share of tokens, so its GPU becomes the bottleneck", "A GPU that overheats", "A long prompt"],
      answer: 0,
      why: "Routing tokens to experts is key-based routing to shards. If the routing is skewed, one expert's device does most of the work while the others idle; training adds load-balancing losses and serving adds capacity limits, much like rebalancing hot keys.",
    },
  ],

  // ---- part 1 ----
  "next-token": [
    {
      q: "What does one run of the model produce?",
      options: ["A probability for every token in the vocabulary, for the next position only", "The whole answer at once", "One word, chosen deterministically"],
      answer: 0,
      why: "A language model is a function from text so far to a distribution over the next token. Longer text comes from calling it in a loop.",
    },
    {
      q: "Why can't one answer's tokens be generated in parallel?",
      options: ["Each token is an input to the next forward pass", "The GPU doesn't allow it", "Tokens are too big to batch"],
      answer: 0,
      why: "Generation is inherently sequential: a dependency chain where step n+1 needs step n's output. That single fact shapes most of inference engineering.",
    },
  ],
  tokens: [
    {
      q: "Why do models read tokens rather than characters or whole words?",
      options: [
        "A middle ground: common words are one token, rare words split into pieces, so the vocabulary stays fixed-size and nothing is out-of-vocabulary",
        "Characters are not supported by computers",
        "Words would be faster in every way",
      ],
      answer: 0,
      why: "Characters make sequences long (more forward passes); whole words make the vocabulary unbounded. Subword tokens compress text roughly 4 characters per token in English.",
    },
    {
      q: "Speeds are quoted in tokens per second. Why is that the natural unit?",
      options: ["Each decode step produces exactly one token, so the cost is per token", "Because tokens are always one word", "It's just marketing"],
      answer: 0,
      why: "One forward pass = one new token (per user). Prices, SLOs and benchmarks all count in tokens because that's the unit of work.",
    },
  ],
  embeddings: [
    {
      q: "What is an embedding, physically?",
      options: ["A row of a big lookup table: token ID → 768 learned numbers", "A compressed image of the word", "A hash of the token's spelling"],
      answer: 0,
      why: "The first operation is a table lookup. Training arranges the rows so tokens used in similar ways get similar vectors.",
    },
    {
      q: "Does the embedding lookup cost much at inference time?",
      options: ["No: it reads one row per token, ~3 KB, out of ~500 MB of weights", "Yes: it's the main cost"],
      answer: 0,
      why: "It's a gather, not a multiply. The expensive work is the matrix multiplies in every layer that follow.",
    },
  ],
  "multiply-add": [
    {
      q: "Roughly how many FLOPs does a forward pass take per token, for a model with N weights?",
      options: ["About 2N: one multiply and one add per weight", "About N²", "About log N"],
      answer: 0,
      why: "Almost all the work is matrix × vector, and each weight takes part in exactly one multiply-add per token. That's why bytes per token and FLOPs per token are both proportional to model size.",
    },
    {
      q: "Why does matrix-vector dominate so completely?",
      options: ["Every layer's big operations are matrices of weights applied to the token's vector", "Because attention is free", "Because the model is mostly lookups"],
      answer: 0,
      why: "Attention projections and the feed-forward block are all weight matrices. The other operations (norms, softmax, adds) touch far fewer numbers.",
    },
  ],
  attention: [
    {
      q: "What does attention let a token do that the rest of the layer can't?",
      options: ["Read information from earlier tokens", "Change the weights", "Skip layers"],
      answer: 0,
      why: "Everything else in a layer processes each token on its own. Attention is the only step where tokens communicate: each one scores earlier tokens and blends their values.",
    },
    {
      q: "Why does attention need per-token state that grows with the conversation?",
      options: [
        "Each new token compares against every earlier token's key and value",
        "Because the weights grow",
        "Because softmax needs memory",
      ],
      answer: 0,
      why: "That growing set of keys and values is what the KV cache stores (chapter 12): the per-session state of an inference server.",
    },
  ],
  layers: [
    {
      q: "What does each of the 12 layers do to the token's vector?",
      options: ["Adds an edit to a shared running vector (the residual stream)", "Replaces it completely", "Shrinks it"],
      answer: 0,
      why: "Each block reads the vector, computes an update with attention and a feed-forward network, and adds it back. The guess sharpens with depth.",
    },
    {
      q: "Why does depth matter for inference cost?",
      options: ["Each layer has its own weights, and all layers run for every token", "Only the last layer runs at inference", "Layers run in parallel for one token"],
      answer: 0,
      why: "Layers are sequential within a token: layer 2 needs layer 1's output. That is also what pipeline parallelism cuts along.",
    },
  ],
  sampling: [
    {
      q: "What does temperature change?",
      options: ["How peaked the probabilities are before rolling the die", "The model's weights", "How fast it runs"],
      answer: 0,
      why: "Low temperature sharpens the distribution toward the top choice; high flattens it. It's a cheap post-processing step on the scores.",
    },
    {
      q: "Does sampling affect speed?",
      options: ["Barely: it's a pass over 50,257 scores, tiny next to the forward pass", "Yes, it dominates"],
      answer: 0,
      why: "The forward pass reads hundreds of MB; sampling reads a few hundred KB. It matters for correctness, not cost.",
    },
  ],
  "the-loop": [
    {
      q: "Why is prefill much faster per token than decode?",
      options: [
        "Prefill processes all prompt tokens in one pass, reusing each weight read many times; decode reads every weight for one token",
        "Prefill uses a smaller model",
        "Decode does more maths per token",
      ],
      answer: 0,
      why: "Same arithmetic per token, very different bytes per token. Prefill is compute-bound; decode is memory-bound. Two phases, two bottlenecks.",
    },
    {
      q: "Which user-facing metric does prefill dominate?",
      options: ["Time to first token", "Time between tokens", "Neither"],
      answer: 0,
      why: "The first token can't appear until the whole prompt is read. After that, decode sets the streaming pace.",
    },
  ],

  // ---- part 2 ----
  "memory-or-maths": [
    {
      q: "Decode does about 0.5 FLOPs per byte read here; the chip can do far more per byte. So the chip is…",
      options: ["Waiting on memory (memory-bound)", "Waiting on arithmetic (compute-bound)", "Waiting on the network"],
      answer: 0,
      why: "Compare work-per-byte with the machine's FLOPs-per-byte ratio (the ridge). Below it, memory sets the pace; above it, arithmetic does.",
    },
    {
      q: "If a job is memory-bound, which fix helps?",
      options: ["Read fewer bytes, or use each byte for more work (smaller weights, batching)", "A faster ALU", "More clever arithmetic"],
      answer: 0,
      why: "Faster maths doesn't help a job waiting on the truck. That's why quantisation and batching are the big levers for decode.",
    },
  ],
  "speed-limit": [
    {
      q: "How do you estimate the best possible decode speed of any model on any machine?",
      options: ["Memory bandwidth ÷ bytes of weights read per token", "Clock speed × cores", "FLOPs ÷ parameters"],
      answer: 0,
      why: "120 GB/s ÷ 495 MB ≈ 243 tok/s for GPT-2 small in f32 on the M4. The same arithmetic gives ~48 tok/s for a 70B bf16 model on an H100 pair: a back-of-envelope SLO you can do before writing code.",
    },
    {
      q: "Halving the bytes per weight does what to that limit?",
      options: ["Doubles it", "Halves it", "Nothing"],
      answer: 0,
      why: "The limit is inversely proportional to bytes per token. That's the whole argument for quantisation.",
    },
  ],
  "starting-line": [
    {
      q: "Why was the first correct engine so slow?",
      options: ["It recomputed every earlier token for every new one, so cost grew with the text", "Rust is slow", "The weights were corrupted"],
      answer: 0,
      why: "No memoisation: O(n) work per token, O(n²) per answer. The fix is caching the state that doesn't change.",
    },
    {
      q: "Why keep that slow version around?",
      options: ["It's the reference oracle: every faster version must produce the same tokens", "For nostalgia", "It's faster on some machines"],
      answer: 0,
      why: "Like a simple reference implementation in a test harness: optimisations that quietly break the maths get caught against it.",
    },
  ],

  // ---- part 3 ----
  "kv-cache": [
    {
      q: "Why can keys and values be cached but not the whole computation?",
      options: [
        "An earlier token's key and value never change once computed; only the new token's work is new",
        "Because the cache is free",
        "Because the weights change each step",
      ],
      answer: 0,
      why: "Causal attention means past tokens never look forward, so their keys/values are immutable. Classic memoisation of an append-only log.",
    },
    {
      q: "What does the KV cache cost?",
      options: ["Memory that grows with conversation length × users, and must be read every step", "Nothing", "Extra training"],
      answer: 0,
      why: "It's per-session state. At long contexts it can outgrow the weights, which is why serving capacity is usually limited by KV memory.",
    },
  ],
  "faster-arithmetic": [
    {
      q: "Why did sixteen running totals beat one?",
      options: [
        "One total makes every add wait for the previous one; sixteen independent chains keep the pipeline full",
        "Sixteen totals use less memory",
        "The compiler likes the number 16",
      ],
      answer: 0,
      why: "It's latency vs throughput at the instruction level: break the dependency chain and the hardware overlaps the work (and can vectorise it with SIMD).",
    },
  ],
  "many-cores": [
    {
      q: "Ten cores gave decode only ~1.5×. Why?",
      options: ["The cores share one memory system, which one core already half-saturates", "Threads are broken in Rust", "Only one core can run maths"],
      answer: 0,
      why: "Like adding app servers in front of a saturated database: throughput stops at the shared bottleneck.",
    },
    {
      q: "Why did prefill scale much better with cores?",
      options: ["Prefill is compute-bound, so more arithmetic units really add speed", "Prefill uses less memory overall", "It doesn't scale better"],
      answer: 0,
      why: "Same hardware, different bottleneck. Scaling advice depends on which resource is saturated.",
    },
  ],
  "smaller-numbers": [
    {
      q: "Why does quantising weights speed up decode even though the maths is the same?",
      options: ["Decode is bandwidth-bound; fewer bytes per weight means fewer bytes per token", "Integer maths is always faster", "It skips layers"],
      answer: 0,
      why: "Lossy compression of the dominant read. 4-bit weights are 8× fewer bytes than 32-bit, so the ceiling rises 8×.",
    },
    {
      q: "What's the cost?",
      options: ["Accuracy: each weight is approximated, and at 4 bits the model gets measurably worse", "Nothing", "The model gets larger"],
      answer: 0,
      why: "8-bit is nearly invisible; 4-bit is a real trade-off; 2-bit falls apart. Blocks with shared scales keep the error small.",
    },
  ],
  "the-gpu": [
    {
      q: "Why was the first GPU version no faster than one CPU core?",
      options: [
        "The CPU waited for the GPU after every operation: an RPC round-trip per tiny job",
        "The GPU was broken",
        "GPUs are bad at matrix maths",
      ],
      answer: 0,
      why: "Chatty synchronous calls kill throughput, on a GPU as over a network. Queue the work, wait once per token, and shape each kernel for thousands of threads.",
    },
  ],
  batching: [
    {
      q: "Why is total throughput much higher with batching while each user gets a bit slower?",
      options: [
        "Each weight read is shared by every user in the batch, but each step does more arithmetic",
        "Batching gives each user a smaller model",
        "The GPU runs at a higher clock",
      ],
      answer: 0,
      why: "It amortises the fixed cost (reading the weights), like group commit amortises an fsync. Past the ridge point it becomes compute-bound and per-user speed drops.",
    },
    {
      q: "What does continuous batching fix?",
      options: ["Short requests waiting for the longest one in their batch, and new arrivals waiting for a whole batch", "Memory bandwidth", "Model accuracy"],
      answer: 0,
      why: "Re-forming the batch every step is a scheduler admitting work each tick instead of in rigid epochs. Median wait for the first token here fell from 3.5 s to 40 ms.",
    },
  ],
  "speculative-decoding": [
    {
      q: "Why can the big model check k guesses in about the time of one step?",
      options: [
        "Checking k tokens is a batch: one read of the weights, k tokens of work, and decode had idle arithmetic to spare",
        "The big model skips layers when checking",
        "The guesses are always right",
      ],
      answer: 0,
      why: "It turns the sequential chain into a small batch for one user. When most guesses are accepted, you get several tokens per weight read.",
    },
    {
      q: "Does speculative decoding change the output?",
      options: ["No: rejected guesses are discarded and the big model's own token is used", "Yes, it lowers quality slightly"],
      answer: 0,
      why: "Validation makes it exact (for greedy decoding here; with sampling, a rejection rule keeps the distribution identical).",
    },
  ],
  "many-gpus": [
    {
      q: "Why would splitting GPT-2 small across GPUs make it slower?",
      options: ["The all-reduces are tiny messages, so latency dominates and costs more than the work saved", "GPUs can't share models", "It would use too much memory"],
      answer: 0,
      why: "Sharding only pays when per-shard work is large next to the communication. Small models stay on one device; big ones split because they must.",
    },
  ],
};

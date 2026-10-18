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
      options: ["Training: weeks on thousands of GPUs", "Inference: it runs on every request", "Storing and shipping the weight files"],
      answer: 1,
      why: "Training is a huge one-off cost; inference is a smaller cost paid on every request, forever. At scale the running cost dominates, which is why a whole field exists to make serving cheaper.",
    },
    {
      q: "Two users chat with the same model at the same time. What do they share, and what is theirs alone?",
      options: [
        "Shared weights; each has their own KV cache",
        "Each needs a private copy of the weights",
        "Everything, including the conversation's state",
      ],
      answer: 0,
      why: "Weights are an immutable, shared asset, like a read-only dataset every request reads. The KV cache is per-request state that grows with the conversation and must be stored, scheduled and evicted, like session state in any stateful service.",
    },
    {
      q: "You send the same prompt twice, at temperature 0, to a busy production server. Will the answers always match?",
      options: [
        "Yes: with no sampling, the maths is fixed",
        "No: batch size changes how floats get added",
        "No: the network adds noise on every pass",
      ],
      answer: 1,
      why: "On one machine with fixed shapes, the forward pass is a pure function of (weights, tokens). A server batches your request with whoever else is online, and picks kernels and reduction orders by batch size. Float addition isn't associative (chapter 13), so a logit can move in its last digit, flip a near-tie, and the text diverges from there. Bit-identical output needs batch-invariant kernels.",
    },
  ],
  "why-its-expensive": [
    {
      q: "Why is generating one token for one user memory-bound on almost any chip?",
      options: [
        "Each weight fetched does one multiply-add, then waits",
        "The arithmetic per weight is long and complicated",
        "Each token's vector is too big to fit in the caches",
      ],
      answer: 0,
      why: "Every token touches every weight once. At bf16 that's 2 bytes moved per multiply-add. An H100 can do ~295 FLOPs in the time it reads one byte, so the maths units sit idle waiting for memory.",
    },
    {
      q: "A 70B model in bf16 on one H100: what's the first problem?",
      options: ["Decode would run at under one token a second", "It doesn't fit: 140 GB vs 80 GB", "The arithmetic would saturate the GPU's lanes"],
      answer: 1,
      why: "Capacity comes before bandwidth. 70 billion × 2 bytes = 140 GB. It must be quantised (int4 ≈ 35 GB) or split across GPUs, and splitting brings the interconnect into every token.",
    },
    {
      q: "Why does batching 64 users barely slow each one down (at first)?",
      options: [
        "One weight read now serves 64 users",
        "The GPU raises its memory clock under load",
        "Each user's request is routed to a smaller model",
      ],
      answer: 0,
      why: "The step time is set by reading the weights. Until the extra arithmetic catches up with that read (the ridge point), more users ride along almost free, so cost per token falls nearly 64×.",
    },
  ],
  "the-hardware": [
    {
      q: "Why do GPUs win at inference, beyond having more cores?",
      options: [
        "Mainly memory bandwidth (HBM)",
        "Much higher clock speeds than CPUs",
        "Better branch prediction for the model's control flow",
      ],
      answer: 0,
      why: "Decode waits on bytes, so the memory system decides speed. GPUs pair thousands of simple lanes with HBM stacked right next to the die. The lanes matter most for prefill and batching, where work per byte is high.",
    },
    {
      q: "You split a model across 8 GPUs. Inside each GPU memory moves at ~3.35 TB/s; between them NVLink moves ~450 GB/s each way. What does that ratio tell you?",
      options: [
        "Send small results between GPUs, never the weights",
        "Eight GPUs will decode about 8× faster",
        "The link only matters while loading the model",
      ],
      answer: 0,
      why: "Every byte crossing a link is ~7× slower than a local one (and ~67× over the network). Good splits send small activations, never weights, and overlap the talking with the work. Same instinct as minimising cross-shard traffic.",
    },
  ],
  "rosetta-stone": [
    {
      q: "Paged attention borrows its key idea from…",
      options: ["Virtual memory paging", "TCP congestion control", "Consistent hashing across GPUs"],
      answer: 0,
      why: "Reserving one contiguous KV region per request for its maximum length wastes most of it. Pages allocated on demand cut fragmentation to the last half-page, exactly as an OS does for processes.",
    },
    {
      q: "Speculative decoding is like optimistic concurrency because…",
      options: [
        "Guess cheaply, validate in bulk, discard conflicts",
        "It locks the KV cache while two models write to it",
        "It retries a failed step on another GPU",
      ],
      answer: 0,
      why: "Execute speculatively, validate in bulk, discard on conflict. It pays off when guesses are usually right (high acceptance), just like optimistic transactions pay off when conflicts are rare.",
    },
    {
      q: "In a mixture-of-experts model, what's the analogue of a hot shard?",
      options: ["An expert the router picks far more than the rest", "A GPU that thermally throttles under load", "One user's very long prompt"],
      answer: 0,
      why: "Routing tokens to experts is key-based routing to shards. If the routing is skewed, one expert's device does most of the work while the others idle; training adds load-balancing losses and serving adds capacity limits, much like rebalancing hot keys.",
    },
  ],

  // ---- part 1 ----
  "next-token": [
    {
      q: "What does one run of the model produce?",
      options: ["A probability for every possible next token", "The whole answer, all positions at once", "One next word, chosen by the network"],
      answer: 0,
      why: "A language model is a function from text so far to a distribution over the next token. Longer text comes from calling it in a loop.",
    },
    {
      q: "Why can't one answer's tokens be generated in parallel?",
      options: ["Each token is an input to the next pass", "GPUs can only run one forward pass at a time", "Tokens of one answer are too large to batch"],
      answer: 0,
      why: "Generation is inherently sequential: a dependency chain where step n+1 needs step n's output. That single fact shapes most of inference engineering.",
    },
  ],
  tokens: [
    {
      q: "Why do models read tokens rather than characters or whole words?",
      options: [
        "A fixed vocabulary that can still spell anything",
        "Characters can't be represented as numbers",
        "Whole words would make the model too accurate",
      ],
      answer: 0,
      why: "Characters make sequences long (more forward passes); whole words make the vocabulary unbounded. Subword tokens compress text roughly 4 characters per token in English.",
    },
    {
      q: "Speeds are quoted in tokens per second. Why is that the natural unit?",
      options: ["Each decode step produces one token", "Because a token is always exactly one word", "It's the unit the GPU's hardware counts"],
      answer: 0,
      why: "One forward pass = one new token (per user). Prices, SLOs and benchmarks all count in tokens because that's the unit of work.",
    },
  ],
  embeddings: [
    {
      q: "What is an embedding, physically?",
      options: ["A row of a lookup table, learned in training", "A hash of the token's spelling into 768 numbers", "A compressed picture of how the word looks"],
      answer: 0,
      why: "The first operation is a table lookup. Training arranges the rows so tokens used in similar ways get similar vectors.",
    },
    {
      q: "Does the embedding lookup cost much at inference time?",
      options: ["No: one ~3 KB row per token", "Yes: it reads the whole 150 MB table every token"],
      answer: 0,
      why: "It's a gather, not a multiply. The expensive work is the matrix multiplies in every layer that follow.",
    },
  ],
  "multiply-add": [
    {
      q: "Roughly how many FLOPs does a forward pass take per token, for a model with N weights?",
      options: ["About 2N", "About N², every weight meets every other", "About log N, thanks to the tree of layers"],
      answer: 0,
      why: "Almost all the work is matrix × vector, and each weight takes part in exactly one multiply-add per token. That's why bytes per token and FLOPs per token are both proportional to model size.",
    },
    {
      q: "Why does matrix-vector dominate so completely?",
      options: ["Nearly every weight lives in a matrix", "Attention's softmax is the expensive part", "The model is mostly table lookups"],
      answer: 0,
      why: "Attention projections and the feed-forward block are all weight matrices. The other operations (norms, softmax, adds) touch far fewer numbers.",
    },
  ],
  attention: [
    {
      q: "What does attention let a token do that the rest of the layer can't?",
      options: ["Read information from earlier tokens", "Update the weights for this conversation", "Decide which layers to skip for this token"],
      answer: 0,
      why: "Everything else in a layer processes each token on its own. Attention is the only step where tokens communicate: each one scores earlier tokens and blends their values.",
    },
    {
      q: "Why does attention need per-token state that grows with the conversation?",
      options: [
        "Each new token reads every earlier key and value",
        "The weights grow a little with every token",
        "Softmax keeps a running total that grows",
      ],
      answer: 0,
      why: "That growing set of keys and values is what the KV cache stores (chapter 12): the per-session state of an inference server.",
    },
  ],
  layers: [
    {
      q: "What does each of the 12 layers do to the token's vector?",
      options: ["Adds an edit to it", "Replaces it with a new vector", "Compresses it to fewer numbers"],
      answer: 0,
      why: "Each block reads the vector, computes an update with attention and a feed-forward network, and adds it back. The guess sharpens with depth.",
    },
    {
      q: "Why does depth matter for inference cost?",
      options: ["Every layer's weights are read every token", "Only the last layer runs at inference time", "All 12 layers run in parallel for one token"],
      answer: 0,
      why: "Layers are sequential within a token: layer 2 needs layer 1's output. That is also what pipeline parallelism cuts along.",
    },
  ],
  sampling: [
    {
      q: "What does temperature change?",
      options: ["How peaked the probabilities are", "The model's weights, slightly, each step", "How many layers the token goes through"],
      answer: 0,
      why: "Low temperature sharpens the distribution toward the top choice; high flattens it. It's a cheap post-processing step on the scores.",
    },
    {
      q: "Does sampling affect speed?",
      options: ["Barely: it touches a few hundred KB", "Yes: sorting 50,257 scores dominates each step"],
      answer: 0,
      why: "The forward pass reads hundreds of MB; sampling reads a few hundred KB. It matters for correctness, not cost.",
    },
  ],
  "the-loop": [
    {
      q: "Why is prefill much faster per token than decode?",
      options: [
        "Each weight read serves the whole prompt",
        "Prefill runs a smaller, faster copy of the model",
        "Decode does far more arithmetic per token",
      ],
      answer: 0,
      why: "Same arithmetic per token, very different bytes per token. Prefill is compute-bound; decode is memory-bound. Two phases, two bottlenecks.",
    },
    {
      q: "Which user-facing metric does prefill dominate?",
      options: ["Time to first token", "Time between tokens", "Neither: sampling dominates both"],
      answer: 0,
      why: "The first token can't appear until the whole prompt is read. After that, decode sets the streaming pace.",
    },
  ],

  // ---- part 2 ----
  "memory-or-maths": [
    {
      q: "Decode does about 0.5 FLOPs per byte read here; the chip can do far more per byte. So the chip is…",
      options: ["Waiting on memory", "Waiting on arithmetic", "Waiting on the operating system's scheduler"],
      answer: 0,
      why: "Compare work-per-byte with the machine's FLOPs-per-byte ratio (the ridge). Below it, memory sets the pace; above it, arithmetic does.",
    },
    {
      q: "If a job is memory-bound, which fix helps?",
      options: ["Fewer bytes, or more work per byte", "A processor with faster multiply units", "Cleverer arithmetic with fewer operations"],
      answer: 0,
      why: "Faster maths doesn't help a job waiting on the truck. That's why quantisation and batching are the big levers for decode.",
    },
  ],
  "speed-limit": [
    {
      q: "How do you estimate the best possible decode speed of any model on any machine?",
      options: ["Bandwidth ÷ bytes per token", "Clock speed × cores × SIMD width", "Peak FLOP/s ÷ parameter count"],
      answer: 0,
      why: "120 GB/s ÷ 495 MB ≈ 243 tok/s for GPT-2 small in f32 on the M4. The same arithmetic gives ~48 tok/s for a 70B bf16 model on an H100 pair: a back-of-envelope SLO you can do before writing code.",
    },
    {
      q: "Halving the bytes per weight does what to that limit?",
      options: ["Doubles it", "Halves it", "Nothing: the arithmetic is the same"],
      answer: 0,
      why: "The limit is inversely proportional to bytes per token. That's the whole argument for quantisation.",
    },
  ],
  "starting-line": [
    {
      q: "Why was the first correct engine so slow?",
      options: ["It redid every earlier token for each new one", "Plain Rust without SIMD is that slow", "It read the weights from disk every token"],
      answer: 0,
      why: "No memoisation: O(n) work per token, O(n²) per answer. The fix is caching the state that doesn't change.",
    },
    {
      q: "Why keep that slow version around?",
      options: ["It's the reference every faster version must match", "It's faster than the others on long prompts", "It uses the least memory, for small devices"],
      answer: 0,
      why: "Like a simple reference implementation in a test harness: optimisations that quietly break the maths get caught against it.",
    },
  ],

  // ---- part 3 ----
  "kv-cache": [
    {
      q: "Why can keys and values be cached but not the whole computation?",
      options: [
        "Past keys and values never change",
        "The cache lives in faster memory than the weights",
        "Each layer's output depends only on the new token",
      ],
      answer: 0,
      why: "Causal attention means past tokens never look forward, so their keys/values are immutable. Classic memoisation of an append-only log.",
    },
    {
      q: "What does the KV cache cost?",
      options: ["Memory that grows with length × users", "Nothing: it only saves work", "Extra training to make the model use it"],
      answer: 0,
      why: "It's per-session state. At long contexts it can outgrow the weights, which is why serving capacity is usually limited by KV memory.",
    },
  ],
  "faster-arithmetic": [
    {
      q: "Why did sixteen running totals beat one?",
      options: [
        "Sixteen independent chains keep the adder busy",
        "Sixteen totals need fewer bytes from memory",
        "The 16 totals run on 16 different cores",
      ],
      answer: 0,
      why: "It's latency vs throughput at the instruction level: break the dependency chain and the hardware overlaps the work (and can vectorise it with SIMD).",
    },
  ],
  "many-cores": [
    {
      q: "Ten cores gave decode only ~1.5×. Why?",
      options: ["They share one memory system", "Rust threads serialise on a global lock", "Only the performance cores can do floating point"],
      answer: 0,
      why: "Like adding app servers in front of a saturated database: throughput stops at the shared bottleneck.",
    },
    {
      q: "Why did prefill scale much better with cores?",
      options: ["Prefill is compute-bound", "Prefill reads fewer bytes in total", "Prefill's threads never need to sleep"],
      answer: 0,
      why: "Same hardware, different bottleneck. Scaling advice depends on which resource is saturated.",
    },
  ],
  "smaller-numbers": [
    {
      q: "Why does quantising weights speed up decode even though the maths is the same?",
      options: ["Fewer bytes to read per token", "Integer arithmetic is always faster", "Small weights let it skip whole layers"],
      answer: 0,
      why: "Lossy compression of the dominant read. 4-bit weights are 8× fewer bytes than 32-bit, so the ceiling rises 8×.",
    },
    {
      q: "What's the cost?",
      options: ["Accuracy: every weight is rounded", "Nothing: the outputs are identical", "More arithmetic, so prefill gets slower"],
      answer: 0,
      why: "8-bit is nearly invisible; 4-bit is a real trade-off; 2-bit falls apart. Blocks with shared scales keep the error small.",
    },
  ],
  "the-gpu": [
    {
      q: "Why was the first GPU version no faster than one CPU core?",
      options: [
        "The CPU waited after every operation",
        "The GPU's memory is slower than the CPU's",
        "GPUs are poor at matrix-vector maths",
      ],
      answer: 0,
      why: "Chatty synchronous calls kill throughput, on a GPU as over a network. Queue the work, wait once per token, and shape each kernel for thousands of threads.",
    },
  ],
  batching: [
    {
      q: "Why is total throughput much higher with batching while each user gets a bit slower?",
      options: [
        "Shared weight reads, more maths per step",
        "Each user is moved to a smaller model",
        "The GPU raises its clock speed when busy",
      ],
      answer: 0,
      why: "It amortises the fixed cost (reading the weights), like group commit amortises an fsync. Past the ridge point it becomes compute-bound and per-user speed drops.",
    },
    {
      q: "What does continuous batching fix?",
      options: ["Waiting on the longest request in a batch", "The memory bandwidth limit on decode", "Accuracy lost when requests share a step"],
      answer: 0,
      why: "Re-forming the batch every step is a scheduler admitting work each tick instead of in rigid epochs. Median wait for the first token here fell from 3.5 s to 40 ms.",
    },
  ],
  "speculative-decoding": [
    {
      q: "Why can the big model check k guesses in about the time of one step?",
      options: [
        "Checking k tokens is one weight read",
        "The big model skips layers when checking",
        "It only checks the first guess of the k",
      ],
      answer: 0,
      why: "It turns the sequential chain into a small batch for one user. When most guesses are accepted, you get several tokens per weight read.",
    },
    {
      q: "Does speculative decoding change the output?",
      options: ["No: wrong guesses are thrown away", "Yes: it trades a little quality for speed"],
      answer: 0,
      why: "Validation makes it exact (for greedy decoding here; with sampling, a rejection rule keeps the distribution identical).",
    },
  ],
  "many-gpus": [
    {
      q: "Why would splitting GPT-2 small across GPUs make it slower?",
      options: ["The talking costs more than the work saved", "GPUs can't hold slices of one model", "Each GPU would need the full KV cache twice"],
      answer: 0,
      why: "Sharding only pays when per-shard work is large next to the communication. Small models stay on one device; big ones split because they must.",
    },
  ],
};

// End-of-chapter questions. Each tests an intuition the chapter built ("why is decode memory-bound?"), not a number to
// memorise. Wrong options are misconceptions a systems engineer could plausibly hold, each with a mechanism, and every
// option in a question is about the same length. Write the right answer first; Check.tsx shuffles the order.

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
      options: ["Inference, because it runs on every request", "Training, because it takes weeks on thousands of GPUs", "Storage, because every replica keeps the weight files"],
      answer: 0,
      why: "Training is a large one-off cost. Inference is a smaller cost paid on every request, for as long as the model is served. At scale the running cost dominates, so most engineering effort goes into making serving cheaper.",
    },
    {
      q: "Two users chat with the same model at the same time. What do they share, and what is theirs alone?",
      options: [
        "They share the weights; each has a KV cache",
        "They share a KV cache; each has a weight copy",
        "They share nothing, so each needs a replica",
      ],
      answer: 0,
      why: "The weights are a read-only asset that every request reads, like a shared dataset. The KV cache is per-request state that grows with the conversation. It has to be stored, scheduled and evicted, like session state in any stateful service.",
    },
    {
      q: "You send the same prompt twice, at temperature 0, to a busy production server. Will the answers always match?",
      options: [
        "No: batch size changes the order of float adds",
        "Yes: with no sampling, every pass is deterministic",
        "No: each server replica holds slightly different weights",
      ],
      answer: 0,
      why: "On one machine with fixed shapes, a forward pass is a pure function of the weights and tokens. A server batches your request with other users and picks kernels by batch size. Float addition isn't associative (chapter 13), so a score can change in its last digit and flip a near-tie. Identical output needs batch-invariant kernels.",
    },
  ],
  "why-its-expensive": [
    {
      q: "Why is generating one token for one user memory-bound on almost any chip?",
      options: [
        "Each weight read feeds only one multiply-add",
        "Each weight needs a long chain of arithmetic",
        "Each token's vector overflows the CPU caches",
      ],
      answer: 0,
      why: "Every token reads every weight once and does one multiply-add with it. At 16-bit that's 2 bytes per multiply-add. An H100 can do about 295 FLOPs in the time it reads 1 byte, so its arithmetic lanes wait on memory.",
    },
    {
      q: "A 70B model in 16-bit on one H100: what's the first problem?",
      options: ["The weights don't fit in its memory", "Decode is far below 1 token per second", "The arithmetic saturates the GPU's lanes"],
      answer: 0,
      why: "Capacity comes before bandwidth. 70 billion weights × 2 bytes = 140 GB, and an H100 has 80 GB. The model must be quantised (4-bit is about 35 GB) or split across GPUs, and a split puts the interconnect into every token.",
    },
    {
      q: "Why does batching 64 users barely slow each one down, at first?",
      options: [
        "One read of the weights serves all 64 users",
        "Each user's work runs on a separate GPU lane",
        "The GPU raises its memory clock when it's busy",
      ],
      answer: 0,
      why: "The step time is set by reading the weights. Until the extra arithmetic takes longer than that read (the ridge point), extra users cost almost nothing. The cost per token falls nearly 64×.",
    },
  ],
  "the-hardware": [
    {
      q: "For decode, what gives a datacentre GPU its biggest advantage over a CPU?",
      options: [
        "Memory bandwidth from HBM beside the die",
        "Thousands of lanes doing arithmetic at once",
        "A much higher clock speed on every lane",
      ],
      answer: 0,
      why: "Decode waits on bytes, so the memory system sets its speed. GPUs pair their lanes with HBM stacked beside the die. The lanes matter most for prefill and batching, where the work per byte is high.",
    },
    {
      q: "You split a model across 8 GPUs. Inside each GPU memory moves at about 3.35 TB/s; NVLink between them moves about 450 GB/s each way. What does that ratio tell you?",
      options: [
        "Send small results across the link, never weights",
        "Eight GPUs will decode about eight times faster",
        "The link matters only while the model is loading",
      ],
      answer: 0,
      why: "A byte over NVLink is about 7× slower than a local one, and about 67× slower over the network. Good splits send small activations, keep the weights local and overlap communication with work. It's the same instinct as keeping cross-shard traffic low.",
    },
  ],
  "rosetta-stone": [
    {
      q: "Paged attention borrows its key idea from…",
      options: ["Virtual memory paging", "TCP congestion control", "Consistent hashing of keys"],
      answer: 0,
      why: "Reserving one contiguous KV region per request, sized for its longest possible answer, wastes most of it. Pages allocated on demand limit the waste to the last part-full page, as an OS does for processes.",
    },
    {
      q: "Speculative decoding is like optimistic concurrency because…",
      options: [
        "Cheap guesses are checked in bulk and rolled back",
        "Two models share the KV cache under a lock",
        "A failed step is retried on another GPU",
      ],
      answer: 0,
      why: "The drafter's guesses are like an optimistic transaction. The big model checks them all in one pass and rolls back from the first mismatch. Like optimistic transactions, it pays when conflicts (wrong guesses) are rare.",
    },
    {
      q: "In a mixture-of-experts model, what's the analogue of a hot shard?",
      options: ["An expert the router picks far more often", "A GPU that throttles when it gets too hot", "A user whose prompt is very long"],
      answer: 0,
      why: "Routing tokens to experts is key-based routing to shards. If routing is skewed, one expert's GPU does most of the work while the others idle. Training adds load-balancing losses and serving adds capacity limits, much like rebalancing hot keys.",
    },
  ],

  // ---- part 1 ----
  "next-token": [
    {
      q: "What does one forward pass of the model produce?",
      options: ["A probability for every possible next token", "The whole answer, every position at once", "One next token, picked inside the network"],
      answer: 0,
      why: "A language model is a function from the text so far to a probability for each possible next token. Longer text comes from calling it in a loop and picking one token each time.",
    },
    {
      q: "Why can't one answer's tokens be generated in parallel?",
      options: ["Each token is an input to the next pass", "A GPU runs one forward pass at a time", "One answer's tokens are too big to batch"],
      answer: 0,
      why: "Generation is a dependency chain: pass n+1 needs the token from pass n. That one fact shapes most of inference engineering.",
    },
  ],
  tokens: [
    {
      q: "Why do models read tokens rather than characters or whole words?",
      options: [
        "A fixed vocabulary that can still spell anything",
        "Characters are too slow to look up in a table",
        "Whole words can't be stored as integer IDs",
      ],
      answer: 0,
      why: "Characters make sequences long, so each answer needs more forward passes. Whole words make the vocabulary unbounded. Subword tokens sit between the two, at about 4 characters per token in English.",
    },
    {
      q: "Speeds are quoted in tokens per second. Why is that the natural unit?",
      options: ["Each decode step writes one token", "A token is always exactly one word", "The GPU's hardware counts tokens"],
      answer: 0,
      why: "One forward pass writes one new token per user. Prices, latency targets and benchmarks count tokens because the token is the unit of work.",
    },
  ],
  embeddings: [
    {
      q: "What is an embedding, physically?",
      options: ["A row of a lookup table, learned in training", "A hash of the token's spelling into 768 numbers", "A vector computed from the token's letters"],
      answer: 0,
      why: "The first operation is a table lookup. Training arranges the rows so that tokens used in similar ways get similar vectors.",
    },
    {
      q: "Does the embedding lookup cost much at inference time?",
      options: ["No: it reads one ~3 KB row per token", "Yes: it reads the whole ~150 MB table"],
      answer: 0,
      why: "A lookup gathers one row; it doesn't multiply by the table. The expensive work is the matrix multiplies in every layer after it.",
    },
  ],
  "multiply-add": [
    {
      q: "Roughly how many FLOPs does a forward pass take per token, for a model with N weights?",
      options: ["About 2N: one multiply-add per weight", "About N²: every weight meets every other", "About N log N, like a sort over weights"],
      answer: 0,
      why: "Almost all the work is matrix times vector, and each weight takes part in one multiply-add per token (2 FLOPs). So bytes per token and FLOPs per token both grow in proportion to model size.",
    },
    {
      q: "Why does matrix-vector work dominate so completely?",
      options: ["Nearly every weight lives in a matrix", "Attention's softmax is the costly part", "Most of the model is table lookups"],
      answer: 0,
      why: "The attention projections and the feed-forward network are all weight matrices. The other operations (norms, softmax, adds) touch far fewer numbers.",
    },
  ],
  attention: [
    {
      q: "What does attention let a token do that the rest of the layer can't?",
      options: ["Read information from earlier tokens", "Update the weights for this conversation", "Choose which layers to skip for itself"],
      answer: 0,
      why: "Everything else in a layer processes each token on its own. Attention is the only step where tokens exchange information: each one scores the earlier tokens and blends their values.",
    },
    {
      q: "Why does attention need per-token state that grows with the conversation?",
      options: [
        "Each new token reads every earlier key and value",
        "The weights gain a new row for every new token",
        "Softmax keeps a running total for every token",
      ],
      answer: 0,
      why: "Each new token compares itself with the keys of all earlier tokens and reads their values. That growing set of keys and values is what the KV cache stores (chapter 12), the per-session state of an inference server.",
    },
  ],
  layers: [
    {
      q: "What does each of the 12 layers do to a token's vector?",
      options: ["Adds an update to it", "Replaces it with a new one", "Shrinks it to fewer numbers"],
      answer: 0,
      why: "Each layer reads the vector, computes an update with attention and a feed-forward network, and adds the update back. The residual stream keeps its 768 numbers all the way through, and the guess sharpens layer by layer.",
    },
    {
      q: "Why does depth matter for inference cost?",
      options: ["Every layer's weights are read every token", "Only the last layer runs at inference time", "The layers run in parallel, one per core"],
      answer: 0,
      why: "Within one token the layers run in order: layer 2 needs layer 1's output, and each reads its own weights. Pipeline parallelism splits a model along this same sequence.",
    },
  ],
  sampling: [
    {
      q: "What does temperature change?",
      options: ["How peaked the probabilities are", "The weights, slightly, at each step", "How many layers each token uses"],
      answer: 0,
      why: "Temperature divides the scores before softmax. A low temperature sharpens the distribution toward the top choice, and a high one flattens it. It's a cheap step after the forward pass.",
    },
    {
      q: "Does sampling affect speed?",
      options: ["Barely: it reads a few hundred KB", "Yes: sorting 50,257 scores is slow"],
      answer: 0,
      why: "A forward pass reads hundreds of MB of weights; sampling reads a few hundred KB of scores. It changes what the model writes, not how long a step takes.",
    },
  ],
  "the-loop": [
    {
      q: "Why is prefill much faster per token than decode?",
      options: [
        "One read of each weight serves the whole prompt",
        "Prefill runs on a smaller copy of the model",
        "Prefill skips attention for the prompt tokens",
      ],
      answer: 0,
      why: "Prefill and decode do the same arithmetic per token. Prefill reads the weights once for the whole prompt, so it is compute-bound. Decode reads them once per token, so it is memory-bound.",
    },
    {
      q: "Which user-facing metric does prefill dominate?",
      options: ["Time to first token", "Time between tokens", "Total tokens per second"],
      answer: 0,
      why: "The first token can't appear until the whole prompt is read. After that, decode sets the pace of the stream.",
    },
  ],

  // ---- part 2 ----
  "memory-or-maths": [
    {
      q: "Decode does about 0.5 FLOPs per byte read here; the chip can do far more per byte. So the chip is…",
      options: ["Waiting on memory", "Waiting on arithmetic", "Waiting on the scheduler"],
      answer: 0,
      why: "Compare the job's FLOPs per byte with the machine's ratio (the ridge point). Below it, memory sets the pace. Above it, arithmetic does.",
    },
    {
      q: "If a job is memory-bound, which fix helps?",
      options: ["Fewer bytes, or more work per byte", "A chip with faster multiply units", "An algorithm with fewer operations"],
      answer: 0,
      why: "Faster arithmetic doesn't help a job that waits on the truck. That's why quantisation (fewer bytes) and batching (more work per byte) are the main levers for decode.",
    },
  ],
  "speed-limit": [
    {
      q: "How do you estimate the best possible decode speed of any model on any machine?",
      options: ["Memory bandwidth ÷ bytes per token", "Clock speed × cores × SIMD width", "Peak FLOP/s ÷ the number of weights"],
      answer: 0,
      why: "120 GB/s ÷ 495 MB ≈ 243 tok/s for GPT-2 small in 32-bit on the M4. The same division gives about 48 tok/s for a 70B 16-bit model on 2 H100s. It's a latency estimate you can make before writing code.",
    },
    {
      q: "Halving the bytes per weight does what to that limit?",
      options: ["Doubles it", "Halves it", "Leaves it the same"],
      answer: 0,
      why: "The speed limit is bandwidth divided by bytes per token, so halving the bytes doubles it. That's the case for quantisation.",
    },
  ],
  "starting-line": [
    {
      q: "Why was the first correct engine so slow?",
      options: ["It reran every earlier token for each new one", "It used plain Rust with no SIMD instructions", "It read the weights from disk for every token"],
      answer: 0,
      why: "It had no memoisation, so each token cost O(n) work and a whole answer O(n²). The fix is to cache the state that doesn't change.",
    },
    {
      q: "Why keep that slow version around?",
      options: ["Every faster stage is tested against it", "It's faster than the others on long prompts", "It uses the least memory, for small devices"],
      answer: 0,
      why: "It works like a reference implementation in a test harness. An optimisation that quietly breaks the maths produces different tokens, and the comparison catches it.",
    },
  ],

  // ---- part 3 ----
  "kv-cache": [
    {
      q: "Why can keys and values be cached but not the whole computation?",
      options: [
        "Past tokens' keys and values never change",
        "The KV cache sits in faster memory than weights",
        "Each layer's output depends only on the new token",
      ],
      answer: 0,
      why: "In causal attention a token never looks at later tokens, so its keys and values are fixed once computed. Caching them is memoisation of an append-only log.",
    },
    {
      q: "What does the KV cache cost?",
      options: ["Memory that grows with length × users", "Nothing, because it only removes work", "Extra training so the model can use it"],
      answer: 0,
      why: "It's per-session state. At long contexts it can outgrow the weights, so serving capacity is usually limited by KV cache memory.",
    },
  ],
  "faster-arithmetic": [
    {
      q: "Why did 16 running totals beat 1?",
      options: [
        "16 independent chains keep the adder busy",
        "16 totals need fewer bytes from memory",
        "The 16 totals run on 16 different cores",
      ],
      answer: 0,
      why: "It's latency vs throughput at the instruction level. One total is a chain of adds that each wait for the last. Independent totals let the hardware overlap the adds, and the compiler can then use SIMD.",
    },
  ],
  "many-cores": [
    {
      q: "10 cores gave decode only about 1.5×. Why?",
      options: ["They all share one memory system", "Rust threads queue on one global lock", "Only 4 of the cores can do floating point"],
      answer: 0,
      why: "It's like adding app servers in front of a saturated database. Decode waits on memory bandwidth, which the cores share, so throughput stops at that shared limit.",
    },
    {
      q: "Why did prefill scale much better with cores?",
      options: ["Prefill is compute-bound", "Prefill reads fewer bytes", "Prefill threads never sleep"],
      answer: 0,
      why: "Prefill does enough arithmetic per byte that the cores, not memory, set its speed. Same hardware, different bottleneck: scaling advice depends on which resource is saturated.",
    },
  ],
  "smaller-numbers": [
    {
      q: "Why does quantising weights speed up decode even though the arithmetic is the same?",
      options: ["Fewer bytes to read per token", "Integer adds are faster than float", "Small weights let it skip layers"],
      answer: 0,
      why: "It's lossy compression of the read that dominates. 4-bit weights are 8× fewer bytes than 32-bit, so the speed limit rises about 8×.",
    },
    {
      q: "What's the cost?",
      options: ["Accuracy, because each weight is rounded", "Nothing, because outputs stay identical", "Prefill, because it needs more arithmetic"],
      answer: 0,
      why: "8-bit is nearly invisible, 4-bit is a real trade-off and 2-bit falls apart. Blocks of 32 weights with a shared scale keep the rounding error small.",
    },
  ],
  "the-gpu": [
    {
      q: "Why was the first GPU stage no faster than one CPU core?",
      options: [
        "The CPU waited after every GPU operation",
        "The GPU's memory is slower than the CPU's",
        "GPUs are weak at matrix-vector arithmetic",
      ],
      answer: 0,
      why: "Chatty synchronous calls kill throughput, on a GPU as over a network. The fix is to queue a token's work, wait once per token, and shape each kernel for thousands of GPU threads.",
    },
  ],
  batching: [
    {
      q: "Why is total throughput much higher with batching while each user gets a little slower?",
      options: [
        "Users share weight reads; steps do more arithmetic",
        "Each user is moved onto a smaller model copy",
        "The GPU raises its clock speed when it's busy",
      ],
      answer: 0,
      why: "Batching spreads the fixed cost (reading the weights) over many users, as group commit spreads an fsync. Past the ridge point the step becomes compute-bound, and each user's speed drops.",
    },
    {
      q: "What does continuous batching fix?",
      options: ["Waiting on the longest request in a batch", "The memory bandwidth limit on decode", "Accuracy lost when users share a step"],
      answer: 0,
      why: "The scheduler re-forms the batch every step, admitting new requests as others finish, instead of in fixed rounds. Here the median wait for the first token fell from 3.5 s to 40 ms.",
    },
  ],
  "speculative-decoding": [
    {
      q: "Why can the big model check k guesses in about the time of one step?",
      options: [
        "All k tokens share one read of the weights",
        "The big model skips layers while checking",
        "It checks only the first of the k guesses",
      ],
      answer: 0,
      why: "Checking k guesses is a small batch for one user, so the weights are read once for all k. When most guesses are kept, each read of the weights writes several tokens.",
    },
    {
      q: "Does speculative decoding change the output?",
      options: ["No: wrong guesses are thrown away", "Yes: it trades some quality for speed"],
      answer: 0,
      why: "The big model keeps only guesses that match its own choice, so the text is exactly what it writes alone. That holds for greedy decoding here; with sampling, a rejection rule keeps the distribution identical.",
    },
  ],
  "many-gpus": [
    {
      q: "Why would splitting GPT-2 small across GPUs make it slower?",
      options: ["Communication costs more than the work saved", "A GPU can't hold only a slice of a model", "Each GPU needs its own full KV cache copy"],
      answer: 0,
      why: "Splitting pays only when each GPU's share of the work is large next to the communication. Small models stay on one GPU; big ones split because they don't fit.",
    },
  ],
};

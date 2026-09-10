# transformer-build-notes

Notes based on my personal experience working through Stanford CS336's Assignment 1 (Basics). Reflects my own learning process, not official course content or solutions.

## 1. Tokenizer

#### 1.1 Why need Tokenizer

- Serves as the tool that converts natural language into a machine-readable representation the model can process.
- Converts the machine-generated representation back into natural language to present the result to the user.
- Must satisfy a **hard structural constraint of the model** — a fixed vocabulary size, since it directly determines the dimensions of the transformer's embedding matrix and the softmax output layer. At the same time, for **computational efficiency**, the tokenizer must go further than simply having a finite vocabulary: it needs to compress the same content into as few tokens as possible, which is what distinguishes it from plain character-level tokenization.

#### 1.2 Steps to build a Tokenizer

##### 1.2.1 Questions before starting

- **Input-format problems** — practical/engineering constraints unrelated to meaning, e.g. text being too long to feed into the model as a single input at once.
- **Semantic-boundary problems** — how to handle characters that carry no semantic content but serve a structural role in text (e.g. punctuation, line breaks); and more generally, where to draw the boundary of a "unit" when raw text is just a continuous stream of characters — if the boundary is drawn inconsistently, the same word could end up encoded differently depending on context.
- **Vocabulary-design problems** — how to map an unbounded space of possible text onto a finite, fixed set of tokens; and specifically, how to guarantee that words or characters never seen during vocabulary construction can still be encoded losslessly at inference time, without errors or missing information.
- **Efficiency problems** — distinct from the "sequence-length efficiency" discussed in the why section (which is about the quality of the output), this is about the runtime efficiency of the encoding process itself: how to design the vocabulary-construction/merging algorithm so it still finishes in reasonable time on large-scale corpora.

##### 1.2.2 Why BPE

Byte pair encoding (BPE) is a data compression and tokenization algorithm that iteratively merges the **most frequent pairs** of bytes or subwords in a text sequence

- Working process:

  <img src="png/bpe_algorithm_steps.png" alt="bpe_algorithm_steps" style="zoom:30%;" />

- Reasons:

  - Because the algorithm repeatedly merges symbol pairs based on their frequency, this mechanism naturally produces a behavior where common words are kept intact as whole tokens while rare words get broken down into finer subword units. This behavior happens to sidestep the problems of both character-level and word-level tokenization at once.
  - BPE's vocabulary isn't defined by hand-crafted rules — it's learned entirely from the statistical frequencies observed in the training corpus. This means it automatically adapts to the statistical distribution of whatever corpus it's trained on, whether that's a different language or a different domain.
  - The byte-level fallback guarantee means that no matter how far subword merging progresses, the worst case always degrades gracefully to a byte-by-byte representation — this is precisely the mechanism that makes the out-of-vocabulary avoidance actually work.

##### 1.2.3 Preparations

- **Preparation 1. Streaming read & Buffering**
  - To address the "input text is too long to feed in all at once" problem, `encode_iterable` reads the input as a stream instead of loading the entire file into memory and encoding it in one shot. It maintains a fixed-size buffer (roughly 4096 characters): it keeps appending incoming pieces of text to the buffer, and once the buffer is full, it encodes that chunk and immediately yields the resulting ids before moving on to accumulate the next chunk. This way, no matter how large the input file is, only **a small buffer needs to stay in memory** at any given time — never the whole file.
  - The cutoff point for the buffer can't be arbitrary, though: if the buffer happens to be cut off in the middle of a word or in the middle of a special token, that word or token would incorrectly get split across two separate encoding calls, corrupting the result. So once the buffer reaches its target length, the code checks whether the last character is a letter, a digit, or a prefix fragment of a special token — if so, it keeps extending the buffer further until **it reaches a safe boundary to cut at.**
  - To achieve this "read a chunk, encode it, yield it" behavior, `encode_iterable` uses `yield from` to emit the token ids produced from each buffer chunk one at a time, rather than accumulating everything into one large list and returning it all at once — this lets the **caller start consuming results while the rest of the file is still being processed.**
  - At the level above, `create_token_id` also wraps its call to `encode_iterable` in a `yield from`. The purpose here is different: it isn't about producing ids one at a time (that's already handled inside `encode_iterable`) — it's about making `create_token_id` itself a lazily-evaluated generator, so that the setup work inside it (such as constructing the `Tokenizer`) only **actually runs once something downstream starts pulling values** (for example, the first internal `next()` call inside `np.fromiter`), rather than running immediately the moment `create_token_id(...)` is called.

<img src="png/streaming_buffer_and_lazy_generator.png" alt="streaming_buffer_and_lazy_generator" style="zoom:30%;" />

- **Preparation 2: Handling semantic boundaries (special tokens and pre-tokenization)**

  To address the problem of "which characters carry a structural function and need to be treated differently from ordinary semantic content," this step operates on two separate layers, each drawing boundaries at a different granularity.

  -  The first layer is **splitting on special tokens**. Text may contain structural markers like `<|endoftext|>`, which carry no semantic meaning at all — they simply mark document or section boundaries — and must therefore be identified as whole, indivisible units before anything else happens, rather than being swept into the byte-merging process. There's a subtlety here that's easy to overlook: when multiple special tokens are present, matching must **prioritize the longest one first**. The reason is that if one special token happens to be a prefix of another, longer special token, matching the shorter one first would incorrectly split the longer token into "the short token plus a leftover fragment," with that leftover then treated as ordinary text — completely destroying the integrity of the longer special token. To prevent this, all special tokens are sorted by length, longest first, before building the matching pattern, so that the more specific, longer match always takes precedence.
  -  The second layer is pre-tokenization. Before handing a stretch of ordinary text (whatever remains after special-token splitting) off to **byte-level merging**, **a set of regex rules first roughly segments it into candidate units** — broadly corresponding to words, runs of digits, runs of punctuation, and runs of whitespace. The purpose of this layer is to define the scope within which byte merging is allowed to happen: merges can only occur inside a single pre-token, never across a pre-token boundary. This prevents a word from accidentally getting merged together with the punctuation that follows it into one inseparable unit, which in turn guarantees that the same word is encoded consistently regardless of what surrounds it in a given context.

- **Preparation 3: Vocabulary construction and efficient lookup (BPE training and index design)**

  This step addresses two categories of problem at once: the vocabulary-design problem (how to map an unbounded space of text onto a finite, fixed-size vocabulary) and the efficiency problem (how to make that construction process finish in reasonable time on a large-scale corpus).

  -  The overall idea behind building the vocabulary is "start from the smallest units, then repeatedly merge by statistical frequency." First, every word in the corpus is not handled as a string at all — it's **converted into a tuple of individual bytes**. The reason for starting from single bytes is that the 256 byte values form a naturally finite set that can nonetheless cover any Unicode text, which is what fundamentally guarantees that any input can be represented no matter what.
  -  On top of that, the construction process maintains three interrelated data structures: `byte_words` records each byte tuple and how often it occurs in the corpus; `pair_frequent_table` records the total weighted frequency of every adjacent byte pair; and `reverse_index_table` records, for each byte pair, which byte tuples contain it. The first two are statistics; the third is a reverse index — extra space deliberately spent in exchange for speed.
  -  Each merge round proceeds as follows: take the highest-frequency byte pair from `pair_frequent_table`, then use `reverse_index_table` to jump straight to the words containing that pair, and for each such word perform the merge while updating all three structures in step. In `byte_words`, the old tuple key is deleted and the merged tuple is inserted as a new key. In `pair_frequent_table`, the pairs that previously sat on either side of the merge point — along with the merged pair itself — have this word's frequency subtracted, while the pairs newly formed around the merged token have that frequency added. In `reverse_index_table`, the old tuple is removed from the sets of all its old byte pairs, and the new tuple is added to the sets of all its new byte pairs. Every byte pair that was touched is recorded, so that once the round finishes, entries whose count dropped to zero and entries whose word set became empty can be cleaned up in one pass. This whole process repeats until the vocabulary reaches its target size.

  <img src="png/bpe_merge_full_pipeline.png" alt="bpe_merge_full_pipeline" style="zoom:30%;" />

##### 1.2.4 Construction

This section is about how the building blocks prepared earlier get organized into three sequential steps that ultimately turn a piece of raw text into a token id sequence ready to feed into the transformer. The three steps are strictly chained — each one's output is exactly the next one's input.

- **Step 1: BPE Training** — takes `input_path`, `vocab_size`, and `special_tokens` as input, and internally runs the full loop described in Preparation 3 ("find the highest-frequency pair → locate the affected words via the reverse index → merge and update the three tables") repeatedly until the vocabulary reaches its target size. This step produces two files: `vocab.json` and the `merges` file — the only two things needed to construct the tokenizer in the next step.

  **Note**: the number of merge operations to run equals `vocab_size` minus the size of the initial vocabulary (`num_init_words`), where the initial vocabulary is the 256 possible byte values plus the number of special tokens

- **Step 2: Building the Tokenizer** — `Tokenizer.from_files` reads in the two files produced by step 1 directly, and constructs the internal state: `vocab`, `vocab_reverse`, and `merges_dict` (this last one is exactly the "pay a one-time preprocessing cost for constant-time lookup later" optimization mentioned at the end of Preparation 3). Once constructed, this `Tokenizer` object exposes three methods with a clear calling relationship between them: `encode` is the core method, internally following the chain from Preparation 2 — "split on special tokens → pre-tokenization → byte-level merging"; `encode_iterable` wraps `encode` with the streaming-buffer logic from Preparation 1, calling `encode` once every time a safe-to-cut buffer chunk has accumulated; and `decode` is simply the inverse of `encode`, turning a token id sequence back into text. The output of this step is a fully functional `Tokenizer` object capable of both encoding and decoding.

- **Step 3: Generating token_ids** — `create_token_id` wraps the `Tokenizer` object from step 2, calling its `encode_iterable`, and itself uses `yield from` to stay fully lazy (matching the design from Preparation 1: nothing executes at call time, everything is deferred until values are actually pulled). The output of this step is a token id generator that can be fed directly into `np.fromiter`, ultimately written out as a `uint16` binary file — this is the endpoint of the entire tokenizer pipeline, and it's exactly the input data that gets read when training the transformer.

## 2. Transformer

## 3. Optimizer


# 1. Large Language Models (LLMs) — Fundamentals

## Core Concepts

### What is a Language Model?
A probabilistic model that predicts the next token given a sequence of previous tokens:

```
P(token_n | token_1, token_2, ..., token_n-1)
```

LLMs are "large" because they have billions of parameters trained on massive text corpora.

---

### Transformer Architecture (Vaswani et al., 2017)

The backbone of all modern LLMs. Key insight: **self-attention** allows every token to attend to every other token in parallel (unlike RNNs which process sequentially).

```
Input → Tokenization → Embedding + Positional Encoding → N × [Multi-Head Attention → Feed-Forward] → Output
```

#### Self-Attention Mechanism

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- **Q** (Query) = what am I looking for?
- **K** (Key) = what do I contain?
- **V** (Value) = what information do I provide?
- **√d_k** = scaling factor (prevents softmax saturation for large dimensions)

#### Multi-Head Attention
Instead of one attention function, run `h` parallel attention heads with different learned projections:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

This allows the model to attend to different types of relationships simultaneously (syntax, semantics, position, etc.).

#### Positional Encoding
Transformers have no inherent notion of order. Position is injected via:
- **Sinusoidal** (original): `PE(pos, 2i) = sin(pos / 10000^(2i/d))` 
- **Learned** (GPT): trainable embedding per position
- **RoPE** (LLaMA): rotary positional encoding — encodes relative position via rotation matrices

---

### GPT Architecture (Decoder-Only)

| Component | Purpose |
|---|---|
| Token Embedding | Maps token IDs → dense vectors |
| Positional Embedding | Injects sequence position |
| Transformer Blocks (×N) | Masked self-attention + FFN |
| Layer Normalization | Stabilizes training |
| Linear + Softmax Head | Projects to vocabulary logits |

**Causal masking**: Each token can only attend to tokens before it (not future tokens). This is what makes it "autoregressive" — generates one token at a time, left to right.

---

### Key LLM Families

| Model | Organization | Parameters | Architecture | Key Innovation |
|---|---|---|---|---|
| GPT-4 | OpenAI | ~1.8T (rumored MoE) | Decoder-only | RLHF, multi-modal |
| GPT-3.5 | OpenAI | 175B | Decoder-only | InstructGPT tuning |
| LLaMA 2/3 | Meta | 7B–70B | Decoder-only | Open weights, RoPE, GQA |
| Claude | Anthropic | Unknown | Decoder-only | Constitutional AI (RLAIF) |
| PaLM/Gemini | Google | 540B+ | Decoder-only | Pathways, multi-modal |
| BERT | Google | 110M–340M | Encoder-only | Bidirectional, MLM |
| T5 | Google | 11B | Encoder-Decoder | Text-to-text framework |

---

### Tokenization

Converts raw text → integer token IDs that the model can process.

| Method | How It Works | Example |
|---|---|---|
| **BPE** (Byte-Pair Encoding) | Iteratively merges most frequent character pairs | "lowest" → ["low", "est"] |
| **WordPiece** | Similar to BPE, uses likelihood-based merging | "unaffable" → ["un", "##aff", "##able"] |
| **SentencePiece** | Language-agnostic, treats input as raw bytes | Works on any language |

**Key facts:**
- GPT-4 uses `cl100k_base` tokenizer (~100K vocab)
- 1 token ≈ 4 characters in English (or ~¾ of a word)
- Tokenization affects cost — you pay per token for API calls

---

### Inference Parameters

| Parameter | Effect | Typical Value |
|---|---|---|
| **Temperature** | Controls randomness. 0 = deterministic, 1 = creative, >1 = chaotic | 0.0–0.7 for factual tasks |
| **Top-p (nucleus)** | Sample from smallest set of tokens whose cumulative probability ≥ p | 0.9–0.95 |
| **Top-k** | Sample from top k highest probability tokens only | 40–50 |
| **Max tokens** | Maximum output length | Varies by task |
| **Frequency penalty** | Reduce probability of repeating tokens | 0.0–1.0 |
| **Presence penalty** | Reduce probability of tokens already used | 0.0–1.0 |

**Temperature formula:**
```
P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)
```
As T → 0, distribution becomes peaked (greedy). As T → ∞, distribution becomes uniform (random).

---

### Context Window

The maximum number of tokens a model can process in a single forward pass.

| Model | Context Window |
|---|---|
| GPT-3.5 | 4K / 16K |
| GPT-4 | 8K / 32K / 128K |
| Claude 3 | 200K |
| LLaMA 3 | 8K (extendable) |
| Gemini 1.5 | 1M+ |

**Why it matters for RAG:** A larger context window means you can stuff more retrieved documents into the prompt, but:
- Attention is O(n²) — longer contexts are slower and more expensive
- "Lost in the middle" problem — models struggle with info in the middle of long contexts

---

### Scaling Laws (Kaplan et al., 2020)

Performance improves predictably with three factors:
1. **Model size** (parameters)
2. **Dataset size** (tokens)
3. **Compute** (FLOPs)

```
Loss ∝ (1/N)^α + (1/D)^β + (1/C)^γ
```

**Chinchilla Scaling** (Hoffmann et al., 2022): For compute-optimal training, scale data and params equally. A 70B model trained on 1.4T tokens beats a 175B model trained on 300B tokens.

---

### Emergent Abilities

Capabilities that appear suddenly at certain model scales:
- **Chain-of-thought reasoning** (~100B params)
- **In-context learning** (~10B params)
- **Instruction following** (~1B params with fine-tuning)
- **Multi-step math** (~100B params)

---

## Quick-Recall Points

- Transformer = self-attention + feed-forward + layer norm + residual connections
- Attention complexity: O(n² × d) where n = sequence length, d = dimension
- GPT = decoder-only (causal mask), BERT = encoder-only (bidirectional)
- Temperature 0 = always pick highest probability token (greedy decoding)
- 1 token ≈ 4 chars ≈ 0.75 words in English
- Scaling laws: 10× more compute → predictable improvement in loss
- Context window ≠ memory; LLMs are stateless between API calls

---

## ⭐ Interview / Exam Q&A

**Q: What is the difference between GPT and BERT?**
> GPT is decoder-only with causal (left-to-right) attention — good for generation. BERT is encoder-only with bidirectional attention — good for understanding/classification. GPT predicts next token; BERT predicts masked tokens.

**Q: Why do we scale by √d_k in attention?**
> Without scaling, dot products grow large for high dimensions, pushing softmax into saturation (very peaked outputs). Dividing by √d_k keeps variance at ~1 regardless of dimension, allowing gradients to flow properly.

**Q: What is the "lost in the middle" problem?**
> LLMs attend most strongly to the beginning and end of the context window. Information placed in the middle of a long prompt is often ignored or given less weight. This impacts RAG — retrieved documents should be placed at the start of context.

**Q: Explain temperature in simple terms.**
> Temperature controls how "confident" the model is. At T=0, it always picks the single most likely word. At T=1, it samples proportional to learned probabilities. At T>1, rare words become more likely — outputs get creative but potentially nonsensical.

**Q: What makes LLMs "auto-regressive"?**
> They generate one token at a time, feeding each generated token back as input for the next step. The causal mask ensures position n can only see positions 1 to n-1. Generation stops at an EOS token or max length.

**Q: How does multi-head attention help?**
> Each head can learn to attend to different types of relationships — one head might learn syntax (subject-verb), another might learn coreference (he → John), another might learn position-based patterns. The model combines all these "perspectives" via concatenation and linear projection.

**Q: What is KV-cache and why does it matter for inference?**
> During autoregressive generation, the Key and Value matrices for all previous tokens don't change. KV-cache stores them so they don't need to be recomputed at each step. This reduces generation from O(n²) to O(n) per token but requires significant GPU memory.

**Q: What is Mixture of Experts (MoE)?**
> Instead of a single large FFN in each transformer block, MoE uses multiple smaller "expert" FFNs and a router/gate that selects 1-2 experts per token. This allows scaling model capacity without proportionally increasing compute — GPT-4 is rumored to use 8 experts with ~220B params each, but only 2 are active per token.

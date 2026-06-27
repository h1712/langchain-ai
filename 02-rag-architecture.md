# 2. Retrieval-Augmented Generation (RAG)

## Core Concept

RAG = **Retrieve** relevant context from an external knowledge base + **Generate** a response conditioned on that context using an LLM.

```
User Query → Embed → Search Vector DB → Retrieve Top-K docs → Inject into LLM Prompt → Generate Answer
```

**Why RAG?**
- LLMs have knowledge cutoff dates (can't know recent info)
- LLMs hallucinate when they don't know the answer
- RAG grounds responses in actual documents
- No expensive fine-tuning needed — just update the knowledge base

**Foundational paper:** Lewis et al. (2020) — *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*, NeurIPS

---

## RAG Pipeline — End to End

### Phase 1: Indexing (Offline)

```
Raw Documents → Chunking → Embedding → Store in Vector DB
```

| Step | What Happens |
|---|---|
| Load | Ingest documents (PDFs, wikis, Slack messages, code, etc.) |
| Chunk | Split into manageable pieces (paragraphs, sentences, fixed-size) |
| Embed | Convert each chunk to a dense vector using embedding model |
| Store | Insert vector + metadata + original text into vector DB |

### Phase 2: Retrieval (Online)

```
User Query → Embed → ANN Search → Top-K Results → Rerank → Context
```

| Step | What Happens |
|---|---|
| Embed query | Same embedding model as indexing |
| ANN search | Approximate nearest neighbor in vector DB |
| Filter | Apply metadata filters (date, source, channel) |
| Rerank | Score results by relevance, freshness, feedback |
| Select | Pick top-K results within token budget |

### Phase 3: Generation (Online)

```
System Prompt + Context + User Query → LLM → Answer
```

The LLM sees:
```
System: You are a helpful assistant. Answer ONLY from the provided context.
Context: [retrieved documents]
Question: [user's query]
Answer:
```

---

## Chunking Strategies

| Strategy | How | Pros | Cons |
|---|---|---|---|
| **Fixed-size** | Split every N tokens/chars | Simple, predictable | Breaks mid-sentence |
| **Recursive** | Split by paragraph → sentence → word | Preserves structure | Variable chunk sizes |
| **Semantic** | Split at topic boundaries (using embeddings) | Best coherence | Expensive, complex |
| **Sentence** | One sentence per chunk | Fine-grained retrieval | Too small for context |
| **Paragraph** | One paragraph per chunk | Natural boundaries | Size varies wildly |
| **Document** | Whole document as one chunk | Full context | May exceed token limits |

**Best practices:**
- Chunk size: 256–512 tokens is a good default
- Overlap: 10–20% overlap between chunks prevents losing info at boundaries
- Metadata: Always store source, page, timestamp alongside the chunk

---

## Retrieval Strategies

### Dense Retrieval (Embedding-based)
- Encode query and docs into same vector space
- Find nearest neighbors by cosine similarity or L2 distance
- Works well for semantic/paraphrase matching
- Fails on keyword-specific or exact-match queries

### Sparse Retrieval (BM25 / TF-IDF)
- Traditional keyword matching with term frequency weighting
- Works well for exact terms, names, error codes
- Fails on paraphrases ("car" won't match "automobile")

### Hybrid Retrieval (Best of Both)
- Run both dense + sparse retrieval
- Combine scores (e.g., Reciprocal Rank Fusion)
- Most production systems use hybrid

```
Hybrid Score = α × dense_score + (1 - α) × sparse_score
```

### Multi-Vector Retrieval
- Store multiple embeddings per document (e.g., question vector + answer vector)
- Query matches against each independently
- Merge results taking the best score per document
- **Your TejaBot uses this** — `question_weight=0.6, answer_weight=0.4`

---

## Reranking

Initial retrieval casts a wide net. Reranking refines the ordering:

| Reranker | Method | Speed | Quality |
|---|---|---|---|
| **Cross-encoder** | Feed (query, doc) pair through BERT-like model | Slow | Best |
| **ColBERT** | Token-level interaction scores | Medium | Very Good |
| **Score fusion** | Combine multiple signals (similarity + feedback + recency) | Fast | Good |
| **LLM-as-judge** | Ask LLM "is this relevant?" | Slowest | Very Good |

**Your TejaBot's approach (score fusion):**
```
Score = 0.65 × similarity + 0.20 × (tanh(feedback/3) + 1)/2 + 0.15 × exp(-age/τ)
```

---

## Advanced RAG Patterns

### 1. Naive RAG (Basic)
```
Query → Retrieve → Generate
```
Simple but prone to: irrelevant retrieval, context overflow, no reasoning.

### 2. Advanced RAG
```
Query → Query Rewriting → Retrieve → Rerank → Generate
```
Adds query transformation and result refinement.

### 3. Modular RAG
```
Query → Router → [Search | SQL | API | Calculator] → Synthesize
```
Multiple retrieval sources, tool use, agentic routing.

### 4. Iterative RAG (Multi-hop)
```
Query → Retrieve → Partial Answer → New Query → Retrieve → Final Answer
```
For complex questions requiring information from multiple documents.

### 5. Self-RAG (Aslan et al., 2023)
The LLM decides:
- Whether to retrieve (or answer from memory)
- Which retrieved passages are relevant
- Whether the generated answer is supported by evidence

---

## RAG Failure Modes

| Failure | Cause | Fix |
|---|---|---|
| **Retrieval miss** | Relevant doc exists but wasn't retrieved | Lower similarity threshold, hybrid search |
| **Context poisoning** | Irrelevant doc retrieved and misleads LLM | Better reranking, raise threshold |
| **Token overflow** | Too many docs exceed context window | Summarize chunks, limit top-K |
| **Lost in the middle** | LLM ignores docs in the middle of context | Place most relevant at start/end |
| **Hallucination despite context** | LLM ignores context and generates from memory | Stronger system prompt, lower temperature |
| **Stale knowledge** | Outdated docs ranked high | Recency weighting (time decay) |
| **Chunk boundary issue** | Answer spans two chunks | Overlapping chunks, larger chunks |

---

## RAG Evaluation Metrics

### Retrieval Quality
| Metric | Measures |
|---|---|
| **Precision@K** | Fraction of top-K results that are relevant |
| **Recall@K** | Fraction of all relevant docs found in top-K |
| **MRR** (Mean Reciprocal Rank) | 1/position of first relevant result, averaged |
| **NDCG** | Normalized Discounted Cumulative Gain — rewards relevant docs at top |
| **Hit Rate** | Does the correct doc appear anywhere in top-K? |

### Generation Quality
| Metric | Measures |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? (no hallucination) |
| **Answer Relevancy** | Does the answer address the question? |
| **Context Relevancy** | Are the retrieved docs actually relevant? |
| **RAGAS** | Open-source framework combining the above metrics |

---

## Quick-Recall Points

- RAG = Retriever + Generator working together
- Chunking: 256–512 tokens with 10–20% overlap is standard
- Hybrid search (dense + sparse) beats either alone in production
- Reranking after retrieval significantly improves quality
- "Lost in the middle" — put important context at start of prompt
- Temperature 0 is standard for RAG (we want faithful, not creative)
- Always include "answer only from context" in system prompt
- Metadata filters (date, source) prevent irrelevant old docs from appearing

---

## ⭐ Interview / Exam Q&A

**Q: What problem does RAG solve that fine-tuning doesn't?**
> RAG solves knowledge freshness and grounding. Fine-tuning bakes knowledge into model weights — it can't be updated without retraining. RAG retrieves from an external store that can be updated in real-time. Also, fine-tuning can't easily cite sources; RAG naturally provides provenance.

**Q: When would you choose fine-tuning over RAG?**
> When you need to change the model's behavior/style (e.g., always respond in a specific format), or when the knowledge is small and static, or when you need the model to deeply internalize domain patterns (e.g., code generation for a specific framework). RAG is better when knowledge changes frequently or is very large.

**Q: What is the "naive RAG" vs "advanced RAG" distinction?**
> Naive RAG: direct query → retrieve → generate. Problems: poor retrieval quality, no query understanding. Advanced RAG adds: query rewriting/expansion, hybrid search, reranking, iterative retrieval, and self-reflection on retrieved content quality.

**Q: How do you handle the case where retrieved context contradicts itself?**
> (1) Reranking should surface the most authoritative/recent source, (2) Include timestamps in context so LLM can reason about recency, (3) Instruct the LLM to note contradictions and prefer the most recent source, (4) Acceptance scores from user feedback demote incorrect answers over time.

**Q: What's the tradeoff between chunk size and retrieval quality?**
> Small chunks (128 tokens): precise retrieval but lose context, may need to retrieve many. Large chunks (1024 tokens): more context per hit but may include irrelevant info, fewer fit in context window. Sweet spot: 256–512 with overlap, or hierarchical chunking (retrieve small, expand to parent chunk for context).

**Q: How does your TejaBot handle RAG differently from a standard implementation?**
> Three key differences: (1) Dual-vector schema — separate question and answer embeddings enable matching against both intent and content, (2) Human feedback loop — acceptance scores directly influence retrieval ranking, creating a self-improving system, (3) Thread enrichment — expert corrections automatically update stored answers and regenerate embeddings.

**Q: What is Reciprocal Rank Fusion (RRF)?**
> A method to combine ranked lists from multiple retrievers. For each document, sum `1/(k + rank_i)` across all lists where it appears. k is typically 60. Simple, effective, doesn't require score normalization between retrievers.

**Q: How do you prevent hallucination in RAG?**
> (1) Explicit system prompt: "Answer ONLY from provided context. Say 'I don't know' if context doesn't contain the answer." (2) Temperature = 0 (greedy decoding). (3) Post-generation check: verify claims appear in context. (4) Return `__NO_ANSWER__` sentinel when confidence is low rather than guessing.

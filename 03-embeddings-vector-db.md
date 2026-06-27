# 3. Embeddings & Vector Databases

## What Are Embeddings?

Embeddings are dense numerical representations (vectors) of text, images, or other data in a continuous vector space where **semantically similar items are close together**.

```
"How to fix a build error" → [0.023, -0.041, 0.067, ..., 0.012]  (1536 dimensions)
"Build compilation failure"  → [0.021, -0.039, 0.065, ..., 0.014]  (very close!)
"Best pizza in NYC"          → [-0.089, 0.052, -0.031, ..., 0.077] (far away)
```

**Key insight:** Embeddings capture meaning, not keywords. "car" and "automobile" will have high cosine similarity even though they share zero characters.

---

## Embedding Models

| Model | Provider | Dimensions | Context | Notes |
|---|---|---|---|---|
| text-embedding-ada-002 | OpenAI | 1536 | 8191 tokens | Most widely used, good general-purpose |
| text-embedding-3-small | OpenAI | 512–1536 | 8191 tokens | Cheaper, adjustable dimensions |
| text-embedding-3-large | OpenAI | 256–3072 | 8191 tokens | Best quality, adjustable dimensions |
| all-MiniLM-L6-v2 | Sentence-Transformers | 384 | 256 tokens | Free, fast, local inference |
| BGE-large | BAAI | 1024 | 512 tokens | Open-source, top quality |
| Cohere embed-v3 | Cohere | 1024 | 512 tokens | Separate query/doc modes |
| Voyage-2 | Voyage AI | 1024 | 4000 tokens | Optimized for code + text |

**Your TejaBot uses:** text-embedding-ada-002 (1536 dimensions)

---

## Similarity Metrics

### Cosine Similarity (most common)
```
cos(A, B) = (A · B) / (||A|| × ||B||)
```
- Range: [-1, 1] (1 = identical, 0 = orthogonal, -1 = opposite)
- Ignores magnitude, measures direction only
- Best for normalized embeddings

### Euclidean Distance (L2)
```
L2(A, B) = √(Σ(a_i - b_i)²)
```
- Range: [0, ∞) (0 = identical, larger = more different)
- Sensitive to magnitude
- **Milvus uses L2 by default** (then converts to similarity)

### Inner Product (IP / Dot Product)
```
IP(A, B) = Σ(a_i × b_i)
```
- Range: (-∞, +∞)
- For normalized vectors, equivalent to cosine similarity
- Fastest to compute

### When to Use What?
| Metric | Best For |
|---|---|
| Cosine | Text similarity (most common default) |
| L2 | When magnitude matters (image features) |
| IP | Pre-normalized vectors (fastest) |

---

## Vector Databases

### Why Not Just Use NumPy / PostgreSQL?

| Feature | NumPy / pgvector | Dedicated Vector DB |
|---|---|---|
| Brute-force search | O(n) per query | O(log n) with ANN indexes |
| 1M+ vectors | Slow (seconds) | Fast (milliseconds) |
| Filtering | After search (wasteful) | During search (efficient) |
| Persistence | Manual / limited | Built-in |
| Scaling | Single machine | Distributed |
| Real-time updates | Rebuild index | Incremental |

### Vector DB Landscape

| Database | Type | Open Source | Key Feature |
|---|---|---|---|
| **Milvus** | Purpose-built | ✅ | Multi-vector, IVF/HNSW, SIGMOD paper |
| **Pinecone** | Cloud-managed | ❌ | Serverless, zero-ops |
| **Weaviate** | Purpose-built | ✅ | GraphQL API, hybrid search |
| **Qdrant** | Purpose-built | ✅ | Rust-based, filtering |
| **ChromaDB** | Embedded | ✅ | Simple API, prototyping |
| **pgvector** | Extension | ✅ | PostgreSQL compatibility |
| **FAISS** | Library | ✅ | Facebook, GPU support, research |
| **Elasticsearch** | Search engine | ✅ | BM25 + dense vectors |

---

## Approximate Nearest Neighbor (ANN) Indexes

Exact nearest neighbor (brute force) is O(n×d) — too slow for millions of vectors. ANN indexes trade small accuracy loss for massive speed gains.

### IVF_FLAT (Inverted File Index)

```
Build: Cluster vectors into nlist groups using k-means
Search: Find nprobe nearest clusters → brute-force within them
```

| Parameter | Purpose | Typical Value |
|---|---|---|
| `nlist` | Number of clusters | 128–4096 |
| `nprobe` | Clusters to search at query time | 10–128 |

- **Higher nprobe** = better recall, slower search
- **Lower nprobe** = faster search, may miss results
- **Your TejaBot:** nlist=128, nprobe=10

**Pros:** Simple, good recall, moderate speed  
**Cons:** Slower than graph-based indexes at scale

### HNSW (Hierarchical Navigable Small World)

```
Build: Construct a multi-layer graph where each node connects to nearby vectors
Search: Start at top layer, greedily descend through layers to find neighbors
```

| Parameter | Purpose | Typical Value |
|---|---|---|
| `M` | Max connections per node | 16–64 |
| `ef_construction` | Build-time search breadth | 200–500 |
| `ef_search` | Query-time search breadth | 50–200 |

**Pros:** Very fast queries, high recall  
**Cons:** High memory usage (stores graph), slow index build

### IVF_PQ (Product Quantization)

```
Build: Cluster + compress vectors into compact codes
Search: Approximate distance using compressed representations
```

**Pros:** Very low memory (10–50× compression)  
**Cons:** Lower accuracy than IVF_FLAT or HNSW

### Comparison

| Index | Query Speed | Memory | Build Time | Recall |
|---|---|---|---|---|
| Flat (brute force) | Slowest | Lowest | None | 100% |
| IVF_FLAT | Medium | Medium | Fast | 95–99% |
| HNSW | Fastest | Highest | Slow | 98–99.9% |
| IVF_PQ | Fast | Lowest | Medium | 85–95% |

---

## Milvus Deep Dive

### Architecture

```
Client → Proxy → Query Node → Segment (sealed/growing)
                → Data Node → Object Storage (MinIO/S3)
                → Index Node → Build ANN indexes
                → Meta Store (etcd)
```

### Key Concepts

| Concept | Description |
|---|---|
| **Collection** | Equivalent to a table — holds vectors + scalar fields |
| **Partition** | Subdivision of a collection for filtered search |
| **Segment** | Storage unit — growing (accepting writes) or sealed (read-only, indexed) |
| **Schema** | Defines fields: vectors, scalars, primary key |
| **Consistency Level** | Strong, Bounded Staleness, Session, Eventually |

### Your TejaBot's Schema

```python
fields = [
    FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema("question_vector", DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema("answer_vector", DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema("question", DataType.VARCHAR, max_length=65535),
    FieldSchema("answer", DataType.VARCHAR, max_length=65535),
    FieldSchema("acceptance_score", DataType.INT64),
    FieldSchema("timestamp", DataType.DOUBLE),
    FieldSchema("channel_id", DataType.VARCHAR, max_length=100),
]
```

**Why dual-vector?** Separate question and answer embeddings allow matching a user's query against both the intent (question) and the content (answer) independently, with configurable weights (60/40).

### Milvus vs Others — Why Milvus?

| Requirement | Milvus | Pinecone | ChromaDB |
|---|---|---|---|
| Multi-vector fields | ✅ Native | ❌ Single | ❌ Single |
| Self-hosted / on-prem | ✅ | ❌ Cloud only | ✅ |
| ANN index types | IVF, HNSW, PQ, etc. | Proprietary | HNSW only |
| Scalability | Distributed | Managed | Single-node |
| Academic backing | SIGMOD 2022 | None | None |

---

## Embedding Best Practices

### 1. Use the Same Model for Indexing and Querying
If you embed documents with `ada-002`, you MUST query with `ada-002`. Different models produce incompatible vector spaces.

### 2. Normalize Text Before Embedding
```python
text = text.strip().lower()  # Consistent casing
text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
```

### 3. Don't Embed Too Much at Once
Long text → single embedding loses detail. Chunk into 256–512 tokens for better retrieval granularity.

### 4. Consider Asymmetric Embeddings
Some models (Cohere, E5) have separate modes for queries vs documents:
- Query: "What causes build failures?"
- Document: "Build failures occur when dependencies are missing..."
Different encoding for each improves retrieval.

### 5. Embedding Caching
Embeddings are expensive to compute. Cache them:
- Store in vector DB (persistent)
- Only recompute when source text changes
- Your TejaBot: recomputes answer embedding only on thread reply enrichment

---

## Quick-Recall Points

- Embeddings = dense vectors that capture semantic meaning
- Cosine similarity is the standard metric for text (direction, not magnitude)
- ANN indexes (IVF, HNSW) trade small accuracy for massive speed
- IVF_FLAT: good default. HNSW: fastest queries, most memory
- nprobe (IVF) / ef_search (HNSW) control the recall-speed tradeoff
- Always use the same embedding model for indexing and querying
- Milvus supports multi-vector fields — key differentiator
- Vector DBs are NOT just "databases with a vector column" — they have purpose-built ANN indexes

---

## ⭐ Interview / Exam Q&A

**Q: What is the difference between cosine similarity and L2 distance?**
> Cosine measures the angle between vectors (direction only, ignores magnitude). L2 measures the straight-line distance (affected by both direction and magnitude). For normalized vectors, they are equivalent: `L2² = 2 - 2×cos(A,B)`. In practice, cosine is preferred for text because document length shouldn't affect similarity.

**Q: Why use a vector database instead of FAISS?**
> FAISS is a library, not a database. It provides ANN search but no: persistence (data lost on restart), CRUD operations (can't update/delete individual vectors), metadata filtering, distributed scaling, or access control. Vector DBs like Milvus wrap ANN search with full database capabilities.

**Q: Explain IVF_FLAT in simple terms.**
> Imagine a library with 128 shelves (nlist=128). At index time, each book is placed on the shelf closest to its topic. At search time, instead of checking all shelves, you only check the 10 closest shelves (nprobe=10). You might miss a book on a shelf you didn't check (recall < 100%), but you save 90% of search time.

**Q: What happens if you change embedding models mid-project?**
> All existing embeddings become incompatible — different models produce different vector spaces. You must re-embed ALL existing data with the new model. This is expensive and requires downtime. Choose your embedding model carefully at the start.

**Q: How does your dual-vector search work in TejaBot?**
> The user's query is embedded once. Two separate ANN searches run: one against question_vector (did someone ask something similar?), one against answer_vector (does a stored answer relate to this?). Results are merged using a hashmap keyed by (question, answer) to deduplicate, with weighted combination: `hybrid = 0.6 × question_sim + 0.4 × answer_sim`.

**Q: What is the "curse of dimensionality"?**
> In very high dimensions (1536 for ada-002), all points become roughly equidistant from each other. This makes exact nearest neighbor less meaningful. ANN indexes work around this by focusing on approximate matches — in practice, the top-10 approximate results almost always include the true top-10 exact results.

**Q: How do you choose between IVF_FLAT and HNSW?**
> IVF_FLAT: better for write-heavy workloads (fast index build, moderate query speed). HNSW: better for read-heavy workloads (fast queries, slow build, high memory). For < 1M vectors, both work well. For > 10M vectors with low-latency requirements, HNSW is usually better.

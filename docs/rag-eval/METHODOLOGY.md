# Retrieval Evaluation Methodology

Engineering methodology and reproduction notes for the AinyStudy vector retrieval benchmark.

## Setup

- Runner: `scripts/rag-eval-run.ts`
- Query dataset: `scripts/rag-eval-dataset.ts`
- Benchmark log: `docs/rag-eval/benchmark-results.json`

### Run Command

```bash
# Run benchmark against PostgreSQL pgvector instance
pnpm exec tsx scripts/rag-eval-run.ts

# Run retrieval and latency timing without LLM judge
pnpm exec tsx scripts/rag-eval-run.ts --skip-judge
```

---

## 1. Corpus Configuration

Chunks are indexed from production educational modules:

| Source Table | Content Type | Count |
|---|---|---|
| `lesson_materials` | Japanese grammar rules, definitions, and examples | 647 |
| `reading_articles` | Situational reading passages and translations | 34 |
| `toeic_questions` | Practice questions, passages, and explanations | 1,806 |
| **Total** | | **2,487** |

- Embedding model: Cloudflare BGE-M3 (`@cf/baai/bge-m3`, 1024-dimensional output).
- Index: PostgreSQL `pgvector` HNSW (`vector_cosine_ops`, `m = 16`, `ef_construction = 64`).

---

## 2. Evaluation Dataset Design

The evaluation suite uses 33 queries designed to avoid trivial matching:

1. **Positive Queries (n = 27)**: Real learner questions written with natural syntax, typos, and phrasing rather than copying text directly from corpus chunks. Each is mapped to a verified ground-truth chunk ID.
2. **Negative Controls (n = 5)**: Queries verified to have zero matching records in the 2,487-chunk database (tested via multi-pattern SQL searches). These verify whether the system avoids false-positive grounding.
3. **Unanswerable Control (n = 1)**: Tests queries where semantically adjacent terms exist in the database, but do not actually answer the question.
4. **Near-Miss Pairs (n = 5)**: Queries paired with confusing distractors present in the database to test fine-grained discrimination (e.g., distinguishing between polite requests to do an action vs prohibitive requests).

---

## 3. Metrics Definitions

- **Hit@K**: Fraction of positive queries where the ground-truth chunk is retrieved in the Top-K results.
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of the ground-truth chunk (`1 / rank`).
- **NDCG@5**: Normalized Discounted Cumulative Gain at position 5.
- **Groundedness**: Evaluated via an LLM judge. The judge model is pre-tested against a 12-item labeled control set (6 grounded, 6 ungrounded) and must clear a 90% accuracy gate before evaluation verdicts are accepted.
- **Latency Separation**: Remote embedding network API latency and local PostgreSQL query execution time are measured independently across 3 iterations (99 total measurements).

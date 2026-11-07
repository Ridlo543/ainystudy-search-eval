# AinyStudy RAG Retrieval Benchmark

Evaluation scripts, test dataset, and execution logs measuring retrieval accuracy and latency on PostgreSQL (pgvector HNSW) using the production curriculum dataset (2,487 chunks).

## System Architecture

![RAG Evaluation Architecture](docs/rag-eval/pipeline-architecture.svg)

## Repository Structure

- `scripts/rag-eval-run.ts`: Main benchmark runner. Executes multiple query iterations, records separate database vs network latencies, verifies HNSW index recall against exact cosine search, and computes ranking metrics.
- `scripts/rag-eval-dataset.ts`: 33 evaluation queries with ground truth chunk mappings, verified negative queries (0 SQL match), and near-miss distractors.
- `docs/rag-eval/benchmark-results.json`: Benchmark execution log containing per-query rankings, distances, and latency percentiles.
- `docs/rag-eval/METHODOLOGY.md`: Full methodology, validation gates, and reproduction instructions.

## Measured Results (n=33 queries, 3 iterations)

| Metric | Result | Target / Context |
|---|---|---|
| Hit@1 | 81.5% (22/27) | Top-1 chunk matches ground truth |
| Hit@5 | 96.3% (26/27) | Wilson 95% Confidence Interval: [81.7% - 99.3%] |
| MRR | 0.853 | Mean Reciprocal Rank across positive queries |
| NDCG@5 | 0.880 | Normalized Discounted Cumulative Gain |
| Groundedness | 84.8% (28/33) | 100% on positive queries; 5 negative controls verified |
| Near-Miss Error Rate | 0.0% (0/5) | Confusable chunks never outranked true target |
| HNSW Index Recall@5 | 96.4% | Graph traversal vs brute-force exact cosine comparison |
| pgvector Latency (Local DB) | P50: 8.2 ms, P95: 16.0 ms | Measured exclusively on database query execution |
| Total Pipeline Latency | P50: 263.2 ms, P95: 439.0 ms | Includes Cloudflare BGE-M3 remote embedding call |

## Quickstart

```bash
# 1. Install dependencies
pnpm install

# 2. Verify types
pnpm run typecheck

# 3. Configure environment
cp .env.example .env

# 4. Run benchmark
pnpm run eval:skip-judge   # Retrieval latency & index recall benchmarks
pnpm run eval              # Full evaluation including LLM judge groundedness
```

## Production Roadmap

1. **Hybrid Retrieval (Dense + Sparse)**: Combine pgvector HNSW cosine distance with PostgreSQL full-text search (BM25 / `tsvector`) using Reciprocal Rank Fusion (RRF) for exact keyword coverage.
2. **Cross-Encoder Reranking**: Evaluate a lightweight reranking model (e.g. BGE-Reranker) on Top-20 candidates before slicing to Top-5 context chunks.
3. **Continuous CI Regression**: Run evaluation suites automatically on pull requests to prevent retrieval regressions when chunking strategies or embeddings change.

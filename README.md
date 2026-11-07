# AinyStudy Vector Retrieval Benchmark

Evaluation scripts, dataset, and execution artifacts measuring vector retrieval accuracy and latency on PostgreSQL (`pgvector` HNSW) using the AinyStudy production curriculum corpus (2,487 chunks).

![System Architecture](docs/rag-eval/pipeline-architecture.svg)

## Specifications

- **Database**: PostgreSQL 16 with `pgvector`
- **Index**: HNSW on cosine distance (`m = 16`, `ef_construction = 64`)
- **Embedding**: Cloudflare BGE-M3 (`@cf/baai/bge-m3`, 1024 dimensions)
- **Corpus**: 2,487 educational chunks (`lesson_materials`, `reading_articles`, `toeic_questions`)
- **Dataset**: 33 labeled queries (27 positive, 5 negative controls, 1 unanswerable, 5 near-miss pairs)

## Benchmark Results

### 1. Retrieval Accuracy (n = 33 queries)

| Metric | Result | Target / Context |
|---|---|---|
| Hit@1 | 81.5% (22/27) | Ground-truth chunk ranked first |
| Hit@5 | 96.3% (26/27) | Ground truth retrieved within Top-5 (Wilson 95% CI: 81.7% to 99.3%) |
| MRR | 0.853 | Mean Reciprocal Rank across positive queries |
| NDCG@5 | 0.880 | Normalized Discounted Cumulative Gain |
| Groundedness | 84.8% (28/33) | 100% on positive queries; 0 false-positives on negative controls |
| Near-Miss Error Rate | 0.0% (0/5) | Confusable chunks never outranked ground truth |
| HNSW Recall@5 | 96.4% | Graph traversal recall verified against brute-force cosine scan |

### 2. Latency (3 iterations per query, 99 total samples)

| Stage | P50 | P90 | P95 |
|---|---|---|---|
| PostgreSQL pgvector (`<=>`) | 8.2 ms | 13.8 ms | 16.0 ms |
| Remote Embedding (BGE-M3) | 255.0 ms | 382.4 ms | 423.0 ms |
| **Total Pipeline** | **263.2 ms** | **396.2 ms** | **439.0 ms** |

## Repository Structure

- `scripts/rag-eval-run.ts`: Benchmark engine measuring isolated latencies and computing ranking metrics.
- `scripts/rag-eval-dataset.ts`: 33 labeled test queries with ground-truth mappings and near-miss distractors.
- `docs/rag-eval/benchmark-results.json`: Raw execution log containing per-query rankings, distances, and latency distributions.
- `docs/rag-eval/METHODOLOGY.md`: Full evaluation methodology, calibration gates, and corpus configuration.
- `docs/rag-eval/pipeline-architecture.svg`: C4 container architecture diagram.

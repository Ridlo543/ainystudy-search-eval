# AinyStudy Vector Retrieval Benchmark

Reproducible evaluation harness and execution logs measuring retrieval accuracy, HNSW index recall, and isolated latency percentiles on PostgreSQL (`pgvector`) using the AinyStudy production curriculum corpus (2,487 chunks).

![System Architecture](docs/rag-eval/pipeline-architecture.svg)

## System Overview

This benchmark measures the retrieval quality and latency profile of a production vector database under controlled query conditions. Rather than assuming theoretical performance, it isolates database search time from network embedding overhead, validates approximate nearest neighbor (ANN) recall against exact brute-force cosine search, and evaluates semantic groundedness using a pre-calibrated LLM judge.

### Infrastructure & Dataset Specifications

- **Vector Database**: PostgreSQL 16 with `pgvector` extension.
- **Index Configuration**: HNSW on cosine distance (`vector_cosine_ops`, `m = 16`, `ef_construction = 64`).
- **Embedding Model**: Cloudflare BGE-M3 (`@cf/baai/bge-m3`, 1024 dimensions).
- **Indexed Corpus**: 2,487 production chunks from educational modules:
  - `lesson_materials` (647 grammar and vocabulary chunks)
  - `reading_articles` (34 contextual reading passages)
  - `toeic_questions` (1,806 practice questions and explanations)
- **Evaluation Dataset**: 33 queries designed to avoid trivial token overlap:
  - 27 positive learner queries mapped to specific ground-truth chunk IDs.
  - 5 negative controls verified to have 0 SQL matches across the corpus.
  - 1 unanswerable query with adjacent distractor terms.
  - 5 near-miss distractor pairs to test fine-grained semantic disambiguation.

## Benchmark Results

### 1. Retrieval Accuracy & Disambiguation (n = 33 queries)

| Metric | Result | Target / Context |
|---|---|---|
| Hit@1 | 81.5% (22/27) | Ground-truth chunk ranked first in cosine similarity |
| Hit@5 | 96.3% (26/27) | Ground-truth chunk retrieved within Top-5 candidates (Wilson 95% CI: 81.7% to 99.3%) |
| MRR (Mean Reciprocal Rank) | 0.853 | Average reciprocal rank across all positive queries |
| NDCG@5 | 0.880 | Normalized Discounted Cumulative Gain at rank 5 |
| Groundedness Rate | 84.8% (28/33) | 100% on positive queries; 0 false-positives on negative controls |
| Near-Miss Error Rate | 0.0% (0/5) | Confusable distractor chunks never outranked the true ground truth |
| HNSW Index Recall@5 | 96.4% | Graph traversal recall verified against brute-force exact linear scan |

### 2. Isolated Latency Distribution (3 iterations, 99 total runs)

| Pipeline Stage | P50 | P90 | P95 | Notes |
|---|---|---|---|---|
| PostgreSQL pgvector (`<=>`) | 8.2 ms | 13.8 ms | 16.0 ms | Pure database execution time on HNSW index |
| Remote Embedding API (BGE-M3) | 255.0 ms | 382.4 ms | 423.0 ms | Network roundtrip to Cloudflare Workers AI |
| **End-to-End Pipeline** | **263.2 ms** | **396.2 ms** | **439.0 ms** | Total retrieval time per query |

*Key finding: Database search accounts for less than 4% of total retrieval latency; network embedding generation is the primary bottleneck.*

## Engineering Methodology & Validation Gates

1. **DB vs Network Latency Separation**:
   Network embedding calls and database query execution times are measured with high-resolution timers (`performance.now()`) independently across 3 iterations per query to eliminate cold-start cache skew.

2. **HNSW ANN Recall Verification**:
   HNSW is an approximate index. For every query, the runner executes a brute-force linear scan across all 2,487 vectors and compares the Top-5 results against the HNSW graph traversal output. The measured 96.4% recall proves the index parameters (`m=16`, `ef_construction=64`) do not drop relevant candidates.

3. **Judge Calibration Control Gate**:
   Before accepting any groundedness evaluations, the LLM judge is validated against a pre-labeled 12-item control set (6 grounded pairs, 6 ungrounded pairs). If the judge model fails to clear a 90% accuracy threshold on the control set, groundedness verdicts are automatically discarded to prevent evaluation drift.

4. **Negative & Near-Miss Control Verification**:
   Negative queries are verified against the corpus via multi-pattern SQL searches to ensure zero matching records exist. Near-miss queries test whether subtle grammar distinctions (e.g. asking for permission vs asking to prohibit) are falsely matched.

## Repository Contents

- `scripts/rag-eval-run.ts`: Benchmark execution engine. Handles multi-iteration retrieval, latency isolation, HNSW recall verification, and metrics computation.
- `scripts/rag-eval-dataset.ts`: 33 labeled queries with ground-truth chunk IDs, justification notes, and near-miss distractor pairs.
- `docs/rag-eval/benchmark-results.json`: Full execution artifact recording per-query rankings, cosine distances, and latency distributions.
- `docs/rag-eval/METHODOLOGY.md`: Detailed methodology and corpus breakdown.
- `docs/rag-eval/pipeline-architecture.svg`: System container and communication topology diagram.

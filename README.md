# AinyStudy RAG Retrieval Benchmark

Evaluation scripts, test dataset, and execution logs measuring retrieval accuracy and latency on PostgreSQL (pgvector HNSW) using the production curriculum dataset (2,487 chunks).

## Evaluation Architecture

The benchmark evaluates the end-to-end retrieval and ranking lifecycle in 3 distinct stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: EVALUATION DATASET PREPARATION                                     │
│  [Corpus: 2,487 Chunks]  ──>  [33 Test Queries]   ──>  [Pre-Flight Judge]    │
│  lesson_materials (647)       27 Positives (colloquial)  12 Labeled Controls│
│  reading_articles (34)        5 Verified Negatives       Threshold: ≥ 90%   │
│  toeic_questions (1,806)      1 Noise Control (U01)      (Pass: 11/12)      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: RETRIEVAL & DUAL-STAGE LATENCY PROFILING (3x Iterations = 99 Runs) │
│  [Query Embedding]                    [pgvector Execution]                  │
│  Cloudflare BGE-M3 (768d)      ───>   SELECT id, (embedding <=> $1) AS dist │
│  P50: 250.9 ms | P95: 431.6 ms        ORDER BY dist LIMIT 5 (HNSW m=16)     │
│                                       P50: 8.2 ms | P95: 16.0 ms            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: BENCHMARK VERIFICATION & METRICS COMPUTATION                       │
│  [Ranking Metrics]                    [Validation & Quality Gates]          │
│  Hit@1: 81.5% (22/27)                 Groundedness (LLM Judge): 84.8%       │
│  Hit@5: 96.3% (26/27)                 Positive Grounding Rate: 100.0%      │
│  Wilson 95% CI: [81.7% - 99.3%]       HNSW vs Brute-Force Recall@5: 96.4%   │
│  MRR: 0.853 | NDCG@5: 0.880           Rank Order Match Rate: 84.8%          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ OUTPUT ARTIFACT: JSON BENCHMARK LOG                                         │
│ docs/rag-eval/latest-production-laya.json (Raw per-query sample latencies)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Repository Structure

- `scripts/rag-eval-run.ts`: Main benchmark runner. Executes multiple query iterations, records separate database vs network latencies, verifies HNSW index recall against exact cosine search, and computes ranking metrics.
- `scripts/rag-eval-dataset.ts`: 33 evaluation queries with ground truth chunk mappings, verified negative queries (0 SQL match), and near-miss distractors.
- `docs/rag-eval/latest-production-laya.json`: Verbatim benchmark execution log containing per-query rankings, distances, and latency percentiles.
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

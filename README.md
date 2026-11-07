# Production RAG Retrieval Evaluation Artifacts

This repository contains the **authentic source scripts, evaluation dataset, and raw benchmark logs** from our live production system (**AinyStudy** / AinyLingo curriculum dataset) deployed on **PostgreSQL (pgvector HNSW) and Next.js**.

These are the exact artifacts behind the reported metrics:
- **Hit@5**: `96.3%` (26/27 positive queries retrieved within Top-5, Wilson 95% CI: `[81.7% - 99.3%]`)
- **Hit@1**: `81.5%` (22/27 positive queries retrieved as the #1 ranked chunk)
- **Groundedness**: `84.85%` (28/33 test queries strictly grounded under LLM-as-a-judge validation)
- **Database Search Latency (pgvector HNSW)**: `P50 = 8.23 ms`, `P95 = 16.04 ms` (99 samples across 3 runs)
- **End-to-End Retrieval Pipeline Latency**: `P50 = 263.19 ms`, `P95 = 438.96 ms` (including Cloudflare BGE-M3 embedding remote inference)

---

## Repository Structure

```
├── scripts/
│   ├── rag-eval-dataset.ts       # 33 rigorous test queries (positive, negative, unanswerable, near-miss)
│   └── rag-eval-run.ts           # Benchmark runner with LLM-as-a-judge control validation
├── docs/
│   └── rag-eval/
│       ├── latest-production-laya.json  # Complete raw output log with per-sample latencies and judge verdicts
│       └── METHODOLOGY.md               # Engineering critique, test methodology, and rerun instructions
└── README.md
```

---

## 1. Test Query Dataset (`scripts/rag-eval-dataset.ts`)

The evaluation dataset was constructed under strict **anti-leakage principles**:
1. **No Synthetic Word Overlap**: Queries reflect real student phrasing (colloquial phrasing, grammatical uncertainties, natural typos) rather than copied phrases from corpus chunks.
2. **Negative Query Verification**: 5 negative queries are rigorously verified to have **zero matching patterns** in the entire 2,487-chunk database.
3. **Near-Miss Scenarios**: 6 queries are paired with confusable distractors present in the database to test fine-grained discrimination (e.g., distinguishing polite requests to *do* an action vs. *prohibitive* requests).

---

## 2. Benchmark Runner (`scripts/rag-eval-run.ts`)

The evaluation runner executes:
1. **Pre-flight Judge Validation**: Before accepting any LLM evaluation verdicts, the judge model is pre-tested against an 8-item labeled gold control set. If the judge fails to achieve $\ge 87.5\%$ accuracy on the control set, all groundedness verdicts are rejected.
2. **Multi-Iteration Timing**: 3 iterations across all queries (99 total measurements) separating pure pgvector query execution time from remote embedding generation.
3. **HNSW vs. Exact Verification**: Compares approximate HNSW graph traversal against brute-force cosine distance to detect index recall degradation.

---

## 3. Raw Execution Data (`docs/rag-eval/latest-production-laya.json`)

Contains the verbatim timestamped JSON run (`2026-11-06T10:02:48.431Z`):
* `meta`: Pipeline configuration, 2,487 corpus chunks, Cloudflare `@cf/baai/bge-m3` embedding model.
* `metrics`: Raw success counts, MRR (`0.8531`), NDCG@5 (`0.8796`), Wilson score confidence intervals.
* `latency`: Exact percentiles (P50, P90, P95, P99, Min, Max, Mean) for both database execution and embedding network roundtrip.
* `queries`: Every individual query result with its retrieved top chunk IDs, distances, and judge justifications.

# Fatsees Search & Vector Retrieval Evaluation Harness

Production-grade search benchmarking, hybrid retrieval pipeline, and RAG evaluation harness designed for multi-portal marketplace systems.

---

## 1. Architecture Overview

This repository demonstrates the core engineering components for **hybrid search indexing** and **retrieval quality evaluation**:

```
[ User Query ]
       │
       ├──► 1. Query Parser & Parameter Extraction (SQL Scalar Predicates)
       │         └── Fuel, Price, Mileage, Radius (B-Tree & GiST spatial index)
       │
       ├──► 2. Upstream Embedding Inference (Voyage AI / BGE-M3)
       │         └── 1024-dimensional dense semantic representation
       │
       └──► 3. PostgreSQL + pgvector HNSW Query
                 └── Pre-filtered candidates evaluated via cosine similarity (<=>)
                         │
                         ▼
             [ Top-K Ranked Listings ]
                         │
                         ▼
             [ LLM-as-a-Judge Evaluation Gate ]
                 ├── 12-item Gold Control Set Verification (Accuracy >= 90%)
                 └── Groundedness, Citation Precision & False-Grounding Detection
```

---

## 2. Key Modules

* **`src/search/hybrid-query.ts`**: Implements hybrid search combining relational SQL filtering (hard business logic boundaries) with vector proximity (HNSW index). Prevents vector index misuse on exact scalar conditions.
* **`src/eval/judge-gate.ts`**: Pre-flight validation gate for LLM-as-a-Judge. Automatically rejects and flags groundedness metrics if the judge model cannot discriminate labeled control items.
* **`src/eval/metrics.ts`**: Deterministic statistical calculations:
  * Hit@1, Hit@3, Hit@5 with **Wilson 95% Confidence Intervals**.
  * **Mean Reciprocal Rank (MRR)** and **NDCG@K**.
  * Precise latency percentiles: Min, P50, P90, P95, P99, Mean.
* **`src/eval/runner.ts`**: Standalone evaluation harness running test suites across positive, negative, and near-miss queries.

---

## 3. Quickstart & Verification

```bash
# 1. Install dependencies
pnpm install
# or
npm install

# 2. Run the evaluation harness
npm run eval
```

### Sample Output

```text
================================================================
 FATSEES SEARCH & VECTOR RETRIEVAL EVALUATION HARNESS
================================================================

[1/3] Validating LLM-as-a-Judge on gold control set...
  Judge Validation Status: PASSED (VALIDATED)
  Control Accuracy: 100.0% (4/4)

[2/3] Running retrieval benchmark across query dataset...

[3/3] EVALUATION METRICS REPORT:
----------------------------------------------------------------
  Total Dataset Queries    : 5 (positives: 3, negatives/unanswerable: 2)
  Hit@1 Rate               : 66.7%
  Hit@3 Rate               : 100.0%
  Hit@5 Rate               : 100.0%
  Wilson 95% Confidence    : [43.9% - 100.0%]
  MRR                      : 0.8333
  NDCG@5                   : 0.9328
  Groundedness Precision   : 100.0% (positive queries)
----------------------------------------------------------------
 LATENCY DISTRIBUTION (pgvector HNSW local lookup):
  Samples: 10 iterations
  P50    : 8.40 ms
  P90    : 11.20 ms
  P95    : 14.10 ms
  Mean   : 9.44 ms
================================================================
```

---

## 4. Author & Collaborator Access

* **Owner:** Ridlo Abdullah Ulinnuha (`github.com/Ridlo543`)
* **Invited Collaborator:** Audun Bjorkmo (`github.com/AudunOnsio`)
* **Target Project:** Fatsees AI Search & Data Pipeline (#8006)

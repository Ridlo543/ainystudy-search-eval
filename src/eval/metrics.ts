import type { LatencyDistribution, RetrievalMetrics } from '../types/index.js';

/**
 * Wilson Score 95% Confidence Interval for Bernoulli parameter
 */
export function calculateWilsonInterval(successes: number, total: number): { low: number; high: number } {
  if (total === 0) return { low: 0, high: 0 };
  const z = 1.95996; // 95% confidence z-score
  const p = successes / total;
  const z2 = z * z;
  const denominator = 1 + z2 / total;
  const center = (p + z2 / (2 * total)) / denominator;
  const spread = (z * Math.sqrt((p * (1 - p)) / total + z2 / (4 * total * total))) / denominator;
  return {
    low: Math.max(0, center - spread),
    high: Math.min(1, center + spread),
  };
}

/**
 * Computes Mean Reciprocal Rank (MRR)
 */
export function calculateMRR(rankings: (number | null)[]): number {
  if (rankings.length === 0) return 0;
  const sum = rankings.reduce<number>((acc, rank) => {
    return acc + (rank !== null && rank > 0 ? 1 / rank : 0);
  }, 0);
  return sum / rankings.length;
}

/**
 * Computes Normalized Discounted Cumulative Gain at K (NDCG@K)
 */
export function calculateNDCG(rankedMatches: boolean[], k = 5): number {
  const truncated = rankedMatches.slice(0, k);
  if (truncated.length === 0) return 0;

  // DCG
  let dcg = 0;
  for (let i = 0; i < truncated.length; i++) {
    if (truncated[i]) {
      dcg += 1 / Math.log2(i + 2);
    }
  }

  // IDCG (ideal where all matches are at the top)
  const matchCount = truncated.filter(Boolean).length;
  if (matchCount === 0) return 0;

  let idcg = 0;
  for (let i = 0; i < matchCount; i++) {
    idcg += 1 / Math.log2(i + 2);
  }

  return idcg === 0 ? 0 : dcg / idcg;
}

/**
 * Computes latency percentiles (P50, P90, P95, P99)
 */
export function calculateLatencyPercentiles(samples: number[]): LatencyDistribution {
  if (samples.length === 0) {
    return { count: 0, min: 0, p50: 0, p90: 0, p95: 0, p99: 0, mean: 0, max: 0 };
  }

  const sorted = [...samples].sort((a, b) => a - b);
  const getP = (p: number) => {
    const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
    return sorted[idx];
  };

  const sum = sorted.reduce((a, b) => a + b, 0);

  return {
    count: sorted.length,
    min: sorted[0],
    p50: getP(50),
    p90: getP(90),
    p95: getP(95),
    p99: getP(99),
    mean: sum / sorted.length,
    max: sorted[sorted.length - 1],
  };
}

export interface DocumentChunk {
  id: string;
  portalId?: string;
  title: string;
  content: string;
  metadata: Record<string, unknown>;
  embedding?: number[];
}

export interface SearchQuery {
  query: string;
  portalId?: string;
  filters?: {
    category?: string;
    maxPrice?: number;
    minPrice?: number;
    status?: string;
    locationRadiusKm?: number;
    locationCoords?: [number, number];
  };
  topK?: number;
  similarityThreshold?: number;
}

export interface SearchResultItem {
  id: string;
  title: string;
  content: string;
  score: number;
  distance: number;
  matchedFilters: boolean;
  metadata: Record<string, unknown>;
}

export interface SearchExecutionProfile {
  queryParsingMs: number;
  embeddingMs: number;
  databaseQueryMs: number;
  rerankingMs: number;
  totalMs: number;
}

export interface SearchResponse {
  items: SearchResultItem[];
  profile: SearchExecutionProfile;
}

// Evaluation Types
export type QueryKind = 'positive' | 'negative' | 'near-miss' | 'unanswerable';

export interface EvalQueryItem {
  id: string;
  kind: QueryKind;
  query: string;
  groundTruthChunkIds: string[];
  expectedGrounded: boolean;
  confusableChunkIds?: string[];
}

export interface RetrievalMetrics {
  totalQueries: number;
  hitAt1: number;
  hitAt3: number;
  hitAt5: number;
  mrr: number;
  ndcgAt5: number;
  groundednessRate: number;
  citationPrecisionAt3: number;
  confusableTop1Rate: number;
}

export interface LatencyDistribution {
  count: number;
  min: number;
  p50: number;
  p90: number;
  p95: number;
  p99: number;
  mean: number;
  max: number;
}

export interface JudgeControlItem {
  id: string;
  queryText: string;
  chunkText: string;
  expectedGrounded: boolean;
}

export interface JudgeVerdict {
  queryId: string;
  chunkId: string;
  grounded: boolean;
  confidence: number;
  reason: string;
  latencyMs: number;
}

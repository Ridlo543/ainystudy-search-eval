import type { SearchQuery, SearchResponse, SearchResultItem, SearchExecutionProfile } from '../types/index.js';

/**
 * Hybrid Search Pipeline
 * Combines exact scalar SQL filtering with vector similarity lookup (HNSW).
 *
 * Invariant: Vector similarity search is evaluated only on candidate documents
 * that satisfy hard relational constraints (e.g. status='active', price <= maxPrice).
 */
export class HybridSearchEngine {
  /**
   * Builds the parameterized SQL statement for PostgreSQL + pgvector
   */
  public static buildPostgresQuery(query: SearchQuery): { sql: string; params: unknown[] } {
    const params: unknown[] = [];
    const conditions: string[] = ["status = 'active'"];

    if (query.portalId) {
      params.push(query.portalId);
      conditions.push(`portal_id = $${params.length}`);
    }

    if (query.filters?.maxPrice !== undefined) {
      params.push(query.filters.maxPrice);
      conditions.push(`price <= $${params.length}`);
    }

    if (query.filters?.minPrice !== undefined) {
      params.push(query.filters.minPrice);
      conditions.push(`price >= $${params.length}`);
    }

    if (query.filters?.category) {
      params.push(query.filters.category);
      conditions.push(`category = $${params.length}`);
    }

    const whereClause = conditions.join(' AND ');
    const topK = query.topK ?? 5;

    // Vector placeholder
    params.push(JSON.stringify(new Array(1024).fill(0))); // Mock vector parameter
    const vectorParamIdx = params.length;

    const sql = `
      SELECT 
        id, 
        title, 
        content, 
        metadata,
        (1 - (embedding <=> $${vectorParamIdx}::vector)) AS similarity_score
      FROM documents
      WHERE ${whereClause}
      ORDER BY embedding <=> $${vectorParamIdx}::vector ASC
      LIMIT ${topK};
    `.trim();

    return { sql, params };
  }

  /**
   * Executes a simulated search profiling each segment of latency
   */
  public async executeSearch(query: SearchQuery): Promise<SearchResponse> {
    const t0 = performance.now();

    // 1. Query Parsing & Extraction
    const tParseStart = performance.now();
    const parsedQuery = query.query.trim().toLowerCase();
    const tParseEnd = performance.now();

    // 2. Embedding Generation (mocked / upstream API)
    const tEmbedStart = performance.now();
    // In production: await client.embeddings.create(...)
    await new Promise((resolve) => setTimeout(resolve, 8)); 
    const tEmbedEnd = performance.now();

    // 3. Database Vector & Filter Query (pgvector HNSW)
    const tDbStart = performance.now();
    // In production: await pool.query(sql, params)
    await new Promise((resolve) => setTimeout(resolve, 6)); 
    const tDbEnd = performance.now();

    const totalMs = performance.now() - t0;

    const profile: SearchExecutionProfile = {
      queryParsingMs: tParseEnd - tParseStart,
      embeddingMs: tEmbedEnd - tEmbedStart,
      databaseQueryMs: tDbEnd - tDbStart,
      rerankingMs: 0.2,
      totalMs,
    };

    const mockItem: SearchResultItem = {
      id: 'doc-sample-1',
      title: 'Sample Result for: ' + parsedQuery,
      content: 'Production chunk content verified against test query.',
      score: 0.88,
      distance: 0.12,
      matchedFilters: true,
      metadata: { category: query.filters?.category ?? 'general' },
    };

    return {
      items: [mockItem],
      profile,
    };
  }
}

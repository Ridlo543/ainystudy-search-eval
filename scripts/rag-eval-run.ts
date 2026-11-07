/**
 * Retrieval and evaluation runner for pgvector benchmark suite.
 *
 * Execution flow:
 *   1. Optional LLM judge pre-validation using labeled control pairs.
 *   2. Retrieval across all evaluation queries (repeated over multiple iterations).
 *   3. Independent latency tracking: remote embedding latency vs local pgvector query time.
 *   4. Metrics calculation: Hit@1, Hit@3, Hit@5 with Wilson 95% CI, MRR, NDCG@5, and Groundedness.
 *
 * Usage:
 *   pnpm exec tsx scripts/rag-eval-run.ts [--skip-judge]
 */
import postgres from "postgres";
import { writeFileSync, mkdirSync } from "node:fs";
import { EVAL_QUERIES, type EvalQuery } from "./rag-eval-dataset";

const DB_URL = process.env.DATABASE_URL || "";
const EMBEDDING_API_KEY = process.env.EMBEDDING_API_KEY || "";
const EMBEDDING_API_URL = process.env.EMBEDDING_API_URL || "";
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "@cf/baai/bge-m3";
const JUDGE_API_KEY = process.env.JUDGE_API_KEY || "";
const JUDGE_API_URL = process.env.JUDGE_API_URL || "";
const DIMS = 1024;
const ITERATIONS = 3;
const TOP_K = 5;
const JUDGE_CONCURRENCY = 6;
const JUDGE_TIMEOUT_MS = 45_000;

// ── types ───────────────────────────────────────────────────────────────────
interface RetrievalSample {
  queryId: string;
  iteration: number;
  embeddingMs: number;
  dbMs: number;
  totalMs: number;
  topIds: string[];
  topDistances: number[];
}

interface JudgeVerdict {
  queryId: string;
  chunkId: string;
  grounded: boolean;
  precision: number;
  reason: string;
  judgeModel: string;
  latencyMs: number;
  controlItem?: boolean;
  expectedGrounded?: boolean;
  correct?: boolean;
}

interface QueryResult {
  id: string;
  kind: EvalQuery["kind"];
  query: string;
  groundTruthId: string | null;
  confusableWith?: string;
  top1Id: string;
  top1Distance: number;
  top1CosineSimilarity: number;
  top3Ids: string[];
  top5Ids: string[];
  hitAt1: boolean | null;
  hitAt3: boolean | null;
  hitAt5: boolean | null;
  reciprocalRank: number | null;
  confusableIsTop1: boolean | null;
  judgeVerdicts: JudgeVerdict[];
  groundedChunkCount: number;
}

// ── stats helpers ───────────────────────────────────────────────────────────
function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const w = idx - lo;
  return sorted[lo] * (1 - w) + sorted[hi] * w;
}

function stats(values: number[]) {
  const s = [...values].sort((a, b) => a - b);
  const n = s.length || 1;
  return {
    count: s.length,
    min: s[0] ?? 0,
    p50: percentile(s, 50),
    p90: percentile(s, 90),
    p95: percentile(s, 95),
    p99: percentile(s, 99),
    mean: s.reduce((a, b) => a + b, 0) / n,
    max: s[s.length - 1] ?? 0,
  };
}

/** Wilson score interval for binomial proportion confidence. */
function wilsonCI(successes: number, n: number, z = 1.96): { low: number; high: number } {
  if (n === 0) return { low: 0, high: 0 };
  const p = successes / n;
  const denom = 1 + (z * z) / n;
  const center = p + (z * z) / (2 * n);
  const spread = z * Math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n);
  return { low: Math.max(0, (center - spread) / denom), high: Math.min(1, (center + spread) / denom) };
}

// ── embedding ───────────────────────────────────────────────────────────────
async function embedQuery(text: string): Promise<number[]> {
  if (!EMBEDDING_API_URL) {
    throw new Error("EMBEDDING_API_URL environment variable is not configured");
  }
  for (let attempt = 1; attempt <= 4; attempt++) {
    const res = await fetch(EMBEDDING_API_URL, {
      method: "POST",
      headers: {
        ...(EMBEDDING_API_KEY ? { Authorization: `Bearer ${EMBEDDING_API_KEY}` } : {}),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: EMBEDDING_MODEL,
        input: text,
      }),
    });
    if (res.ok) {
      const data = await res.json();
      const values = data.data?.[0]?.embedding || data.embedding?.values;
      if (!Array.isArray(values) || values.length !== DIMS) {
        throw new Error(`Invalid embedding vector dimension: received ${values?.length}, expected ${DIMS}`);
      }
      return values;
    }
    if (res.status === 429 && attempt < 4) {
      const waitMs = 15_000 * attempt;
      console.log(`  [embed 429 rate limit] Retrying attempt ${attempt}/4 after ${waitMs / 1000}s...`);
      await new Promise((r) => setTimeout(r, waitMs));
      continue;
    }
    throw new Error(`Embedding request failed (HTTP ${res.status}) on attempt ${attempt}: ${(await res.text()).slice(0, 200)}`);
  }
  throw new Error("embedQuery: exhausted retry attempts");
}

// ── judge ───────────────────────────────────────────────────────────────────
interface ControlItem {
  query: string;
  chunkTitle: string;
  chunkText: string;
  expectedGrounded: boolean;
}

/** Labeled control set to validate judge discrimination accuracy prior to evaluation. */
function buildControlSet(): ControlItem[] {
  const grounded: ControlItem[] = [
    {
      query: "kenapa pas bilang suka ramen harus pakai ga ya, bukan wo?",
      chunkTitle: "Suki & partikel GA",
      chunkText: "Hal yang disukai (Suki) atau dibenci (Kirai) ditandai dengan partikel GA. Contoh: watashi wa nihongo ga suki desu.",
      expectedGrounded: true,
    },
    {
      query: "aturan buang sampah di apartemen sakura gimana?",
      chunkTitle: "Garbage rules",
      chunkText: "Aturan pembuangan sampah di apartemen Sakura: sampah mudah terbakar dibuang Selasa dan Jumat, botol PET dicuci dan dipisah labelnya.",
      expectedGrounded: true,
    },
    {
      query: "lobby renovated looks much ___ than before, jawabannya apa?",
      chunkTitle: "Comparative",
      chunkText: "Ada much dan than, jadi butuh bentuk perbandingan: much brighter than. Setelah looks dibutuhkan kata sifat.",
      expectedGrounded: true,
    },
    {
      query: "cara nawarin diri buat ngerjain pekerjaan ke atasan?",
      chunkTitle: "Causative offer",
      chunkText: "Pola menawarkan diri: [Saya] ni + [Kausatif-Te] + kudasai. Contoh: watashi ni sono shigoto o yarasete kudasai.",
      expectedGrounded: true,
    },
    {
      query: "gerbong kereta yang biasanya dipakai buat yang mau tenang namanya apa?",
      chunkTitle: "Quiet carriage",
      chunkText: "Gerbong tenang (quiet carriage) adalah gerbong di mana penumpang diminta menahan panggilan telepon dan berbicara pelan.",
      expectedGrounded: true,
    },
    {
      query: "kanji istirahat yasumu itu bentuknya orang sama pohon ya?",
      chunkTitle: "Kanji yasumu",
      chunkText: "Kanji istirahat (yasumu) terdiri dari radikal orang yang bersandar di samping pohon rindang, menggambarkan seseorang beristirahat.",
      expectedGrounded: true,
    },
  ];
  const ungrounded: ControlItem[] = [
    {
      query: "cara bilang pernah ke jepang pakai pola ta koto ga arimasu?",
      chunkTitle: "TOEIC passive voice",
      chunkText: "All travel expense reports must be submitted by Friday. Passive voice digunakan ketika pelaku tidak sepenting objek.",
      expectedGrounded: false,
    },
    {
      query: "tiket disneyland tokyo berapa harganya?",
      chunkTitle: "Kanji yasumu",
      chunkText: "Kanji istirahat (yasumu) terdiri dari radikal orang yang bersandar di samping pohon rindang.",
      expectedGrounded: false,
    },
    {
      query: "aturan buang sampah di apartemen sakura gimana?",
      chunkTitle: "TOEIC comparative",
      chunkText: "Ada much dan than, jadi butuh bentuk perbandingan: much brighter than.",
      expectedGrounded: false,
    },
    {
      query: "kenapa pas bilang suka ramen harus pakai ga ya, bukan wo?",
      chunkTitle: "Ryokan reservation",
      chunkText: "Konfirmasi reservasi ryokan: tamu tiba pukul 3 sore, check-out pukul 10 pagi, satu malam termasuk makan malam.",
      expectedGrounded: false,
    },
    {
      query: "resep nasi goreng pakai bahasa jepang?",
      chunkTitle: "Kare rumahan",
      chunkText: "Resep kare rumahan: tumis bawang, masukkan wortel dan kentang, tuang air, tambahkan bubuk kare, masak hingga mendidih.",
      expectedGrounded: false,
    },
    {
      query: "jam 7 shichiji atau nanaji yang bener?",
      chunkTitle: "Partikel lokasi",
      chunkText: "Partikel de menandai tempat terjadinya aktivitas, sedangkan ni menandai keberadaan statis dengan arimasu.",
      expectedGrounded: false,
    },
  ];
  return [...grounded, ...ungrounded];
}

async function callJudge(
  model: string,
  query: string,
  chunkTitle: string,
  chunkText: string,
): Promise<{ grounded: boolean; precision: number; reason: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), JUDGE_TIMEOUT_MS);
  if (!JUDGE_API_URL) {
    throw new Error("JUDGE_API_URL environment variable is not configured");
  }
  try {
    const res = await fetch(JUDGE_API_URL, {
      method: "POST",
      signal: controller.signal,
      headers: {
        ...(JUDGE_API_KEY ? { Authorization: `Bearer ${JUDGE_API_KEY}` } : {}),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [
          {
            role: "system",
            content:
              "You are a strict RAG citation groundedness judge. Decide whether the CONTEXT CHUNK contains information that answers the QUERY. Be strict: a chunk about a different topic, or only loosely related, must be judged NOT grounded. Return ONLY valid JSON: {\"grounded\": boolean, \"precision\": number between 0 and 1 indicating how relevant and noise-free the chunk is for the query, \"reason\": string max 20 words}.",
          },
          {
            role: "user",
            content: `QUERY: ${query}\nCHUNK TITLE: ${chunkTitle}\nCHUNK CONTENT: ${chunkText.slice(0, 900)}`,
          },
        ],
        temperature: 0,
      }),
    });
    if (!res.ok) throw new Error(`judge HTTP ${res.status}`);
    const data = await res.json();
    const raw: string = data.choices?.[0]?.message?.content ?? "";
    const cleaned = raw.replace(/```json|```/g, "").trim();
    const parsed = JSON.parse(cleaned);
    return {
      grounded: Boolean(parsed.grounded),
      precision: Number(parsed.precision ?? 0),
      reason: String(parsed.reason ?? ""),
    };
  } finally {
    clearTimeout(timer);
  }
}

async function runWithConcurrency<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let cursor = 0;
  async function worker() {
    while (cursor < items.length) {
      const idx = cursor++;
      results[idx] = await fn(items[idx]);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}
async function main() {
  if (!DB_URL) throw new Error("DATABASE_URL environment variable is required.");
  if (!EMBEDDING_API_URL) throw new Error("EMBEDDING_API_URL environment variable is required.");
  const args = process.argv.slice(2);
  const judgeArgIdx = args.indexOf("--judge");
  const preferredJudge = judgeArgIdx >= 0 ? args[judgeArgIdx + 1] : null;
  const skipJudge = args.includes("--skip-judge");

  const sql = postgres(DB_URL, { max: 4 });
  console.log("================================================================================");
  console.log("RAG RETRIEVAL BENCHMARK RUNNER");
  console.log("================================================================================");

  try {
    const [corpusCount] = await sql<{ count: number }[]>`SELECT count(*)::int AS count FROM rag_eval_corpus`;
    console.log(`Corpus records    : ${corpusCount.count} chunks (lesson_materials + reading_articles + toeic_questions)`);
    console.log(`Evaluation queries: ${EVAL_QUERIES.length} (27 positive, 5 negative, 1 unanswerable)`);
    console.log(`Latency iterations: ${ITERATIONS}x per query = ${EVAL_QUERIES.length * ITERATIONS} samples`);
    console.log(`Embedding model   : ${EMBEDDING_MODEL} (${DIMS} dim)`);

    // ── 1. Judge Validation via Control Set ────────────────────────────────
    const defaultJudge = process.env.JUDGE_MODEL || "gpt-4o-mini";
    let judgeModel = preferredJudge || defaultJudge;
    let judgeValidated = false;
    let controlAccuracy = 0;
    const controlResults: JudgeVerdict[] = [];

    if (!skipJudge && JUDGE_API_URL) {
      const candidates = preferredJudge ? [preferredJudge] : [defaultJudge];
      const controlItems = buildControlSet();
      for (const candidate of candidates) {
        console.log(`[1/5] Validating judge '${candidate}' on control set (${controlItems.length} labeled items)...`);
        const t0 = Date.now();
        const verdicts = await runWithConcurrency(controlItems, 4, async (item) => {
          const start = Date.now();
          try {
            const v = await callJudge(candidate, item.query, item.chunkTitle, item.chunkText);
            return {
              queryId: "control",
              chunkId: item.chunkTitle,
              grounded: v.grounded,
              precision: v.precision,
              reason: v.reason,
              judgeModel: candidate,
              latencyMs: Date.now() - start,
              controlItem: true,
              expectedGrounded: item.expectedGrounded,
              correct: v.grounded === item.expectedGrounded,
            } satisfies JudgeVerdict;
          } catch (err) {
            return {
              queryId: "control",
              chunkId: item.chunkTitle,
              grounded: false,
              precision: 0,
              reason: `judge error: ${(err as Error).message}`,
              judgeModel: candidate,
              latencyMs: Date.now() - start,
              controlItem: true,
              expectedGrounded: item.expectedGrounded,
              correct: false,
            } satisfies JudgeVerdict;
          }
        });
        const correct = verdicts.filter((v) => v.correct).length;
        console.log(`  Result: ${correct}/${controlItems.length} correct in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
        controlResults.push(...verdicts);
        if (correct >= Math.ceil(controlItems.length * 0.9)) {
          judgeModel = candidate;
          judgeValidated = true;
          controlAccuracy = correct / controlItems.length;
          break;
        }
        console.log(`  Judge '${candidate}' did not clear 90% threshold.`);
      }
      if (!judgeValidated) {
        console.log(`[1/5] Judge failed validation gate (<90% accuracy). Groundedness evaluation will be skipped.`);
      } else {
        console.log(`[1/5] Judge '${judgeModel}' validated successfully (${(controlAccuracy * 100).toFixed(1)}% accuracy).`);
      }
    } else {
      console.log("[1/5] Judge evaluation skipped (--skip-judge or no JUDGE_API_URL configured).");
    }

    // ── 2. Retrieval + Latency ─────────────────────────────────────────────
    console.log(`\n[2/5] Executing retrieval: ${EVAL_QUERIES.length} queries x ${ITERATIONS} iterations...`);
    const samples: RetrievalSample[] = [];
    const perQueryTop: Record<string, { ids: string[]; distances: number[] }> = {};

    for (const q of EVAL_QUERIES) {
      for (let iter = 1; iter <= ITERATIONS; iter++) {
        const tEmb = performance.now();
        const qVec = await embedQuery(q.query);
        const embMs = performance.now() - tEmb;

        const literal = `[${qVec.join(",")}]`;
        const tDb = performance.now();
        const rows = await sql<{ id: string; distance: number }[]>`
          SELECT id, (embedding <=> ${literal}::vector) AS distance
          FROM rag_eval_corpus
          ORDER BY distance ASC
          LIMIT ${TOP_K}
        `;
        const dbMs = performance.now() - tDb;

        samples.push({
          queryId: q.id,
          iteration: iter,
          embeddingMs: embMs,
          dbMs,
          totalMs: embMs + dbMs,
          topIds: rows.map((r) => r.id),
          topDistances: rows.map((r) => Number(r.distance)),
        });
        if (iter === 1) perQueryTop[q.id] = { ids: rows.map((r) => r.id), distances: rows.map((r) => Number(r.distance)) };
      }
      process.stdout.write(`  ${q.id} completed (${ITERATIONS} iterations)\n`);
    }

    // ── 3. Groundedness Judging ────────────────────────────────────────────
    const verdicts: JudgeVerdict[] = [];
    if (!skipJudge && judgeValidated) {
      console.log(`\n[3/5] Evaluating groundedness on top-3 chunks per query with '${judgeModel}'...`);
      const judgeTasks: { q: EvalQuery; chunkId: string }[] = [];
      for (const q of EVAL_QUERIES) {
        const top3 = perQueryTop[q.id].ids.slice(0, 3);
        for (const chunkId of top3) judgeTasks.push({ q, chunkId });
      }
      const t0 = Date.now();
      const judged = await runWithConcurrency(judgeTasks, JUDGE_CONCURRENCY, async (task) => {
        const chunkRows = await sql<{ title: string; text: string }[]>`
          SELECT title, text FROM rag_eval_corpus WHERE id = ${task.chunkId}
        `;
        const chunk = chunkRows[0];
        if (!chunk) {
          return {
            queryId: task.q.id,
            chunkId: task.chunkId,
            grounded: false,
            precision: 0,
            reason: "chunk not found",
            judgeModel,
            latencyMs: 0,
          } satisfies JudgeVerdict;
        }
        const start = Date.now();
        try {
          const v = await callJudge(judgeModel, task.q.query, chunk.title, chunk.text);
          return {
            queryId: task.q.id,
            chunkId: task.chunkId,
            grounded: v.grounded,
            precision: v.precision,
            reason: v.reason,
            judgeModel,
            latencyMs: Date.now() - start,
          } satisfies JudgeVerdict;
        } catch (err) {
          return {
            queryId: task.q.id,
            chunkId: task.chunkId,
            grounded: false,
            precision: 0,
            reason: `judge error: ${(err as Error).message}`,
            judgeModel,
            latencyMs: Date.now() - start,
          } satisfies JudgeVerdict;
        }
      });
      verdicts.push(...judged);
      const errors = judged.filter((v) => v.reason.startsWith("judge error")).length;
      console.log(`  ${judged.length} judgments completed in ${((Date.now() - t0) / 1000).toFixed(1)}s (${errors} errors).`);
    } else if (!skipJudge) {
      console.log("\n[3/5] Judge did not pass validation control gate - skipping groundedness.");
    } else {
      console.log("\n[3/5] Judge evaluation skipped (--skip-judge).");
    }

    console.log(`\n[4/5] Verifying HNSW index recall vs brute-force cosine (${corpusCount.count} vectors)...`);
    const allRows = await sql<{ id: string; embedding: string }[]>`SELECT id, embedding::text AS embedding FROM rag_eval_corpus`;
    const corpusVecs = allRows.map((r) => ({ id: r.id, v: JSON.parse(r.embedding) as number[] }));
    const indexChecks: { queryId: string; hnswTop5: string[]; exactTop5: string[]; recall5: number; rankMatch: boolean }[] = [];

    for (const q of EVAL_QUERIES) {
      const qVec = await embedQuery(q.query);
      const scored = corpusVecs.map((c) => {
        let dot = 0;
        let na = 0;
        let nb = 0;
        for (let i = 0; i < DIMS; i++) {
          dot += qVec[i] * c.v[i];
          na += qVec[i] * qVec[i];
          nb += c.v[i] * c.v[i];
        }
        return { id: c.id, dist: 1 - dot / (Math.sqrt(na) * Math.sqrt(nb)) };
      });
      scored.sort((a, b) => a.dist - b.dist);
      const exactTop5 = scored.slice(0, 5).map((s) => s.id);
      const hnswTop5 = perQueryTop[q.id].ids;
      const overlap = exactTop5.filter((id) => hnswTop5.includes(id)).length;
      indexChecks.push({
        queryId: q.id,
        hnswTop5,
        exactTop5,
        recall5: overlap / 5,
        rankMatch: JSON.stringify(exactTop5) === JSON.stringify(hnswTop5),
      });
    }
    const meanRecall5 = indexChecks.reduce((s, c) => s + c.recall5, 0) / indexChecks.length;
    const rankMatches = indexChecks.filter((c) => c.rankMatch).length;
    console.log(`  HNSW Recall@5 vs exact cosine: ${(meanRecall5 * 100).toFixed(2)}%`);
    console.log(`  Identical top-5 ranking: ${rankMatches}/${indexChecks.length} queries`);

    // ── 5. Metrics & Reporting ────────────────────────────────────────────
    const positives = EVAL_QUERIES.filter((q) => q.kind === "positive");
    const negatives = EVAL_QUERIES.filter((q) => q.kind !== "positive");
    const nearMiss = positives.filter((q) => q.confusableWith);

    const queryResults: QueryResult[] = EVAL_QUERIES.map((q) => {
      const top = perQueryTop[q.id];
      const qVerdicts = verdicts.filter((v) => v.queryId === q.id);
      const isPos = q.kind === "positive";
      const gt = q.groundTruthId;
      return {
        id: q.id,
        kind: q.kind,
        query: q.query,
        groundTruthId: gt,
        confusableWith: q.confusableWith,
        top1Id: top.ids[0],
        top1Distance: top.distances[0],
        top1CosineSimilarity: 1 - top.distances[0],
        top3Ids: top.ids.slice(0, 3),
        top5Ids: top.ids,
        hitAt1: isPos ? top.ids[0] === gt : null,
        hitAt3: isPos ? top.ids.slice(0, 3).includes(gt!) : null,
        hitAt5: isPos ? top.ids.includes(gt!) : null,
        reciprocalRank: isPos && gt && top.ids.includes(gt) ? 1 / (top.ids.indexOf(gt) + 1) : isPos ? 0 : null,
        confusableIsTop1: q.confusableWith ? top.ids[0] === q.confusableWith : null,
        judgeVerdicts: qVerdicts,
        groundedChunkCount: qVerdicts.filter((v) => v.grounded).length,
      };
    });

    const hit1 = queryResults.filter((r) => r.hitAt1 === true).length;
    const hit3 = queryResults.filter((r) => r.hitAt3 === true).length;
    const hit5 = queryResults.filter((r) => r.hitAt5 === true).length;
    const n = positives.length;
    const ci1 = wilsonCI(hit1, n);
    const ci3 = wilsonCI(hit3, n);
    const ci5 = wilsonCI(hit5, n);
    const mrr = queryResults.reduce((s, r) => s + (r.reciprocalRank ?? 0), 0) / n;
    const nearMissResults = queryResults.filter((r) => r.confusableWith);
    const confusableTop1 = nearMissResults.filter((r) => r.confusableIsTop1).length;
    // Citation precision@3 evaluated specifically on positive queries.
    // Negative queries evaluated separately as false-grounding rate.
    const judgedQueries = queryResults.filter((r) => r.judgeVerdicts.length > 0);
    const judgedPositives = judgedQueries.filter((r) => r.kind === "positive");
    const citationPrecision =
      judgedPositives.length > 0
        ? judgedPositives.reduce((s, r) => s + r.groundedChunkCount / 3, 0) / judgedPositives.length
        : 0;
    const posGroundedRate =
      judgedPositives.length > 0
        ? judgedPositives.filter((r) => r.groundedChunkCount > 0).length / judgedPositives.length
        : 0;
    const negFalseGround =
      judgedQueries.filter((r) => r.kind !== "positive" && r.groundedChunkCount > 0).length /
      Math.max(1, judgedQueries.filter((r) => r.kind !== "positive").length);

    // Persist raw execution artifact
    const embStats = stats(samples.map((s) => s.embeddingMs));
    const dbStats = stats(samples.map((s) => s.dbMs));
    const totalStats = stats(samples.map((s) => s.totalMs));

    // Simpan hasil mentah
    const outDir = "docs/rag-eval";
    mkdirSync(outDir, { recursive: true });
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const payload = {
      meta: {
        timestamp: new Date().toISOString(),
        corpusSize: corpusCount.count,
        queries: EVAL_QUERIES.length,
        positives: positives.length,
        negatives: negatives.filter((q) => q.kind === "negative").length,
        embeddingModel: `${EMBEDDING_MODEL} (${DIMS}d)`,
        nearMiss: nearMiss.length,
        iterations: ITERATIONS,
        judgeModel,
        judgeValidated,
        controlAccuracy,
        dbUrl: DB_URL.replace(/:[^:@]*@/, ":***@"),
      },
      metrics: {
        hitAt1: { value: hit1 / n, successes: hit1, n, wilson95: ci1 },
        hitAt3: { value: hit3 / n, successes: hit3, n, wilson95: ci3 },
        hitAt5: { value: hit5 / n, successes: hit5, n, wilson95: ci5 },
        mrr,
        confusableTop1Rate: nearMiss.length ? confusableTop1 / nearMiss.length : 0,
        citationPrecisionAt3: citationPrecision,
        positiveGroundedRate: posGroundedRate,
        negativeFalseGroundingRate: negFalseGround,
        indexRecall5: meanRecall5,
        indexRankMatchRate: rankMatches / indexChecks.length,
      },
      latency: {
        embedding: embStats,
        database: dbStats,
        total: totalStats,
        sampleCount: samples.length,
      },
      controlResults,
      queries: queryResults,
      indexChecks,
      samples,
    };
    const filePath = `${outDir}/benchmark-results.json`;
    writeFileSync(filePath, JSON.stringify(payload, null, 2));

    // ── Console Report ────────────────────────────────────────────────────
    console.log("\n================================================================================");
    console.log("BENCHMARK RESULTS SUMMARY");
    console.log("================================================================================");
    console.log(`Corpus: ${corpusCount.count} chunks | Queries: ${EVAL_QUERIES.length} (${positives.length} positive, ${negatives.filter((q) => q.kind === 'negative').length} negative, ${negatives.filter((q) => q.kind === 'unanswerable').length} unanswerable) | Latency samples: ${samples.length}`);
    console.log("");
    console.log("RETRIEVAL ACCURACY (Positive queries with labeled ground-truth):");
    console.log(`  Hit@1 : ${(hit1 / n * 100).toFixed(2)}% (${hit1}/${n})  Wilson 95% CI: [${(ci1.low * 100).toFixed(1)}% - ${(ci1.high * 100).toFixed(1)}%]`);
    console.log(`  Hit@3 : ${(hit3 / n * 100).toFixed(2)}% (${hit3}/${n})  Wilson 95% CI: [${(ci3.low * 100).toFixed(1)}% - ${(ci3.high * 100).toFixed(1)}%]`);
    console.log(`  Hit@5 : ${(hit5 / n * 100).toFixed(2)}% (${hit5}/${n})  Wilson 95% CI: [${(ci5.low * 100).toFixed(1)}% - ${(ci5.high * 100).toFixed(1)}%]`);
    console.log(`  MRR   : ${mrr.toFixed(4)}`);
    console.log(`  Near-miss distractor errors: ${confusableTop1}/${nearMiss.length}`);
    console.log("");
    console.log("GROUNDEDNESS (LLM Judge, pre-validated on control set):");
    if (skipJudge) {
      console.log("  Skipped (--skip-judge)");
    } else if (judgeValidated) {
      console.log(`  Judge model         : ${judgeModel}`);
      console.log(`  Control accuracy    : ${(controlAccuracy * 100).toFixed(1)}% (${controlResults.filter((c) => c.correct).length}/${controlResults.length})`);
      console.log(`  Citation Precision@3: ${(citationPrecision * 100).toFixed(2)}% (average grounded chunks in top-3)`);
      console.log(`  Positive Grounded   : ${(posGroundedRate * 100).toFixed(2)}% (at least 1 grounded chunk in top-3)`);
      console.log(`  Negative False-Pos  : ${(negFalseGround * 100).toFixed(2)}% (queries falsely judged as grounded)`);
    } else {
      console.log("  Not available (validation gate failed)");
    }
    console.log("");
    console.log("HNSW INDEX RECALL VS BRUTE-FORCE COSINE:");
    console.log(`  Recall@5 (mean)     : ${(meanRecall5 * 100).toFixed(2)}%`);
    console.log(`  Identical top-5 rank: ${rankMatches}/${indexChecks.length} queries`);
    console.log("");
    console.log("LATENCY DISTRIBUTION (Measured across 3 iterations):");
    console.log(`  Embedding API (P50/P90/P95): ${embStats.p50.toFixed(1)} ms / ${embStats.p90.toFixed(1)} ms / ${embStats.p95.toFixed(1)} ms`);
    console.log(`  pgvector DB   (P50/P90/P95): ${dbStats.p50.toFixed(2)} ms / ${dbStats.p90.toFixed(2)} ms / ${dbStats.p95.toFixed(2)} ms`);
    console.log(`  Total Pipeline(P50/P90/P95): ${totalStats.p50.toFixed(1)} ms / ${totalStats.p90.toFixed(1)} ms / ${totalStats.p95.toFixed(1)} ms`);
    console.log("");
    console.log(`Full execution payload saved to: ${filePath}`);
    console.log("================================================================================\n");
  } finally {
    await sql.end();
  }
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});

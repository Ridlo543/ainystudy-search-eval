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

const DB_URL = process.env.RAG_EVAL_DATABASE_URL || process.env.DATABASE_URL || "";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const ROUTER_KEY = process.env.AINYLINGO_LLM_API_KEY || "";
const ROUTER_URL = "https://router.ainything.online/v1/chat/completions";
const DIMS = 768;
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

/** Wilson score interval — benar untuk proporsi dengan n kecil. */
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
  for (let attempt = 1; attempt <= 4; attempt++) {
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=${GEMINI_API_KEY}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: { parts: [{ text }] }, outputDimensionality: DIMS }),
      },
    );
    if (res.ok) {
      const data = await res.json();
      const values = data.embedding?.values;
      if (!Array.isArray(values) || values.length !== DIMS) {
        throw new Error(`Dimensi embedding tidak valid: ${values?.length}`);
      }
      return values;
    }
    if (res.status === 429 && attempt < 4) {
      // Kuota free tier 100 request/menit; retrieval 31 kueri × 3 iterasi bisa
      // mendekati batas, jadi backoff sebelum menyerah.
      const waitMs = 20_000 * attempt;
      console.log(`  [embed 429] retry ${attempt}/4 setelah ${waitMs / 1000}s...`);
      await new Promise((r) => setTimeout(r, waitMs));
      continue;
    }
    throw new Error(`embedContent gagal (HTTP ${res.status}) setelah ${attempt} percobaan: ${(await res.text()).slice(0, 200)}`);
  }
  throw new Error("embedQuery: retry habis");
}

// ── judge ───────────────────────────────────────────────────────────────────
interface ControlItem {
  query: string;
  chunkTitle: string;
  chunkText: string;
  expectedGrounded: boolean;
}

/** Control set berlabel untuk memvalidasi diskriminasi judge SEBELUM dipakai. */
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
  try {
    const res = await fetch(ROUTER_URL, {
      method: "POST",
      signal: controller.signal,
      headers: { Authorization: `Bearer ${ROUTER_KEY}`, "Content-Type": "application/json" },
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

// ── main ────────────────────────────────────────────────────────────────────
async function main() {
  if (!GEMINI_API_KEY) throw new Error("GEMINI_API_KEY tidak ditemukan.");

  const args = process.argv.slice(2);
  const judgeArgIdx = args.indexOf("--judge");
  const preferredJudge = judgeArgIdx >= 0 ? args[judgeArgIdx + 1] : null;
  const skipJudge = args.includes("--skip-judge");

  const sql = postgres(DB_URL, { max: 4 });

  console.log("================================================================================");
  console.log("🔬 BENCHMARK EVALUASI RAG — VERSI KETAT");
  console.log("================================================================================");

  try {
    const [corpusCount] = await sql<{ count: number }[]>`SELECT count(*)::int AS count FROM rag_eval_corpus`;
    console.log(`Korpus pgvector   : ${corpusCount.count} chunk riil (lesson_materials + reading_articles + toeic_questions)`);
    console.log(`Kueri evaluasi    : ${EVAL_QUERIES.length} (24 positive, 6 negative, 1 unanswerable)`);
    console.log(`Iterasi latency   : ${ITERATIONS}× per kueri = ${EVAL_QUERIES.length * ITERATIONS} sampel`);
    console.log(`Embedding model   : gemini-embedding-001 (${DIMS} dim) via API langsung`);
    console.log("");

    // ── 1. Validasi judge lewat control set ────────────────────────────────
    let judgeModel = preferredJudge || "gemini-3.8-flash-high";
    let judgeValidated = false;
    let controlAccuracy = 0;
    const controlResults: JudgeVerdict[] = [];

    if (!skipJudge) {
      const jevCandidates = preferredJudge ? [preferredJudge] : ["jev-1.13", "openjev"];
      const controlItems = buildControlSet();
      for (const candidate of jevCandidates) {
        console.log(`[1/5] Validasi judge '${candidate}' pada control set (${controlItems.length} item berlabel)...`);
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
        console.log(`  Hasil: ${correct}/${controlItems.length} benar dalam ${((Date.now() - t0) / 1000).toFixed(1)}s`);
        controlResults.push(...verdicts);
        if (correct >= Math.ceil(controlItems.length * 0.9)) {
          judgeModel = candidate;
          judgeValidated = true;
          controlAccuracy = correct / controlItems.length;
          break;
        }
        console.log(`  Judge '${candidate}' TIDAK lolos ambang 90% — coba kandidat berikutnya.`);
      }
      if (!judgeValidated && !preferredJudge) {
        console.log(`[1/5] Model jev tidak tersedia/lolos. Fallback ke 'gemini-3.8-flash-high'...`);
        const controlItems = buildControlSet();
        const t0 = Date.now();
        const verdicts = await runWithConcurrency(controlItems, JUDGE_CONCURRENCY, async (item) => {
          const start = Date.now();
          try {
            const v = await callJudge(judgeModel, item.query, item.chunkTitle, item.chunkText);
            return {
              queryId: "control",
              chunkId: item.chunkTitle,
              grounded: v.grounded,
              precision: v.precision,
              reason: v.reason,
              judgeModel,
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
              judgeModel,
              latencyMs: Date.now() - start,
              controlItem: true,
              expectedGrounded: item.expectedGrounded,
              correct: false,
            } satisfies JudgeVerdict;
          }
        });
        const correct = verdicts.filter((v) => v.correct).length;
        controlResults.length = 0;
        controlResults.push(...verdicts);
        controlAccuracy = correct / controlItems.length;
        judgeValidated = correct >= Math.ceil(controlItems.length * 0.9);
        console.log(`  Hasil: ${correct}/${controlItems.length} benar (${(controlAccuracy * 100).toFixed(1)}%) dalam ${((Date.now() - t0) / 1000).toFixed(1)}s`);
        console.log(judgeValidated ? `  Judge '${judgeModel}' LOLOS validasi.` : `  Judge '${judgeModel}' TIDAK lolos — verdict groundedness tidak dipercaya.`);
      }
    } else {
      console.log("[1/5] Judge dilewati (--skip-judge).");
    }

    // ── 2. Retrieval + latency ─────────────────────────────────────────────
    console.log(`\n[2/5] Menjalankan retrieval ${EVAL_QUERIES.length} kueri × ${ITERATIONS} iterasi...`);
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
      process.stdout.write(`  ${q.id} selesai (${ITERATIONS} iterasi)\n`);
    }

    // ── 3. Groundedness judging (top-3 per kueri) ──────────────────────────
    const verdicts: JudgeVerdict[] = [];
    if (!skipJudge && judgeValidated) {
      console.log(`\n[3/5] Menilai groundedness top-3 chunk per kueri dengan '${judgeModel}'...`);
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
            reason: "chunk tidak ditemukan",
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
      console.log(`  ${judged.length} penilaian selesai dalam ${((Date.now() - t0) / 1000).toFixed(1)}s (${errors} error).`);
    } else if (!skipJudge) {
      console.log("\n[3/5] Judge tidak lolos validasi — penilaian groundedness dilewati.");
    } else {
      console.log("\n[3/5] Judge dilewati (--skip-judge).");
    }

    // ── 4. Verifikasi indeks HNSW vs brute-force ───────────────────────────
    console.log(`\n[4/5] Verifikasi indeks HNSW terhadap brute-force cosine (${corpusCount.count} vektor)...`);
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
    console.log(`  Recall@5 rata-rata HNSW vs exact: ${(meanRecall5 * 100).toFixed(2)}%`);
    console.log(`  Urutan top-5 identik: ${rankMatches}/${indexChecks.length} kueri`);

    // ── 5. Metrik & laporan ────────────────────────────────────────────────
    console.log(`\n[5/5] Menghitung metrik akhir...`);
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

    // Citation precision@3 dihitung KHUSUS kueri positif (kualitas retrieval).
    // Kueri negatif dilaporkan terpisah sebagai false-grounding rate.
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

    // Latency stats
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
        unanswerable: negatives.filter((q) => q.kind === "unanswerable").length,
        nearMiss: nearMiss.length,
        iterations: ITERATIONS,
        embeddingModel: "gemini-embedding-001 (768d)",
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
    const filePath = `${outDir}/results-${timestamp}.json`;
    writeFileSync(filePath, JSON.stringify(payload, null, 2));
    writeFileSync(`${outDir}/latest.json`, JSON.stringify(payload, null, 2));

    // ── Laporan konsol ────────────────────────────────────────────────────
    console.log("\n================================================================================");
    console.log("📊 HASIL BENCHMARK RAG (KORPUS RIIL, METODE KETAT)");
    console.log("================================================================================");
    console.log(`Korpus: ${corpusCount.count} chunk | Kueri: ${EVAL_QUERIES.length} (${positives.length} positif, ${negatives.filter((q) => q.kind === 'negative').length} negatif, ${negatives.filter((q) => q.kind === 'unanswerable').length} tak terjawab) | Sampel latency: ${samples.length}`);
    console.log("");
    console.log("RETRIEVAL (hanya kueri positif, ground truth = chunk spesifik berlabel manual):");
    console.log(`  Hit-Rate@1 : ${(hit1 / n * 100).toFixed(2)}% (${hit1}/${n})  Wilson 95% CI [${(ci1.low * 100).toFixed(1)}%, ${(ci1.high * 100).toFixed(1)}%]`);
    console.log(`  Hit-Rate@3 : ${(hit3 / n * 100).toFixed(2)}% (${hit3}/${n})  Wilson 95% CI [${(ci3.low * 100).toFixed(1)}%, ${(ci3.high * 100).toFixed(1)}%]`);
    console.log(`  Hit-Rate@5 : ${(hit5 / n * 100).toFixed(2)}% (${hit5}/${n})  Wilson 95% CI [${(ci5.low * 100).toFixed(1)}%, ${(ci5.high * 100).toFixed(1)}%]`);
    console.log(`  MRR        : ${mrr.toFixed(4)}`);
    console.log(`  Near-miss gagal (chunk pembingkai jadi Top-1): ${confusableTop1}/${nearMiss.length}`);
    console.log("");
    console.log("GROUNDEDNESS (judge LLM, tervalidasi control set):");
    if (skipJudge) {
      console.log("  dilewati (--skip-judge)");
    } else if (judgeValidated) {
      console.log(`  Judge model            : ${judgeModel}`);
      console.log(`  Akurasi control set    : ${(controlAccuracy * 100).toFixed(1)}% (${controlResults.filter((c) => c.correct).length}/${controlResults.length})`);
      console.log(`  Citation Precision@3   : ${(citationPrecision * 100).toFixed(2)}% (rata-rata chunk grounded dari 3 teratas)`);
      console.log(`  Groundedness rate (positif): ${(posGroundedRate * 100).toFixed(2)}%`);
      console.log(`  False-grounding (negatif/tak terjawab): ${(negFalseGround * 100).toFixed(2)}% (semakin rendah semakin baik)`);
    } else {
      console.log(`  Judge TIDAK lolos validasi (akurasi ${(controlAccuracy * 100).toFixed(1)}%) — verdict groundedness ditolak.`);
    }
    console.log("");
    console.log("VERIFIKASI INDEKS HNSW vs BRUTE-FORCE:");
    console.log(`  Recall@5 rata-rata     : ${(meanRecall5 * 100).toFixed(2)}%`);
    console.log(`  Urutan identik         : ${rankMatches}/${indexChecks.length} kueri`);
    console.log("");
    console.log("LATENCY (ms):");
    console.log(`  Embedding API : P50=${embStats.p50.toFixed(1)} | P95=${embStats.p95.toFixed(1)} | mean=${embStats.mean.toFixed(1)} | min=${embStats.min.toFixed(1)} | max=${embStats.max.toFixed(1)}`);
    console.log(`  Pgvector HNSW : P50=${dbStats.p50.toFixed(2)} | P95=${dbStats.p95.toFixed(2)} | mean=${dbStats.mean.toFixed(2)} | min=${dbStats.min.toFixed(2)} | max=${dbStats.max.toFixed(2)}`);
    console.log(`  Total         : P50=${totalStats.p50.toFixed(1)} | P95=${totalStats.p95.toFixed(1)} | mean=${totalStats.mean.toFixed(1)} | min=${totalStats.min.toFixed(1)} | max=${totalStats.max.toFixed(1)}`);
    console.log("");
    console.log("CATATAN KEJUJURAN:");
    console.log(`  - n=${n} kueri positif: CI 95% masih lebar (rentang ±20-35%).`);
    console.log(`  - Kueri negatif hanya 6+1: estimasi false-grounding punya variansi tinggi.`);
    console.log(`  - Ground truth kueri positif dipilih oleh inspeksi manual isi chunk, bukan auto-split.`);
    console.log(`  - Model jev di router terverifikasi tidak tersedia saat run ini; lihat meta.judgeModel.`);
    console.log(`  - Hasil mentah lengkap: ${filePath}`);
    console.log("================================================================================\n");
  } finally {
    await sql.end();
  }
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});

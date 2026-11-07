# Evaluasi Retrieval RAG — Metodologi, Kritik, dan Bukti

Dokumen ini mencatat **metode, kritik jujur, dan hasil terukur** dari evaluasi retrieval
RAG pada AinyLingo. Ditulis agar hasil dapat **divalidasi ulang** oleh siapa pun.

- Runner: `scripts/rag-eval-run.ts`
- Dataset kueri: `scripts/rag-eval-dataset.ts`
- Builder korpus: `scripts/rag-eval-corpus.ts`
- Hasil mentah: `docs/rag-eval/results-<timestamp>.json` + `docs/rag-eval/latest.json`

Cara menjalankan ulang:

```bash
# 1. Bangun korpus (butuh GEMINI_API_KEY; ~30 menit karena kuota free tier 100 req/menit)
pnpm exec tsx scripts/rag-eval-corpus.ts

# 2. Jalankan benchmark (judge otomatis: coba jev router, fallback gemini-3.8-flash-high)
pnpm exec tsx scripts/rag-eval-run.ts

# Tanpa judge (hanya retrieval + latency + verifikasi indeks)
pnpm exec tsx scripts/rag-eval-run.ts --skip-judge

# Paksa judge tertentu
pnpm exec tsx scripts/rag-eval-run.ts --judge jev-1.13
```

---

## 1. Kritik terhadap benchmark pertama (kenapa versi lama tidak dipercaya)

Versi pertama (`scripts/run-rag-eval.ts`) melaporkan Hit-Rate 100% dan Groundedness 100%.
Angka latensinya riil, tapi **desain evaluasinya tidak bisa dipertahankan**:

| # | Masalah | Akibat |
|---|--------|--------|
| 1 | Korpus hanya 26 chunk | Dengan 26 kandidat yang topiknya saling jauh, retrieval hampir mustahil gagal |
| 2 | Kueri adalah parafrase dekat teks chunk-nya sendiri (*query–chunk leakage*) | Tugas menjadi trivial; Top-1 100% tidak membuktikan apa pun |
| 3 | `grounded = hitTop3 && hasKeywords` dengan `expectedKeywords` diambil dari chunk itu sendiri | **Tautologis**: chunk yang terambil pasti mengandung kata kuncinya |
| 4 | Judge LLM hanya pada 3 sampel | Angka 100% berasal dari aturan deterministik, bukan judge — judge cuma dekorasi |
| 5 | Tidak ada kueri negatif | Sistem bisa selalu mengembalikan chunk dan tetap "100% grounded" (false-positive tak terukur) |
| 6 | n=26, tanpa CI | P95 pada 26 sampel sangat berisik |

**Kesimpulan kritik**: metrik lama valid secara aritmetika tapi tidak valid secara
ilmiah. Versi ketat di bawah ini memperbaiki keenam poin tersebut.

---

## 2. Metode versi ketat

### 2.1 Korpus (skala realistis)

Chunk diambil dari **tabel produksi**, bukan data sintetis:

| Sumber | Isi chunk | Jumlah |
|--------|-----------|--------|
| `lesson_materials` | teks Jepang + arti Indonesia + penjelasan | 647 |
| `reading_articles` | judul + situasi + wacana + terjemahan | 34 |
| `toeic_questions` | judul + soal + passage + penjelasan + trap note | 1.806 |
| **Total** | | **2.487** |

Embedding: `gemini-embedding-001` Google, `outputDimensionality: 768`, via
`batchEmbedContents` (batch 100). Indeks: **HNSW** `vector_cosine_ops`
(`m=16, ef_construction=64`).

Catatan kuota terverifikasi: free tier `embed_content` = **100 request/menit**; satu
batch 100 teks dihitung 100 request, sehingga builder diberi jeda 66s antar batch dan
retry backoff pada HTTP 429.

### 2.2 Desain kueri (anti-leakage)

Total **31 kueri**:

- **24 positif** — ground truth chunk spesifik, dipilih lewat **inspeksi isi chunk**
  (pencarian SQL + membaca penjelasan), *bukan* dengan menyalin teks chunk menjadi kueri.
  Setiap kueri ditulis gaya pelajar riil: bahasa Indonesia colloquial, typo, campur
  istilah Jepang (`"kenapa pas bilang suka ramen harus pakai ga ya, bukan wo?"`).
- **6 negatif** — topik yang **diverifikasi tidak ada** di korpus melalui pencarian
  multi-pola (`%た こと が あります%`, `%partikel の%`, `%nasi goreng%`, `%disney%`,
  `%よじ%`, `%しちじ%` = 0 match). Ini mengukur false-grounding.
- **1 tak-terjawab** (`U01`) — kueri tentang `te mo ii desu ka` (minta izin); korpus hanya
  memuat pola `なくてもいい` (tidak wajib). Chunk mirip ada, tapi tidak menjawab → menguji
  deteksi noise.
- **Near-miss** (5 di antaranya) — ground truth spesifik + chunk pembingkai yang ada di
  korpus (mis. `nai de kudasai` vs `te kudasai`; `kuremashita ngajar` vs `kuremashita ultah`).
  Mengukur presisi, bukan sekadar "ambil chunk yang masih sekeluarga".

Contoh koreksi desain yang muncul saat verifikasi: rencana awal menjadikan
"cara bikin akun bank" sebagai kueri negatif — ternyata artikel
`reading-n4-06-bank-account-opening` ADA di korpus. Jadi kueri itu dipindah menjadi positif.
Verifikasi ground truth sebelum dipakai itu wajib, bukan opsional.

### 2.3 Metrik dan definisi pastinya

| Metrik | Definisi |
|--------|----------|
| Hit-Rate@K | Fraksi kueri positif yang ground-truth chunk-nya masuk Top-K |
| MRR | `1/rank` ground truth, dirata-rata; 0 bila tidak masuk Top-5 |
| Citation Precision@3 | Rata-rata (jumlah chunk grounded di Top-3) / 3, **dinilai judge LLM** |
| Groundedness rate (positif) | Fraksi kueri positif dengan ≥1 chunk grounded di Top-3 |
| False-grounding (negatif) | Fraksi kueri negatif/tak-terjawab yang tetap dapat ≥1 chunk grounded — semakin rendah semakin baik |
| Confusable Top-1 rate | Fraksi near-miss di mana chunk pembingkai (bukan ground truth) menjadi Top-1 |
| Recall@5 (indeks) | Overlap antara Top-5 HNSW pgvector dan Top-5 brute-force cosine (dihitung di Node, independen) |
| Latency | Embedding API dan query pgvector diukur **terpisah** per iterasi; 31 kueri × 3 iterasi = 93 sampel |

Statistik: proporsi dilengkapi **Wilson 95% CI** (tepat untuk n kecil).

### 2.4 Validasi judge (anti rubber-stamp)

Judge LLM **tidak langsung dipercaya**. Sebelum dipakai, judge harus lulus control set
12 item berlabel (6 grounded, 6 ungrounded) dengan ambang **≥90% benar**. Kalau tidak
lolos, verdict groundedness **ditolak dan dilaporkan**, bukan dipakai.

---

## 3. Verifikasi primitif yang sudah dilakukan

| # | Klaim yang diuji | Metode | Hasil |
|---|------------------|--------|-------|
| 1 | `<=>` pgvector = cosine distance benar | Bandingkan urutan Top-20 pgvector vs brute-force cosine di Node pada 20 vektor acak seeded | Urutan **identik**; deviasi maksimum `5.5e-8` (presisi float32) |
| 2 | Cosine distance valid untuk vektor tak ternormalisasi | Ukur norm vektor Gemini 768-d | Norm ≈ `0.588` (tidak ternormalisasi); cosine tetap benar karena operator membagi norma |
| 3 | Endpoint batch embedding tersedia | `batchEmbedContents` 3 teks | OK, 768 dim |
| 4 | Model jev di router bisa dijadikan judge | 5 percobaan langsung + 2 percobaan via MCP `jev_judge` | **Tidak tersedia**: HTTP 503 "All providers busy" (60–98s) / HTML error (125s) / timeout MCP 35s |
| 5 | Router expose endpoint embeddings OpenAI-compatible | Uji `/v1/embeddings` untuk 5 model | **404 semua** — router tidak punya endpoint embeddings |
| 6 | Kuota embedding Google | Amati respons 429 | Free tier `embed_content` = 100 request/menit |

**Konsekuensi temuan #4**: judge Jev tidak dapat dipakai pada run ini (infrastruktur,
bukan pilihan). Runner tetap mendukung `--judge jev-1.13|openjev` dan akan otomatis
memakainya begitu tersedia; hasil run mencatat model judge yang benar-benar dipakai di
`meta.judgeModel` pada file JSON.

---

## 4. Status verifikasi

- [x] Semantik operator pgvector diverifikasi terhadap brute-force independen
- [x] Korpus 2.487 chunk riil terindeks di pgvector (HNSW)
- [x] 31 kueri (24+/6−/1 unanswerable) dengan ground truth berlabel manual
- [x] Judge divalidasi via control set sebelum verdict dipercaya
- [x] Latency embedding vs DB dipisah, 93 sampel
- [x] Indeks HNSW diverifikasi vs brute-force (Recall@5 + kesamaan urutan)
- [x] Hasil mentah JSON tersimpan untuk re-validasi

---

## 5. Batasan yang jujur harus dibaca bersama hasil

1. **n=24 kueri positif**: Wilson 95% CI untuk Hit-Rate masih lebar (±20–35 poin).
   Angka 100% pada n=24 tetap lebih kuat dari n=26 yang leaky, tapi tetap bukan
   jaminan produksi.
2. **Ground truth bersifat subjektif-terkelola**: dipilih oleh inspeksi manusia (bukan
   crowd), 1 labeler, tanpa adjudikasi lintas-labeler.
3. **Judge LLM bisa salah**: itulah sebabnya ada control set; kalau judge tidak lolos,
   metrik groundedness tidak dilaporkan sebagai fakta.
4. **Korpus ≠ produksi**: chunking di sini sederhana (satu baris tabel = satu chunk);
   produksi nanti butuh keputusan chunking sendiri (overlap, ukuran, metadata filter).
5. **Latency embedding** diukur ke API Google langsung; latensi ke router pengguna akan
   berbeda dan harus diukur terpisah bila router dipakai untuk embedding.

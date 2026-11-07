/**
 * Evaluation dataset for RAG retrieval benchmarking.
 *
 * Query design notes:
 * 1. Positive queries: realistic conversational questions paired with verified ground-truth chunk IDs.
 * 2. Negative queries: verified absent from the 2,487-chunk database to test false-positive retrieval.
 * 3. Near-miss queries: specifically paired with confusable chunks to evaluate disambiguation accuracy.
 */

export type QueryKind = "positive" | "negative" | "unanswerable";

export interface EvalQuery {
  id: string;
  kind: QueryKind;
  /** Natural learner-style query phrasing (not verbatim chunk text). */
  query: string;
  /** Ground-truth chunk ID; null for negative controls. */
  groundTruthId: string | null;
  /** Human-verifiable explanation of why the chunk satisfies the query. */
  groundTruthJustification: string;
  /** Adjacent confusable chunk present in corpus for near-miss discrimination testing. */
  confusableWith?: string;
  confusableNote?: string;
}

export const EVAL_QUERIES: EvalQuery[] = [
  // ── POSITIVE: Japanese Grammar (ground truth = lesson_materials) ─────────
  {
    id: "P01",
    kind: "positive",
    query: "kenapa pas bilang suka ramen harus pakai ga ya, bukan wo?",
    groundTruthId: "232f664d-bb71-4413-8456-0fde6f4ba029",
    groundTruthJustification:
      "Chunk 'わたし は にほんご が すき です' dengan penjelasan: 'Hal yang disukai (Suki) atau dibenci (Kirai) ditandai dengan partikel GA' — menjawab langsung pertanyaan partikel GA vs WO untuk suki.",
  },
  {
    id: "P02",
    kind: "positive",
    query: "gimana cara minta tolong orang nunggu sebentar dengan sopan?",
    groundTruthId: "e3bd244e-5289-425a-a7ad-79499ad3f931",
    groundTruthJustification:
      "Chunk '[Bentuk-Te] + ください' dengan contoh 'Chotto matte kudasai (Tolong tunggu sebentar ya)' — menjawab cara meminta menunggu dengan sopan.",
  },
  {
    id: "P03",
    kind: "positive",
    query: "kalau mau bilang jangan lupa ke teman gimana ya?",
    groundTruthId: "5805912a-135c-4f95-ae46-09b252810df3",
    groundTruthJustification:
      "Chunk '[Bentuk-Nai] + で ください' dengan contoh 'Wasurenaide kudasai (Tolong jangan lupa ya)' — menjawab permintonan 'jangan lupa'.",
    confusableWith: "e3bd244e-5289-425a-a7ad-79499ad3f931",
    confusableNote:
      "Chunk te-kudasai (P02) memakai pola Bentuk-Te + kudasai; mirip secara bentuk tetapi menjawab permintaan LAKUKAN, bukan JANGAN lakukan.",
  },
  {
    id: "P04",
    kind: "positive",
    query: "kenapa keterlambatan gara gara hujan deras kereta mogok dijelasin pakai pola apa?",
    groundTruthId: "45b14e3b-19ff-42f7-8465-acffeabd21dc",
    groundTruthJustification:
      "Chunk bacaan 'けさ、おおあめ が ふって、ちかてつ が とまりました。その ため...' dengan penjelasan hubungan sebab-akibat — menjawab pola penyebab akibat hujan/kereta mogok.",
  },
  {
    id: "P05",
    kind: "positive",
    query: "teman ngajarin aku bahasa jepang, kenapa pakai oshiete kuremashita?",
    groundTruthId: "e5dc79bf-6df8-408e-b568-7983a0011e5b",
    groundTruthJustification:
      "Chunk 'さとうさん は わたし に にほんご を おしえて くれました' dengan penjelasan pola kuremashita saat orang lain berbuat baik untuk pembicara — menjawab arah kebaikan masuk ke diri pembicara.",
    confusableWith: "9258b8d2-a076-4a19-bea2-565eca4cba10",
    confusableNote:
      "Chunk P06 juga memakai kuremashita tetapi konteksnya hadiah ulang tahun, bukan diajari bahasa.",
  },
  {
    id: "P06",
    kind: "positive",
    query: "ibu buatin kue ultah, pakai tsukutte kuremashita ya?",
    groundTruthId: "9258b8d2-a076-4a19-bea2-565eca4cba10",
    groundTruthJustification:
      "Chunk bacaan ulang tahun dengan penjelasan pola 'tsukutte kuremashita' dan 'okutte kuremashita' — menjawab kebaikan orang lain berupa pembuatan kue.",
  },
  {
    id: "P07",
    kind: "positive",
    query: "cara nawarin diri buat ngerjain pekerjaan ke atasan?",
    groundTruthId: "2a73292f-b30c-460f-b5fa-3e1c0c8236a9",
    groundTruthJustification:
      "Chunk 'わたし に その しごと を やらせて ください' dengan penjelasan pola menawarkan diri (sasete kudasai) — menjawab penawaran diri mengerjakan tugas.",
  },
  {
    id: "P08",
    kind: "positive",
    query: "kanji istirahat yasumu itu bentuknya orang sama pohon ya? ceritanya gimana",
    groundTruthId: "ebdc9fa4-42f6-4296-9746-a7035b639008",
    groundTruthJustification:
      "Chunk kanji '休' dengan penjelasan 'Seseorang (亻) bersandar santai di samping pohon rindang (木)' — menjawab asosiasi visual kanji istirahat.",
  },
  {
    id: "P09",
    kind: "positive",
    query: "mau cerita kegiatan minggu kayak baca buku denger musik gitu caranya?",
    groundTruthId: "161a0f56-359f-4ce9-b3ee-248c4b59da44",
    groundTruthJustification:
      "Bacaan penutup N5 yang memadukan pola '~tari ~tari shitai desu' — menjawab cara menyebut kegiatan contoh.",
  },
  {
    id: "P10",
    kind: "positive",
    query: "kalo bilang nggak usah maksak diri gimana?",
    groundTruthId: "2f49d1e0-0337-4e76-8405-e1e60a492072",
    groundTruthJustification:
      "Chunk '無理をしなくてもいいですよ' dengan penjelasan meyakinkan lawan bicara tidak perlu memaksakan diri — menjawab ungkapan 'nggak usah maksak diri'.",
  },

  // ── POSITIVE: Reading Articles ────────────────────────────────────────────
  {
    id: "P11",
    kind: "positive",
    query: "aturan buang sampah di apartemen sakura gimana?",
    groundTruthId: "reading-n4-01-garbage-rules",
    groundTruthJustification:
      "Artikel N4 'マンションさくらのゴミ出し' membahas aturan pembuangan sampah apartemen Sakura — menjawab langsung.",
    confusableWith: "reading-n4-07-water-tank-cleaning",
    confusableNote:
      "Artikel pengumuman apartemen lain (pemadaman air) juga bernuansa papan informasi penghuni.",
  },
  {
    id: "P12",
    kind: "positive",
    query: "mau buat akun bank sakura harus reservasi dulu?",
    groundTruthId: "reading-n4-06-bank-account-opening",
    groundTruthJustification:
      "Artikel 'さくら銀行の相談予約' membahas pembukaan akun dan reservasi konsultasi bank Sakura — menjawab langsung.",
    confusableWith: "reading-n4-03-ryokan-reservation",
    confusableNote:
      "Artikel reservasi ryokan juga memuat pola konfirmasi janji, tetapi objeknya penginapan, bukan bank.",
  },
  {
    id: "P13",
    kind: "positive",
    query: "gerbong kereta yang biasanya dipakai buat yang mau tenang namanya apa?",
    groundTruthId: "reading-n2-04-quiet-carriage",
    groundTruthJustification:
      "Artikel N2 '静かな車両の試み' membahas gerbong tenang — menjawab nama/konsep gerbong hening.",
  },
  {
    id: "P14",
    kind: "positive",
    query: "ada tempat ngobrol dulu sebelum buang barang rusak, itu apa?",
    groundTruthId: "reading-n3-01-repair-cafe",
    groundTruthJustification:
      "Artikel N3 '捨てる前に相談を' membahas konsultasi perbaikan sebelum membuang barang — menjawab konsep repair cafe.",
  },
  {
    id: "P15",
    kind: "positive",
    query: "cara bikin kare rumahan yang gampang?",
    groundTruthId: "reading-03",
    groundTruthJustification:
      "Artikel N5 'おいしい おうちカレーの つくりかた' — resep kare rumahan.",
    confusableWith: "reading-09",
    confusableNote: "Artikel onigiri juga merupakan resep makanan Jepang sederhana.",
  },
  {
    id: "P16",
    kind: "positive",
    query: "kenapa bangku umum di taman jarang dipakai?",
    groundTruthId: "reading-n1-02-public-bench",
    groundTruthJustification:
      "Artikel N1 '利用率の低いベンチ' membahas bangku publik dengan utilisasi rendah — menjawab penyebab.",
  },
  {
    id: "P17",
    kind: "positive",
    query: "cara bikin onigiri tuna mayo?",
    groundTruthId: "reading-09",
    groundTruthJustification: "Artikel 'かんたん ツナマヨおにぎりの つくりかた' — resep onigiri tuna mayo.",
  },
  {
    id: "P18",
    kind: "positive",
    query: "ketentuan balikin buku perpustakaan apa aja?",
    groundTruthId: "reading-10",
    groundTruthJustification:
      "Artikel 'としょかんの やすみと 本のかえしかた' membahas aturan pengembalian buku perpustakaan.",
  },

  // ── POSITIVE: TOEIC ───────────────────────────────────────────────────────
  {
    id: "P19",
    kind: "positive",
    query: "lobby renovated looks much ___ than before, jawabannya apa?",
    groundTruthId: "t07-r-p5-119",
    groundTruthJustification:
      "Soal 'The renovated lobby looks much ------- than it did before' dengan penjelasan 'much brighter than' — menjawab grammar perbandingan.",
    confusableWith: "t05-r-p5-130",
    confusableNote: "Soal 'similar to' juga grammar kata sifat, tetapi konteksnya perbandingan kesetaraan.",
  },
  {
    id: "P20",
    kind: "positive",
    query: "the shipment was delayed because the customs documents were ___",
    groundTruthId: "t07-r-p5-129",
    groundTruthJustification:
      "Soal 'The shipment was delayed because the customs documents were -------' dengan penjelasan pemilihan kata sifat — menjawab langsung.",
  },
  {
    id: "P21",
    kind: "positive",
    query: "sales figures this quarter are ___ to those of previous, pilih apa?",
    groundTruthId: "t05-r-p5-130",
    groundTruthJustification:
      "Soal 'The sales figures for this quarter are ------- to those of the previous...' dengan penjelasan 'similar to' — menjawab langsung.",
  },
  {
    id: "P22",
    kind: "positive",
    query: "the board has asked the finance department to prepare an ___",
    groundTruthId: "t04-r-p5-128",
    groundTruthJustification:
      "Soal 'The board has asked the finance department to prepare an ------- of ne...' dengan penjelasan 'an estimate' — menjawab langsung.",
  },
  {
    id: "P23",
    kind: "positive",
    query: "kapan reportnya di submit?",
    groundTruthId: "t03-l-p3-38",
    groundTruthJustification:
      "Soal listening 'When was the report submitted?' — menjawab pertanyaan waktu pengumpulan laporan.",
  },
  {
    id: "P24",
    kind: "positive",
    query: "kapan hillcrest kirim produk pertama ke japan?",
    groundTruthId: "t06-r-p7-175",
    groundTruthJustification:
      "Soal 'When will Hillcrest send its first products to Japan?' dengan penjelasan jadwal pengiriman — menjawab langsung.",
  },

  {
    id: "P25",
    kind: "positive",
    query: "partikel no buat nunjuk punya siapa gimana?",
    groundTruthId: "cf5b5192-af85-4614-8b50-8f65db34ecc4",
    groundTruthJustification:
      "Chunk 'これ は だれ の [Benda] です か。' memuat aturan partikel NO untuk kepemilikan (Watashi no kagi).",
  },
  {
    id: "P26",
    kind: "positive",
    query: "jam 7 shichiji atau nanaji yang bener?",
    groundTruthId: "0e773ee5-eeb7-4451-b395-63bb163d2c2c",
    groundTruthJustification:
      "Chunk '半' memuat pelafalan Shichiji-han (jam 7 lewat 30) dan chunk jam memuat 7-ji.",
  },
  {
    id: "P27",
    kind: "positive",
    query: "cara minta izin boleh foto pakai te mo ii desu ka?",
    groundTruthId: "9b98c78a-0632-42f3-bb3a-f482790f1bd5",
    groundTruthJustification:
      "Chunk '[Bentuk-Te] + も いい です か。' memuat pola sopan meminta izin (Bolehkah saya melakukan aksi ini?).",
  },

  // ── NEGATIVE: Verified absent from corpus (0 SQL matches across text patterns) ──
  {
    id: "N01",
    kind: "negative",
    query: "cara bilang pernah ke jepang pakai pola ta koto ga arimasu?",
    groundTruthId: null,
    groundTruthJustification:
      "Verifikasi ketiadaan: pola '%た こと が あります%' di lesson_materials, toeic_questions, reading_articles = 0 match.",
  },
  {
    id: "N02",
    kind: "negative",
    query: "resep nasi goreng pakai bahasa jepang?",
    groundTruthId: null,
    groundTruthJustification: "Verifikasi ketiadaan: '%nasi goreng%' = 0 match di seluruh korpus.",
  },
  {
    id: "N03",
    kind: "negative",
    query: "tiket disneyland tokyo berapa harganya?",
    groundTruthId: null,
    groundTruthJustification: "Verifikasi ketiadaan: '%disney%' = 0 match di seluruh korpus.",
  },
  {
    id: "N04",
    kind: "negative",
    query: "jam 4 yoji dibaca gimana pengecualiannya?",
    groundTruthId: null,
    groundTruthJustification: "Verifikasi ketiadaan: '%よじ%' = 0 match di seluruh korpus.",
  },
  {
    id: "N05",
    kind: "negative",
    query: "syarat dokumen visa kerja tokutei ginou apa saja?",
    groundTruthId: null,
    groundTruthJustification: "Verifikasi ketiadaan: '%tokutei%' = 0 match di seluruh korpus.",
  },

  // ── UNANSWERABLE: Nearest semantic chunk exists but does not answer the question ──
  {
    id: "U01",
    kind: "unanswerable",
    query: "apa perbedaan mendalam partikel wo dan ni pada objek penderita kalimat kausatif aktif?",
    groundTruthId: null,
    groundTruthJustification:
      "Kurikulum N4 hanya memuat contoh permintaan izin kausatif (watashi ni yarasete kudasai), tidak memuat teori tata bahasa perbedaan transitif-intransitif partikel wo vs ni pada kausatif aktif. Menguji deteksi noise.",
  },
];

/** Total 33 queries: 27 positive, 5 negative controls, 1 unanswerable control. */
export const EVAL_SUMMARY = {
  total: EVAL_QUERIES.length,
  positive: EVAL_QUERIES.filter((q) => q.kind === "positive").length,
  negative: EVAL_QUERIES.filter((q) => q.kind === "negative").length,
  unanswerable: EVAL_QUERIES.filter((q) => q.kind === "unanswerable").length,
  nearMiss: EVAL_QUERIES.filter((q) => q.confusableWith).length,
};

# Tahap 2 — hasil ekstraksi landmark

Konfigurasi terpilih adalah `landmarks-v1` (MediaPipe 0.10.21, RGB tanpa CLAHE, tanpa resize, static mode). Keputusan dibekukan setelah eksperimen validation-only di [`pilot_comparison.json`](pilot_comparison.json); identitas konfigurasi dan hash model dicatat di `extraction_lock.json`. Ringkasan agregat dan per kelas tersedia di [`summary.md`](summary.md); data mesin lengkap disimpan pada `summary.json`.

Pilot validation: SIBI baseline valid secara geometri 215/216 (CLAHE 214/216); BISINDO baseline 762/780 (CLAHE 763/780). CLAHE mengubah 17 sampel BISINDO, 9 menjadi valid dan 8 tidak lagi valid, sementara median waktu deteksi lebih tinggi pada sampel pilot untuk kedua bahasa. Validitas di sini bukan akurasi pengenalan. Ekstraksi penuh: SIBI 993/1.001 train, 215/216 validation, 213/216 test; BISINDO 4.235/6.480 train, 762/780 validation, 770/780 test. Rincian dan cakupan per kelas ada di [`summary.md`](summary.md).

Interpretasi penting:

- SIBI tidak memiliki deteksi multi-tangan pada split ini; masing-masing split memiliki 8, 1, dan 3 sampel tanpa tangan terdeteksi (train/validation/test).
- BISINDO mencatat satu atau dua tangan sebagai fitur geometris, tetapi pose mana yang secara resmi memerlukan satu/dua tangan dan siapa pemilik tangan dalam frame belum diverifikasi.
- Split BISINDO berdasarkan seri nama file bukan split peserta terverifikasi. Hasil tidak boleh dinyatakan sebagai generalisasi ke peserta baru.
- Akurasi landmark dan classifier belum diukur. Tahap 3 hanya boleh disebut eksperimen kandidat sampai validasi alfabet/pose dilengkapi.
- Cakupan BISINDO train hanya 65,35%, dibanding 97,69% validation dan 98,72% test. Overlay train memperlihatkan beberapa frame diambil jauh atau berisi beberapa orang; kondisi ini menunjukkan pergeseran data antarsplit dan perlu ditangani sebelum membandingkan classifier.

Untuk mereproduksi, ikuti [`FEATURE_SCHEMA.md`](../../docs/FEATURE_SCHEMA.md). Manifest input terkunci oleh Tahap 1; keluaran fitur, log per gambar, checkpoint, dan overlay detail tetap lokal.

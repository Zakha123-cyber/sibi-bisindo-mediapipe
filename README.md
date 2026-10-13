# Service AI alfabet SIBI dan BISINDO

MVP yang disepakati: mengenali pose huruf statis dari webcam secara realtime, dengan mode SIBI/BISINDO dipilih pengguna. Audit awal serta manifest/protokol evaluasi teknis sudah tersedia; classifier dan service FastAPI belum dibuat.

**Acuan pekerjaan:** [roadmap MVP dan kriteria selesai](docs/MVP_ROADMAP.md). Langkah berikutnya adalah mengunci detektor/preprocessing dan mengekstrak landmark dari manifest aktif, sambil menjaga batas metadata yang belum tersedia.

Lihat [hasil dan keputusan audit](reports/dataset_audit/FINDINGS.md) atau buka [notebook audit](notebooks/01_dataset_audit.ipynb). Ringkasan kecil disimpan di Git; inventory lengkap, daftar duplikasi, serta contoh gambar di `reports/dataset_audit/` dibuat ulang secara lokal dengan skrip audit.

## Menjalankan audit

Gunakan lingkungan Python terpisah bila dependensi belum tersedia:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-audit.txt
.\.venv\Scripts\python scripts/audit_dataset.py
```

Audit membaca dataset dan menulis laporan. Dataset mentah tidak diubah. Pemeriksaan gambar lengkap dapat memakan beberapa menit. Tinjauan label dan sumber otomatis dijalankan setelah inventory selesai.

Untuk mereproduksi sampling detektor yang sudah diuji:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-landmark-audit.txt
.\.venv\Scripts\python scripts/audit_landmarks.py --per-class 20 --seed 42
```

Sampling ini memakai MediaPipe 0.10.21 dengan API legacy yang tersedia saat audit. Pilihan API produksi belum ditetapkan. Deteksi tangan pada sampel bukan ukuran akurasi pengenalan huruf.

Notebook dapat dibuka melalui editor yang mendukung Jupyter dengan kernel dari lingkungan tersebut. Secara default notebook membaca hasil tersimpan; aktifkan flag yang tersedia jika ingin menjalankan audit ulang.

## File untuk GitHub

`.gitignore` mengecualikan dataset, fitur, model, rekaman, environment, secrets, cache, hasil eksperimen, serta laporan detail hasil generate. Kode, dokumentasi, notebook sumber, dan ringkasan audit tetap dapat di-commit.

Notebook sumber disimpan tanpa output besar. Salinan hasil eksekusi lokal berada di `notebooks/executed/` dan diabaikan Git. Setelah clone, contoh gambar dan laporan detail yang ditautkan dalam laporan tersedia setelah audit dijalankan dengan dataset lokal.

## Manifest data dan protokol evaluasi

Lihat [hasil Tahap 1](reports/data_preparation/README.md), [sumber dan atribusi dataset](docs/DATASET_SOURCES.md), serta [protokol evaluasi](docs/EVALUATION_PROTOCOL.md). Keputusan kurasi tersimpan di [curation_decisions_v1.csv](docs/curation_decisions_v1.csv); aturan split ada di [data_v1.json](configs/data_v1.json).

Setelah inventory audit tersedia:

```powershell
python scripts/prepare_data.py --verify-images
python -m unittest discover -s tests -v
```

Manifest rinci dibuat di `data/manifests/data-v1/` dan diabaikan Git. Ringkasan serta lock snapshot disimpan di `reports/data_preparation/`. Baseline SIBI aktif memakai mono tanpa augmentasi. Split BISINDO memisahkan seri nama file; metadata peserta/sesi belum tersedia sehingga hasilnya tidak boleh disebut evaluasi pada peserta berbeda.

Pengulangan dengan input sama menghasilkan manifest identik. Jika input/config mengubah lock, gunakan versi dan direktori keluaran baru dengan keputusan yang terdokumentasi. Belum ada file dataset yang dikoreksi langsung.

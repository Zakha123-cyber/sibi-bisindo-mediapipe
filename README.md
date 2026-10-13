# Service AI alfabet SIBI dan BISINDO

MVP yang disepakati: mengenali pose huruf statis dari webcam secara realtime, dengan mode SIBI/BISINDO dipilih pengguna. Proyek saat ini berada pada tahap audit dataset; classifier dan service FastAPI belum dibuat.

**Acuan pekerjaan:** [roadmap MVP dan kriteria selesai](docs/MVP_ROADMAP.md). Langkah aktif berikutnya adalah membuat manifest data dan protokol evaluasi berdasarkan hasil audit.

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

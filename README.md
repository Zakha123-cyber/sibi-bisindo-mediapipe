# Service AI alfabet SIBI dan BISINDO

MVP yang disepakati: mengenali pose huruf statis dari webcam secara realtime, dengan mode SIBI/BISINDO dipilih pengguna. Audit, manifest, ekstraksi landmark, dan training baseline gambar BISINDO sudah tersedia. Checkpoint BISINDO sudah dievaluasi satu kali pada test; demo webcam lokal menjadi langkah integrasi berikutnya. Model SIBI dan service FastAPI belum terintegrasi.

**Acuan pekerjaan:** [roadmap MVP dan kriteria selesai](docs/MVP_ROADMAP.md). Langkah berikutnya adalah menjalankan baseline SVM/MLP sambil menjaga batas metadata yang belum tersedia.

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

## Tahap 2: ekstraksi landmark

Kontrak fitur, konfigurasi terpilih, batas interpretasi, dan perintah reproduksi ada di [FEATURE_SCHEMA.md](docs/FEATURE_SCHEMA.md) dan [laporan Tahap 2](reports/landmarks/README.md). Instal `requirements-features.txt`, lalu jalankan `python scripts/extract_landmarks.py --splits train val test`. Ekstraksi test hanya diizinkan setelah konfigurasi dibekukan dari validation. Cache dan log per gambar tetap lokal; ringkasan ukuran kecil dapat dilacak Git.

## Tahap 3: baseline klasifikasi

Panduan training interaktif per epoch tersedia di [TRAINING_BASELINES.md](docs/TRAINING_BASELINES.md). Jalankan `python scripts/train_baselines.py --languages sibi bisindo` untuk membandingkan kandidat SVM dan MLP; proses ini tidak membaca test. Setelah pilihan validation final, evaluasi test dilakukan terpisah satu kali.

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


## Kandidat dataset BISINDO dari Kaggle

Dataset tambahan dari [Kaggle](https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo) sudah diaudit secara lokal. Hasil audit dan aturan split ada di [laporan Kaggle BISINDO](reports/bisindo_kaggle_v1/README.md); konfigurasi ada di `configs/bisindo_kaggle_v1.json`. Untuk mereproduksi manifest setelah dataset tersedia di direktori sumber yang sama:

```powershell
python scripts/prepare_bisindo_kaggle.py
```

Manifest rinci dan lock BISINDO Kaggle v2 berada di `data/manifests/bisindo-kaggle-v2/` dan diabaikan Git. Split mengelompokkan kandidat gambar sumber berdasarkan pHash agar variasi augmentasi tidak tersebar ke split berbeda. Karena ID peserta/sesi tidak tersedia, evaluasinya bukan evaluasi signer-independent. Metadata lisensi/atribusi juga perlu diverifikasi di halaman sumber sebelum dataset atau turunannya dibagikan. Split v1 memiliki jumlah grup validation yang timpang; gunakan split v2 untuk eksperimen BISINDO Kaggle. Ekstraksi train/validation v2 selesai (76,61% dan 84,99% feature-valid); baseline confidence 0,5 mendapat system macro-F1 validation 0,877; pipeline confidence 0,3 dengan SVM linear C=1 mendapat 0,898. Keduanya validation-only; test belum dibuka. Lihat [laporan v2](reports/bisindo_kaggle_v2/README.md). Perintah ekstraksi dan training terpisah tersedia di `docs/TRAINING_BASELINES.md`.

## Diagnostik webcam

Gunakan [skrip diagnostik detektor webcam](docs/WEBCAM_DETECTOR_DIAGNOSTIC.md) untuk memeriksa landmark, jumlah tangan, status, dan FPS pada kamera langsung. Skrip tidak menyimpan frame atau video; ringkasan sesi lokal tersimpan di `data/webcam_diagnostics/`.

Untuk BISINDO, baseline yang mengklasifikasikan gambar penuh tanpa MediaPipe dijelaskan di [IMAGE_CLASSIFIER_BASELINE.md](docs/IMAGE_CLASSIFIER_BASELINE.md). Evaluasi test final dijalankan sekali; laporan lokal ada di `artifacts/bisindo-image-run-01/test_metrics.json`. Preview model dan collector webcam berlabel tersedia melalui [panduan demo webcam](docs/WEBCAM_IMAGE_CLASSIFIER.md), `scripts/webcam_image_classifier.py`, dan `scripts/collect_bisindo_webcam_samples.py`. Hasil training dan laporan rinci tetap lokal di `artifacts/`.

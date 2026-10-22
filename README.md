# Service AI alfabet SIBI dan BISINDO

FastAPI untuk dua checkpoint terpilih tersedia: [kontrak developer web](docs/API_CONTRACT.md), [deployment Docker + GitHub Actions](docs/DEPLOYMENT.md), dan [pemilahan file GitHub](docs/GITHUB_PUBLISH_CHECKLIST.md). Bundle model disimpan terpisah dari Git di `models/selected-sibi-bisindo-v1/`. Service sudah berjalan di `https://ai.signtech.tech`.

Untuk mencoba deteksi webcam di komputer ini, jalankan `& '.\.venv-service\Scripts\python.exe' scripts/run_local_demo.py` dari direktori proyek, lalu buka `http://localhost:8000`. Pilih SIBI atau BISINDO dan klik **Mulai kamera**. [Petunjuk lokal lengkap](docs/DEPLOYMENT.md#menjalankan-lokal).

Untuk menguji **API yang sudah dideploy**, buka [halaman simulasi produksi](https://ai.signtech.tech/production.html) di Chrome atau Edge. Halaman memakai HTTPS/WSS `ai.signtech.tech` dan menyediakan pemeriksaan koneksi tanpa kamera. Alternatif lokal: jalankan `& '.\.venv-service\Scripts\python.exe' scripts/run_production_demo.py` lalu buka `http://localhost:5173/production.html`. [Panduan simulasi produksi](docs/PRODUCTION_DEMO.md).

MVP yang disepakati: mengenali pose huruf statis dari webcam, dengan mode SIBI/BISINDO dipilih pengguna. Dua model koordinat MediaPipe terpilih kini memiliki service FastAPI dan halaman HTML untuk uji integrasi; UI produk menjadi tanggung jawab aplikasi web. Latensi realtime, terutama fallback BISINDO, masih perlu diperhatikan saat deployment.

**Acuan aktif:** [reset model koordinat](docs/COORDINATE_RESET_PLAN.md). SIBI 26 kelas memakai 24 huruf dataset SIBI dan pose statis J/Z sumber ASL. Training CNN1D/MLP, lalu SVM pada embedding, memakai cache landmark tanpa ekstraksi ulang atau koleksi webcam tambahan. Jalankan `python scripts/train_coordinates.py --language sibi` atau gunakan [notebook Colab](notebooks/02_coordinate_reset_colab.ipynb). Pilot webcam dan eksperimen sebelumnya menjadi arsip pembanding.

Langkah terbaru: setelah perbandingan webcam, pengguna menilai model BISINDO recovery v2 bagus secara umum, dengan kendala D/F/P terutama D. Percobaan webcam belum memiliki rekaman atau metrik tersimpan. Validation epoch 69 (system accuracy 91,21%, macro-F1 92,62%) belum menjelaskan kegagalan D pada webcam. [Review dan tindak lanjut](docs/BISINDO_RECOVERY_TRAINING_RUN01.md) mencatat perbedaannya. Panduan data dan training ada di [recovery v2](docs/BISINDO_RECOVERY_TRAINING_V2.md).

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

Eksperimen SIBI lama (`svm_linear_c1`, 24 kelas) dapat diuji melalui [panduan uji webcam SIBI](docs/WEBCAM_SIBI_CLASSIFIER.md) dan `scripts/webcam_sibi_classifier.py`. Service dan halaman uji saat ini memakai checkpoint koordinat terpilih yang dijelaskan di bagian awal dokumen.

Eksperimen baru dengan dataset landmark NumPy sumber ASL memakai 63 fitur dan 24 kelas kandidat, dengan kesesuaian pose SIBI belum diverifikasi. Panduan audit, persiapan split, serta training SVM/MLP per epoch ada di [NPY_LANDMARK_EXPERIMENT.md](docs/NPY_LANDMARK_EXPERIMENT.md). Artefaknya terpisah dari model SIBI 132 fitur sebelumnya.

Model 63 fitur terpilih dapat dijalankan melalui `python scripts/webcam_npy_landmarks.py`; [panduan webcam 63 fitur](docs/WEBCAM_NPY_LANDMARKS.md) menjelaskan pemeriksaan instalasi dan uji pose berlabel. Ringkasan trial disimpan tanpa gambar/video.

Kelanjutan model SIBI lama menambahkan modul gerakan J/Z dari video lokal melalui [panduan temporal J/Z](docs/SIBI_JZ_TEMPORAL.md). Jalankan `python scripts/webcam_sibi_classifier.py --temporal-model artifacts/sibi-jz-v1/templates.npz` setelah ekstraksi dan fitting template. Modul bersifat eksperimental; penolakan gerakan lain belum terkalibrasi.

Persentase keberhasilan ekstraksi dataset, cakupan gambar yang sudah/belum diproses, dan visualisasi per huruf tersedia di [laporan cakupan landmark](docs/LANDMARK_COVERAGE_REPORT.md). Reproduksi grafik dari log tersimpan dengan `python scripts/report_landmark_coverage.py`.

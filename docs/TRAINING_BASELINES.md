# Tahap 3 — menjalankan baseline klasifikasi

Skrip membandingkan SVM (kernel linear/RBF, `C` 0.1/1/10) dengan MLP kecil (hidden layers 128 dan 128×64). Scaler hanya di-fit pada contoh train yang feature-valid. Contoh dengan fitur gagal dikeluarkan dari fit classifier, tetapi tetap dihitung sebagai penolakan pada metrik sistem validation/test. Untuk BISINDO, hasil tetap eksploratif: train hanya 65,35% feature-valid sedangkan validation 97,69%; seri nama file bukan identitas peserta, pose resmi dan kepemilikan tangan belum diverifikasi.

SVM tidak mempunyai epoch yang sebanding dengan MLP; log mencatat kandidat, validation, dan waktu fit. MLP memakai `partial_fit` satu pass train per epoch dan mencetak loss, macro-F1 train, macro-F1 validation, skor sistem, waktu epoch, serta epoch terbaik. Log CSV ditulis dan di-flush tiap epoch, dan checkpoint terbaik diperbarui setiap kali skor sistem validation meningkat. Ini menggunakan API `partial_fit` scikit-learn yang melakukan satu iterasi/update atas data pada setiap pemanggilan. [Dokumentasi MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

## Menyiapkan environment

Gunakan environment Python yang sama dengan versi NumPy fitur, atau environment terpisah dengan paket training:

```powershell
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -v
```

Dependensi training dipatok pada scikit-learn 1.5.2 dan joblib 1.4.2. Estimasi awal pada perangkat lokal: training CPU cukup untuk baseline ini; SVM RBF pada seluruh data dan dua MLP mungkin memerlukan beberapa menit. Tidak perlu GPU untuk skrip yang disediakan.

## Training dan pemantauan

Untuk dataset Kaggle BISINDO, gunakan snapshot `bisindo-kaggle-v2`. Snapshot v1 menghasilkan jumlah grup validation yang timpang pada beberapa kelas (Q hanya 2 grup, U hanya 1), sehingga hasil training v1 tetap historis dan jangan dipakai untuk evaluasi per kelas. V2 menyeimbangkan jumlah grup pHash per kelas, lalu memilih alokasi yang menjaga jumlah gambar sedekat mungkin ke rasio 70/15/15. Ini tetap bukan split signer-independent. Cache dan hasil disimpan terpisah dari baseline Mendeley.

Ekstrak fitur train/validation v2 (cache dapat dilanjutkan dari checkpoint):

```powershell
python scripts/extract_landmarks.py --config configs/landmarks_bisindo_kaggle_v2.json --languages bisindo --splits train val
```

Setelah ekstraksi selesai, training dapat dipantau per epoch seperti biasa:

```powershell
python scripts/train_baselines.py --languages bisindo --features-dir data/features/landmarks-bisindo-kaggle-v2 --output-dir artifacts/classification-bisindo-kaggle-v2
```

Log epoch dan checkpoint berada di `artifacts/classification-bisindo-kaggle-v2/bisindo/`. Ekstraksi test tidak dijalankan pada langkah ini. Setelah model dipilih berdasarkan validation dan pilihan dibekukan, konfigurasi ekstraksi yang sama harus dikunci sebelum test Kaggle dibuka.

Ringkasan audit awal dan hasil training historis v1 ada di [laporan v1](../reports/bisindo_kaggle_v1/README.md); split baru dan hasil ekstraksi v2 dicatat di [laporan v2](../reports/bisindo_kaggle_v2/README.md).

```powershell
python scripts/train_baselines.py --languages sibi bisindo
```

Skrip hanya membaca `train` dan `val`, tidak membuka cache test. MLP menampilkan satu baris per epoch ke terminal. Jika ingin menyimpan output terminal:

```powershell
New-Item -ItemType Directory -Force artifacts/classification-v1 | Out-Null
python scripts/train_baselines.py --languages sibi bisindo 2>&1 | Tee-Object -FilePath artifacts/classification-v1/training_console.log
```

Hasil berada di `artifacts/classification-v1/`: file `*_epochs.csv` untuk grafik/analisis tiap epoch, checkpoint MLP terbaik, model tiap kandidat SVM, metrik validation lengkap, dan `selection.json` per bahasa. Kandidat dipilih berdasarkan macro-F1 sistem validation (seluruh gambar; kegagalan landmark diperlakukan sebagai reject), lalu valid-only macro-F1 sebagai tie-break. Skor SVM adalah decision score, bukan probabilitas; MLP juga tidak dikalibrasi sebagai probabilitas.

Laporan memisahkan metrik classifier pada fitur valid dari metrik sistem seluruh gambar, menyertakan precision/recall/F1 tiap kelas dan confusion matrix. Tidak ada threshold reject yang dipilih karena belum ada set negatif yang diverifikasi. Pilihan model/threshold berdasarkan validation wajib ditinjau sebelum test dibuka.

## Evaluasi test satu kali

Setelah selection validation dianggap final, jalankan:

```powershell
python scripts/evaluate_classifier_test.py --languages sibi bisindo --confirm-frozen-selection
```

Skrip memeriksa hash artefak dan cache, membaca test satu kali, menyimpan hasil akhir ke `artifacts/classification-v1/<bahasa>/test_metrics.json`, lalu menandai selection sebagai sudah memakai test. Skrip training menolak tuning ulang setelah test diakses. Jika hasil test dipakai untuk mengubah model, buat split test baru dan versi eksperimen baru; jangan menamai ulang hasil lama sebagai test independen.

Training dan evaluasi ini tidak mengesahkan alfabet BISINDO, signer-independent generalization, atau performa webcam. Tahap webcam masih membutuhkan sampel percobaan independen dan evaluasi negatif sesuai [protokol evaluasi](EVALUATION_PROTOCOL.md).


Untuk eksperimen confidence 0,3 yang terpisah, train/validation sudah diekstrak dengan konfigurasi sama. Jangan campur cache baseline confidence 0,5. Jalankan training ulang dan pantau epoch dengan:

```powershell
python scripts/train_baselines.py --languages bisindo --features-dir data/features/landmarks-bisindo-kaggle-v2-conf03 --output-dir artifacts/classification-bisindo-kaggle-v2-conf03
```

Hasil detector-only belum membuktikan classifier atau pose lebih benar; verifikasi system validation dan contoh landmark sebelum memilih konfigurasi. Test tetap tidak digunakan untuk pemilihan.

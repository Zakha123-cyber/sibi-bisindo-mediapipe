# Eksperimen landmark NumPy untuk kandidat SIBI

Tanggal audit: 20 Oktober 2026. Pengguna meminta penggunaan dataset landmark baru sebagai eksperimen untuk target SIBI. Asal data tetap ASL; kesetaraan pose/label ASL dengan SIBI belum diverifikasi. Baseline SIBI sebelumnya tetap menjadi pembanding dan tidak diganti oleh hasil audit ini.

## Sumber dan hasil audit

Folder lokal: `dataset/ASL Dataset`. Metadata bawaan mengidentifikasi **ASL Alphabet Hand Landmarks - MediaPipe 3D**, pembuat Boris Graudt, ID Kaggle `borisgraudt/asl-alphabet-hand-landmarks`. README bawaan menyebut satu signer dan sampel statis. Itu adalah keterangan sumber, bukan verifikasi independen atas proses pengumpulan. Sumber daring: [dataset Kaggle](https://www.kaggle.com/datasets/borisgraudt/asl-alphabet-hand-landmarks), [proyek pembuat](https://github.com/bgraudt/helios).

Audit membaca semua 10.508 file menggunakan `np.load(..., allow_pickle=False)`:

| Pemeriksaan | Hasil |
|---|---|
| Kelas | 26 label A–Z, 400–411 sampel per kelas |
| Bentuk array aktual | Seluruh file `(63,)`, dapat dibentuk ulang menjadi `(21, 3)` |
| Tipe angka | Seluruh file `float64` |
| Validitas numerik | Seluruh file finite, tanpa geometri tangan degenerat |
| Duplikat numerik identik | Tidak ditemukan, termasuk lintas label |
| Pergelangan bernilai nol | 0 dari 10.508 sampel |
| Jarak maksimum dari wrist bernilai 1 | 0 dari 10.508 sampel |
| Metadata ukuran frame, signer/sesi per sampel | Tidak tersedia dalam paket lokal |

README menyebut `(21, 3)` dan normalisasi wrist/max-distance sudah diterapkan. Isi lokal berbeda: array diratakan dan tidak memenuhi dua sifat normalisasi tersebut. Karena itu preprocessing harus dibangun dari isi array aktual, bukan menganggap file siap dipakai langsung.

Fingerprint snapshot inventaris: `f3923cd2be8b47ef631202cd4cc14ecb285c2316e57b32feb874d0a98f0cf7a3`. Inventaris rinci dan statistik lokal ada di `data/audits/asl-landmarks-v1/`, yang diabaikan Git. Reproduksi:

```powershell
python scripts/audit_asl_landmarks.py
```

## Usulan baseline

1. Gunakan eksperimen dan artefak baru dengan metadata `source_language=asl`, `target_language=sibi`, dan `pose_mapping=unverified`. Jangan menyebut metrik ASL sebagai akurasi SIBI sebelum pose dicocokkan dengan referensi SIBI.
2. Bentuk ulang menjadi 21×3, kurangi koordinat wrist, lalu bagi jarak maksimum 3D dari wrist; flatten kembali menjadi 63 fitur. Gunakan satu fungsi yang sama untuk dataset dan webcam. Pertahankan orientasi. Ukuran frame asli tidak diketahui sehingga jangan menerapkan koreksi aspect ratio lama secara diam-diam pada dataset ini.
3. Bandingkan StandardScaler + SVM linear/RBF dengan MLP kecil 128→64. Fit scaler hanya pada train. Log MLP per epoch, pilih kandidat dengan macro-F1 validation. Model urutan seperti LSTM belum sesuai karena file hanya berisi satu pose, tanpa urutan gerakan.
4. Untuk pembandingan awal dengan baseline SIBI lama, prioritaskan kandidat 24 kelas A–I dan K–Y; kehadiran J/Z di dataset tidak membuktikan kemampuan mengenali gerakannya. Tinjau pose per kelas sebelum menetapkan label untuk produk.
5. Jangan memakai random split per frame sebagai bukti generalisasi. Penomoran `frame_<n>` bukan ID trial/sesi. Jika memakai holdout blok nomor file, dokumentasikan sebagai heuristik sementara; metadata peserta/sesi independen tetap belum tersedia. Tidak adanya duplikat identik tidak meniadakan near-duplicate.
6. Nilai model pada validation webcam berlabel dari pengguna yang berbeda dari signer sumber bila memungkinkan. Metrik dataset hanya mengukur classifier atas landmark yang sudah tersedia; kegagalan detektor webcam perlu dihitung tersendiri.

MediaPipe tetap diperlukan untuk menghasilkan landmark webcam. Dataset `.npy` mempercepat eksperimen classifier, tetapi tidak melatih ulang detektor MediaPipe. Pipeline 63 fitur ini juga berbeda dari `hand-geometry-v1` 132 fitur pada baseline lama; model lama tidak dapat menerima input baru secara langsung.

## Pipeline eksperimen yang tersedia

Konfigurasi: [`npy_landmarks_v1.json`](../configs/npy_landmarks_v1.json). Modul fitur bersama ada di `src/sibi_bisindo/npy_landmarks.py`. Fitur baru berjumlah 63 dan tidak memakai koreksi aspect ratio ataupun perubahan handedness. Sumber array mentah dipertahankan.

Split memakai urutan **numerik** `frame_<n>` di setiap kelas: batas awal pada 70% dan 85% urutan, dengan 10 sampel dikeluarkan pada setiap sisi dari kedua batas. Karena ada gap, jumlah akhir bukan rasio tepat 70/15/15. J/Z dicatat sebagai `excluded`, bukan diganti label. Identitas file, config, implementasi fitur, dan cache dikunci dengan SHA-256.

Snapshot yang disiapkan menghasilkan **6.541 train**, **978 validation**, **1.224 test yang dicadangkan**, **960 sampel gap**, dan **805 sampel J/Z yang dikeluarkan**. Pemeriksaan cache aktual dan pengulangan persiapan dengan input sama sudah lulus.

Urutan nomor file belum diverifikasi sebagai urutan waktu. Gap hanya pengaman praktis terhadap frame dekat dalam penomoran; bukan jaminan bebas near-duplicate atau pemisahan sesi. Keterbatasan satu signer dan label ASL tetap berlaku.

Pasang dependensi training pada environment yang digunakan proyek bila belum tersedia:

```powershell
python -m pip install -r requirements-training.txt
```

Siapkan cache, kemudian validasi input tanpa melatih model:

```powershell
python scripts/prepare_npy_landmarks.py
python scripts/train_npy_landmarks.py --check-only
```

Perintah persiapan hanya memuat dan menormalisasi array **train dan validation**. Test dicadangkan dalam `manifest.csv`; tidak ada cache `test.npz` yang dibuat. Direktori cache default: `data/features/asl-source-sibi-candidate-v1/`. Pengulangan dengan snapshot yang sama memverifikasi hasil yang sudah dikunci; perubahan konfigurasi memerlukan versi/direktori baru.

Jalankan training CPU sendiri:

```powershell
python scripts/train_npy_landmarks.py
```

Default membandingkan SVM linear dan RBF pada C=0.1/1/10, lalu MLP 128→64 hingga 80 epoch dengan patience 12. Setiap epoch MLP menampilkan loss, train accuracy, validation accuracy, dan macro-F1 validation. SVM menampilkan hasil tiap kandidat karena tidak memiliki epoch seperti MLP. Untuk membatasi durasi eksperimen MLP, misalnya:

```powershell
python scripts/train_npy_landmarks.py --epochs 40 --patience 8
```

Setiap run membuat direktori unik `artifacts/asl-source-sibi-candidate-v1_<timestamp>/`; direktori hasil yang sudah berisi file ditolak agar tidak tertimpa. Isi hasil:

- Model SVM dan checkpoint MLP pada epoch validation terbaik, masing-masing membawa pipeline scaler dan metadata preprocessing.
- `mlp_128x64_epochs.csv`: log per epoch.
- `best_validation_metrics.json`: confusion matrix dan precision/recall/F1 per kelas dari kandidat terpilih.
- `selection.json`: kandidat, hash model, kontrak fitur, sumber ASL/target SIBI, dan batas interpretasi.
- `training_state.json`: status run dan penanda `test_accessed=false`.

Pemilihan model berdasarkan macro-F1 validation, dengan accuracy sebagai tie-break. Scaler hanya di-fit pada train. Test tidak dipakai selama training atau pemilihan. Tidak ada threshold unknown yang dipilih karena belum ada set negatif berlabel. Angka validation adalah hasil classifier atas landmark ASL yang tersedia, bukan akurasi sistem SIBI webcam.

## Hasil training dan langkah webcam

Pengguna menyelesaikan run `asl-source-sibi-candidate-v1_20261020_093347`. Kandidat terpilih adalah SVM RBF C=10: accuracy validation 0.987730 dan macro-F1 0.987689 pada 978 sampel. Kesalahan: M→N sebanyak 3 dan V→R sebanyak 9; recall V 0.780488. Hash artefak terpilih: `46162acf68dd095843645b379c8991134b8a1232fb19874ead964d921042f60c`. Hash dan metrik validation telah diverifikasi ulang dari artefak yang tersimpan.

MLP berhenti setelah epoch 26, dengan epoch terbaik 14; hasil validation-nya lebih rendah dari SVM terpilih. Test tetap belum diakses. Skor tersebut adalah validation classifier landmark sumber ASL, bukan bukti akurasi SIBI webcam.

Demo kontrak baru tersedia di [`webcam_npy_landmarks.py`](../scripts/webcam_npy_landmarks.py); instruksi uji berlabel tanpa rekaman ada di [WEBCAM_NPY_LANDMARKS.md](WEBCAM_NPY_LANDMARKS.md). Uji pada 10 sampel validation mentah menghasilkan fitur dan prediksi yang identik dengan jalur cache training. Skrip webcam SIBI lama tetap memakai kontrak 132 fitur dan tidak membaca model baru secara otomatis.

Uji webcam pertama mendapat 0/6 trial benar dan keluaran dominan O, meskipun pengguna melaporkan rangka landmark mengikuti tangan kanan dengan baik. Kecocokan jalur kode terhadap file dataset belum memverifikasi kecocokan orientasi dataset dengan webcam. Counterfactual mirror-x pada validation menyebabkan prediksi O mendominasi, sehingga demo sekarang menyediakan `--compare-mirror` untuk menguji dua orientasi pada frame dan trial yang sama. Belum ada bukti live bahwa mirror-x menyelesaikan masalah; endpoint/UI belum dilanjutkan.

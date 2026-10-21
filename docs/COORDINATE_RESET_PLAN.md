# Acuan aktif: model koordinat SIBI dan BISINDO

Keputusan pengguna 20 Oktober 2026: mulai ulang alur eksperimen dan fokus SIBI dahulu. Dataset/foto, artefak, dan hasil lama tetap diarsipkan sebagai pembanding; tidak dihapus atau ditimpa. Ini reset eksperimen, bukan klaim test baru yang independen.

## Ruang lingkup

- Input realtime webcam, output satu huruf, mode SIBI/BISINDO dipilih pengguna.
- Classifier hanya menerima koordinat MediaPipe beserta penanda slot/relasi tangan; tidak menerima piksel foto.
- SIBI baru: 24 huruf dari dataset SIBI + pose statis J/Z sumber ASL NPY, sesuai persetujuan pengguna. Label J/Z adalah kandidat lintas sumber; gerakan tidak dimodelkan dan kesesuaian linguistik tidak diklaim sudah terverifikasi.
- BISINDO: pipeline dua tangan disiapkan sekarang, eksperimen modelnya setelah hasil SIBI ditinjau. Cache Kaggle v2 confidence 0,3 dipakai sebagai sumber koordinat.
- Tidak mengambil sampel tambahan, mengulang seluruh ekstraksi, atau membuka test untuk memilih arsitektur.

## Input bersama

Versi `two-hand-coordinates-v1`, bentuk `(42, 7)`: dua slot × 21 titik; kanal local xyz, wrist xy relatif terhadap pusat kedua tangan, rasio skala telapak, dan present. Posisi serta skala absolut dihilangkan tetapi orientasi dan hubungan posisi kedua tangan dipertahankan. Slot diurutkan menurut posisi pergelangan pada gambar, bukan label anatomis kiri/kanan. Tangan tidak ada menjadi slot nol dengan present=0, bukan koordinat tangan palsu. Kedalaman lintas tangan tidak direkayasa.

Gambar SIBI/BISINDO dikoreksi rasio aspek memakai dimensi dari cache. Dimensi asli ASL tidak tersedia; asumsi aspek 1,0 dicatat eksplisit dalam konfigurasi. Normalisasi yang sama diterapkan pada koordinat saat webcam. Hasil MediaPipe masih mungkin salah atau tidak lengkap; classifier tidak memulihkan titik yang hilang.

## Data dan split

Konfigurasi: `configs/coordinate_reset_v1.json`. Persiapan menolak direktori output yang sudah terisi; perubahan eksperimen memerlukan versi/path baru. NPZ hanya train/validation. Tidak mengubah atau memuat array test cache lama. SIBI lama sudah pernah dievaluasi pada test, sehingga hasil test tersebut tidak menjadi test segar untuk eksperimen baru.

SIBI train native: 1.001 gambar, 993 valid. Ditambah 42 J dan 42 Z terpilih dari bagian train ASL: total 1.085, valid 1.077. Validation: 216 native + 40 J + 40 Z = 296, valid 295. Sampel gagal dipertahankan dengan mask untuk menghitung performa sistem, tetapi tidak menjadi contoh training classifier.

ASL diurutkan berdasarkan nomor filename, split 70/15/15 dengan gap sepuluh sampel di kedua sisi batas. Ini heuristik, bukan pemisahan pemeraga/sesi terverifikasi. Hanya koordinat train/validation yang dimuat; file test/gap hanya diberi hash/metadata. Training J/Z dibatasi menjadi 42 per kelas dengan pemilihan tersebar pada urutan train. Duplikat identik lintas split dikarantina; duplikat dalam split tidak dihitung berulang. Keterkaitan sumber dengan label J/Z dapat mempermudah evaluasi secara semu; metrik per sumber wajib dilaporkan.

## Model dan training

1. CNN1D kecil sebagai kandidat utama: kanal koordinat → Conv1D 32 → Conv1D 64 → pooling → embedding 64 dimensi → Softmax.
2. MLP 128→64 sebagai kontrol pada **tensor baru yang sama**, bukan disebut metode yang belum pernah dicoba di proyek. MLP sebelumnya pada fitur lama tidak mengalahkan SVM lama.
3. Setelah checkpoint neural terbaik dibekukan, embedding training dipakai melatih SVM linear C=1 dan RBF C=1. Embedding validation hanya digunakan untuk evaluasi/pemilihan kandidat, tidak untuk fit scaler/SVM.
4. Loss berbobot kelas, augmentasi rotasi kecil/noise koordinat hanya pada training. Tidak ada flip horizontal otomatis atau imputasi tangan hilang. Belum ada fine-tuning model pretrained eksternal; jaringan dilatih dari awal pada data gabungan.
5. CSV dan terminal memuat loss, train accuracy, validation accuracy, macro-F1, waktu, serta best epoch. Early stopping menggunakan macro-F1 sistem validation. Tidak menjalankan training penuh otomatis; pengguna memantau sendiri.

Epoch CSV/checkpoint, metrik per kelas/sumber, confusion matrix angka, kurva training, dan selection.json tersimpan pada direktori run baru. Nilai classifier-only dan sistem termasuk kegagalan ekstraksi dipisahkan. Skor belum terkalibrasi dan belum ada kelas unknown/background.

## Perintah lokal

Environment ekstraksi lama tetap digunakan untuk webcam. Training membutuhkan PyTorch; environment terpisah disarankan bila memasang dependency training yang berbeda. Jangan mengganti NumPy/MediaPipe environment webcam hanya demi Colab.

```powershell
# Sudah disiapkan pada implementasi awal; jangan ulangi bila output masih ada.
python scripts/prepare_coordinates.py --languages sibi

# Jalankan sendiri dan pantau setiap epoch. Default: CNN dan MLP, maksimal 100 epoch.
python scripts/train_coordinates.py --language sibi

# Fokus CNN saja bila ingin satu run lebih singkat:
python scripts/train_coordinates.py --language sibi --models cnn

# Ganti RUN_DIR dengan path run yang dicetak setelah training selesai.
python scripts/webcam_coordinates.py --run-dir RUN_DIR --check-only
python scripts/webcam_coordinates.py --run-dir RUN_DIR
```

Webcam memakai model terpilih dalam selection.json, bukan otomatis run terbaru. Validasi hash artefak dan implementasi mencegah checkpoint dipakai dengan praproses berbeda. MediaPipe tetap memakai konfigurasi asli dari cache. J/Z adalah pose statis. Preview dicerminkan sebelum teks ditulis agar teks terbaca; inferensi tidak dicerminkan. Tidak merekam foto/video.

Untuk BISINDO gunakan `--language bisindo`; persiapan terpisah dengan `--languages bisindo`. Data pipeline BISINDO boleh disiapkan sebelum training, tetapi hasil satu tangan tetap belum menjamin kelengkapan pose. Bandingkan waktu classifier dan detektor pada perangkat target, bukan menganggap CNN otomatis lebih cepat/akurat.

## Colab GPU

```powershell
python scripts/package_coordinates_colab.py --language sibi
```

Buka `notebooks/02_coordinate_reset_colab.ipynb` di Colab, pilih runtime GPU, lalu upload `data/exports/coordinate-reset-v1-sibi-colab.zip` saat diminta. Paket hanya kode dan koordinat train/validation, bukan dataset foto/test arrays. Notebook menampilkan epoch, kurva, selection, dan menyediakan ZIP hasil untuk diunduh. Ekstrak ZIP hasil ke akar proyek agar folder `artifacts/...` dipertahankan. Setelah itu gunakan path run yang dihasilkan untuk webcam lokal.

## Urutan penyelesaian

- [x] Pisahkan eksperimen reset dan kontrak koordinat dua tangan.
- [x] Siapkan data SIBI 26 kelas dan jalur persiapan BISINDO dari cache.
- [x] Sediakan training CNN/MLP dan SVM embedding, logging per epoch, notebook Colab.
- [x] Sediakan inferensi webcam yang memakai kontrak koordinat yang sama.
- [x] Pengguna menjalankan training penuh SIBI; hasil run pertama ditinjau tanpa membuka test. CNN–Softmax terpilih; lihat [review run 01](COORDINATE_RESET_SIBI_RUN01.md).
- [ ] Periksa perilaku webcam SIBI, khususnya kebingungan J/Z kandidat ASL dengan huruf lain.
- [x] Pengguna menilai webcam cukup bagus pada huruf lain; audit seluruh train J/Z dan [kandidat fine-tuning terpisah](SIBI_JZ_FINETUNE.md) disiapkan. Validation tetap identik; training kandidat dijalankan pengguna.
- [x] Siapkan [transfer encoder SIBI ke checkpoint BISINDO terpisah](BISINDO_TRANSFER.md), dengan classifier baru, training dua tahap, dan paket Colab.
- [x] Pengguna menjalankan training transfer BISINDO; [review run 01](BISINDO_TRANSFER_RUN01.md) selesai, checkpoint terbaik epoch total 55.
- [ ] Pengguna mencoba webcam BISINDO dari checkpoint transfer; hasil realtime belum dinilai.
- [x] FastAPI HTTP/WebSocket dan halaman HTML uji tersedia untuk dua checkpoint terpilih; Docker/Compose serta GitHub Actions disiapkan. Lihat [deployment](DEPLOYMENT.md).
- [ ] Konfigurasikan domain/origin, bundle VPS, dan GitHub secrets; verifikasi build Docker dan deployment pertama di server.
- [x] Pemulihan ekstraksi BISINDO train/validation menggunakan rotasi/ROI dan penyaringan duplikasi; lihat [hasil dan batasan](BISINDO_LANDMARK_RECOVERY.md). Dataset test tidak dibuka.
- [x] Tambahkan filter duplikasi kumpulan titik, perbaiki contoh X/O otomatis, dan samakan recovery offline/webcam; [paket training v2](BISINDO_RECOVERY_TRAINING_V2.md) siap. Pemeriksaan sampel tidak mengesahkan seluruh landmark.
- [x] Pengguna menjalankan training BISINDO recovery v2, lalu hasil checkpoint baru ditinjau sebelum uji webcam.
- [x] Training recovery v2 selesai; [hasil run 01](BISINDO_RECOVERY_TRAINING_RUN01.md) lebih baik dari transfer v1 pada validation yang sama, dan pemeriksaan konfigurasi webcam lulus.
- [x] Pengguna mencoba webcam recovery v2: prediksi dilaporkan lebih buruk daripada model sebelumnya meskipun rangka mengikuti tangan. Ini regresi penggunaan nyata; skor validation belum membuktikan generalisasi webcam.
- [ ] Bandingkan bobot lama/baru pada frame dan landmark baseline yang sama; bandingkan juga hasil recovery melalui `scripts/webcam_compare_bisindo.py`. Jangan melatih ulang atau menaikkan confidence berdasarkan tebakan.
- [x] Pembaruan pengguna setelah perbandingan: model baru dinilai bagus secara umum; kendala terutama D/F/P, dengan D belum berhasil dikenali pada percobaan tersebut. Tidak ada hasil webcam tersimpan karena mode perbandingan tidak merekam. Validation v2 D=52/54, F=57/59, P=63/67 benar; kegagalan D di webcam belum terjelaskan oleh metrik dataset.

Tidak ada janji bahwa skor validation tinggi akan memperbaiki webcam. Keputusan lanjut didasarkan pada hasil run dan perilaku aktual; tidak kembali ke pengambilan sampel besar secara otomatis.

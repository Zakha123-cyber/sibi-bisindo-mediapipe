# Tinjauan training SIBI reset, run 01

Run: `artifacts/coordinate-reset-v1/sibi_20261020_130607_889055`, ditinjau 20 Oktober 2026. Hash paket train/validation dan checkpoint terpilih cocok dengan selection.json. Prediksi validation checkpoint diulang tanpa training; seluruh metrik tersimpan berhasil direproduksi. Test tidak dibuka.

## Hasil validation

Metrik sistem mencakup satu gambar validation yang gagal ekstraksi sebagai penolakan.

| Kandidat | Akurasi | Macro-F1 | Median classifier satu sampel |
|---|---:|---:|---:|
| CNN–Softmax, terpilih | 95,61% | 96,44% | 0,267 ms |
| CNN–SVM linear | 94,26% | 94,88% | 0,698 ms |
| CNN–SVM RBF | 95,61% | 96,32% | 0,696 ms |
| MLP–Softmax | 90,54% | 91,70% | 0,183 ms |
| MLP–SVM linear | 91,22% | 91,35% | 0,283 ms |
| MLP–SVM RBF | 91,22% | 91,75% | 0,316 ms |

Waktu di atas berasal dari benchmark saat training pada CPU, tidak mencakup MediaPipe, kamera, praproses, dan tampilan. Belum merupakan FPS sistem webcam.

CNN mencapai batas 100 epoch dengan checkpoint terbaik epoch 89. Checkpoint terbaik tersebut dipakai untuk evaluasi dan SVM embedding, bukan bobot terakhir epoch 100. Pada epoch 89 train accuracy 99,07%, validation valid-only accuracy 95,93%, dan validation system macro-F1 96,44%. MLP berhenti dini pada epoch 77, terbaik epoch 62. Loss turun dan validation mulai mendatar; belum ada alasan dari kurva ini saja untuk memperpanjang training.

## Rincian checkpoint CNN–Softmax

- Semua kelas: 283/296 benar (95,61%). Terdapat 12 salah klasifikasi dan satu gagal ekstraksi.
- SIBI asli, 24 kelas: 212/216 benar (98,15%), terdiri dari tiga salah klasifikasi dan satu gagal ekstraksi. Prediksi tetap memilih di antara 26 kelas, tidak dibatasi ulang menjadi 24.
- J sumber ASL: 32/40 benar (recall 80%). Salah sebagai G tiga sampel dan T lima sampel.
- Z sumber ASL: 39/40 benar (recall 97,5%). Satu sampel salah sebagai A.
- Kesalahan native: C→O satu, M→N satu, S→A satu, dan satu D ditolak karena ekstraksi gagal.

SVM lama pada validation 24 kelas yang sama sebelumnya benar 211/216 (97,69%). Kandidat baru benar satu gambar lebih banyak pada subset native tersebut, tetapi eksperimennya menggunakan representasi/training 26 kelas berbeda. Selisih satu gambar belum membuktikan perbaikan generalisasi. Tidak ada prediksi native validation yang menjadi J/Z pada checkpoint terpilih ini; perilaku tersebut belum diperiksa pada webcam.

## Keputusan berikut

Pertahankan checkpoint CNN–Softmax terpilih untuk mencoba inferensi webcam. Tidak mengganti menjadi CNN–SVM hanya karena penelitian memakai SVM: pada run ini Softmax memberi macro-F1 sedikit lebih tinggi dan classifier lebih cepat. Nilai tersebut masih validation-selected; sumber J/Z berbeda dari kelas lain dan skor bukan confidence terkalibrasi.

```powershell
python scripts/webcam_coordinates.py --run-dir artifacts/coordinate-reset-v1/sibi_20261020_130607_889055
```

Tidak ada kamera yang dibuka selama review. J/Z tetap pengenalan pose statis, bukan lintasan gerakan.

Review juga memperbaiki pemeriksaan konfigurasi pada loader webcam: hash sekarang memakai JSON canonical yang sama dengan ekstraksi, sehingga perubahan format file tidak dianggap perubahan detektor. Perubahan nilai konfigurasi tetap ditolak. Bobot/fitur tidak diubah dan training ulang tidak diperlukan.

[Kurva training](../artifacts/coordinate-reset-v1/sibi_20261020_130607_889055/learning_curves.png), [selection](../artifacts/coordinate-reset-v1/sibi_20261020_130607_889055/selection.json), dan [rincian validation terpilih](../artifacts/coordinate-reset-v1/sibi_20261020_130607_889055/cnn_softmax_validation.json).

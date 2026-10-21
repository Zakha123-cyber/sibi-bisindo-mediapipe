# Review fine-tuning J/Z run 01

Run `artifacts/coordinate-jz-ft-v1/sibi_20261020_143704_789723`, ditinjau 20 Oktober 2026. Training berlangsung 12 epoch lalu early stopping. Checkpoint terpilih adalah **epoch 0**, yaitu bobot awal baseline, bukan pembaruan epoch training.

| Ukuran validation | Baseline / checkpoint tersimpan | Percobaan epoch 1–12 |
|---|---:|---:|
| Macro-F1 sistem 26 kelas | 96,44% | Maksimum 95,56% |
| Native SIBI benar | 212/216 | 208–210/216 |
| J benar | 32/40 | 31–32/40 |
| Z benar | 39/40 | 39/40 |

Akurasi checkpoint tersimpan tetap 283/296 = 95,61%. Semua epoch percobaan memiliki macro-F1 lebih rendah dan native benar lebih sedikit daripada baseline. Jadi tidak ada peningkatan yang terhalang hanya oleh penjagaan native: macro-F1 sendiri juga belum meningkat.

Parent selection, hash paket train/validation, pemuatan artefak, dan kesamaan setiap tensor bobot tersimpan dengan baseline sudah diperiksa. Hash file checkpoint berbeda karena metadata/serialisasi berbeda; parameter model identik. Baseline validation dan validation terpilih sama. Tidak ada training ulang atau pembukaan test saat review.

## Keputusan

Pertahankan baseline `artifacts/coordinate-reset-v1/sibi_20261020_130607_889055`. Tidak perlu mengulang webcam run fine-tuning untuk mencari perbedaan prediksi karena bobot terpilih sama. Tidak menambah epoch atau mengganti ambang otomatis pada validation yang sama.

Memperbanyak contoh dari pool training yang hampir seragam, dengan sampling seimbang dan learning rate lebih kecil, belum memperbaiki J/Z. Ini bukan bukti semua fine-tuning pasti gagal. Keterbatasan cakupan pose J dan potensi perbedaan dengan webcam tetap belum terselesaikan. Langkah berikut perlu membawa variasi/informasi berbeda atau perubahan representasi yang beralasan, bukan mengulang training identik.

J tetap fitur eksperimen dengan recall validation 80%; Z 97,5% pada sumber ASL belum menjamin hasil webcam. Tidak ada perubahan label/split atau penonaktifan kelas pada aplikasi.

[Log epoch](../artifacts/coordinate-jz-ft-v1/sibi_20261020_143704_789723/cnn_epochs.csv) dan [selection](../artifacts/coordinate-jz-ft-v1/sibi_20261020_143704_789723/selection.json).

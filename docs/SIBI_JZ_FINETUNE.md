# Kandidat fine-tuning J/Z, tanpa mengganti baseline

Status: training run pertama sudah ditinjau; tidak ada peningkatan dan checkpoint epoch 0 dipertahankan. Lihat [hasil review](SIBI_JZ_FINETUNE_RUN01.md). Perintah di bawah tetap tersedia untuk reproduksi, bukan rekomendasi mengulang training yang sama.

Audit dan persiapan 20 Oktober 2026. Baseline tetap `artifacts/coordinate-reset-v1/sibi_20261020_130607_889055`, CNN–Softmax epoch 89, yang sudah dinilai cukup bagus pada webcam pengguna. Training penuh kandidat baru dijalankan pengguna agar dapat dipantau per epoch.

## Temuan audit training yang belum digunakan

| Kelas | Training lama | Belum digunakan | Seluruh pool training |
|---|---:|---:|---:|
| J | 42 | 229 | 271 |
| Z | 42 | 230 | 272 |

Audit hanya membaca bagian train/unused_train dari manifest lama. Validation, gap, dan test tidak dibaca koordinatnya. File NPY dicocokkan dengan hash manifest sebelum diproses.

J tetap sangat seragam meski seluruh pool dimasukkan. Median RMS sebaran sekitar median kelas hanya berubah dari 0,0516 menjadi 0,0534; median jarak sampel J tambahan ke contoh lama terdekat 0,0110. Angka ini ukuran koordinat ternormalisasi, bukan batas noise terkalibrasi atau ukuran kebenaran pose. Pada Z, median sebaran tidak meningkat (0,1249 menjadi 0,1216), walau ada contoh tambahan yang jauh dari contoh terpilih sebelumnya. Lebih banyak sampel tidak otomatis berarti pose lebih beragam.

[Grafik perbandingan pool](../data/reports/sibi-jz-training-pool/training_pool_comparison.png) dan [hasil angka audit](../data/reports/sibi-jz-training-pool/summary.json).

## Kandidat yang disiapkan

`data/features/coordinate-jz-ft-v1/sibi/` memuat 1.544 baris train, 1.536 valid: 993 native SIBI + 271 J + 272 Z. Seluruh contoh native lama dipertahankan. Validation **disalin byte-for-byte** dari baseline: 296 baris, 295 valid. Tidak memindahkan validation/gap/test ke training.

- Arsitektur/normalisasi sama, mulai dari bobot CNN baseline, bukan training dari nol atau modul temporal.
- Learning rate 0,0001; maksimal 60 epoch; patience 12.
- Sampling kelas seimbang dengan replacement, probabilitas contoh berbanding terbalik jumlah kelas. Loss tidak sekaligus diberi bobot kelas agar efek penyeimbangan tidak diterapkan dua kali. Jumlah kelas seimbang dalam ekspektasi, bukan tepat sama setiap batch.
- Augmentasi koordinat tetap seperti baseline; eksperimen ini mengubah cakupan train dan prosedur fine-tuning, bukan menambah distorsi yang ditargetkan pada kesalahan validation.
- Semua 26 kelas tetap dilatih. Model J/Z saja tidak dibuat.
- Checkpoint epoch 0 berisi baseline. Checkpoint hanya diganti jika macro-F1 sistem validation meningkat dan jumlah prediksi benar pada subset native tidak turun dari nilai awal (pada baseline nyata: 212/216). Ini penjagaan agregat pada validation, bukan jaminan setiap huruf atau webcam tidak turun.
- Hanya head Softmax dibandingkan untuk fine-tuning ini. Epoch 0 dipertahankan bila tidak ada kandidat yang lolos. Tidak memperpanjang training otomatis jika pool ternyata tidak cukup.

## Menjalankan lokal

Data dan paket telah dibuat; tidak perlu persiapan ulang. Jalankan:

```powershell
python scripts/train_coordinates.py --language sibi --models cnn --features-dir data/features/coordinate-jz-ft-v1/sibi --init-run artifacts/coordinate-reset-v1/sibi_20261020_130607_889055
```

Terminal/CSV menampilkan loss, macro-F1, jumlah native benar serta J/Z benar per epoch. Direktori hasil baru berada di `artifacts/coordinate-jz-ft-v1/`. `baseline_validation.json` mencatat hasil sebelum fine-tuning; `selection.json` mencatat asal bobot dan floor native. Jika best epoch=0, training belum menghasilkan peningkatan yang memenuhi syarat.

Gunakan path run kandidat saat mencoba webcam. Path baseline tidak berubah:

```powershell
python scripts/webcam_coordinates.py --run-dir PATH_RUN_KANDIDAT
```

## Colab

Buka [notebook fine-tuning](../notebooks/03_sibi_jz_finetune_colab.ipynb), pilih GPU dan upload [paket siap pakai](../data/exports/coordinate-jz-ft-v1-sibi-colab.zip). Paket berisi koordinat baru, kode, selection baseline dan checkpoint awal; tidak ada foto atau array test.

Reproduksi paket:

```powershell
python scripts/package_coordinates_colab.py --language sibi --features-dir data/features/coordinate-jz-ft-v1/sibi --init-run artifacts/coordinate-reset-v1/sibi_20261020_130607_889055
```

## Batas keputusan

Perbaikan J tidak dijanjikan karena pose tambahan hampir sama. Validation yang sama digunakan lagi untuk pemilihan kandidat, sehingga peningkatan nanti bukan evaluasi independen. Jika hasil tidak meningkat, pertahankan baseline yang sudah berjalan dan catat keterbatasan pose ASL J/Z; jangan memindahkan sampel validation ke train atau menyebut pelatihan lebih lama sebagai perbaikan yang pasti.

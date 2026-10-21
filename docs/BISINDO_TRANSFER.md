# Transfer encoder SIBI ke checkpoint BISINDO

Status: training run pertama sudah selesai dan ditinjau. CNN transfer mencapai macro-F1 sistem validation 86,41%; checkpoint SIBI tetap utuh. Model belum mengalahkan SVM lama pada validation yang sama. Lihat [review run 01](BISINDO_TRANSFER_RUN01.md).

Keputusan pengguna: dua checkpoint terpisah dalam satu rancangan aplikasi dengan pilihan mode bahasa. Checkpoint SIBI yang sudah cukup bagus tidak ditimpa. Training penuh BISINDO dijalankan pengguna agar setiap epoch dapat dipantau.

## Model dan data

Sumber: `artifacts/coordinate-reset-v1/sibi_20261020_130607_889055`, CNN–Softmax epoch 89. Hanya bobot **encoder** yang disalin. Classifier BISINDO 26 kelas diinisialisasi baru; pemetaan huruf SIBI tidak dianggap sama dengan BISINDO. Arsitektur dan praproses tetap `(42, 7)` seperti model koordinat sebelumnya, sehingga kode inferensi bersama dapat dipakai tanpa mengubah kontrak model SIBI.

Data: `data/features/coordinate-reset-v1/bisindo/`, dari cache Kaggle split v2 confidence deteksi 0,3. Tidak mengulang ekstraksi atau menggabungkan dataset SIBI/ASL ke training BISINDO.

| Split | Semua gambar | Fitur valid | Satu tangan valid | Dua tangan valid |
|---|---:|---:|---:|---:|
| Train | 8.341 | 6.680 | 3.844 | 2.836 |
| Validation | 1.593 | 1.401 | 808 | 593 |

Terdapat 1.661 train dan 192 validation gagal ekstraksi. Baris tersebut tidak dipakai melatih classifier, tetapi tetap dihitung sebagai penolakan pada metrik sistem. Karena itu classifier yang benar pada semua 1.401 baris valid sekalipun hanya mencapai sekitar 87,95% akurasi sistem pada seluruh 1.593 validation gambar dengan cache ini. Macro-F1 tidak memiliki batas numerik yang sama persis. Satu tangan valid belum menjamin kelengkapan suatu pose; dua tangan pun belum menjamin pemeraga/label benar.

## Dua tahap training

Konfigurasi: `configs/bisindo_transfer_v1.json`.

1. **Head**: maksimal 10 epoch, encoder dibekukan, classifier baru belajar dengan learning rate 0,001.
2. **Fine-tune**: mulai dari checkpoint head terbaik; encoder dibuka dengan LR 0,0001, classifier LR 0,0003, maksimal 60 epoch, patience 12.

Sampling kelas seimbang, loss tidak sekaligus diberi bobot kelas. Augmentasi rotasi kecil/noise hanya pada training. Checkpoint terbaik dipilih menurut macro-F1 sistem validation dari kedua tahap, sehingga hasil akhir bisa berasal dari tahap head bila fine-tuning menurunkannya. Tidak ada guard akurasi SIBI pada validation BISINDO karena checkpoint SIBI tetap terpisah dan tidak dilatih.

Tidak ada janji transfer lebih baik daripada training BISINDO dari nol; manfaat tersebut belum dibandingkan. Model SIBI mempelajari satu tangan saja, sedangkan training BISINDO perlu mengadaptasi hubungan dua tangan.

## Jalankan lokal

Gunakan environment training PyTorch yang sudah berhasil dipakai. Data dan konfigurasi sudah tersedia:

```powershell
python scripts/train_bisindo_transfer.py
```

Terminal/CSV menampilkan tahap, epoch, loss, train accuracy, validation valid accuracy, system macro-F1, dan best epoch. Hasil tersimpan pada direktori baru `artifacts/bisindo-transfer-v1/bisindo_<waktu>/`. Jangan mengganti path baseline SIBI dengan run ini pada aplikasi mode SIBI.

Artefak berisi `head_best.pt`, `finetune_best.pt`, `cnn_best.pt`, `selection.json`, `dataset.json`, `transfer_config.json`, `cnn_epochs.csv`, kurva, dan metrik per huruf/sumber/jumlah tangan. `selection.json` tetap kompatibel dengan loader webcam yang ada. Hash sumber SIBI dicek sebelum/sesudah training; fase head juga memeriksa encoder tetap identik.

## Colab GPU

Buka [notebook transfer BISINDO](../notebooks/04_bisindo_transfer_colab.ipynb), aktifkan GPU, lalu upload [paket training](../data/exports/bisindo-transfer-v1-bisindo-colab.zip), sekitar 3,15 MiB. Paket berisi train/validation koordinat, kode, konfigurasi, serta salinan checkpoint/selection SIBI sebagai sumber encoder. Tidak memuat foto atau array test.

Reproduksi paket:

```powershell
python scripts/package_coordinates_colab.py --language bisindo --init-run artifacts/coordinate-reset-v1/sibi_20261020_130607_889055 --transfer-config configs/bisindo_transfer_v1.json
```

Unduh ZIP hasil notebook, ekstrak ke akar proyek dengan mempertahankan struktur `artifacts/`. Untuk run Colab:

```powershell
python scripts/webcam_coordinates.py --run-dir artifacts/bisindo-transfer-v1/bisindo_colab --check-only
python scripts/webcam_coordinates.py --run-dir artifacts/bisindo-transfer-v1/bisindo_colab
```

Untuk run lokal, ganti path dengan direktori hasil yang dicetak trainer. Loader memilih bahasa dari metadata run: SIBI menggunakan konfigurasi deteksi lama SIBI, BISINDO menggunakan confidence 0,3 yang sesuai cache. Deteksi dan normalisasi tidak ditukar hanya karena bobot encoder berawal dari SIBI.

## Dua checkpoint yang dipakai

- Mode SIBI: tetap run `coordinate-reset-v1/sibi_20261020_130607_889055`.
- Mode BISINDO: run terbaik hasil training transfer yang akan dijalankan pengguna; belum disebut model siap pakai sebelum training dan review selesai.

Endpoint FastAPI/HTML belum ditambahkan pada langkah ini. Setelah hasil training dan webcam BISINDO diperiksa, kedua checkpoint dapat dipilih melalui satu service.

# BISINDO recovery v2: data training dan webcam

Status 21 Oktober 2026: ekstraksi otomatis, persiapan tensor, dan paket Colab selesai. Training dataset nyata belum dijalankan. Checkpoint SIBI dan BISINDO sebelumnya tidak ditimpa.

## Perbaikan kualitas

Filter v1 membandingkan landmark dengan nomor yang sama. V2 menambahkan jarak simetris titik terdekat antarkumpulan 21 landmark, sehingga rangka yang bertumpuk tetapi susunan jarinya berbeda juga diperiksa. Ambang jarak sepadan 0,85 telapak dan jarak kumpulan titik 0,55 telapak adalah heuristik konservatif; tangan sungguhan yang berimpitan juga bisa ditolak.

Aturan digunakan pada fallback view, tanpa label huruf, anotasi kotak, atau pengecualian per foto. Proposal mencurigakan dicoba ulang dari baseline menggunakan aturan yang sama pada setiap view. Baseline dua tangan tetap dipertahankan. Semua perubahan masih menyimpan flag `review_required`; hasil bukan anotasi ruas jari yang seluruhnya diverifikasi manusia.

| Data | Jumlah gambar | Valid baseline | Valid v2 | Tidak ada landmark v2 | Dua tangan v2 |
|---|---:|---:|---:|---:|---:|
| Train | 8.341 | 6.680 (80,09%) | 7.871 (94,37%) | 470 | 3.593 |
| Validation | 1.593 | 1.401 (87,95%) | 1.537 (96,48%) | 56 | 765 |

Sebanyak 82 proposal train dan 17 validation dicoba ulang. Contoh X train indeks 5977 akhirnya kembali menjadi gagal deteksi, menggantikan dua rangka yang salah. O validation indeks 1457 kembali ke baseline satu tangan secara otomatis. Tidak memakai CSV koreksi manual v1. Penurunan satu gambar valid train dibanding v1 adalah akibat penolakan hasil X tersebut. Gambar tidak dibuang dari paket atau denominator evaluasi.

[Grafik](../data/reports/bisindo-recovery-v2/overview.png), [per huruf](../data/reports/bisindo-recovery-v2/per_letter.png), [laporan](../data/reports/bisindo-recovery-v2/report.md), dan [perbandingan koreksi](../data/reports/bisindo-recovery-v2/quality_corrections.jpg).

## Data dan kontrak inferensi

- Cache koordinat: `data/features/bisindo-recovery-clean-v2/`.
- Tensor classifier: `data/features/coordinate-bisindo-recovery-v2/bisindo/`, bentuk `(N,42,7)`.
- Policy lengkap dan hash implementasi: `runtime_policy.json` pada cache, disalin ke metadata dataset training.
- Semua 26 kelas tersedia pada train/validation. Gambar tanpa fitur tetap memiliki mask false dan tensor nol; tidak digunakan sebagai sampel classifier, tetapi tetap dihitung pada metrik sistem validation.
- Foto, label, split, dan cache lama tetap disimpan. Array test tidak dibuka.

`RuntimeRecoveryExtractor` dipakai bersama untuk retry offline dan webcam. RGB tidak dimirror; hanya preview yang dicerminkan. Rotasi/crop dipetakan kembali ke koordinat asli sebelum normalisasi. Model baru menyimpan hash policy dan metadata dataset; webcam menolak metadata yang berubah atau jalur recovery yang hilang. Model lama tetap memakai baseline lama.

Pemeriksaan 13 gambar train/validation mencakup hasil nol/satu/dua tangan, ROI, rotasi, dan kasus duplikasi. Hasil runtime cocok dengan cache pada jumlah tangan, valid mask, dan tensor classifier (toleransi 1e-4). Ini pemeriksaan konsistensi, bukan evaluasi akurasi model. [Hasil pemeriksaan](../data/reports/bisindo-recovery-v2/runtime_parity.json).

## Training lokal

Gunakan environment Python 3.12 yang sudah memuat dependency proyek:

```powershell
& 'C:\Program Files\Python312\python.exe' scripts/train_bisindo_transfer.py --config configs/bisindo_transfer_recovery_v2.json
```

Tetap memakai encoder SIBI terbaik sebagai inisialisasi, classifier BISINDO baru, 10 epoch head lalu maksimal 60 epoch fine-tuning. Learning rate dan augmentasi sama dengan eksperimen transfer v1 agar perubahan data lebih mudah dibandingkan. Loss, train accuracy, validation accuracy dan system macro-F1 dicetak tiap epoch. Early stopping aktif. Pemilihan checkpoint berdasarkan validation system macro-F1.

Run baru otomatis disimpan di `artifacts/bisindo-transfer-recovery-v2/bisindo_<timestamp>/`. Model SIBI sumber `artifacts/coordinate-reset-v1/sibi_20261020_130607_889055` tetap utuh. Belum ada klaim bahwa transfer lebih baik daripada training dari awal.

## Training Colab GPU

Buka [notebook 05](../notebooks/05_bisindo_recovery_v2_colab.ipynb), aktifkan runtime GPU, lalu upload [paket ZIP lokal](../data/exports/bisindo-transfer-recovery-v2-bisindo-colab.zip). Paket sekitar 3,7 MiB hanya berisi kode training, tensor train/validation, dan checkpoint sumber; foto dan array test tidak disertakan.

Notebook menampilkan hasil per epoch dan mengharuskan CUDA pada perintah training. Unduh ZIP hasil lalu ekstrak ke akar proyek, mempertahankan direktori `artifacts/`.

## Webcam setelah training

Contoh jika menggunakan output default notebook:

```powershell
& 'C:\Program Files\Python312\python.exe' scripts/webcam_coordinates.py --run-dir artifacts/bisindo-transfer-recovery-v2/bisindo_colab --check-only
& 'C:\Program Files\Python312\python.exe' scripts/webcam_coordinates.py --run-dir artifacts/bisindo-transfer-recovery-v2/bisindo_colab
```

Untuk training lokal, ganti run-dir dengan folder timestamp hasilnya. Preview menampilkan metode recovery dan jumlah view. Webcam tidak merekam gambar/video. Skor softmax belum terkalibrasi dan tidak ada kelas unknown/background.

Kebijakan multi-view menambah latensi. Pada 13 gambar pemeriksaan, satu pemrosesan menghabiskan sekitar 32–408 ms pada CPU sesi ini; ini bukan benchmark FPS webcam. Jangan mengganti runtime diam-diam menjadi baseline agar lebih cepat karena input classifier akan berbeda dari training. Optimasi latensi adalah langkah berikutnya setelah kualitas kandidat model dinilai.

## Reproduksi persiapan

Hasil sudah tersedia; tidak perlu menjalankan ulang sebelum training:

```powershell
& 'C:\Program Files\Python312\python.exe' scripts/clean_bisindo_recovery.py
& 'C:\Program Files\Python312\python.exe' scripts/prepare_coordinates.py --config configs/coordinate_bisindo_recovery_v2.json --languages bisindo
& 'C:\Program Files\Python312\python.exe' scripts/package_coordinates_colab.py --language bisindo --features-dir data/features/coordinate-bisindo-recovery-v2/bisindo --init-run artifacts/coordinate-reset-v1/sibi_20261020_130607_889055 --transfer-config configs/bisindo_transfer_recovery_v2.json
```

Persiapan tensor menolak output yang sudah terisi; gunakan versi baru jika policy berubah. Validasi source hash menjaga agar hasil lama tidak dipakai bersama kode recovery yang berbeda.

# Ringkasan cakupan ekstraksi landmark v1

Konfigurasi: `landmarks-v1`; hash: `efdbc7028b89c7825d8a9dd1ea45e5e49ac841ec9887e1fade10712ad555b5d4`. Angka feature-valid mengukur geometri dan policy tangan saja, bukan akurasi huruf atau kebenaran landmark. Median/p95 adalah waktu preprocessing+deteksi lokal, tanpa I/O dan jaringan.

| Bahasa | Split | Valid / total | Cakupan | Median ms | P95 ms | Jumlah tangan (0/1/2) |
|---|---|---:|---:|---:|---:|---|
| SIBI | train | 993 / 1001 | 99.20% | 21.49 | 26.09 | 8/993/0 |
| SIBI | val | 215 / 216 | 99.54% | 21.42 | 25.07 | 1/215/0 |
| SIBI | test | 213 / 216 | 98.61% | 22.30 | 27.21 | 3/213/0 |
| BISINDO | train | 4235 / 6480 | 65.35% | 22.32 | 33.27 | 2245/2778/1457 |
| BISINDO | val | 762 / 780 | 97.69% | 27.59 | 33.68 | 18/320/442 |
| BISINDO | test | 770 / 780 | 98.72% | 28.70 | 33.16 | 10/150/620 |

## Cakupan per kelas

Sel yang tidak penuh menunjukkan jumlah contoh yang tidak feature-valid pada kelas tersebut. Untuk BISINDO, valid satu tangan tetap bersifat sementara karena kelengkapan pose belum terverifikasi.

### SIBI

| Kelas | Train valid/total | Validation valid/total | Test valid/total |
|---|---:|---:|---:|
| A | 42/42 | 9/9 | 8/9 |
| B | 41/41 | 9/9 | 9/9 |
| C | 42/42 | 9/9 | 9/9 |
| D | 40/41 | 8/9 | 9/9 |
| E | 42/42 | 9/9 | 9/9 |
| F | 42/42 | 9/9 | 9/9 |
| G | 42/42 | 9/9 | 9/9 |
| H | 42/42 | 9/9 | 9/9 |
| I | 41/41 | 9/9 | 9/9 |
| K | 42/42 | 9/9 | 9/9 |
| L | 42/42 | 9/9 | 9/9 |
| M | 42/42 | 9/9 | 9/9 |
| N | 42/42 | 9/9 | 9/9 |
| O | 40/41 | 9/9 | 9/9 |
| P | 35/41 | 9/9 | 7/9 |
| Q | 42/42 | 9/9 | 9/9 |
| R | 42/42 | 9/9 | 9/9 |
| S | 42/42 | 9/9 | 9/9 |
| T | 42/42 | 9/9 | 9/9 |
| U | 42/42 | 9/9 | 9/9 |
| V | 41/41 | 9/9 | 9/9 |
| W | 42/42 | 9/9 | 9/9 |
| X | 41/41 | 9/9 | 9/9 |
| Y | 42/42 | 9/9 | 9/9 |

### BISINDO

| Kelas | Train valid/total | Validation valid/total | Test valid/total |
|---|---:|---:|---:|
| A | 247/280 | 30/30 | 30/30 |
| B | 240/280 | 30/30 | 30/30 |
| C | 154/280 | 30/30 | 30/30 |
| D | 171/280 | 30/30 | 30/30 |
| E | 154/280 | 18/30 | 22/30 |
| F | 147/280 | 30/30 | 30/30 |
| G | 148/240 | 29/30 | 30/30 |
| H | 157/240 | 30/30 | 30/30 |
| I | 117/240 | 30/30 | 30/30 |
| J | 111/240 | 30/30 | 30/30 |
| K | 205/240 | 30/30 | 30/30 |
| L | 125/240 | 30/30 | 30/30 |
| M | 220/240 | 30/30 | 30/30 |
| N | 221/240 | 30/30 | 30/30 |
| O | 135/240 | 29/30 | 30/30 |
| P | 208/240 | 30/30 | 30/30 |
| Q | 190/240 | 30/30 | 30/30 |
| R | 116/240 | 30/30 | 30/30 |
| S | 224/240 | 30/30 | 30/30 |
| T | 182/240 | 29/30 | 30/30 |
| U | 108/240 | 28/30 | 30/30 |
| V | 99/240 | 30/30 | 28/30 |
| W | 132/240 | 30/30 | 30/30 |
| X | 168/240 | 30/30 | 30/30 |
| Y | 117/240 | 29/30 | 30/30 |
| Z | 139/240 | 30/30 | 30/30 |

## Interpretasi dan batasan

- SIBI: feature-valid 993/1.001 train (99,20%), 215/216 validation (99,54%), dan 213/216 test (98,61%). Semua deteksi lolos adalah satu tangan; contoh gagal tidak dijadikan fitur nol.
- BISINDO: train 4.235/6.480 (65,35%), validation 762/780 (97,69%), test 770/780 (98,72%). Perbedaan train terhadap validation/test sangat besar dan terlihat pula pada contoh visual: beberapa frame train diambil dari jarak jauh atau memperlihatkan banyak orang, sehingga MediaPipe sering tidak mendeteksi tangan. Split berdasarkan seri nama berpotensi menggabungkan karakteristik pengambilan yang berbeda.
- Banyak frame BISINDO terdeteksi dua tangan, tetapi gambar dapat mencakup orang lain. Jumlah tangan dan actor assignment tidak membuktikan bahwa tangan pose berasal dari pelaku yang sama.
- Jangan lanjut ke klaim performa BISINDO atau merangking classifier tanpa membahas cakupan per kelas dan bias split tersebut. Untuk MVP webcam, ukur ulang dengan data webcam terkurasi dari satu pelaku utama dan pose referensi terkonfirmasi.
- Hasil test diekstrak setelah preprocessing dikunci dan tidak dipakai memilih konfigurasi. Angka di atas adalah cakupan ekstraksi, bukan metrik classifier.

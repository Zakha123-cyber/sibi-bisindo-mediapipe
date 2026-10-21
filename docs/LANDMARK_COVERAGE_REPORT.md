# Cakupan ekstraksi landmark tersimpan

Pemeriksaan 20 Oktober 2026. Angka dihitung dari CSV hasil ekstraksi, dicocokkan dengan manifest dan `summary.json`. Tidak menjalankan ulang MediaPipe, training, prediksi test, atau membuka gambar sumber. Persentase berikut memakai penyebut **gambar yang sudah diproses**, bukan semua varian dataset.

| Dataset/cache | Diproses | Fitur valid | Tidak valid/no_hand | Valid (%) |
|---|---:|---:|---:|---:|
| SIBI Mendeley, landmarks-v1-final | 1.433 | 1.421 | 12 | 99,16 |
| BISINDO Mendeley, landmarks-v1-final | 8.040 | 5.767 | 2.273 | 71,73 |
| BISINDO Kaggle v2, confidence 0,3 | 9.934 | 8.081 | 1.853 | 81,35 |

## Apakah semua gambar sudah diproses?

- **SIBI:** semua 1.433 gambar pada subset kurasi telah diproses. Inventaris manifest awal memuat 31.949 gambar dari beberapa varian; 30.516 berada di luar protokol terpilih. Jadi 99,16% bukan tingkat keberhasilan seluruh varian SIBI. Subset hanya mencakup 24 kelas tanpa J/Z.
- **BISINDO Mendeley:** seluruh 8.040 gambar pada manifest telah diproses.
- **BISINDO Kaggle:** seluruh train/validation terpilih telah diproses. Sebanyak 1.534 gambar test pada split v2 belum diekstrak di cache confidence 0,3. Dari 11.470 gambar pada manifest, dua dikecualikan karena masalah anotasi. Tidak memproses ulang cache/split v1 atau konfigurasi lain untuk mengisi kekosongan ini.

Ini cakupan terhadap inventaris manifest yang sudah ada, bukan penghitungan ulang setiap gambar baru pada filesystem. Video J/Z, file landmark ASL dari pihak lain, dan koleksi webcam tidak digabung ke penyebut dataset gambar.

## Per split

| Dataset | Split | Valid / diproses | Valid (%) |
|---|---|---:|---:|
| SIBI Mendeley | Train | 993 / 1.001 | 99,20 |
| SIBI Mendeley | Validation | 215 / 216 | 99,54 |
| SIBI Mendeley | Test historis | 213 / 216 | 98,61 |
| BISINDO Mendeley | Train | 4.235 / 6.480 | 65,35 |
| BISINDO Mendeley | Validation | 762 / 780 | 97,69 |
| BISINDO Mendeley | Test historis | 770 / 780 | 98,72 |
| BISINDO Kaggle | Train | 6.680 / 8.341 | 80,09 |
| BISINDO Kaggle | Validation | 1.401 / 1.593 | 87,95 |
| BISINDO Kaggle | Test | Belum diekstrak | — |

Kesenjangan ekstraksi train/validation BISINDO Mendeley besar; nilai keseluruhan saja menyembunyikan perbedaan kondisi antarsplit. Penyebab per gambar tidak dapat disimpulkan hanya dari log ini. Angka test dilaporkan ulang dari hasil lama, tidak digunakan untuk memilih konfigurasi baru.

## Visualisasi dan tabel rinci

Artefak lokal yang dapat dibuka:

- [Grafik cakupan](../data/reports/landmark-coverage/overview.png): valid, gagal, dan belum diproses; persentase grafik memakai semua gambar terpilih agar kategori belum diproses tetap terlihat.
- [Heatmap per huruf/split](../data/reports/landmark-coverage/per_letter.png): persentase valid dari yang diproses, abu-abu berarti kelas/split belum tersedia.
- [Distribusi 0/1/2 tangan](../data/reports/landmark-coverage/hand_counts.png).
- [Laporan lengkap](../data/reports/landmark-coverage/report.md), [CSV per huruf](../data/reports/landmark-coverage/by_letter.csv), [CSV per split](../data/reports/landmark-coverage/by_split.csv), dan [riwayat konfigurasi Kaggle](../data/reports/landmark-coverage/previous_settings.csv).

Huruf dengan persentase valid train terendah:

- SIBI: P 85,37% (35/41), D dan O 97,56% (40/41).
- BISINDO Mendeley: V 41,25%, U 45,00%, J 46,25%, R 48,33%.
- BISINDO Kaggle confidence 0,3: A 61,56%, Y 65,00%, K 68,71%, Q 69,60%.

**Valid bukan akurasi prediksi atau ketepatan landmark.** Semua kegagalan pada tiga cache ini berstatus no_hand. Namun, BISINDO satu tangan bisa tetap berstatus valid; ini tidak memastikan tangan kedua yang dibutuhkan suatu pose terdeteksi. Sebanyak 3.248 gambar BISINDO Mendeley dan 4.652 gambar Kaggle hanya menghasilkan satu tangan. Dua tangan terdeteksi juga tidak otomatis membuktikan label pose atau pemeraganya benar.

Kesimpulan untuk eksperimen berikutnya: landmark SIBI pada subset yang dipakai sudah tersedia hampir seluruhnya. BISINDO mengalami kehilangan data lebih besar dan sebagian hasil satu tangan belum terverifikasi kelengkapannya; mengganti classifier tidak memulihkan koordinat tangan yang gagal diekstrak.

Reproduksi laporan: `python scripts/report_landmark_coverage.py`. Artefak lokal berada di `data/reports/landmark-coverage/` dan diabaikan Git; skrip dan dokumen ringkasan tetap dapat dilacak.

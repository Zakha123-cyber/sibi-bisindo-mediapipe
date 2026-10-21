# Review transfer BISINDO run 01

Run `artifacts/bisindo-transfer-v1/bisindo_20261020_152850_281288`, ditinjau 20 Oktober 2026. Hash paket data, artefak, dan sumber SIBI cocok; prediksi validation checkpoint terpilih berhasil direproduksi. Encoder pada checkpoint head tetap identik dengan sumber SIBI. Tidak membuka test atau melatih ulang saat review.

## Hasil

Training: 10 epoch head + 57 epoch fine-tuning, berhenti setelah 12 epoch tanpa perbaikan. Checkpoint terbaik: epoch total 55, yaitu epoch 45 tahap fine-tuning. Checkpoint terakhir epoch 67 tidak dipakai sebagai bobot terbaik.

| Ukuran | Nilai |
|---|---:|
| Validation seluruh gambar | 1.593 |
| Landmark/fitur valid | 1.401 (87,95%) |
| Prediksi benar | 1.300 |
| Salah klasifikasi pada fitur valid | 101 |
| Ditolak akibat ekstraksi gagal | 192 |
| Akurasi classifier, hanya fitur valid | 92,79% |
| Macro-F1 classifier, hanya fitur valid | 92,54% |
| Akurasi sistem, semua gambar | 81,61% |
| Macro-F1 sistem, semua gambar | 86,41% |
| Akurasi pada 808 gambar valid satu tangan | 92,45% |
| Akurasi pada 593 gambar valid dua tangan | 93,25% |
| Median classifier satu sampel CPU | 0,193 ms |

Waktu classifier tidak mencakup kamera/MediaPipe/praproses/UI dan bukan FPS sistem. Satu tangan valid belum membuktikan kelengkapan pose; angka kelompok jumlah tangan tidak sama dengan keberhasilan pose satu/dua tangan yang diverifikasi.

## Perubahan antar tahap

| Tahap terbaik | Akurasi fitur valid | Macro-F1 sistem | Akurasi valid dua tangan |
|---|---:|---:|---:|
| Head, encoder beku, epoch 9 | 73,38% | 67,19% | 60,88% |
| Fine-tuning, epoch total 55 | 92,79% | 86,41% | 93,25% |

Membuka encoder membantu dalam run ini. Namun, manfaat memulai dari encoder SIBI dibanding training BISINDO dari nol belum dibuktikan karena kontrol dari nol belum dijalankan dengan representasi/config yang sama.

## Pembanding yang sudah tersedia

SVM linear C=1 lama, pada cache BISINDO Kaggle v2 confidence 0,3, mencatat macro-F1 sistem 89,81%, dibanding CNN transfer 86,41%. Kedua cache validation dicocokkan menurut path, label dan valid mask: sama 1.593 gambar. Representasinya berbeda (geometri 132 vs tensor koordinat 42×7), sehingga ini pembanding pipeline, bukan isolasi dampak transfer.

Model transfer belum menjadi pemenang offline keseluruhan. Skor yang lebih rendah tidak otomatis membuktikan webcam lebih buruk, tetapi tidak ada dasar menyatakan model baru sudah lebih andal di webcam sebelum dicoba.

## Kesalahan utama

Di antara fitur valid, recall A terendah: 25/37 = 67,57%; S 41/51 = 80,39%; B 45/53 = 84,91%. Contoh kebingungan: A→D lima, D→C lima, N→M lima, B→E empat, H→R empat. Untuk A secara sistem, 25/66 benar: 29 gambar gagal ekstraksi dan 12 salah klasifikasi. Jadi masalah A terdapat pada deteksi dan classifier sekaligus.

Kurva loss turun dan validation mendatar pada akhir training; tidak perlu menambah epoch secara otomatis. Langkah berikutnya mencoba run ini di webcam, mempertahankan SIBI dan SVM BISINDO lama sebagai pembanding, tanpa klaim model siap produksi.

```powershell
python scripts/webcam_coordinates.py --run-dir artifacts/bisindo-transfer-v1/bisindo_20261020_152850_281288
```

[Kurva](../artifacts/bisindo-transfer-v1/bisindo_20261020_152850_281288/learning_curves.png), [metrik](../artifacts/bisindo-transfer-v1/bisindo_20261020_152850_281288/cnn_softmax_validation.json), dan [selection](../artifacts/bisindo-transfer-v1/bisindo_20261020_152850_281288/selection.json).

# Pemeriksaan keseimbangan kelas dan format fitur SIBI

Pemeriksaan 20 Oktober 2026, tanpa training ulang. CSV split dicocokkan dengan array training NPZ; semua fitur training bertanda valid bersifat finite dan memiliki satu slot tangan terisi. Test hanya dihitung ulang dari log historis, tidak dibuka array/prediksinya.

## Keseimbangan kelas

| Split | Gambar | Fitur valid | Sampel/kelas sebelum ekstraksi | Sampel valid/kelas | Rasio terbanyak/tersedikit |
|---|---:|---:|---:|---:|---:|
| Train | 1.001 | 993 | 41–42 | 35–42 | 1,20 |
| Validation | 216 | 215 | 9 | 8–9 | 1,125 |
| Test historis | 216 | 213 | 9 | 7–9 | 1,286 |

Tidak ada ketimpangan jumlah kelas yang besar pada subset ini. Huruf P paling banyak kehilangan training: 41 gambar menjadi 35 valid. D/O masing-masing 41 menjadi 40; kelas lain tidak kehilangan training akibat ekstraksi. Kelas training terbesar hanya 4,23% dari 993 sampel valid. Pipeline SVM sebelumnya juga memakai `class_weight='balanced'`.

[Grafik per kelas](../data/reports/sibi-balance/class_balance.png), [CSV](../data/reports/sibi-balance/per_class.csv), dan [ringkasan JSON](../data/reports/sibi-balance/summary.json) dibuat oleh `python scripts/report_sibi_balance.py`.

## Mengapa hasil webcam masih bisa buruk?

- Keberhasilan ekstraksi 99,16% hanya menyatakan geometri/policy valid, bukan landmark tepat atau classifier mengenali pose dengan benar.
- Model hanya dilatih dengan 993 contoh valid, sekitar 35–42 per huruf, dari subset mono-background. Jumlahnya hampir seimbang tetapi belum membuktikan keragaman tangan, orientasi, atau kondisi webcam.
- ID pemeraga pada manifest kosong. Pemisahan grup sumber tersedia, tetapi generalisasi pada pemeraga baru belum terverifikasi.
- Normalisasi mengurangi pengaruh posisi dan ukuran, namun mempertahankan orientasi tangan. Perbedaan arah hadap/perspektif, oklusi jari, dan jitter tetap dapat menggeser fitur. Ini kandidat penyebab, bukan diagnosis yang sudah terbukti per trial webcam.
- SVM lama memilih salah satu kelas ketika ekstraksi valid; belum ada ambang penolakan pose di luar kelas yang terkalibrasi. Tidak semua gerakan biasa merupakan huruf.
- Log webcam yang tersimpan tidak memiliki label acuan per frame, sehingga histogram prediksi tidak cukup menghitung akurasi atau membuktikan satu penyebab kegagalan.

Kegagalan modul temporal J/Z terpisah dari analisis keseimbangan 24 kelas statis ini.

## MLP sudah pernah dicoba

Artefak lama `artifacts/classification-v1/sibi/selection.json` mencatat hasil validation pada **fitur geometri 132 dimensi yang sama**:

| Kandidat | Akurasi sistem validation | Macro-F1 sistem validation |
|---|---:|---:|
| SVM linear C=1 | 97,69% | 97,87% |
| MLP 128 | 94,91% | 95,07% |
| MLP 128→64 | 95,83% | 96,06% |

Jadi MLP bukan pendekatan yang sepenuhnya baru pada proyek ini. Eksperimen berikutnya perlu menyebut perubahan representasi/augmentasi secara jelas, bukan menjanjikan perbaikan hanya dari mengganti nama model. CNN1D/CNN–SVM dapat menjadi pembanding baru pada split yang sama.

## Format penyimpanan

Hasil proyek ini berupa **NPZ terkompresi**, bukan satu file NPY per gambar. NPZ mengemas beberapa array NumPy dengan nama; format berkas dipilih kode ekstraksi kita, bukan format keluaran wajib MediaPipe. MediaPipe menghasilkan objek landmark di memori.

Contoh `data/features/landmarks-v1-final/sibi_train.npz`:

| Array | Shape | Isi |
|---|---|---|
| raw_landmarks | (1001, 2, 21, 3) | Koordinat mentah per gambar, maksimal dua slot tangan, 21 titik xyz |
| X | (1001, 132) | Fitur geometri terolah: penanda slot, koordinat lokal ternormalisasi, fitur pasangan tangan |
| valid_mask | (1001,) | Penanda baris fitur yang layak dipakai |
| labels | (1001,) | Huruf acuan |
| hand_count | (1001,) | Jumlah tangan yang terdeteksi |
| width, height | masing-masing (1001,) | Dimensi gambar untuk koreksi rasio aspek |

Array lain menyimpan path, hash, grup sumber, status, handedness, dan provenance ekstraksi. Slot tangan yang tidak terisi memakai NaN. Pada 993 baris SIBI valid, tepat satu slot tangan terisi. Jangan memasukkan slot kosong atau baris gagal sebagai koordinat tangan normal.

```python
import numpy as np

with np.load('data/features/landmarks-v1-final/sibi_train.npz', allow_pickle=False) as data:
    valid = data['valid_mask']
    X = data['X'][valid]                  # (993, 132), fitur geometri lama
    y = data['labels'][valid]             # (993,)
    raw = data['raw_landmarks'][valid]    # (993, 2, 21, 3), masih ada slot kosong
```

Tidak perlu mengekstrak ulang gambar hanya untuk mendapatkan koordinat: `raw_landmarks` sudah tersimpan. Pemakaian raw untuk representasi baru tetap perlu normalisasi konsisten dan penanganan slot kosong, terutama bila diperluas ke BISINDO.

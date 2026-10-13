# Audit dataset SIBI dan BISINDO

Audit membaca seluruh gambar tanpa mengubah dataset. Detail tersedia di `inventory.csv`, `summary.json`, dan `duplicates.json`.

| Varian | Gambar | Kelas | Gambar rusak | Masalah anotasi |
|---|---:|---:|---:|---:|
| BISINDO/Mendeley BISINDO | 8040 | 26 | 0 | 0 |
| SIBI/sibi_alphabet_mono_bg_no_augmentation | 1440 | 24 | 0 | 0 |
| SIBI/sibi_alphabet_mono_bg_with_augmentation | 3870 | 24 | 0 | 0 |
| SIBI/sibi_alphabet_multi_bg_no_augmentation | 7200 | 24 | 0 | 4 |
| SIBI/sibi_alphabet_multi_bg_with_augmentation | 19439 | 24 | 0 | 10 |

## Duplikasi

```json
{
  "sha256": {
    "groups": 8639,
    "cross_split_groups": 0,
    "cross_variant_groups": 8639,
    "conflicting_letter_groups": 0
  },
  "origin": {
    "groups": 8640,
    "cross_split_groups": 0,
    "cross_variant_groups": 8640,
    "conflicting_letter_groups": 0
  }
}
```

SHA-256 mendeteksi file identik, bukan kemiripan visual. Origin adalah kandidat sumber berdasarkan nama file; perlu verifikasi sebelum digunakan sebagai ID grup.
ID kelas SIBI disandingkan dengan huruf dari nama file, bukan metadata resmi. Periksa pemetaan dalam summary.json.
Label tanpa pasangan gambar: 1.

Pemeriksaan struktur ini belum mengukur keberhasilan MediaPipe, kesamaan visual, identitas pengguna, atau kebenaran linguistik pose. Jumlah kelas bukan jaminan dukungan pose statis.
Lihat review_summary.json untuk ketidaksesuaian kelas dan kandidat kebocoran sumber lintas latar. Tidak adanya duplikasi hash antar split bukan bukti split bebas kebocoran.
Hasil audit landmark terpisah tersedia di landmark_summary.json setelah menjalankan scripts/audit_landmarks.py. Interpretasi snapshot tersedia di FINDINGS.md.
Gunakan varian tanpa augmentasi untuk baseline. Pisahkan sumber/pengguna/sesi sebelum augmentasi; jangan gabungkan seluruh varian secara langsung.

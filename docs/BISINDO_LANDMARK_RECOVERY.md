# Pemulihan ekstraksi landmark BISINDO

Kelanjutan aktif: [recovery otomatis v2 dan paket training](BISINDO_RECOVERY_TRAINING_V2.md). Dokumen ini mencatat hasil v1; v2 menggantikan koreksi manual dengan aturan geometri bersama untuk offline/webcam.

Eksperimen 21 Oktober 2026. Tujuan: memperbaiki deteksi landmark pada gambar BISINDO Kaggle train/validation sebelum training classifier berikutnya. Dataset mentah, split, label, cache lama, dan checkpoint SIBI/BISINDO tetap disimpan. Tidak ada training classifier atau pembukaan test pada pekerjaan ini.

## Metode

Baseline tetap MediaPipe Hands 0.10.21, model full, confidence 0,3. Hasil dua tangan baseline langsung dipertahankan. Gambar dengan nol atau satu tangan dicoba melalui view tambahan berurutan: rotasi 90/270/180/±45 derajat, crop tengah, crop sisi yang saling overlap, CLAHE, dan pembesaran 1,5 kali. Proses tidak menggunakan label huruf atau kotak anotasi dataset untuk memilih ROI. Karena itu kebijakan yang sama bisa diimplementasikan pada frame webcam tanpa kotak acuan.

Semua xy hasil view dipetakan kembali memakai invers transformasi affine ke koordinat gambar asli. Relative z dikoreksi untuk perubahan lebar view dan zoom; tidak digabung menjadi kedalaman global antartangan. Dasar normalisasi xyz dijelaskan pada [dokumentasi MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python#handle_and_display_results). Geometri hubungan kedua tangan tetap dihitung pada gambar asli. Tidak ada flip/mirror tambahan pada input inferensi.

Kandidat harus finite, memiliki skala telapak tidak degenerat, dan tidak terlalu jauh keluar gambar. Kandidat dua tangan harus cukup terpisah. Untuk pemulihan satu menjadi dua tangan, salah satu kandidat harus cocok secara spasial dengan tangan yang sudah ditemukan baseline. Tidak menjahit beberapa potongan dari gambar berbeda menjadi satu pose dan tidak menyalin satu tangan menjadi dua slot.

View pertama yang meningkatkan jumlah tangan dan lolos pemeriksaan dipakai. Pada nol menjadi satu, pencarian berhenti; belum menjamin semua tangan pada gambar tersebut ditemukan. Ini batas metode saat ini, bukan klaim maksimal teoritis MediaPipe.

## Penyaringan duplikasi

Inspeksi overlay menemukan satu tangan fisik yang menghasilkan dua rangka, termasuk contoh J dan O serta beberapa P. Meningkatkan jumlah deteksi tanpa menyaring kasus ini akan menurunkan kualitas data.

Tahap kedua menghitung RMS xy pasangan landmark yang sepadan, dibagi mean skala telapak. Kandidat dua tangan baru di bawah jarak 0,85 telapak dianggap terlalu mirip dan dicoba ulang dengan aturan duplikasi aktif pada setiap view. Bila tidak ditemukan perbaikan yang lolos, baseline dipertahankan. Baseline dua tangan tidak diubah secara diam-diam. Ambang ini heuristik geometri; masih mungkin menolak tangan sungguhan yang saling menutupi atau melewatkan kesalahan lain. Semua perubahan tetap diberi penanda untuk review, bukan otomatis dianggap anotasi tangan benar.

## Hasil dan artefak

[Laporan angka/grafik](../data/reports/bisindo-recovery/report.md), [ringkasan](../data/reports/bisindo-recovery/overview.png), dan [per huruf](../data/reports/bisindo-recovery/per_letter.png).

- Proposal mentah: `data/features/bisindo-recovery-v1/`.
- Hasil setelah pemeriksaan duplikasi: **`data/features/bisindo-recovery-filtered-v1/`**. Gunakan ini sebagai calon sumber data berikutnya, bukan proposal mentah.
- NPZ mempertahankan `raw_landmarks`, `X` 132 dimensi, label, mask, path, hash, dan metadata. Terdapat method/attempts/waktu fallback serta `review_required`; hasil tetap koordinat, bukan dataset foto baru.
- CSV per gambar dan overlay sebelum/sesudah tersedia pada direktori cache final. Jenis grid: nol→satu, nol→dua, dan satu→dua tangan. Label huruf hanya dipakai pada laporan, tidak sebagai input detektor.

Validation setelah filter dan tinjauan teknis: **1.537/1.593 (96,48%)** memiliki fitur valid, dibanding 1.401/1.593 (87,95%) baseline. Gagal deteksi turun 192→56. Sepuluh proposal duplikat dicoba ulang. Satu contoh O yang masih mengandung dua rangka pada tangan yang sama dikembalikan ke baseline satu tangan, dengan keputusan berbasis path/hash pada [CSV tinjauan teknis](bisindo_recovery_review.csv). Ini tidak mengubah label atau menghapus gambar. Peningkatan bukan akurasi classifier atau sertifikasi landmark per ruas. Rincian train dan distribusi tangan akhir berada pada laporan final.

## Reproduksi

Train selesai: **7.872/8.341 (94,38%)**, dibanding 6.680/8.341 (80,09%); tanpa landmark turun dari 1.661 menjadi 469. Sebanyak 38 proposal duplikat dicoba ulang. Jumlah dua tangan akhir train 3.622 dan validation 771. Angka ini cakupan fitur, bukan akurasi pose.

Inspeksi grid akhir masih menunjukkan kandidat X train indeks 5977 dengan rangka bertumpuk pada satu sisi. Filter geometri belum menangkap seluruh kesalahan; semua 1.947 perubahan train dan 311 perubahan validation tetap ditandai `review_required`. Cache ini kandidat hasil pemulihan, belum anotasi bersih terverifikasi. Koreksi manual O pada validation membuat laporan ini pemulihan data semiotomatis, bukan evaluasi detektor otomatis. Jangan menganggap dua tangan terdeteksi pasti berarti dua tangan benar.

Tambahan waktu recovery rata-rata seluruh gambar train 192 ms dan validation 152 ms; p95 gabungan perkiraan masing-masing 553 dan 368 ms. Ini biaya ekstraksi offline, bukan pengukuran FPS webcam. Kebijakan banyak view perlu dioptimalkan sebelum dipakai realtime.

Pada sesi ini `python` default mengarah ke Python 3.10 Laragon yang tidak memiliki MediaPipe. Environment baseline yang cocok tersedia di Python 3.12; tidak ada pemasangan dependency baru:

```powershell
& 'C:\Program Files\Python312\python.exe' scripts/recover_bisindo_landmarks.py --splits train val --workers 3
& 'C:\Program Files\Python312\python.exe' scripts/filter_bisindo_recovery.py --splits train val
& 'C:\Program Files\Python312\python.exe' scripts/report_bisindo_recovery.py
```

Ekstraksi pertama resumable melalui partial NPZ dan mask per gambar. Worker hanya memparalelkan gambar independen, memakai detektor terpisah dengan kebijakan identik. Hasil final yang sudah ada dipakai kembali setelah pemeriksaan provenance. Perubahan konfigurasi/implementasi recovery memerlukan versi/path output baru; cache lama tidak ditimpa. Jumlah worker bukan penentu pose atau label.

Konfigurasi: `configs/bisindo_recovery_v1.json` dan `configs/bisindo_recovery_quality_v1.json`. Kode bersama: `landmark_recovery.py` dan `recovery_quality.py`.

## Sebelum training berikutnya

1. Tinjau overlay pemulihan, terutama satu menjadi dua tangan dan pose dengan oklusi. Pemeriksaan sampel pada pekerjaan ini tidak mengesahkan seluruh gambar.
2. Persiapan tensor training baru harus mempertahankan provenance recovery/filter serta skema koordinat yang sama. Jangan mengubah file dalam paket training lama.
3. Jalur inferensi model baru harus memanggil recovery/filter yang sama; jangan training dengan crop lalu webcam diam-diam memakai baseline tanpa fallback. Modul disiapkan agar dapat dipakai bersama, tetapi demo model lama belum diubah karena bobotnya dilatih pada praproses lama.
4. Ukur biaya tambahan. Percobaan banyak view lebih lambat; cache menyimpan baseline `process_ms` dan tambahan `recovery_ms` terpisah. Perkiraan gabungan bukan FPS webcam, terlebih ekstraksi diparalelkan. Keberhasilan offline ini belum menjamin latensi realtime yang sesuai.
5. Baru jalankan training kandidat BISINDO terpisah setelah format/policy offline–webcam disamakan. Tidak ada training otomatis dalam eksperimen ekstraksi ini.

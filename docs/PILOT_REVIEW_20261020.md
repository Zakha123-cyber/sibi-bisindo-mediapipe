# Pemeriksaan koleksi pilot 20 Oktober 2026

Audit lokal: `python scripts/audit_webcam_pilot.py`. Laporan, tabel cakupan, dan grid berada di `data/pilot_audit/v1/` (tidak diunggah Git). Foto asli tidak diubah. Foto peserta test tidak dibuka atau dirender oleh auditor.

## Hasil teknis

| Mode, peserta/sesi | Positif | Negatif | Foto |
|---|---:|---:|---:|
| SIBI p001/s01 | 25/25 | 4/8 | 348 |
| BISINDO p001/s01 | 25/25 | 4/8 | 348 |

58 trial selesai, 696 JPEG berhasil dibaca. Tidak ditemukan ketidakcocokan hash, jumlah frame, resolusi, peran peserta, atau protokol. Ini pemeriksaan penyimpanan, bukan persetujuan label atau kualitas model. Semua 50 trial positif masih memiliki tinjauan pose pending; semua 58 trial masih menunggu kurasi. Lighting, background, dan signing_hand belum diisi pada metadata; jangan menebaknya untuk mengubah rekaman asli.

## Temuan visual development

- Sejumlah foto tampak buram/berkabut, dengan area latar sangat terang. Detail jari, khususnya pembeda M/N, sulit diverifikasi. Penyebab pasti belum diketahui; periksa kebersihan lensa, fokus, dan pencahayaan dari depan sebelum merekam ulang. Ada orang lain di latar beberapa foto; gunakan satu pemeraga dengan latar sederhana untuk pilot awal.
- BISINDO C trial `773dadf3de844c6b9a00400e2c1c90cb`: frame awal/akhir menunjukkan satu tangan berbentuk C, tetapi frame tengah memasukkan tangan kedua. Label statis sepanjang trial tidak dapat langsung dipercaya. Tinjau interval lengkap dan catat keputusan terpisah; jangan otomatis mengubah label atau menghapus sumber. Contoh temporal: `data/pilot_audit/v1/bisindo_C_review.jpg`.
- SIBI neutral terlihat berpindah posisi selama interval rekam. Ulangi posisi istirahat yang stabil atau tandai trial untuk ditinjau sebagai transisi.
- Trial transition BISINDO memuat pose yang menyerupai kelas positif pada sebagian interval. Jangan memberikan label negatif pada seluruh fotonya; simpan untuk evaluasi temporal sampai tersedia anotasi interval/frame yang tepat.
- Label non_sign_pose tetap memerlukan peninjauan: pose di luar lima kelas pilot tidak otomatis bukan isyarat. Referensi kandidat belum merupakan pengesahan linguistik.

## Langkah berikut

1. Perbaiki ketajaman dan pencahayaan, lalu tinjau/ulang trial bermasalah. Simpan keputusan kurasi terpisah memakai `docs/pilot_trial_review.example.csv`; rekaman ulang tidak otomatis membatalkan trial lama.
2. Untuk p001/s01, masih kurang satu trial masing-masing no_hand, neutral, non_sign_pose, dan transition pada setiap bahasa. Pastikan negatif sesuai definisinya; transition bukan sumber negatif per-frame otomatis.
3. Setelah contoh sesi pertama konsisten, lanjut p001/s02 serta peserta berbeda p002/p003, masing-masing dua sesi sesuai protokol. Dari target keseluruhan 396 trial, saat audit masih kurang 338 berdasarkan jumlah, sebelum pengecualian kualitas.
4. Selesaikan verifikasi pose dan kurasi development sebelum training. Jangan memakai p003 untuk memilih model.

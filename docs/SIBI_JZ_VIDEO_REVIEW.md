# Data gerakan J/Z untuk melengkapi model SIBI

Pemeriksaan lokal 20 Oktober 2026. Model huruf statis SIBI lama tetap menjadi baseline; belum ada training atau perubahan prediksi pada pemeriksaan ini.

| Sumber | J | Z | Bentuk |
|---|---:|---:|---|
| ASL Dataset/landmarks | 402 | 403 | Semua sampel `(63,)`, satu frame |
| Video baru | 22 | 15 | Video berurutan 1920×1080 |

Seluruh 37 video berhasil didekode sampai jumlah frame yang dilaporkan kontainer. J berdurasi 1,08–2,28 detik, 50 FPS; Z berdurasi 1,12–1,92 detik, 25 FPS. Perbedaan FPS harus diperhitungkan saat menyamakan durasi urutan.

`J_001.MP4` dan `J_007.MP4` identik berdasarkan SHA-256. Terdapat 21 video J unik dan 15 video Z unik. Keduanya tidak boleh dipisahkan ke train/validation sebagai contoh independen. Video asli tidak diubah.

Hasil teknis dan grid temporal lokal tersimpan di `data/audits/jz-video-v1/summary.json`, `J_temporal_grid.jpg`, dan `Z_temporal_grid.jpg`. Grid memakai enam posisi waktu per video. Pemeriksaan visual menunjukkan gerakan dan variasi posisi awal/akhir; label folder belum merupakan verifikasi bahwa gerakan mengikuti referensi SIBI. Khusus contoh Z, jangan mengunci implementasi pada asumsi menggambar huruf Z dengan telunjuk: bentuk tangan dan gerakan yang tampak perlu mengikuti sumber sebenarnya.

ASL `.npy` tidak menyediakan batas klip, timestamp, FPS, atau pemetaan ke gerakan pada video baru. Nama `frame_N` sendiri tidak membuktikan urutan satu gerakan utuh. Data tersebut tidak digunakan sebagai ground truth urutan J/Z SIBI.

Langkah implementasi yang disarankan:

1. Gunakan video baru sebagai kandidat sumber urutan landmark J/Z; keluarkan duplikat lewat manifest tanpa menghapus file.
2. Pertahankan koordinat lintasan relatif terhadap awal klip dan ukuran tangan. Normalisasi setiap frame ke pergelangan saja akan menghilangkan perpindahan tangan yang mungkin dibutuhkan.
3. Mulai dengan pencocokan urutan/template temporal sederhana karena jumlah klip kecil. Model temporal besar belum memiliki dasar data yang cukup.
4. Pisahkan video utuh, dan kelompok pemeraga/sesi bila pemetaannya tersedia. Jangan membagi frame satu video secara acak ke train/validation. Identitas pemeraga tidak ditetapkan hanya dari penampilan.
5. Gabungkan modul temporal dengan classifier statis lama setelah ada mekanisme penolakan gerakan lain. Classifier dua kelas J/Z saja akan selalu memilih salah satunya; belum cukup untuk aktivasi otomatis pada webcam.

Belum perlu pengambilan sampel tambahan untuk memulai ekstraksi. Sumber video/referensi gerakan masih perlu dicatat agar label SIBI dapat diverifikasi, dan akurasi webcam belum dinilai dari koleksi ini.

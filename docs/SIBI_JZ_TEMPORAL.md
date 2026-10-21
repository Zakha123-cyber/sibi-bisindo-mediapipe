# Eksperimen pengenal gerakan J/Z

Modul ini mengikuti label dan gerakan video lokal `dataset/J` dan `dataset/Z`, sesuai keputusan proyek. Tidak memerlukan pencarian sumber tambahan. Modul melengkapi classifier SIBI `svm_linear_c1` lama; file fitur dan model statis tidak diganti.

## Menjalankan

Gunakan environment MediaPipe/fitur dan training yang sudah dipakai model SIBI. Ekstraksi dan fitting cukup dengan CPU:

```powershell
python scripts/prepare_sibi_jz.py
python scripts/train_sibi_jz.py
python scripts/webcam_sibi_classifier.py --temporal-model artifacts/sibi-jz-v1/templates.npz --check-only
python scripts/webcam_sibi_classifier.py --temporal-model artifacts/sibi-jz-v1/templates.npz
```

Tanpa `--temporal-model`, webcam tetap memakai perilaku model statis sebelumnya. Pada mode gabungan, baris `HASIL` menampilkan event J/Z yang lolos gerbang selama satu detik; setelah itu kembali ke huruf statis. Baris prediksi SVM tetap tersedia untuk membandingkan. Preview dicerminkan setelah inferensi dan sebelum teks ditulis. Foto/video webcam tidak disimpan.

## Cara kerja

Ekstraksi memakai konfigurasi MediaPipe lama yang sama dengan webcam: RGB asli, mode statis, satu tangan valid. Video disampel berdasarkan waktu sekitar 12,5 FPS; frame gagal disimpan sebagai NaN beserta statusnya, bukan diganti tangan palsu. Duplikat byte video dikeluarkan lewat manifest. Hasil disimpan di `data/features/sibi-jz-v1/`.

Template menggabungkan bentuk tangan relatif terhadap pergelangan dan lintasan pergelangan, telunjuk, serta kelingking. Skala dan rasio gambar diperhitungkan; lintasan tidak dihapus dengan normalisasi posisi per-frame. Urutan diresampel ke 32 langkah waktu. Klip dengan kurang dari 60% frame valid, gap antarobservasi di atas 0,3 detik, atau gerakan terlalu kecil dikeluarkan dengan alasan dalam laporan.

Fitting menyimpan contoh urutan unik sebagai template dan mengukur jarak urutan. Tidak ada optimisasi neural network, GPU, atau epoch. Evaluasi awal leave-one-video-out mempertahankan setiap klip utuh dan mengeluarkan duplikat persis. Belum ada pemetaan pemeraga/sesi, sehingga angka ini bukan akurasi pada orang baru dan bukan pengganti evaluasi webcam. Semua klip yang layak akhirnya digunakan sebagai template.

Pada webcam, buffer paling lama 2,5 detik dicoba dalam beberapa durasi, dengan pemeriksaan paling sering tiap 0,2 detik. Buffer direset bila tangan hilang/lebih dari satu atau jeda observasi di atas 0,3 detik. Event harus memenuhi gerbang gerak minimum, jarak template, dan selisih antar-kelas; ada jeda satu detik antar-event. Semua batas tersebut masih heuristik karena belum tersedia urutan negatif berlabel khusus. Gerakan biasa masih mungkin memicu J/Z; jarak bukan confidence probabilitas.

Artefak: `artifacts/sibi-jz-v1/templates.npz` dan `report.json`. Laporan memuat klip yang dikeluarkan, jumlah template, jarak, dan prediksi tiap klip pada evaluasi awal. Jangan menimpa hasil dengan konfigurasi lain lalu membandingkannya seolah berasal dari protokol identik.

## Hasil awal lokal

Ekstraksi selesai untuk 36 video unik; satu duplikat J dilewati. Template memakai 21 klip J dan 11 klip Z. Z_006/Z_009 dikeluarkan karena landmark valid di bawah 60%, sedangkan Z_007/Z_011 memiliki gerak terukur di bawah gerbang awal.

Pencocokan klip utuh leave-one-video-out menghasilkan 32/32 benar pada dua kelas ini. Angka tersebut tidak mengukur penolakan gerakan lain atau keberhasilan event streaming. Pada replay buffer streaming atas seluruh 36 video sumber unik (termasuk video pembentuk template), 30 video menghasilkan sedikitnya satu event dengan label sesuai, dan J_005 juga menghasilkan event Z yang salah. Ini pemeriksaan alur pada sumber training, bukan evaluasi independen. Rinciannya di `artifacts/sibi-jz-v1/stream_replay.json`; hasil webcam masih dapat berbeda.

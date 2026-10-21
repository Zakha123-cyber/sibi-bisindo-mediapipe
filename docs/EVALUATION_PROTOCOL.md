# Protokol evaluasi MVP — versi 1

Status: protokol awal dikunci sebelum training; pengambilan data webcam belum dilakukan. Target awal tetap mengikuti [roadmap](MVP_ROADMAP.md). SIBI mencakup 24 kelas tersedia; BISINDO memuat kandidat A–Z yang kelayakan statisnya diverifikasi pada Tahap 2.

Pembaruan 20 Oktober 2026: uji webcam development telah mengungkap kegagalan generalisasi. Pekerjaan berikutnya memakai [pilot lima huruf per bahasa](WEBCAM_PILOT.md), dengan pemisahan p001 train/p002 validation/p003 test. Pilot tiga orang ini tidak menggantikan kebutuhan evaluasi final pada minimal tiga peserta test baru di bawah.

## Split dataset offline

SIBI: kelompokkan nama sumber inti `huruf_nomor` lintas mono, latar pengganti, dan augmentasi. Karantina seluruh turunan sumber yang memiliki kasus anotasi belum pasti. Urutkan sumber per kelas berdasarkan SHA-256 dari `42:<source_group>`, lalu alokasikan sekitar 15% test, 15% validation, dan sisanya train. Pembulatan ke bilangan terdekat diterapkan per kelas. Baseline hanya mengaktifkan mono tanpa augmentasi.

BISINDO: satu seri nama ditempatkan utuh pada satu split lintas seluruh kelas. Seri tanpa sufiks dan `AR` menjadi train; `2` validation; `3` test. Pembagian ini berdasarkan pola penamaan, **bukan ID orang/sesi yang terverifikasi**. Jangan membagi acak frame dalam seri yang sama. Hasil disebut *provisional filename-series holdout*. Perbedaan sudut/latar antar seri dapat memengaruhi skor; peserta atau pose sangat mirip masih mungkin muncul antar seri.

Kedua protokol memeriksa overlap kelompok sumber dan file identik. Pemeriksaan itu belum menjamin tidak ada near-duplicate atau peserta sama lintas split. Tidak ada klaim signer-independent berdasarkan dataset ini.

## Pemisahan tugas dan penguncian test

- Train dipakai untuk fit fitur yang dipelajari, scaler, serta classifier. Augmentasi baru hanya diterapkan setelah split dan hanya pada train.
- Validation dipakai memilih detektor/preprocessing, classifier, hyperparameter, ambang penolakan, dan parameter stabilisasi. Perbandingan cakupan yang dilakukan pada audit awal merupakan eksplorasi sebelum protokol dikunci; jangan melanjutkan tuning terhadap test baru.
- Test hanya untuk penilaian akhir konfigurasi yang telah dipilih. Jika test digunakan untuk iterasi, tandai sebagai development dan siapkan test baru; jangan terus menyebutnya test independen.
- `split_lock.json` mengunci hash input, konfigurasi, assignment kelompok, dan manifest termasuk test. Perubahan yang mengubah lock ditolak pada direktori versi yang sama. Buat versi dan direktori baru dengan alasan tertulis jika data berubah.
- File test dapat dibaca oleh kode evaluasi, sehingga lock ini adalah pengaman reproducibility, bukan kontrol akses. Disiplin eksperimen tetap diperlukan.

## Rekaman webcam terpisah

Sebelum pengambilan, verifikasi referensi pose dan daftar kelas, serta minta persetujuan peserta untuk merekam dan menyimpan data lokal. Gunakan ID samaran; nama/kontak tidak dimasukkan dalam manifest teknis. Kamera dan sesi tidak dijalankan otomatis oleh skrip Tahap 1.

1. Siapkan peserta development untuk tuning webcam yang terpisah dari minimal **3 peserta final**. Pastikan peserta final menyatakan tidak ikut pengumpulan dataset training; jika hal ini tidak dapat dikonfirmasi, catat keterbatasannya.
2. Untuk setiap bahasa/kelas yang didukung, rekam minimal **10 percobaan pose terpisah per peserta final**, dibagi ke minimal dua sesi. Satu percobaan: posisi netral, bentuk pose, tahan sekitar 2–3 detik, lalu kembali netral. Jangan menghitung puluhan frame satu percobaan sebagai puluhan sampel independen.
3. Variasikan kondisi dalam batas penggunaan yang disepakati: jarak dan pencahayaan cukup agar seluruh tangan terlihat, latar berbeda, serta orientasi/mirror yang dicatat. Sertakan kondisi sulit dan laporkan terpisah, tanpa mengganti label demi memperbaiki skor.
4. Rekam minimal **100 percobaan negatif final** di luar set positif. Sertakan sedikitnya 25 tanpa tangan, 25 tangan netral, 25 pose di luar kelas, dan 25 transisi. Hindari label negatif untuk pose yang sebenarnya termasuk kelas yang didukung. Distribusikan kondisi ini pada peserta/sesi final.
5. Catat `participant_id`, `session_id`, `trial_id`, `mode`, `label_or_negative_type`, perangkat, resolusi, orientasi/mirror, pencahayaan, serta waktu mulai/akhir. Semua frame satu trial tetap dalam split yang sama; seluruh data satu peserta final hanya masuk test final.

Template metadata: [webcam_trials.example.csv](webcam_trials.example.csv). Data aktual disimpan di `recordings/` atau `data/`, yang diabaikan Git.

## Cara menghitung hasil

### Dataset gambar

Laporkan metrik classifier pada subset dengan landmark valid dan metrik sistem pada seluruh gambar test secara terpisah. Pada metrik sistem, kegagalan detektor atau penolakan untuk gambar huruf dihitung sebagai kegagalan mengenali kelas tersebut, bukan dihapus dari denominator. Prediksi ditolak ditangani sebagai keluaran kosong di confusion matrix; jangan menjadikannya kelas huruf baru.

Sajikan macro-F1, precision/recall per kelas, confusion matrix, jumlah sampel per kelas, serta cakupan ekstraksi 0/1/2 tangan. Dua tangan terdeteksi tidak selalu berarti dua tangan pelaku target benar. Jangan menyebut skor SVM atau handedness MediaPipe sebagai probabilitas kebenaran huruf.

### Webcam per percobaan

Kunci batas awal/akhir interval pose stabil berdasarkan label manusia sebelum melihat hasil model. Pada tiap trial positif, ambil keluaran stabil sistem yang paling sering muncul pada interval tersebut sebagai prediksi trial. Jika terjadi seri/tidak ada keluaran stabil, hitung sebagai penolakan. Jangan menambahkan majority vote evaluasi yang tidak tersedia di demo.

Laporkan juga proporsi waktu interval pose dengan huruf benar, salah, dan ditolak, serta waktu menuju keluaran stabil pertama. Dengan demikian, prediksi trial yang benar tidak menutupi tampilan yang sering berkedip ke huruf lain.

Pada trial negatif, hitung *false accept* jika sistem mengeluarkan huruf stabil apa pun selama interval negatif. Laporkan tingkat salah terima total dan per jenis negatif. Penanda stabil ditetapkan pada validation dan berlaku sama pada demo serta evaluasi.

### Latensi dan keandalan

Ukur satu koneksi lokal pada perangkat sasaran. Setelah warm-up, catat waktu monotonic sejak frame dikirim hingga respons frame tersebut diterima, ID frame, frame yang dibuang, dan jumlah hasil per detik. Ukur p50/p95 dan throughput pada sesi 10 menit. Laporkan waktu stabilisasi terpisah; jangan mengganti pengukuran ujung ke ujung dengan waktu detektor saja.

Catat CPU/RAM, resolusi, versi model, dan konfigurasi. Target awal: ≥5 hasil/detik, p95 ≤200 ms, tanpa crash atau antrean yang terus bertambah. Laporkan angka nyata sekalipun target belum terpenuhi.

## Bukti yang harus disimpan

Versi manifest/lock, kode/config training, versi preprocessing/model, daftar kelas, aturan penolakan/stabilisasi, jumlah peserta/sesi/trial, hasil per kelas, dan keterbatasan. Ringkasan dapat masuk Git; rekaman serta hasil rinci tetap lokal. Belum ada angka akurasi classifier atau klaim kualitas webcam pada Tahap 1.

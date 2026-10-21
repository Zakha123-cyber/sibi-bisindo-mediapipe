# Uji webcam kandidat landmark 63 fitur

Demo ini memakai `svm_rbf_c10` dari run `asl-source-sibi-candidate-v1_20261020_093347`. Label sumbernya ASL, dengan pemetaan pose SIBI belum diverifikasi. Model dan demo lama 132 fitur tetap terpisah.

## Jalankan

Environment harus memiliki paket yang dikunci dalam `requirements-features.txt` dan `requirements-training.txt`. Model, scaler, urutan kelas, versi dependency, dan SHA-256 implementasi preprocessing diverifikasi saat dimuat. Pemeriksaan tanpa membuka kamera:

```powershell
python scripts/webcam_npy_landmarks.py --check-only
```

Mulai preview:

```powershell
python scripts/webcam_npy_landmarks.py
```

Gunakan `--device 1` bila kamera berada di indeks lain. Untuk memakai run training lain yang menggunakan kontrak sama:

```powershell
python scripts/webcam_npy_landmarks.py --run-dir artifacts/NAMA_RUN
```

Default merujuk tepat ke run yang ditinjau, bukan otomatis memilih run terbaru. Tekan **Esc** untuk keluar. **Q merupakan label huruf Q**, bukan tombol keluar.

## Input dan keluaran

Kamera menghasilkan BGR yang diubah ke RGB. MediaPipe menghasilkan 21 landmark xyz untuk satu tangan. Fungsi `normalized_features` yang dipakai training mengurangi wrist, membagi jarak maksimum 3D, dan meratakan menjadi 63 fitur sebelum scaler/model. Tidak ada koreksi aspect ratio atau crop. Frame kamera tidak dicerminkan sebelum deteksi; secara default landmark juga tidak dicerminkan. Opsi diagnostik orientasi di bawah dapat mencerminkan x landmark setelah deteksi. Preview dicerminkan secara terpisah; teks digambar setelah pencerminan agar terbaca normal.

Detektor webcam dipatok melalui `configs/webcam_npy_landmarks_v1.json` pada MediaPipe 0.10.21, static mode, maksimal dua tangan dan confidence 0.5. Versi/handedness serta resolusi pengambilan dataset sumber tidak diketahui. Konfigurasi webcam yang dipilih ini tidak diklaim identik dengan proses pengumpulan sumber.

Tidak ada tangan menghasilkan `no_hand`; lebih dari satu menghasilkan `unexpected_extra_hand`; geometri degenerat menghasilkan `invalid_geometry`. Ketiganya meniadakan huruf dan menghapus riwayat stabilisasi. Satu tangan valid diklasifikasikan memakai seluruh pipeline scaler/model yang tersimpan.

Label stabil membutuhkan sekurangnya 4 suara pada jendela 5 frame, dengan label frame terbaru harus sama dengan mayoritas. Label lama tidak ditampilkan sebagai stabil ketika prediksi terbaru berubah. `Space` mereset riwayat. Skor SVM bukan probabilitas; satu tangan netral/pose di luar kelas masih dapat diprediksi sebagai huruf karena belum ada kebijakan unknown yang dikalibrasi.

## Uji pose dengan label

1. Letakkan satu tangan dan bentuk pose yang ingin diuji. Klik jendela preview agar menerima tombol keyboard.
2. Setelah pose terbentuk, tekan tombol hurufnya (A–I atau K–Y), lalu tahan sampai trial 3 detik berakhir. Label trial berasal dari tombol pengguna, bukan prediksi model.
3. Lihat hasil trial pada terminal, kembali netral, lalu ulangi untuk pose berikutnya. Trial aktif tidak diganti oleh tombol huruf lain. `Space` membatalkan trial; trial terpotong tidak dihitung dalam ringkasan trial selesai.
4. Prioritaskan juga V/R dan M/N yang tertukar pada validation, serta coba kondisi tanpa tangan. Ini observasi development, bukan evaluasi final orang/sesi independen.

Trial menyimpan hitungan prediksi frame, label stabil, dan kegagalan deteksi. Prediksi trial adalah label stabil yang paling sering tampil selama trial; penolakan ikut dihitung. Jika penolakan paling sering atau terdapat seri, prediksi trial kosong. Trial salah/ditolak tetap tercatat; tidak dihapus dari hasil.

## Hasil lokal

Tidak ada frame, gambar, atau video yang disimpan. Ringkasan JSON ditulis ke `data/webcam_npy_landmarks/` saat keluar normal. Ringkasan memuat hash model, konfigurasi detektor, orientasi input, prediksi/status per frame, trial beserta label sebenarnya, dan jumlah trial selesai/benar. Data lokal ini diabaikan Git.

`pipeline_ms` mengukur konversi warna, deteksi, normalisasi, dan classifier; tidak mencakup waktu menunggu capture dan menggambar preview. `throughput_fps` menghitung keseluruhan loop. Angka ini bukan latensi endpoint; API belum dibuat.

Sesudah hasil webcam ditinjau, model yang layak dapat dibungkus ke FastAPI dan UI HTML. Test dataset yang dicadangkan tetap tidak dibuka untuk pembuatan demo ini.

## Diagnostik orientasi asli dan mirror

Sesi `webcam_npy_20261020_095705_146169.json` memperoleh 0/6 trial benar: A/B/C/D diprediksi O. Dari 736 frame dengan fitur valid, 710 diprediksi O. Pengguna mengonfirmasi tangan kanan dan rangka landmark mengikuti jari.

Uji counterfactual pada **validation saja**: ketika x relatif landmark dicerminkan, accuracy kandidat turun dari 0.987730 menjadi 0.146217, dengan 637/978 prediksi menjadi O. Ini mendukung hipotesis ketidaksesuaian orientasi, tetapi tidak membuktikan orientasi kamera adalah penyebab sesi tadi. Sumber tidak memberi metadata tangan atau mirroring saat pengambilan.

Bandingkan kedua orientasi pada pose webcam yang sama:

```powershell
python scripts/webcam_npy_landmarks.py --compare-mirror
```

Bentuk A, tekan A, tahan sampai trial selesai; lakukan juga B/C/D. Jalur utama tetap `original`. Jalur pembanding `mirror-x` mengganti koordinat x mentah menjadi `1-x` sebelum normalisasi yang sama; y dan z dipertahankan, landmark sumber tidak diubah. Model, scaler, dan jumlah frame trial sama. Penambahan pembanding tidak memilih orientasi secara otomatis dari label sebenarnya ataupun skor model.

Terminal, preview, dan JSON menampilkan hasil asli serta pembanding. Ringkasan menyertakan `landmark_orientation`, `mirror_comparison_enabled`, `comparison_orientation`, dan `correct_comparison_trials`. Latensi classifier saat mode ini aktif mencakup dua pemanggilan model; bukan pengukuran biaya satu cabang.

Jika validation webcam menunjukkan `mirror-x` sesuai, jalankan satu cabang tersebut secara eksplisit:

```powershell
python scripts/webcam_npy_landmarks.py --landmark-orientation mirror-x
```

Pencerminan preview agar teks/gambar nyaman dilihat tidak mengubah fitur classifier. Pencerminan landmark ini merupakan perubahan orientasi fitur yang disengaja dan tercatat; normalisasi, model, dan default orientasi lama dipertahankan sampai hasil pembanding ditinjau.

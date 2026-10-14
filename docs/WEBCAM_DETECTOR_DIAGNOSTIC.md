# Diagnostik detektor MediaPipe melalui webcam

Skrip ini menguji jalur kamera langsung dengan MediaPipe dan konfigurasi yang sama dengan ekstraksi BISINDO confidence 0,3. Skrip menampilkan landmark, jumlah tangan, status deteksi, waktu proses detektor, resolusi aktual, dan FPS. Frame tidak disimpan; hanya ringkasan jumlah frame yang ditulis ke `data/webcam_diagnostics/` (direktori lokal yang diabaikan Git).

## Menjalankan

Aktifkan environment Python yang memiliki dependensi dari `requirements-features.txt`, lalu jalankan dari root proyek:

```powershell
python scripts/webcam_detector_diagnostic.py
```

Secara default, skrip memakai device kamera `0`, resolusi yang diminta `640x480`, mode BISINDO, dan konfigurasi `landmarks_bisindo_kaggle_v2_conf03.json`. Preview dicerminkan seperti kamera selfie, tetapi frame inferensi tetap RGB dan tidak dicerminkan.

Opsi yang tersedia:

```powershell
python scripts/webcam_detector_diagnostic.py --device 1
python scripts/webcam_detector_diagnostic.py --language sibi
python scripts/webcam_detector_diagnostic.py --no-mirror-preview
python scripts/webcam_detector_diagnostic.py --width 1280 --height 720
```

## Prosedur uji awal

Satu sesi dibagi otomatis menjadi segmen 10 detik. Siapkan tangan kanan sebelum menjalankan skrip, lalu ikuti urutan ini dan tekan `q` setelah 30 detik:

1. Tangan terbuka di tengah frame, seluruh tangan terlihat, pencahayaan cukup.
2. Tahan beberapa pose huruf statis yang jelas, misalnya A, B, dan Y BISINDO.
3. Ulangi dengan tangan sedikit lebih jauh dari kamera.
4. Opsional: ulangi di pencahayaan ruangan yang biasa digunakan.

Periksa overlay: garis landmark seharusnya mengikuti jari dan telapak. Ringkasan JSON memisahkan `segment_1`, `segment_2`, dan `segment_3`, sesuai tiga interval tersebut. Setiap segmen menyertakan jumlah status/jumlah tangan, waktu proses detektor, waktu baca kamera, serta jarak antarframe. Hitungan frame hanya menunjukkan kestabilan deteksi saat tangan bergerak; hitungan itu bukan akurasi pengenalan huruf. Skrip ini belum menjalankan classifier.

Jika kamera tidak terbuka, coba `--device 1` atau indeks lain. Tutup aplikasi lain yang sedang memakai kamera. Jangan kirim rekaman video; ringkasan JSON cukup untuk evaluasi awal.

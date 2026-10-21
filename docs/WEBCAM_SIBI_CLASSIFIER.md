# Uji webcam classifier SIBI

Skrip ini menjalankan kandidat terpilih `svm_linear_c1` bersama pipeline `landmarks-v1` dan MediaPipe yang sama dengan ekstraksi offline. Frame webcam tidak direkam. Hanya ringkasan hitungan frame dan latensi yang disimpan ke `data/webcam_sibi_classifier/` (diabaikan Git).

## Persiapan dan jalankan

Environment harus cocok dengan dependensi yang dikunci di `requirements-features.txt` dan `requirements-training.txt`. Validasi instalasi model tanpa membuka kamera:

```powershell
python scripts/webcam_sibi_classifier.py --check-only
```

Jika pemeriksaan lulus, jalankan demo:

```powershell
python scripts/webcam_sibi_classifier.py --device 0
```

Tekan `q` atau `Esc` untuk berhenti; gunakan `--device 1` bila webcam berada pada indeks lain. Preview dicerminkan agar terasa seperti webcam selfie, tetapi inferensi tetap memakai frame RGB yang tidak dicerminkan. Letakkan satu tangan di area kamera. Baseline SIBI ini menolak frame saat lebih dari satu tangan terdeteksi.

Preview memperlihatkan status deteksi, prediksi frame, prediksi stabil sederhana dari lima frame, margin keputusan SVM, waktu detektor, dan FPS. Margin SVM bukan probabilitas keyakinan. J dan Z belum termasuk kelas yang didukung; belum ada aturan `unknown` yang dikalibrasi.

## Membaca ringkasan

File `webcam_sibi_*.json` mencatat jumlah frame per status dan per prediksi, resolusi, orientasi preview/inferensi, throughput, serta median/p95 waktu detektor dan classifier. Prediksi per frame bukan percobaan pose independen. Untuk menilai akurasi per huruf, catat trial terpisah dan bandingkan hasil stabil tiap trial terhadap label sebenarnya. Uji ini juga belum menggantikan evaluasi pada peserta/sesi baru atau input negatif.

Setelah hasil webcam diperiksa, tahap berikutnya dapat membungkus classifier yang sama dengan endpoint FastAPI dan halaman HTML sederhana. Pastikan kontrak status `no_hand`, `unexpected_extra_hand`, dan prediksi valid tetap dipertahankan.

# Kontrak fitur landmark v1

Konfigurasi terpilih: [`landmarks_v1.json`](../configs/landmarks_v1.json). Pilihan baseline RGB tanpa peningkatan kontras dibekukan berdasarkan validation saja; hash dan alasan tercatat pada [`landmark_selection_v1.json`](../configs/landmark_selection_v1.json). Eksperimen CLAHE disimpan sebagai pembanding, bukan konfigurasi terpilih.

## Input dan detektor

Frame diproses sebagai RGB, orientasi EXIF diterapkan pada gambar dataset, tidak ada resize atau CLAHE, dan gambar offline tidak dicerminkan. Detector MediaPipe `solutions.hands.Hands` memakai `static_image_mode=true`, maksimum dua tangan, model complexity 1, dan ambang deteksi/tracking 0.5. Versi dependency dan SHA-256 berkas model/graph dipatok dalam konfigurasi dan diperiksa saat proses mulai. API legacy `Hands` menerima RGB dan menyediakan 21 landmark per tangan; label handedness MediaPipe tidak menjadi fitur. [Panduan MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python) dan [implementasi v0.10.21](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/python/solutions/hands.py) menjadi rujukan API.

Webcam produk belum diuji. `static_image_mode` membuat frame independen tanpa tracking antarbabak; ini menghindari perbedaan perilaku ekstraksi offline dan baseline awal, tetapi latensi serta kestabilan webcam harus diukur lagi ketika service dibuat. Mirroring preview adalah urusan UI; frame untuk deteksi/fitur harus tetap pada orientasi yang dipakai training.

## Vektor 132 dimensi

Urutan fitur stabil dan ditulis ke setiap cache:

1. Dua bit kehadiran slot tangan (slot 0, slot 1).
2. Untuk masing-masing dua slot, 21 landmark dengan koordinat `(x,y,z)`: titik wrist dikurangi dari seluruh titik; `y` dikalikan `height/width`; lalu seluruh koordinat dibagi rata-rata jarak 2D wrist ke MCP 5, 9, 13, dan 17. Orientasi pose dipertahankan.
3. Delta `(x,y)` antarpusat wrist dalam unit skala telapak bersama, diikuti rasio skala telapak setiap tangan terhadap rata-ratanya.

Slot diurutkan dari wrist dengan `x` gambar terkecil, lalu `y`, bukan handedness anatomi. Z dipertahankan relatif dalam tiap tangan; fitur ini tidak mengklaim kedalaman relatif antara dua tangan. Slot yang tidak ada diisi angka nol hanya sebagai padding struktur, dengan bit hadir `0`; satu tangan tetap menghasilkan vektor 132 dimensi yang sah. Bila tidak ada tangan atau geometri degenerat, seluruh vektor disimpan sebagai NaN dan `valid_mask=false`, bukan sebagai contoh kelas biasa.

`feature_valid` berarti hanya lolos pemeriksaan geometri dan kebijakan jumlah tangan. Itu bukan penilaian benar/salahnya landmark, label huruf, atau prediksi classifier. Log menyimpan jumlah tangan, status, peringatan, waktu proses, label, dan path. NPZ juga menyimpan koordinat landmark mentah dan provenance cache. Fitur offline dibangun ulang dengan `scripts/extract_landmarks.py`; pipeline webcam harus memanggil implementasi preprocessing/normalisasi yang sama.

## Kebijakan dataset

- SIBI: 24 kelas data tersedia (A–I dan K–Y); pose yang menghasilkan lebih dari satu tangan ditandai `unexpected_extra_hand`. Kecocokan 24 label dengan referensi alfabet resmi belum disahkan.
- BISINDO: A–Z merupakan kandidat label dari dataset. Ekstraksi mempertahankan satu atau dua tangan agar tidak menghapus sampel sebelum referensi pose diverifikasi. Status satu tangan menyatakan kelengkapan belum diverifikasi; status dua tangan menyatakan penetapan pelaku belum diverifikasi. Maksimum dua tangan berarti tangan tambahan tidak teramati.
- Kelas BISINDO yang didukung produk dan jumlah tangan yang diwajibkan per pose belum dikunci. Karena itu ekstraksi ini menyiapkan eksperimen classifier, bukan sertifikasi kelas/bahasa atau evaluasi pada peserta baru.

## Reproduksi

```powershell
python -m pip install -r requirements-features.txt
python scripts/extract_landmarks.py --splits train val test
python scripts/review_landmark_features.py --full
python -m unittest discover -s tests -v
```

Test hanya diekstrak setelah pilihan konfigurasi dibekukan; tidak digunakan memilih konfigurasi. Hasil besar masuk ke `data/features/landmarks-v1-final/` dan tidak dilacak Git. Ringkasan dan keputusan kecil disimpan di `reports/landmarks/`.

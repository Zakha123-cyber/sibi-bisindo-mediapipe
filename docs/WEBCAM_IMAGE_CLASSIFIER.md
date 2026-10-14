# Demo webcam classifier gambar BISINDO

Demo ini menjalankan checkpoint `best.pt` dari baseline MobileNetV3-Small langsung pada frame webcam. Model menerima seluruh frame, mengubah ukurannya menjadi 224×224, lalu memakai normalisasi yang tersimpan bersama checkpoint. Tidak ada deteksi/crop tangan dan tidak ada frame yang disimpan.

## Menjalankan

Pastikan PyTorch, TorchVision, Pillow, OpenCV, dan scikit-learn tersedia. Bila dependensi training belum dipasang, pasang sesuai instruksi [baseline classifier](IMAGE_CLASSIFIER_BASELINE.md); instalasi PyTorch/TorchVision harus cocok dengan CUDA perangkat.

Dari root repository jalankan:

```powershell
python scripts/webcam_image_classifier.py
```

Tekan `q` atau `Esc` untuk keluar. Untuk kamera lain gunakan `--device 1`; resolusi dapat diubah dengan `--width` dan `--height`. Secara default preview dicerminkan seperti webcam selfie, sementara inferensi menggunakan frame kamera asli. Ringkasan sesi tanpa gambar disimpan di `data/webcam_image_classifier/`.

Tampilan menyertakan prediksi frame, prediksi stabil dari lima frame terakhir, top-3 skor softmax, latensi inferensi, dan FPS keseluruhan. Skor softmax bukan probabilitas terkalibrasi. Prediksi stabil hanya membantu observasi demo dan bukan hasil per percobaan independen. Saat ini model selalu memilih salah satu huruf A–Z; belum ada kelas tanpa tangan atau penolakan unknown.

## Uji awal

Mulai dengan satu pose yang ditahan, lalu ulangi pose tersebut setelah kembali ke posisi netral. Coba beberapa kelas dan kondisi kamera, termasuk tanpa tangan. Catat apakah huruf frame berubah-ubah, apakah preview/inferensi terasa terbalik, serta nilai `inference_ms` dan `throughput_fps` pada ringkasan sesi. Ringkasan hitungan per frame tidak menggantikan pencatatan percobaan pose terpisah dan tidak membuktikan generalisasi.

Model ini dilatih pada gambar dataset, bukan citra webcam. Jika hasil live lemah, jangan sesuaikan threshold dari test dataset yang sudah dievaluasi. Kumpulkan validation webcam terpisah untuk menguji framing, orientasi, batas penolakan, dan stabilisasi. Model SIBI serta service FastAPI belum terintegrasi dalam demo ini.

## Mengumpulkan contoh webcam berlabel

Karena evaluasi gambar dataset belum memprediksi hasil webcam dengan baik, contoh tambahan dapat dikumpulkan secara eksplisit untuk eksperimen adaptasi domain. Collector hanya menyimpan still JPEG saat trial huruf aktif; tidak membuat video. File ditulis lokal di `data/webcam_collection/bisindo/`, yang diabaikan Git. Jalankan:

```powershell
python scripts/collect_bisindo_webcam_samples.py --save --participant-id p001 --session-id session_01
```

Bentuk pose terlebih dahulu, lalu tekan tombol huruf A–Z yang sesuai untuk mulai menyimpan gambar setiap setengah detik. Tahan pose sekitar 2–3 detik, tekan **Space**, lalu kembali ke posisi netral sebelum trial berikutnya. Tekan **Esc** untuk menyelesaikan sesi. Jangan menekan huruf sampai pose sudah benar-benar terbentuk. Setiap trial memiliki ID sendiri di `manifest.csv`; beberapa frame dari satu trial tetap berkorelasi dan tidak boleh dipisah ke train dan validation.

Ambil sesi lain dengan `--session-id session_02` untuk semua atau sebagian kelas. Untuk mengevaluasi pada orang berbeda, gunakan `--participant-id` samaran berbeda dan jangan masukkan orang itu ke data fine-tuning. Koleksi satu orang hanya mendukung evaluasi awal pada orang/sesi tersebut, bukan klaim generalisasi ke signer lain. Tinjau label dan pose sebelum training; gambar tidak dikirim atau diunggah otomatis.

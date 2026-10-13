# Temuan audit dataset — 13 Oktober 2026

Catatan tindak lanjut: laporan ini menyimpan temuan audit awal. Metadata sumber dan keputusan kurasi setelah audit ada di [sumber dataset](../../docs/DATASET_SOURCES.md) dan [hasil Tahap 1](../data_preparation/README.md). Pernyataan informasi belum tersedia di bawah menggambarkan kondisi saat audit awal.

## Keputusan yang disarankan

Dataset dapat digunakan untuk eksperimen awal, tetapi pembagian SIBI multi-background perlu diperbaiki dan keberhasilan ekstraksi BISINDO perlu ditingkatkan sebelum baseline classifier dinilai. Mulai baseline landmark SIBI dari **mono-background tanpa augmentasi**; simpan multi-background sebagai eksperimen terpisah setelah sumber gambarnya dikelompokkan dengan benar. Kualitas pada webcam nyata tetap harus diuji dengan data baru.

Temuan ini merevisi dugaan awal bahwa multi-background otomatis menjadi pilihan terbaik: pada sampel yang diuji, MediaPipe lebih sering gagal pada varian tersebut. Penyebabnya belum diisolasi melalui eksperimen terkontrol.

## Cakupan dan integritas

Seluruh **39.989 JPG** berhasil didekode oleh Pillow. Dataset tidak diubah. Audit meliputi nama kelas, resolusi, split, format dan batas koordinat anotasi YOLO, pasangan gambar–label, hash SHA-256, kandidat sumber berdasarkan nama file, dan pemeriksaan visual sejumlah contoh.

| Dataset | Gambar | Kelas | Train / val / test | Resolusi |
|---|---:|---:|---|---|
| BISINDO | 8.040 | 26, A–Z | Belum dibagi | 640 × 480 |
| SIBI mono tanpa augmentasi | 1.440 | 24 | 1.215 / 150 / 75 | 640 × 640 |
| SIBI mono dengan augmentasi | 3.870 | 24 | 3.645 / 150 / 75 | 640 × 640 |
| SIBI multi tanpa augmentasi | 7.200 | 24 | 6.120 / 720 / 360 | 640 × 640 |
| SIBI multi dengan augmentasi | 19.439 | 24 | 18.359 / 720 / 360 | 640 × 640 |

- Seluruh varian SIBI tidak memiliki J dan Z berdasarkan nama file; ID anotasi yang ditemukan juga hanya mencakup 24 kelas.
- BISINDO memiliki 340 gambar per kelas A–F dan 300 per kelas G–Z. SIBI tanpa augmentasi seimbang: 60 per kelas pada mono dan 300 per kelas pada multi.
- Semua split SIBI memuat 24 kelas. Test mono hanya memiliki 3–6 gambar per kelas, sehingga estimasi metrik per kelas akan mudah berubah oleh sedikit kesalahan.
- Train SIBI multi dengan augmentasi memiliki subfolder huruf; skrip menangani struktur bersarang ini.
- Tidak ada video yang ditemukan. Label A–Z tidak membuktikan semua kelas dapat dipahami sebagai pose statis.

## Kasus anotasi untuk ditinjau

Ada **26 gambar yang perlu ditinjau**, terdiri dari:

- **14 anotasi kosong**: 4 di multi tanpa augmentasi dan 10 di multi dengan augmentasi.
- **12 ketidaksesuaian nama huruf dengan ID anotasi**: 3 di multi tanpa augmentasi dan 9 pada versi augmentasi. Nama B, P, dan X berpasangan dengan ID `18`, sementara pemetaan mayoritas menunjukkan `18` adalah T.

Pemetaan ID diperoleh dari konsistensi mayoritas nama file, belum dari konfigurasi kelas resmi. Gambar sampel mendukung perlunya meninjau anotasi tersebut, tetapi audit tidak mengubah label otomatis. Untuk classifier landmark, anotasi bounding box tidak wajib; nama file tetap perlu diverifikasi sebelum menjadi label training. Anotasi kosong tidak boleh otomatis dianggap contoh tanpa tangan.

Ada **satu anotasi tanpa gambar** pada `labels/train/H/bg_04__H_116_png.rf.cbbab358c33ed017969777e07bc10fd0.txt` di multi dengan augmentasi. Ini konsisten dengan total lokal 19.439 dibanding 19.440 pada deskripsi sumber. Penyebab absennya gambar belum diketahui.

Daftar: [review_required.csv](review_required.csv), [summary.json](summary.json), dan [contoh anotasi](annotation_review_samples.jpg).

## Duplikasi dan kebocoran antar split

**Jangan mengartikan nol duplikasi hash antar split sebagai split yang aman.**

- Ada **8.639 kelompok file identik** berdasarkan SHA-256, semuanya melibatkan varian dataset berbeda. Menggabungkan semua varian dapat menggandakan sampel.
- Untuk nama sumber lengkap yang masih menyertakan latar, tidak ditemukan duplikasi sumber antar split dalam varian yang sama.
- Setelah awalan latar seperti `bg_01__` diabaikan, ada **810 kandidat sumber lintas split pada setiap varian multi-background**. Total 1.620 entri audit mencakup dua varian; jangan dianggap 1.620 sumber independen.
- Contoh kandidat `A_10`: latar 01, 02, 04, dan 05 berada di train, sedangkan latar 03 berada di val. Pemeriksaan visual memperlihatkan bentuk tangan dan pakaian yang sama dengan latar pengganti berbeda. Ini merupakan bukti kuat kebocoran sumber pada contoh tersebut; seluruh 810 kandidat belum diverifikasi visual satu per satu.

![Contoh sumber sama pada latar dan split berbeda](background_source_candidate_sample.jpg)

Sebelum training multi-background, verifikasi aturan nama sumber lalu buat split berbasis kelompok sumber, dengan seluruh latar dan augmentasi sumber tersebut berada pada satu split. Jika metadata orang/sesi tersedia, prioritaskan juga pemisahan orang/sesi untuk menguji generalisasi. Jangan menggabungkan mono dan multi sebelum hubungan sumber keduanya diperiksa.

Detail: [background_source_candidates.json](background_source_candidates.json), [contoh sumber](background_source_candidate_sample.json), [duplicates.json](duplicates.json).

## Uji awal ekstraksi landmark

Sampling acak deterministik: **20 gambar per kelas per varian tanpa augmentasi**, seed `42`, total **1.480 gambar**. Sampel diambil dari gabungan split untuk audit ekstraksi, bukan evaluasi classifier.

Menggunakan instalasi lokal **MediaPipe 0.10.21, API legacy `solutions.hands.Hands`**, mode gambar statis, maksimal dua tangan, `model_complexity=1`, `min_detection_confidence=0.5`. Hasil perlu diperiksa ulang bila implementasi produksi beralih ke Tasks API. API tidak diunduh atau di-upgrade dalam audit ini.

| Dataset sampel | N | 0 tangan | 1 tangan | 2 tangan | ≥1 tangan terdeteksi |
|---|---:|---:|---:|---:|---:|
| BISINDO | 520 | 145 | 217 | 158 | 72,1% |
| SIBI mono tanpa augmentasi | 480 | 3 | 477 | 0 | 99,4% |
| SIBI multi tanpa augmentasi | 480 | 191 | 289 | 0 | 60,2% |

**Angka ini bukan akurasi klasifikasi huruf atau akurasi landmark.** Pada gestur dua tangan, satu tangan terdeteksi dapat berarti masukan belum lengkap. Dataset juga memuat contoh lebih dari satu orang dalam frame, sehingga dua tangan terdeteksi belum membuktikan keduanya milik pelaku target. Belum ada anotasi jumlah tangan yang semestinya atau kualitas landmark sebagai ground truth.

Contoh BISINDO yang gagal memperlihatkan distorsi lensa lebar, variasi ukuran tangan, pencahayaan, dan tangan saling menutupi. Hal-hal ini menjadi kandidat penyebab, bukan kesimpulan kausal. Pada sampel BISINDO, D, I, V, dan Z gagal mendeteksi tangan pada setidaknya 10 dari 20 gambar masing-masing. Untuk multi SIBI, P gagal pada 14 dari 20 gambar.

Median pemrosesan per gambar sekitar 19–22 ms pada perangkat lokal. Pengukuran ini hanya mencakup pemanggilan detektor dalam mode statis, belum IO, jaringan, classifier, atau tracking webcam; tidak boleh dilaporkan sebagai FPS sistem realtime.

Detail dan contoh: [landmark_summary.json](landmark_summary.json), [landmark_sample.csv](landmark_sample.csv), [kegagalan BISINDO](BISINDO_Mendeley%20BISINDO_no_hand_samples.jpg), [kegagalan SIBI multi](SIBI_sibi_alphabet_multi_bg_no_augmentation_no_hand_samples.jpg).

## Langkah berikutnya

1. Tinjau 26 kasus anotasi, verifikasi pemetaan kelas dan kandidat sumber, lalu susun manifest bersih. Simpan keputusan koreksi terpisah dari dataset mentah.
2. Siapkan split sumber dan, bila tersedia, orang/sesi. Metadata tersebut belum disertakan pada struktur lokal yang diperiksa. Urutan nomor file tidak otomatis menunjukkan identitas atau sesi.
3. Ekstrak landmark penuh untuk SIBI mono dan jalankan baseline SVM/MLP; pertahankan laporan kegagalan ekstraksi per kelas. Jangan hanya melaporkan akurasi pada gambar yang berhasil diekstrak.
4. Untuk BISINDO, tinjau kegagalan dan kelengkapan dua tangan terlebih dahulu. Bandingkan konfigurasi detektor, preprocessing yang juga tersedia saat inferensi, atau model berbasis gambar bila diperlukan. Crop memakai bounding box ground truth saja tidak merepresentasikan kondisi webcam.
5. Rekam set uji webcam baru dari orang/sesi yang tidak masuk training. Sertakan tangan netral, transisi pose, dan tanpa tangan untuk menguji penolakan prediksi.

Audit ini belum memverifikasi kebenaran linguistik seluruh pose, lisensi resmi, duplikasi visual menyeluruh, atau generalisasi ke orang baru. Tautan sumber Mendeley dan referensi alfabet masih diperlukan untuk melengkapi dokumentasi.

## Reproduksi

Lingkungan yang digunakan: Windows, Python 3.12.5, Pillow 11.3.0, NumPy 1.26.4, MediaPipe 0.10.21; Intel i5-1135G7, RAM sekitar 11,7 GB. Notebook berfungsi sebagai pembaca hasil dan pemicu audit opsional.

```powershell
python scripts/audit_dataset.py
python scripts/audit_landmarks.py --per-class 20 --seed 42
```

`audit_dataset.py` juga menjalankan `review_dataset_audit.py`. Untuk menyusun ulang daftar tinjauan dari inventory yang sudah ada, jalankan `python scripts/review_dataset_audit.py`.

Laporan ini adalah interpretasi snapshot hasil audit tanggal di atas. Skrip memperbarui JSON/CSV/gambar; interpretasi Markdown ini perlu ditinjau kembali bila dataset atau konfigurasi berubah.

Referensi API: [kode resmi Hands yang digunakan](https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/python/solutions/hands.py), [panduan Hand Landmarker Tasks](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python).

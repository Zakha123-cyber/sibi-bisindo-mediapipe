# Perbandingan landmark J/Z dengan G/T

Review 20 Oktober 2026 atas checkpoint CNN–Softmax epoch 89, run `sibi_20261020_130607_889055`. Menggambar ulang koordinat **input model yang sebenarnya**, bukan gambar buatan atau foto sumber. Hash train/validation cocok dengan run terpilih. Tidak ada training, perubahan bobot/label, koleksi webcam, atau pembukaan test.

## Gambar yang dapat dibuka

1. [Ringkasan empat pose](../data/reports/sibi-jz-landmarks/key_poses.png): satu contoh training nyata yang paling dekat median koordinat tiap kelas.
2. [Grid variasi training J/G/T/Z](../data/reports/sibi-jz-landmarks/training_comparison.png): enam contoh/kelas, dipilih dari contoh dekat median lalu contoh yang berbeda secara koordinat. Ini bukan pemilihan contoh terbaik menurut classifier.
3. [Sebaran seluruh pose XY dan XZ](../data/reports/sibi-jz-landmarks/pose_spread_xy_xz.png): seluruh 42 contoh training per kelas ditumpuk transparan; garis tebal merupakan satu sampel nyata.
4. [Fokus J: train vs validation benar/salah](../data/reports/sibi-jz-landmarks/j_train_vs_validation.png): 42 training, 32 validation benar, delapan validation salah.
5. [Sembilan kesalahan validation J/Z](../data/reports/sibi-jz-landmarks/validation_errors.png): sampel salah, contoh training terdekat dari kelas benar, dan contoh training terdekat dari kelas prediksi. Jarak RMS dihitung pada koordinat xyz lokal, bukan embedding CNN; kedekatan itu tidak langsung menjelaskan keputusan jaringan.

## Membaca rangka

Oranye = ibu jari, biru = telunjuk, ungu = tengah, hijau = manis, merah = kelingking. Angka 4/8/12/16/20 adalah indeks ujung jari MediaPipe. Pergelangan berada di titik asal karena normalisasi. Plot memakai skala sumbu yang sama serta orientasi dataset, tanpa flip atau penyelarasan rotasi tambahan. XY mengikuti gambar: y meningkat ke bawah. XZ menampilkan kedalaman relatif MediaPipe, bukan jarak fisik yang terkalibrasi. Ketiadaan foto berarti landmark saja tidak cukup mengesahkan ketepatan pose linguistik atau akurasi detektor.

Koordinat yang diplot adalah 21 titik local xyz dari slot tangan terisi pada tensor `(42, 7)`; slot kedua kosong pada SIBI sehingga tidak digambar. J/Z berasal dari ASL, G/T dari SIBI. Asumsi rasio aspek ASL 1,0 tetap mengikuti kontrak training, bukan diubah untuk mempercantik grafik.

## Temuan

- **J training sangat seragam pada koordinat yang sudah dinormalisasi.** Sebaran orientasi/posisi jari dalam kelas tersebut jauh lebih sempit dibanding G/T pada data SIBI ini.
- Seluruh delapan kesalahan J ada pada `frame_310.npy` sampai `frame_317.npy`: lima menjadi T, tiga menjadi G. Pada rangka kelompok ini susunan ujung/ruas jari berbeda dari kelompok J training, termasuk arah ibu jari dan konfigurasi telunjuk. Nomor file berurutan tidak membuktikan waktu/identitas pemeraga karena metadata klip tidak tersedia.
- J validation lain masih 32/40 benar. Ini mendukung dugaan bahwa contoh J training belum mencakup semua variasi pada validation; tidak membuktikan bahwa menambah epoch akan memperbaikinya atau bahwa label J pasti salah.
- Satu kesalahan Z, `frame_313.npy`, diprediksi A; Z lainnya 39/40 benar. Pada input yang tersimpan, variasi Z lebih lebar daripada J training, tetapi hasil dataset tersebut belum menjelaskan kegagalan Z webcam per frame.
- Di antara contoh dekat median, ibu jari pada J dataset tampak dominan memanjang ke atas; model dilatih pada pola koordinat tersebut. Jangan menganggap gambar ini sebagai panduan pose SIBI resmi. Pose pengguna di webcam belum direkam pada review ini, sehingga tidak ada perbandingan langsung dengan tangan pengguna.

**Implikasi:** pertahankan baseline yang sudah cukup baik. Bukti yang baru ditemukan mengarah ke cakupan variasi koordinat J, bukan ketimpangan jumlah kelas (J/G/T/Z masing-masing 42 training). Sebelum mengganti model/menambah epoch, tinjau bentuk J/Z yang benar-benar diajarkan dataset. Normalisasi/augmentasi lanjutan atau perubahan contoh training harus menjadi eksperimen versi baru, tidak mengubah checkpoint ini diam-diam.

Identitas setiap panel dan path sampel tersimpan di [samples.csv](../data/reports/sibi-jz-landmarks/samples.csv); prediksi dan hash sumber laporan ada di [summary.json](../data/reports/sibi-jz-landmarks/summary.json).

Reproduksi: `python scripts/visualize_sibi_jz.py`.

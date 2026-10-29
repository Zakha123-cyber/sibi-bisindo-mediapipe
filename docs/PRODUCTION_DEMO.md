# Simulasi browser memakai API produksi

Buka **https://ai.signtech.tech/production.html** di Chrome atau Edge. Halaman disajikan oleh service FastAPI produksi. Untuk menyajikan halaman yang sama dari komputer lokal, jalankan dari direktori proyek:

```powershell
& '.\.venv-service\Scripts\python.exe' scripts/run_production_demo.py
```

Buka **http://localhost:5173/production.html** untuk versi lokal. Launcher hanya memakai pustaka standar Python untuk menyajikan HTML/CSS/JavaScript. Pada kedua alamat, semua prediksi dikirim langsung oleh browser ke `https://ai.signtech.tech` dan `wss://ai.signtech.tech`; launcher lokal tidak menjalankan model atau API lokal.

1. Klik **Uji koneksi lengkap**. Browser memeriksa health, metadata model, HTTP, serta WSS kedua bahasa menggunakan JPEG putih sintetis. Hasil yang diharapkan adalah `no_hand`, karena gambar memang tidak berisi tangan. Uji ini memeriksa integrasi, bukan akurasi alfabet.
2. Pilih SIBI atau BISINDO, klik **Mulai kamera**, lalu berikan izin kamera pada browser.
3. Bentuk pose dan amati huruf, skor, jumlah tangan, waktu server, jeda pengiriman frame, serta FPS respons.
4. Klik **Berhenti** sebelum mengganti bahasa. Hanya satu frame menunggu respons, sehingga gambar tidak menumpuk dalam antrean.

Preview sinkron memasangkan landmark dengan frame yang dikirim. Jika dimatikan, video live terus bergerak dan landmark dapat tertinggal akibat inferensi/jaringan. Preview cermin tidak membalik piksel input API. Halaman tidak merekam atau menyimpan gambar. Frame kamera dikirim ke VM untuk diproses.

Untuk mencoba kondisi gelap, nyalakan **Bantu cahaya redup** di bawah preview. Browser menaikkan kecerahan 1,25× dan kontras 1,10× pada frame yang dikirim ke API; preview sinkron menampilkan frame yang sama. Opsi ini mati secara default agar input sesuai alur asli. Bandingkan pose yang sama dengan opsi mati/nyala, amati apakah landmark dan huruf membaik, serta matikan jika detail jari menjadi memutih atau noise makin kuat. Opsi ini masih eksperimen; peningkatan akurasi belum terukur. Pencahayaan fisik yang lebih baik atau pengaturan eksposur kamera tetap lebih baik jika tersedia.

Origin `http://localhost:5173` dan `http://127.0.0.1:5173` telah ditambahkan secara spesifik ke `ALLOWED_ORIGINS` VM, bersama kedua origin produksi. Jangan membuka berkas melalui `file://`. Jika port diubah dengan `--port`, tambahkan origin port tersebut ke konfigurasi VM; jangan memakai wildcard atau origin `null`.

Jika port 5173 terpakai, hentikan server lain pada port tersebut atau gunakan port lain yang sudah diizinkan. Jika CORS ditolak atau WSS tidak tersambung, periksa origin halaman, koneksi internet, dan konfigurasi Nginx. Skor model belum dikalibrasi; keberhasilan koneksi tidak menjamin setiap pose diklasifikasikan benar. J/Z SIBI masih statis dan D/F/P BISINDO dapat tertukar.

`Ctrl+C` pada terminal menghentikan server halaman lokal. Halaman produksi ikut dirilis bersama image service API melalui CI/CD. Kontrak developer tersedia pada [OpenAPI JSON](api/openapi.json).

Pemeriksaan 22 Oktober 2026: browser Chrome headless dari origin localhost berhasil menjalankan pemeriksaan HTTP/WSS kedua model, start/stop kamera sintetis, pergantian bahasa, dan layout seluler tanpa galat JavaScript. Satu contoh gambar A dari validation SIBI serta satu contoh A dari validation BISINDO menghasilkan label A dan landmark masing-masing satu serta dua tangan. Ini pemeriksaan integrasi dengan sampel terbatas, bukan evaluasi akurasi atau pengujian webcam fisik.

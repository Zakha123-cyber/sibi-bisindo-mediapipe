# Hasil Tahap 1 — data-v1

Tanggal: 13 Oktober 2026. **Manifest dan protokol evaluasi teknis selesai**, dengan keterbatasan metadata yang dicatat. Belum ada training model atau rekaman webcam. Tautan dan lisensi dataset sudah diverifikasi; referensi alfabet BISINDO spesifik, konfigurasi kelas resmi SIBI, serta pemetaan peserta/sesi belum tersedia.

## Hasil yang siap digunakan

| Dataset aktif | Train | Validation | Test | Total | Kelas |
|---|---:|---:|---:|---:|---:|
| SIBI mono tanpa augmentasi | 1.001 | 216 | 216 | 1.433 | 24 |
| BISINDO, seri nama sementara | 6.480 | 780 | 780 | 8.040 | 26 kandidat |

Semua kelas terwakili di setiap split. Kelas statis BISINDO belum dikunci sebagai dukungan produk; peninjauannya tetap pada Tahap 2.

Manifest lokal berada di `data/manifests/data-v1/`:

- `all_samples.csv`: seluruh 39.989 gambar, termasuk status tidak aktif/karantina.
- `sibi_train.csv`, `sibi_val.csv`, `sibi_test.csv`: baseline aktif SIBI.
- `bisindo_train.csv`, `bisindo_val.csv`, `bisindo_test.csv`: baseline kandidat BISINDO.
- `excluded.csv`: gambar tidak aktif dan yang dikarantina; **bukan daftar gambar yang semuanya rusak**.
- `split_lock.json` dan `summary.json`: fingerprint serta hasil pemeriksaan.

Path dalam manifest relatif terhadap `dataset/` dan menggunakan `/` agar dapat dibaca lintas sistem operasi. Kolom `include` berisi `1` atau `0`; pembaca CSV harus membandingkan nilainya secara eksplisit, bukan memakai `bool('0')`. ID peserta/sesi kosong karena tidak diketahui.

## Keputusan kurasi

Seluruh 26 gambar bermasalah ditinjau pada tiga lembar visual lokal. Anotasi kosong tetap menunjukkan tangan; kasus B/P/X berkonflik dengan ID 18. Kita tidak menyimpulkan bahwa nama file selalu benar atau langsung mengoreksi label. Keputusan awal seluruhnya `exclude`, dengan alasan dan hash tiap gambar tersimpan pada [curation_decisions_v1.csv](../../docs/curation_decisions_v1.csv). Satu anotasi tanpa gambar juga diberi keputusan `exclude`; total 27 catatan.

Pengecualian diperluas secara konservatif ke seluruh varian/turunan tujuh sumber: `B_0`, `D_92`, `I_115`, `O_76`, `P_106`, `V_76`, `X_11`. Akibatnya **158 gambar** dikarantina, termasuk tujuh gambar mono asli. Ini mengurangi baseline SIBI dari 1.440 menjadi 1.433 gambar. Tidak ada file mentah yang diubah atau dihapus.

30.516 gambar berstatus tidak aktif dalam manifest: 158 dikarantina dan 30.358 dari varian yang belum diaktifkan untuk baseline. Multi-background dan augmentasi tetap terinventarisasi serta memiliki assignment kelompok yang konsisten, tetapi tidak digunakan untuk training/evaluasi baseline v1.

## Pengelompokan sumber

**SIBI:** seluruh empat varian memiliki 1.440 nama sumber inti yang sama. Setiap sumber memiliki satu gambar mono tanpa augmentasi dan lima latar pada multi tanpa augmentasi. Contoh `A_10`, `B_0`, dan `X_11` diperiksa visual lintas mono dan kelima latar, mendukung bahwa nama inti menghubungkan sumber yang sama. Pemeriksaan ini bukan verifikasi visual satu per satu dari 1.440 kelompok dan tidak menjamin independensi orang/sesi.

Setelah karantina, sumber per kelas diurutkan menggunakan hash seed 42. Sekitar 70/15/15 dibagi ke train/validation/test; seluruh varian dan augmentasi tetap mengikuti assignment sumber. Augmentasi pra-generasi tidak diaktifkan pada v1, termasuk apabila berasal dari sumber test baru.

**BISINDO:** ditemukan empat seri nama, tanpa sufiks, `AR`, `2`, dan `3`. Seluruh kelas memiliki keempat seri tersebut. Seri tanpa sufiks dan `AR` berada di train, seri `2` di validation, serta seri `3` di test. Tidak ada pembagian frame acak di dalam seri.

Pembuat menyebut tujuh peserta, tetapi empat seri nama tidak dapat dipetakan ke tujuh peserta berdasarkan informasi tersedia. Karena itu, split BISINDO merupakan **holdout seri nama sementara**, bukan pemisahan orang yang terverifikasi. Pose/partisipan sangat mirip masih mungkin melintasi seri. Menambahkan ID peserta hasil tebakan akan memberi rasa aman yang keliru, sehingga kolom identitas dibiarkan kosong.

## Validasi

- Seluruh **39.989 hash gambar** dihitung ulang dan cocok dengan inventory audit.
- Tidak ada kelompok sumber melintasi split, termasuk di varian tidak aktif.
- Tidak ada file identik berdasarkan SHA-256 melintasi split atau memiliki label berbeda.
- Seluruh kelas kandidat tersedia di setiap split aktif.
- **14 tes otomatis lulus**, mencakup kebocoran kelompok, kebocoran hash, kelas hilang, sufiks tak dikenal, karantina seluruh sumber, input dibalik urutannya, fingerprint lintas pemisah path, perubahan split lock, dan pembuatan versi snapshot baru.
- Manifest dikunci melalui [split_lock.json](split_lock.json). Input/config berubah tidak boleh menimpa snapshot versi yang sama, termasuk saat direktori manifest lokal belum ada.

Hash gambar tidak memverifikasi seluruh kebenaran anotasi atau kedekatan visual. Build menggunakan snapshot anotasi dari audit; jika file anotasi diubah, jalankan audit ulang dan revisi keputusan dengan versi baru. Identitas orang/sesi tetap belum diketahui.

Lock di direktori manifest melindungi snapshot versi tersebut; lock dalam laporan Git merekam versi terbaru. Versi baru boleh memperbarui ringkasan laporan, tetapi tetap harus memakai direktori manifest baru agar snapshot/test lama tidak tertimpa.

## Reproduksi

```powershell
python scripts/prepare_data.py --verify-images
python -m unittest discover -s tests -v
```

Skrip menggunakan standard library Python; inventory audit harus tersedia. Pada checkout baru, sediakan dataset dan jalankan `python scripts/audit_dataset.py` terlebih dahulu. Fingerprint input dihitung dari konten terstruktur yang dinormalisasi, sehingga perubahan CRLF/LF, urutan baris, atau pemisah path Windows/POSIX tidak mengganti snapshot. Perubahan konten tetap ditolak meskipun jumlah gambar sama. Periksa penyebabnya dan catat versi baru, jangan menghapus lock untuk menghindari pemeriksaan.

Untuk mengulang tanpa menghitung ulang seluruh hash gambar, jalankan `python scripts/prepare_data.py`; skrip tetap memeriksa keberadaan file, konsistensi manifest, dan lock, tetapi tidak membuktikan isi gambar belum berubah. Gunakan `--verify-images` untuk validasi snapshot lengkap gambar.

## Acuan berikutnya

- [Sumber dataset dan metadata yang belum tersedia](../../docs/DATASET_SOURCES.md)
- [Aturan split dalam konfigurasi](../../configs/data_v1.json)
- [Pemetaan label empiris](../../configs/label_map_v1.json)
- [Protokol evaluasi dan rekaman webcam](../../docs/EVALUATION_PROTOCOL.md)

Langkah berikutnya adalah **Tahap 2: mengunci detektor/preprocessing lalu mengekstrak landmark dari manifest aktif**. Tetap tinjau cakupan dan kelengkapan tangan BISINDO sebelum mengunci classifier atau daftar kelas produk.

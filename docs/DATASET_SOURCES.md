# Sumber dan atribusi dataset

Metadata halaman diverifikasi pada 13 Oktober 2026. Konfigurasi terstruktur tersedia di [dataset_sources.json](../configs/dataset_sources.json). Tanggal unduh dan checksum arsip unduhan asli tidak tersedia; URL versi berasal dari pengguna. Audit memberi fingerprint isi lokal, bukan bukti bahwa arsip lokal identik penuh dengan arsip server.

## SIBI

Teguh Andriyanto (2024), **Labeled Official Indonesian Sign Language (SIBI) Alphabet**, Mendeley Data, versi 1. Terbit 22 Mei 2024. DOI: [10.17632/yhsgmynccf.1](https://doi.org/10.17632/yhsgmynccf.1). Halaman sumber mencantumkan lisensi **CC BY 4.0**. [Sumber Mendeley](https://data.mendeley.com/datasets/yhsgmynccf/1).

Pembuat menyebut empat varian: satu latar, satu latar dengan augmentasi, banyak latar, serta banyak latar dengan augmentasi. Audit lokal menemukan 31.949 gambar SIBI; varian multi dengan augmentasi kurang satu gambar dibanding jumlah yang dideskripsikan. Detail per varian ada di [laporan audit](../reports/dataset_audit/FINDINGS.md).

Rujukan institusional yang ditemukan adalah [Kamus SIBI resmi](https://pkplk.kemendikdasmen.go.id/sibi/). Ini tidak menyediakan pemetaan ID kelas khusus dataset yang telah berhasil kita verifikasi. Pemetaan `0..23` ke A–I/K–Y di proyek merupakan hasil pemeriksaan empiris anotasi lokal; kebenaran linguistik seluruh pose belum disertifikasi.

## BISINDO

Arya Raden dan MUHAMMAD ASSHAFI (2026), **BISINDO DATASET**, Mendeley Data, versi 1. Terbit 3 Juli 2026. DOI: [10.17632/4xnkvr88tk.1](https://doi.org/10.17632/4xnkvr88tk.1). Halaman sumber mencantumkan lisensi **CC BY 4.0**. [Sumber Mendeley](https://data.mendeley.com/datasets/4xnkvr88tk/1).

Pembuat melaporkan tujuh peserta, kamera USB ELP fisheye 180°, jarak tangan sekitar 30–60 cm, dan kelas A–Z. Klaim referensi alfabet kementerian ada pada deskripsi, tetapi dokumen spesifiknya belum teridentifikasi. Kita tidak menyamakan klaim pembuat dengan validasi independen atas seluruh pose.

Isi lokal: 8.040 gambar dalam folder huruf. Pengguna mengonfirmasi belum ada metadata tambahan untuk pemetaan gambar ke peserta/sesi. Sufiks nama `AR`, `2`, `3`, atau tanpa sufiks tidak boleh ditafsirkan otomatis sebagai identitas tujuh peserta.

## Penggunaan dan perubahan lokal

Sertakan nama kontributor, judul, DOI/URL, tautan lisensi, dan keterangan perubahan saat menggunakan atau membagikan materi turunan. Ketentuan atribusi dan penandaan perubahan tercantum pada [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Proyek ini mempertahankan gambar/anotasi mentah. Perubahan pada Tahap 1 hanya berupa keputusan pengecualian dan pembagian sumber dalam manifest. Belum ada relabeling file mentah. Gambar dataset, lembar contoh, dan manifest rinci tetap lokal sesuai `.gitignore`; kode, keputusan kurasi, metadata sumber, serta ringkasan hasil dilacak Git.

## Hal yang masih belum tersedia

| Informasi | Status / dampak |
|---|---|
| Tanggal unduh dan hash arsip asli | Tidak tersedia; gunakan fingerprint snapshot lokal dan jangan mengklaim kecocokan arsip server |
| File kelas resmi SIBI | Belum tersedia; mapping lokal diverifikasi secara empiris dengan pengecualian eksplisit |
| Dokumen alfabet BISINDO spesifik dari pembuat | Belum teridentifikasi; validasi kelas statis tetap pekerjaan Tahap 2 |
| ID peserta/sesi per gambar | Belum tersedia untuk kedua dataset; evaluasi dataset tidak disebut signer-independent |
| Makna sufiks BISINDO | Belum terverifikasi; pemisahan seri adalah protokol sementara |

Ketiadaan informasi tersebut tidak diisi dengan tebakan. Bila metadata baru tersedia, lakukan tinjauan lalu buat versi manifest baru jika pembagian berubah.

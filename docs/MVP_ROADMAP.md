# Acuan penyelesaian MVP AI SIBI dan BISINDO

Status: aktif, tahap audit awal selesai. Terakhir diperbarui: 13 Oktober 2026.

Dokumen ini menjadi acuan ruang lingkup, urutan kerja, dan kriteria selesai. Perbarui checklist berdasarkan bukti hasil pekerjaan. Jika pendekatan berubah, catat alasan dan dampaknya pada tabel keputusan di akhir dokumen sebelum melanjutkan pekerjaan yang bergantung padanya.

## 1. Hasil akhir yang dituju

Service AI mengenali **satu huruf dari pose tangan statis pada webcam secara realtime**. Pengguna memilih mode `sibi` atau `bisindo`. Service memberikan huruf yang didukung atau status belum dapat memprediksi, lalu antarmuka demo menampilkan hasil yang stabil.

Tanggung jawab proyek ini adalah pipeline data, model, evaluasi, service FastAPI, dan demo teknis webcam untuk menguji integrasi. Tampilan produk lengkap menjadi pekerjaan tim frontend.

MVP tidak mencakup penyusunan kata/kalimat, penerjemahan percakapan, pengenalan lintasan gerakan, deteksi otomatis jenis bahasa isyarat, pengenalan banyak pelaku sekaligus, atau deployment berskala besar.

## 2. Keputusan dasar

| Aspek | Keputusan |
|---|---|
| Input | Frame dari webcam; satu pelaku utama dalam frame |
| Bahasa | Mode dipilih pengguna; model SIBI dan BISINDO terpisah |
| SIBI tahap awal | 24 kelas yang tersedia: A–I dan K–Y; J dan Z belum didukung |
| BISINDO tahap awal | Audit kandidat A–Z; daftar pose statis yang layak dikunci setelah peninjauan kelas dan ekstraksi |
| Baseline | Landmark tangan → fitur → bandingkan SVM dan MLP kecil |
| Data awal SIBI | Mono-background tanpa augmentasi |
| Data awal BISINDO | Dataset folder A–Z, setelah masalah ekstraksi ditinjau |
| Service | FastAPI; pemrosesan frame di backend untuk integrasi awal |
| Perangkat awal | Training lokal CPU i5-1135G7, RAM sekitar 11,7 GB; Colab bila eksperimen yang diperlukan terlalu berat |
| Kondisi tangan | Maksimal dua tangan; pertahankan posisi relatif antartangan |
| Ketidakpastian | Hasil boleh ditolak; sistem tidak wajib mengeluarkan huruf setiap frame |

MediaPipe versi produksi belum dikunci. Audit eksplorasi menggunakan **0.10.21, API legacy Hands**. Sebelum ekstraksi penuh, pilih API/model detektor, kunci versinya, dan gunakan konfigurasi preprocessing yang sama pada training serta inferensi. Hasil audit perlu diperiksa ulang jika detektornya berubah.

## 3. Bukti audit yang menjadi dasar

Lihat [laporan audit](../reports/dataset_audit/FINDINGS.md) untuk detail, batas kesimpulan, dan cara reproduksi.

- 39.989 gambar dapat dibaca; semua varian SIBI hanya memuat 24 huruf.
- Ada 14 anotasi kosong, 12 ketidaksesuaian nama huruf dengan ID anotasi, serta satu anotasi tanpa gambar. Jumlah ini mencakup varian dan turunan augmentasi.
- Setiap varian SIBI multi-background memiliki 810 kandidat sumber yang melintasi split. Contoh visual menunjukkan pose yang sama dengan latar berbeda masuk ke train dan validation.
- Sampling 20 gambar per kelas menunjukkan minimal satu tangan terdeteksi pada 99,4% SIBI mono, 60,2% SIBI multi, dan 72,1% BISINDO. Ini bukan akurasi klasifikasi atau bukti kelengkapan dua tangan.

Konsekuensi: jangan menggabungkan semua varian, jangan langsung menggunakan split multi-background untuk menyimpulkan akurasi, dan jangan menghapus seluruh kegagalan deteksi dari perhitungan kinerja sistem.

## 4. Urutan pekerjaan dan kriteria selesai

### Tahap 0 — Audit awal

- [x] Inventarisasi seluruh gambar, kelas, split, resolusi, serta anotasi.
- [x] Periksa file rusak, pasangan label, dan duplikasi file identik.
- [x] Catat kandidat ketidaksesuaian label dan kebocoran sumber antar split.
- [x] Uji MediaPipe pada sampel setiap kelas.
- [x] Simpan notebook, skrip reproduksi, dan temuan.

Selesai untuk audit awal. Validasi linguistik dan ekstraksi seluruh gambar termasuk pekerjaan tahap berikutnya.

### Tahap 1 — Manifest data dan protokol evaluasi

- [ ] Lengkapi tautan Mendeley, versi unduhan, lisensi, dan referensi alfabet dalam dokumentasi sumber.
- [ ] Tinjau 26 gambar bermasalah dan satu label tanpa gambar. Catat keputusan `keep`, `exclude`, atau `override_label` beserta alasan; jangan ubah dataset mentah.
- [ ] Verifikasi pemetaan ID anotasi SIBI. Label dari nama file tidak dianggap otomatis benar.
- [ ] Verifikasi aturan kandidat sumber gambar, termasuk hubungan antar latar dan varian; dokumentasikan bila identitas orang/sesi tidak tersedia.
- [ ] Susun manifest dengan minimal `path`, `language`, `label`, `source_group`, `split`, `include`, serta alasan pengecualian. Tambahkan `signer_id`/`session_id` hanya jika diketahui.
- [ ] Buat train/validation/test yang memisahkan kelompok sumber; seluruh latar dan augmentasi satu sumber berada pada split yang sama. Gunakan pemisahan orang/sesi bila metadata memungkinkan.
- [ ] Pastikan kelas yang didukung terwakili dalam tiap split; simpan seed dan aturan pembagian. Kunci test untuk evaluasi akhir.
- [ ] Tentukan protokol rekaman webcam terpisah, mencakup orang/sesi baru dan kondisi negatif.

**Selesai bila:** ada manifest yang dapat direproduksi, keputusan kasus bermasalah tercatat, tidak ada kelompok sumber yang melintasi split, dan batas evaluasi akibat metadata yang belum tersedia ditulis jelas. Split sumber saja tidak membuktikan evaluasi pada orang baru.

### Tahap 2 — Ekstraksi landmark dan fitur

- [ ] Kunci detektor, versi dependency, konfigurasi, serta perlakuan frame yang dicerminkan.
- [ ] Ekstrak seluruh data SIBI mono yang disertakan; simpan hasil agar eksperimen tidak mengulang deteksi.
- [ ] Tinjau dan ukur ekstraksi BISINDO per kelas, termasuk kelengkapan tangan dan pemilihan tangan pelaku utama.
- [ ] Tentukan fitur: landmark ternormalisasi, keberadaan tangan, urutan tangan konsisten, dan posisi relatif antartangan. Normalisasi tidak boleh membuang orientasi yang membedakan huruf.
- [ ] Simpan log kegagalan deteksi dan contoh visual. Gambar gagal tidak diubah menjadi vektor nol berlabel huruf biasa.
- [ ] Validasi ulang daftar kelas statis BISINDO bersama referensi/pihak yang memahami alfabetnya; kelas yang belum layak ditandai belum didukung.

**Selesai bila:** fitur dan preprocessing memiliki versi, cakupan ekstraksi per kelas diketahui, sampel landmark ditinjau visual, dan perilaku ketika tangan hilang/tidak lengkap ditetapkan.

**Jika BISINDO belum memadai:** lakukan eksperimen terukur pada konfigurasi detektor atau preprocessing yang juga tersedia saat webcam. Jika masih tidak memadai, bandingkan pendekatan berbasis gambar atau tambahkan data. Catat perubahan ini; jangan menganggap kegagalan detektor dapat diselesaikan hanya dengan classifier lebih besar.

### Tahap 3 — Baseline klasifikasi

- [ ] Latih dan bandingkan SVM serta MLP kecil untuk tiap bahasa yang siap.
- [ ] Fit scaler dan transformasi yang dipelajari hanya pada train; pilih model dan parameter menggunakan validation.
- [ ] Laporkan macro-F1, precision/recall per kelas, confusion matrix, cakupan ekstraksi, serta latensi.
- [ ] Bedakan metrik classifier pada fitur valid dengan metrik sistem pada seluruh masukan, termasuk kegagalan detektor.
- [ ] Tentukan penolakan berdasarkan skor model menggunakan validation dan contoh negatif. Jika skor belum dikalibrasi, jangan menamainya probabilitas kebenaran.
- [ ] Simpan model, preprocessing, daftar kelas, versi fitur, seed, dan metadata training sebagai satu paket yang konsisten.

**Selesai bila:** pemilihan model didukung evaluasi, eksperimen bisa diulang, artefak dapat dimuat untuk inferensi, dan daftar kelas yang didukung jelas. Evaluasi test akhir dilakukan setelah pemilihan model dan ambang selesai.

### Tahap 4 — Service FastAPI

- [ ] Sediakan health check dan endpoint prediksi satu frame untuk menguji model terlepas dari webcam.
- [ ] Implementasikan aliran frame webcam menggunakan WebSocket dengan antrean terbatas; buang frame lama ketika pemrosesan tertinggal.
- [ ] Kontrak keluaran memuat `mode`, `status`, `letter` yang boleh kosong, skor beserta jenisnya, dan versi model. Gunakan status seperti `ok`, `no_hand`, `incomplete_hand`, atau `uncertain` sesuai bukti yang tersedia.
- [ ] Validasi mode, format gambar, ukuran input, dan model yang tersedia; muat model sekali saat service dimulai.
- [ ] Pisahkan state tiap koneksi untuk tracking/stabilisasi; jangan berbagi state antar pengguna.
- [ ] Gunakan stabilisasi beberapa frame dan reset saat mode berubah/tangan hilang agar huruf lama tidak tertinggal.
- [ ] Uji jalur penting: input valid, input rusak, tanpa tangan, mode salah, ganti mode, putus koneksi, dan isolasi state.

**Selesai bila:** service menghasilkan respons sesuai kontrak, preprocessing sesuai training, dan demo menerima hasil tanpa antrean yang terus bertambah. Batas pengguna serentak MVP awal adalah satu koneksi aktif untuk pengukuran performa; scaling bukan syarat tahap ini.

### Tahap 5 — Validasi webcam dan serah terima

- [ ] Buat demo teknis sederhana: buka webcam, pilih mode, tampilkan prediksi/status, dan tutup kamera/koneksi dengan benar.
- [ ] Uji pada orang/sesi yang tidak digunakan training. Catat pencahayaan, jarak, orientasi kamera, serta jumlah percobaan per kelas.
- [ ] Sertakan tanpa tangan, tangan netral, pose di luar kelas, dan perpindahan pose. Ukur salah terima pada kondisi negatif.
- [ ] Ukur latensi ujung ke ujung, stabilitas prediksi, serta konsumsi resource pada perangkat sasaran.
- [ ] Dokumentasikan cara setup, training, menjalankan service, kontrak integrasi frontend, kelas yang didukung, dan keterbatasan.

**Selesai bila:** kriteria penerimaan di bawah diperiksa dengan bukti, kedua mode menjalankan kelas yang disepakati, dan tim dapat menjalankan ulang demo dari dokumentasi. Selesainya baseline SIBI saja belum berarti seluruh MVP selesai.

## 5. Target penerimaan awal

Angka berikut adalah **target kerja awal**, bukan hasil yang telah tercapai atau jaminan performa. Kunci protokol pengukuran sebelum training. Bila target perlu direvisi, tulis alasan dan bukti di log keputusan; jangan menggantinya hanya karena hasil test rendah.

| Aspek | Target awal / bukti yang diperlukan |
|---|---|
| Fungsi | Dua mode, prediksi pose statis, status penolakan, dan daftar kelas eksplisit |
| Kualitas | Macro-F1 ≥ 0,90 dan recall setiap kelas ≥ 0,80 pada pengujian terpisah untuk kelas yang didukung; masukan huruf yang gagal dideteksi/ditolak dihitung gagal dikenali |
| Generalisasi | Uji webcam pada minimal 3 peserta di luar data training, dengan minimal 10 percobaan pose terpisah per kelas per peserta; jika belum tersedia, klaim generalisasi tetap belum terverifikasi |
| Negatif | Sertakan minimal 100 percobaan negatif terpisah; target salah menerima sebagai huruf ≤ 5%, dengan jumlah dan jenis negatif dilaporkan |
| Realtime | Minimal 5 hasil inferensi per detik dan latensi p95 ≤ 200 ms pada satu koneksi lokal; ukur sejak frame dikirim hingga respons diterima; waktu stabilisasi dilaporkan terpisah |
| Stabilitas | Sesi demo 10 menit tanpa crash atau antrean frame yang terus bertambah |
| Reproduksi | Split, konfigurasi, versi model, laporan evaluasi, dan instruksi penggunaan tersedia |

Percobaan pose terpisah berarti pengguna mengulang pose, bukan menghitung banyak frame berdekatan sebagai sampel independen. Angka metrik harus disertai jumlah sampel dan hasil per kelas. Evaluasi pada dataset lama tidak menggantikan uji webcam.

## 6. Aturan menjaga fokus

1. Pekerjaan berikutnya adalah **Tahap 1: manifest data dan protokol evaluasi**. Jangan langsung membangun API atau mengoptimalkan model sebelum fondasi datanya siap.
2. Gunakan baseline sederhana terlebih dahulu. Tambahkan CNN/model temporal hanya bila ada bukti kebutuhan; model temporal akan mengubah ruang lingkup MVP statis.
3. Pertahankan data mentah. Koreksi, pengecualian, dan split disimpan sebagai manifest terpisah.
4. Test tidak digunakan untuk memilih hyperparameter atau ambang. Jika sudah dipakai untuk iterasi, perlakukan hasil itu sebagai development dan siapkan test baru.
5. Setiap tahap menghasilkan bukti yang dapat ditinjau. Tandai selesai berdasarkan kriteria, bukan sekadar karena kode berhasil dijalankan.
6. Perubahan kelas yang didukung atau ruang lingkup produk dibicarakan dengan tim dan dicatat; jangan mengurangi cakupan secara diam-diam.

## 7. Struktur dan kebijakan Git

Struktur sasaran berikut bersifat panduan; direktori dibuat ketika tahap terkait dikerjakan.

```text
docs/                 acuan MVP, sumber dataset, kontrak integrasi, keputusan
scripts/              audit, preprocessing, training, evaluasi
notebooks/            eksplorasi yang dapat dijalankan ulang, tanpa output besar
reports/              ringkasan hasil; detail hasil generate disimpan lokal
src/                  pipeline inferensi dan service, dibuat pada tahap implementasi
tests/                verifikasi perilaku penting
dataset/              data mentah lokal, diabaikan Git
data/                 manifest lengkap, fitur, split lokal, diabaikan Git
artifacts/            paket model dan hasil training lokal, diabaikan Git
```

Git menyimpan kode, dokumentasi, konfigurasi contoh, notebook tanpa output, serta ringkasan audit kecil. Dataset, rekaman pengguna, fitur, bobot model, secrets, cache, dan laporan detail berukuran besar diabaikan melalui `.gitignore`. Aturan dan alasan keputusan kurasi tetap dirangkum dalam dokumentasi yang dilacak Git meskipun manifest lengkap berada di folder lokal.

Notebook hasil eksekusi disimpan lokal di `notebooks/executed/`; `.gitignore` tidak dapat menyaring output yang tertanam dalam notebook sumber. Sebelum commit, bersihkan output notebook sumber dan periksa `git status`. File yang sudah tracked tidak otomatis dilepas oleh `.gitignore`.

Artefak model dapat dibagikan terpisah saat integrasi memerlukannya, dengan metadata versi dan checksum dicatat di dokumentasi. Belum ada publikasi atau upload otomatis dalam alur ini.

## 8. Log keputusan

| Tanggal | Keputusan | Dasar / konsekuensi |
|---|---|---|
| 2026-10-13 | MVP webcam, pose statis, keluaran huruf | Disepakati pengguna; kata/kalimat dan lintasan gerakan di luar tahap awal |
| 2026-10-13 | Mulai training lokal | Perangkat memadai untuk baseline landmark; belum ada kebutuhan GPU untuk baseline ini |
| 2026-10-13 | SIBI mono tanpa augmentasi sebagai baseline pertama | Sampling deteksi 99,4%, dibanding 60,2% pada multi; hasil masih perlu divalidasi pada webcam |
| 2026-10-13 | Tinjau pembagian sumber multi-background | Ada 810 kandidat lintas split per varian dan contoh visual kebocoran sumber |
| 2026-10-13 | Audit deteksi BISINDO sebelum mengunci classifier | 72,1% sampel memiliki minimal satu tangan terdeteksi; belum menjamin kelengkapan pose |
| 2026-10-13 | Target evaluasi awal ditulis sebagai acuan | Angka pada bagian 5 merupakan target engineering, belum hasil pengujian |

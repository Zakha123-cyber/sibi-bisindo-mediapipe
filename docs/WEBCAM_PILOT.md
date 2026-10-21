# Pilot webcam SIBI dan BISINDO

Status 20 Oktober 2026: protokol dan collector siap; kandidat pose belum disahkan dan pengambilan data pilot asli belum dijalankan. Pilot ini disepakati setelah beberapa model memperoleh skor dataset tinggi tetapi gagal pada webcam. Tujuannya menilai pipeline pada orang berbeda dengan label dan kondisi input yang dapat ditelusuri.

## 1. Kandidat pose dan peninjauan

| Mode | Kandidat lima huruf | Alasan pemilihan teknis |
|---|---|---|
| SIBI | A, B, C, M, N | Variasi bentuk A/B/C dan pasangan M/N yang mirip |
| BISINDO | A, B, C, R, T | Variasi bentuk serta pasangan C/R yang tertukar pada model gambar sebelumnya |

Ini kandidat, bukan pengesahan bahwa setiap pose dataset benar. ASL tidak dipakai sebagai rujukan label SIBI. Lembar contoh diambil hanya dari manifest train, dua kelompok sumber berbeda per huruf, dengan hash gambar diperiksa:

- [Kandidat pose SIBI](../data/pilot_references/v1/sibi_candidate_poses.jpg).
- [Kandidat pose BISINDO](../data/pilot_references/v1/bisindo_candidate_poses.jpg).

Gambar tersebut lokal dan diabaikan Git. Reproduksi setelah dataset/manifest lama tersedia:

```powershell
python scripts/render_pilot_references.py
```

Cocokkan dengan referensi bahasa yang disepakati dan, bila tersedia, praktisi/penutur atau pengajar. [Kamus SIBI](https://pkplk.kemendikdasmen.go.id/sibi/) dapat menjadi titik awal untuk SIBI; tautan itu tidak mengesahkan gambar lokal secara otomatis. Referensi alfabet BISINDO spesifik masih perlu ditetapkan. Jangan menyamakan jumlah tangan yang diperlukan semua huruf: contoh C/R BISINDO lokal tampak satu tangan, sedangkan contoh A/B/T melibatkan dua tangan.

Catat hasil di [`pilot_pose_review.csv`](pilot_pose_review.csv): isi `status=verified` hanya setelah pemeriksaan, beserta URL/dokumen referensi, reviewer samaran, tanggal, dan catatan orientasi/jumlah tangan. Saat ini semua entri `pending`. Collector dapat menyimpan bahan tinjauan dengan label yang dimaksud, tetapi data berstatus pending belum boleh dianggap ground truth untuk training. Pose `rejected` tidak dapat dikumpulkan sebagai positif. Jika kandidat harus diganti setelah koleksi dimulai, buat versi protokol/direktori baru.

## 2. Orang, sesi, dan target

Peran berikut berlaku **untuk orang yang sama di kedua bahasa dan kedua sesi**:

| ID samaran | Peran | Pemakaian |
|---|---|---|
| p001 | train | Fit model; boleh digabung data lama yang sudah lolos kurasi |
| p002 | val | Pilih preprocessing, kandidat model, threshold dan stabilisasi |
| p003 | test | Evaluasi setelah seluruh pilihan dibekukan |

Gunakan orang yang berbeda untuk ketiga ID. Orang yang sama tidak boleh berganti ID demi memenuhi jumlah peserta. Bila baru satu orang tersedia, mulai dari p001 dan catat bahwa evaluasi antarorang belum mungkin.

Jika training nanti memakai dataset lama, periksa juga apakah peserta validation/test pernah ikut pengumpulan sumber tersebut. Bila tidak bisa dipastikan, catat ketidakpastiannya; pemisahan ID pilot saja tidak membuktikan orang itu tidak ada pada data lama.

Ambil `s01` dan `s02` pada waktu berbeda; variasikan latar/pencahayaan secara wajar dan catat. Target per **orang × bahasa × sesi**:

- Lima trial terpisah per huruf, jadi 25 trial positif per sesi.
- Dua trial untuk masing-masing empat jenis negatif, jadi delapan trial negatif per sesi.

Total rencana: 300 trial positif dan 96 negatif di kedua bahasa. Beberapa foto dari satu trial adalah sampel berkorelasi; tidak dihitung sebagai percobaan independen. Pilot dengan satu peserta test bukan evaluasi final MVP pada minimal tiga peserta final; target akhir pada roadmap tetap berlaku.

## 3. Menjalankan collector

Collector hanya memerlukan OpenCV dan NumPy yang sudah ada pada environment fitur. Tidak memakai classifier/MediaPipe sehingga kegagalan deteksi tidak menyebabkan foto dibuang.

Periksa protokol tanpa membuka kamera atau menulis file:

```powershell
python scripts/collect_webcam_pilot.py --mode sibi --participant-id p001 --session-id s01 --check-only
```

Setelah peserta setuju gambar disimpan lokal, mulai sesi SIBI:

```powershell
python scripts/collect_webcam_pilot.py --mode sibi --participant-id p001 --session-id s01 --save --consent --hand right --lighting room_lamp --background plain_wall
```

Sesi BISINDO menggunakan protokol yang sama dan direktori mode terpisah:

```powershell
python scripts/collect_webcam_pilot.py --mode bisindo --participant-id p001 --session-id s01 --save --consent --hand both --lighting room_lamp --background plain_wall
```

`--hand both` mencatat penggunaan tangan pada sesi, bukan aturan wajib dua tangan per huruf. Isikan kondisi sebenarnya; metadata ini tidak mengubah label maupun memfilter gambar. Gunakan `--device 1` bila kamera berbeda. Ulangi dengan `--session-id s02` pada sesi berikutnya; peserta lain memakai ID peran masing-masing.

## 4. Urutan satu trial

Klik jendela preview agar menerima keyboard. Tekan huruf atau tombol negatif ketika siap; ada hitung mundur dua detik untuk menyiapkan tangan. Setelah tulisan **REKAM**, tahan pose tiga detik sampai kembali ke status netral. Kembalikan tangan ke posisi netral sebelum memulai trial berikutnya. Untuk BISINDO, jeda persiapan memungkinkan kedua tangan kembali ke pose setelah menekan tombol.

Frame disimpan sekitar setiap 0,25 detik (jumlah aktual mengikuti kamera). Tidak ada prediksi model yang ditampilkan untuk menghindari pengambilan ulang hanya ketika model memberikan jawaban yang diinginkan. Pertahankan tangan/ujung jari yang diperlukan dalam frame; variasikan jarak secara wajar antartrial tanpa mengubah label.

| Tombol | Label yang dimaksud |
|---|---|
| Huruf kandidat | Pose positif sesuai mode |
| 1 | `no_hand`: tidak ada tangan dalam area kamera |
| 2 | `neutral`: tangan pada posisi istirahat yang bukan pose target |
| 3 | `non_sign_pose`: pose yang telah ditinjau sebagai bukan isyarat target |
| 4 | `transition`: perpindahan dari/ke posisi pose selama interval rekam |
| Space | Batalkan countdown/trial; foto trial yang dibatalkan tetap ditandai aborted |
| Esc | Keluar; trial yang belum selesai ditandai aborted |

Jangan menandai huruf lain yang sah sebagai `non_sign_pose` untuk sistem alfabet lengkap. Bila status pose ragu, tandai pada kurasi dan jangan gunakan sebagai negatif otomatis.

Trial `transition` dapat mencakup pose positif pada awal/akhir rekaman. Jangan melabeli seluruh framenya sebagai negatif untuk classifier statis. Gunakan untuk evaluasi temporal sampai interval/frame transisi telah dianotasi secara terpisah.

Preview dicerminkan, sedangkan foto yang disimpan tetap frame kamera asli tanpa teks, landmark, atau mirror. Protokol ini dicatat pada metadata. Tidak ada rekaman video atau unggahan otomatis. Collector dapat dilanjutkan pada mode/orang/sesi yang sama; trial baru mendapat ID unik, dan hitungan trial selesai sebelumnya ditampilkan kembali.

## 5. File dan peninjauan data

Semua foto/metadata asli masuk `data/webcam_pilot_v1/`, yang diabaikan Git:

```text
protocol.json
sibi/p001/s01/
  summary.json
  trials/<trial_id>/
    trial.json
    frames.csv
    frames/frame_0000.jpg
bisindo/p001/s01/...
```

`trial.json` menyimpan peran peserta, bahasa, label yang dimaksud, status selesai/batal, snapshot tinjauan pose, kondisi pengambilan, jumlah frame, dan hash protokol. `frames.csv` berisi waktu relatif serta hash JPEG. Koleksi yang terputus mendadak dapat meninggalkan `status=recording`; trial itu harus ditinjau dan tidak otomatis masuk training. Trial aborted tidak dihitung sebagai target selesai dan tidak digunakan sebagai positif/negatif yang sah.

Setiap trial tetap `trial_review_status=pending` saat diambil. Setelah pemeriksaan ketepatan label, tangan terlihat, dan interval pose stabil, simpan keputusan keep/exclude pada manifest kurasi terpisah, memakai format [`pilot_trial_review.example.csv`](pilot_trial_review.example.csv). Jangan menimpa foto/metadata asli. Kelengkapan hitungan tidak sama dengan kelulusan label.

Jangan mengedit `protocol.json` untuk mengganti peran peserta/kelas di koleksi berjalan. Collector menolak protokol yang berubah pada direktori sama. ASL, model lama, dan dataset pilot tidak digabung secara diam-diam.

## 6. Setelah data tersedia

Jalankan `python scripts/audit_webcam_pilot.py` untuk memeriksa integritas, cakupan, dan membuat grid development di `data/pilot_audit/v1/`. Auditor tidak membuka foto peserta test. Hasil pemeriksaan awal dan perbaikan pengambilan tercatat di [PILOT_REVIEW_20261020.md](PILOT_REVIEW_20261020.md). Pastikan detail jari tajam dan cahaya dari depan sebelum memperbanyak rekaman.

1. Tinjau pose dan trial p001/p002; siapkan manifest train/validation dengan seluruh frame satu trial tetap bersama.
2. Bandingkan baseline SIBI landmark dan kandidat BISINDO landmark/crop gambar pada validation yang sama. Model crop memerlukan deteksi/anotasi area tangan yang diverifikasi; collector belum menyediakan bounding box otomatis.
3. Pilih model, orientasi, penolakan dan stabilisasi dari p002. Laporkan juga tanpa tangan, transisi, penolakan, serta semua kegagalan detektor.
4. Bekukan pilihan sebelum p003 dipakai mengevaluasi prediksi. Hasil test tidak dipakai untuk mengganti model tanpa menyediakan test baru.
5. Perluas alfabet dan data orang/sesi setelah pilot layak. Kemudian buat satu service dengan dua model/mode, endpoint FastAPI, dan UI HTML.

Dokumen ini mencatat rencana kerja; belum ada angka performa model dari dataset pilot.

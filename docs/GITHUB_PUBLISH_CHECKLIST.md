# File yang dikirim ke GitHub

| Boleh di Git | Tetap lokal / server |
|---|---|
| `src/`, `scripts/`, `tests/` | `dataset/`, `data/` |
| `configs/` tanpa secret | `artifacts/`, `models/`, checkpoint/bobot |
| `docs/`, README, requirements | foto/video, grid dataset, hasil ekstraksi `.npy/.npz` |
| Notebook sumber tanpa output | Notebook yang sudah dieksekusi dan berisi output |
| Dockerfile, `.dockerignore`, `.github/`, `deploy/` | `.env`, SSH key, token, virtual environment |
| `.env.example` dan `deploy/.env.example` | ZIP/arsip paket training dan bundle model |
| Ringkasan audit kecil yang sudah diizinkan `.gitignore` | inventory/cache/raw report besar |

`.dockerignore` juga memakai daftar yang diizinkan agar `docker build .` tidak mengirim dataset/model/secret ke builder. Image hanya mengandung runtime code, dua config detector, dan dependency. Model di-mount terpisah di VPS.

## Pemeriksaan sebelum push

```powershell
& 'C:\Program Files\Python312\python.exe' scripts/audit_github_files.py --include-untracked --check
git status --short
```

Audit meninjau daftar tracked + untracked yang tidak diabaikan, menolak file model/data/media/arsip/private key, `.env`, file >10 MiB, notebook dengan output, serta beberapa pola kredensial. Ini pemeriksaan batas file, bukan pemindai secret yang menjamin seluruh histori bersih.

Pada persiapan ditemukan lima JPG dataset yang sudah tercatat di indeks Git. `.gitignore` diperketat dan gambar tersebut dilepas dari indeks; salinan lokal tetap ada. Jika perlu mengulang pada checkout lama:

```powershell
git rm --cached --ignore-unmatch -- reports/bisindo_kaggle_v2/detector-review/conf03_recovered_A.jpg reports/bisindo_kaggle_v2/detector-review/conf03_recovered_I.jpg reports/bisindo_kaggle_v2/detector-review/conf03_recovered_Y.jpg reports/bisindo_kaggle_v2/landmark-comparison-conf03/A-M.jpg reports/bisindo_kaggle_v2/landmark-comparison-conf03/N-Z.jpg
```

Tindakan ini membersihkan snapshot berikutnya; tidak menulis ulang histori commit lama. Link gambar laporan lama masih dapat dibuka pada workspace yang memiliki artefak lokal, tetapi gambar tersebut tidak ikut clone baru. Tidak ada bobot/model/dataset mentah pada daftar tracked saat audit persiapan.

Setelah audit lulus, tinjau perubahan lalu commit/push sendiri:

```powershell
git add .
git diff --cached --stat
git diff --cached --check
git commit -m "feat: add SIBI BISINDO inference API and Docker deployment"
git push origin main
```

Perubahan riset yang sebelumnya belum di-commit juga dapat muncul pada `git add .`; periksa daftar sebelum commit. Persiapan ini tidak membuat commit, mengubah remote, atau melakukan push. CI akan mengulang audit file pada setiap PR/push.

## Model di luar Git

Gunakan bundle `models/selected-sibi-bisindo-v1/`, lalu transfer ke VPS sesuai [DEPLOYMENT.md](DEPLOYMENT.md). Bundle berisi dua bobot terpilih serta metadata tepatnya, bukan seluruh direktori training. Perubahan kode lewat GitHub Actions tetap bisa memperbarui service tanpa mengunggah ulang model selama hash kontrak preprocessing masih cocok.

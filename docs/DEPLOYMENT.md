# Deployment Docker dan GitHub Actions

Target yang disepakati: VPS Ubuntu/Linux amd64, Docker Compose v2, GitHub Actions + SSH, branch `main`, dan browser frontend mengakses API langsung. API sudah diperiksa lokal dengan kedua model terpilih. Docker tidak terpasang pada mesin pengembangan saat persiapan ini, sehingga build image dan deployment VPS belum diverifikasi.

## Pemisahan kode dan model

GitHub menyimpan kode, konfigurasi, Dockerfile, workflow, pengujian, notebook sumber, dan dokumentasi. Dataset, cache, hasil training, bobot, environment, SSH key, dan `.env` tidak dikirim ke Git.

Bundle terpilih sudah tersedia lokal di `models/selected-sibi-bisindo-v1/` (sekitar 0,8 MiB):

```text
bundle.json
sibi/selection.json
sibi/dataset.json
sibi/cnn_best.pt
bisindo/selection.json
bisindo/dataset.json
bisindo/cnn_best.pt
```

Bundle mempertahankan metadata asli dan hash checkpoint. Server memeriksa seluruh file bundle serta kontrak preprocessing sebelum sehat. Tidak membutuhkan folder `dataset/`, `data/`, atau hasil eksperimen lain.

Untuk mengekspor ulang ke direktori versi baru:

```powershell
& 'C:\Program Files\Python312\python.exe' scripts/export_service_models.py --output models/selected-sibi-bisindo-v2
```

Tidak menimpa output yang ada. Konfigurasi model sumber ada di `configs/service_models.json`. Bobot baru perlu diekspor dan dikirim ke server secara terpisah; push kode saja tidak mengganti bobot secara diam-diam.

## Menjalankan lokal

Environment service terpisah sudah dibuat pada `.venv-service`. Dari direktori proyek, jalankan API dan halaman uji dalam satu terminal:

```powershell
& '.\.venv-service\Scripts\python.exe' scripts/run_local_demo.py
```

Buka `http://localhost:8000` atau `http://localhost:8000/demo/webcam.html`. Pilih SIBI/BISINDO, klik **Mulai kamera**, lalu izinkan browser menggunakan kamera. Halaman menampilkan preview, rangka landmark, huruf, skor mentah model, jumlah tangan, waktu proses, dan frame per detik. Tombol **Berhenti** mematikan kamera; `Ctrl+C` menghentikan server. API docs tersedia pada `http://localhost:8000/docs`.

Jika port 8000 sedang dipakai, tentukan port lain, misalnya:

```powershell
& '.\.venv-service\Scripts\python.exe' scripts/run_local_demo.py --port 8001
```

Launcher lokal melayani halaman dari proses FastAPI yang sama dan mengizinkan origin localhost pada port yang dipilih. Bila halaman web lain perlu mengakses API, atur `$env:ALLOWED_ORIGINS` sebelum menjalankan service. `.env.example` merupakan contoh pengaturan; factory membaca environment proses, bukan otomatis membaca file `.env` lokal.

Instalasi dari awal di mesin lain: Python 3.12, buat virtual environment, pasang PyTorch CPU 2.6.0 dari indeks resmi CPU, lalu `pip install -r requirements-service.txt`. Dependency MediaPipe harus sesuai pin karena model demo sebelumnya sudah mengunci versi dan hash model bawaannya.

## Persiapan VPS satu kali

Mulai dengan 2 vCPU/4 GiB RAM untuk satu container CPU; ini saran kapasitas awal, bukan hasil benchmark server. Instal Docker Engine + Compose v2 serta reverse proxy dengan HTTPS. Akun SSH deployment harus dapat memakai Docker.

Contoh direktori dengan akun bernama `deploy`:

```bash
sudo install -d -o deploy -g deploy /opt/sibi-bisindo-ai /opt/sibi-bisindo-ai/models /opt/sibi-bisindo-ai/releases
```

Kirim bundle dari komputer lokal (ganti USER dan HOST):

```powershell
scp -r models/selected-sibi-bisindo-v1 USER@HOST:/opt/sibi-bisindo-ai/models/
scp deploy/.env.example USER@HOST:/opt/sibi-bisindo-ai/.env
```

Edit `.env` di VPS:

```dotenv
AI_IMAGE=ghcr.io/owner/repository:sha-COMMIT_40_KARAKTER
AI_PORT=8000
MODEL_BUNDLE_PATH=/opt/sibi-bisindo-ai/models/selected-sibi-bisindo-v1
ALLOWED_ORIGINS=https://frontend-domain-anda.com,https://api-domain-anda.com
```

Origin harus lengkap dengan skema/port bila ada, tanpa path atau slash di akhir. Beberapa origin dipisahkan koma. Sertakan origin API sendiri bila ingin memakai Try it out di Swagger. Domain frontend dan API sebenarnya belum diberikan, jadi konfigurasi tidak memakai domain produksi rekaan. Path bundle harus sudah ada; Compose tidak membuat folder model kosong secara otomatis.

Workflow mempublikasikan image ke GHCR. Jika package private, login ke GHCR **sekali di VPS** dengan PAT `read:packages` milik akun yang punya akses:

```bash
read -rsp 'GHCR token: ' ghcr_token
printf '%s' "$ghcr_token" | docker login ghcr.io -u GITHUB_USERNAME --password-stdin
unset ghcr_token
chmod 600 /opt/sibi-bisindo-ai/.env
```

Jangan taruh token ini di repo. CI build memakai `GITHUB_TOKEN`; server memakai kredensial pull-nya sendiri. Untuk package public, login pull tidak diperlukan.

## GitHub Actions

Workflow: `.github/workflows/deploy.yml`.

1. Pull request ke `main`: audit file Git dan pengujian kontrak API/isolasikan inferensi.
2. Push ke `main`: pemeriksaan yang sama, lalu build image dan publish `ghcr.io/owner/repository:sha-<commit>`.
3. Bila repository variable `DEPLOY_ENABLED=true`, job deployment mengirim Compose/script lewat SSH dan menjalankan image baru.
4. Deployment menunggu healthcheck. Bila gagal, script mengembalikan image dan Compose deployment sehat sebelumnya jika tersedia.

Gunakan environment GitHub bernama **production**. Isi secrets berikut (jangan kirim nilainya ke chat/repo):

| Secret | Nilai |
|---|---|
| `DEPLOY_HOST` | IP atau hostname VPS |
| `DEPLOY_USER` | Akun SSH deployment |
| `DEPLOY_PORT` | Port SSH; default 22 |
| `DEPLOY_SSH_KEY` | Private key khusus deployment |
| `DEPLOY_KNOWN_HOSTS` | Entri known_hosts server yang fingerprint-nya sudah diverifikasi |

Repository variables:

| Variable | Nilai |
|---|---|
| `DEPLOY_ENABLED` | `true` setelah VPS/model/env siap; jika belum, build/publish saja |
| `DEPLOY_PATH` | `/opt/sibi-bisindo-ai` atau path absolut sesuai VPS |

Public key pasangan `DEPLOY_SSH_KEY` dipasang di `authorized_keys` akun VPS. Host key checking tetap aktif. Tidak memakai password SSH atau `StrictHostKeyChecking=no`. Repository variable `DEPLOY_ENABLED` ditempatkan pada scope repository agar dapat dipakai pada kondisi job sebelum environment production aktif.

Workflow sudah diserialkan per branch agar dua deployment `main` tidak bertabrakan. Akun registry build mendapat izin `packages:write`, job lain hanya `contents:read`.

## HTTPS dan akses browser

Container hanya bind pada `127.0.0.1:8000` di VPS. Reverse proxy menerima HTTPS/WSS dari browser. Contoh konfigurasi Nginx ada pada `deploy/nginx.conf.example`; sesuaikan domain dan sertifikat TLS yang sudah tersedia. `map` dan `limit_*_zone` ditempatkan pada konteks `http` Nginx. Terapkan setelah `nginx -t` berhasil.

Konfigurasi contoh membatasi ukuran body, request per IP, jumlah koneksi, dan meneruskan upgrade WebSocket. HTTPS diperlukan untuk kamera pada frontend publik. CORS tidak menggantikan autentikasi aplikasi; integrasi login/JWT perlu disepakati dengan developer web jika endpoint harus dibatasi ke pengguna tertentu.

Docker berjalan sebagai UID 10001, filesystem aplikasi read-only, model mount read-only, dan satu worker Uvicorn. File model harus dapat dibaca UID tersebut (misalnya folder 0755 dan file 0644). Tidak menyimpan frame pengguna. Jangan menambah worker untuk throughput tanpa menghitung penggunaan RAM dan menguji kapasitas.

## Operasional dan rollback

Periksa:

```bash
docker compose --env-file /opt/sibi-bisindo-ai/.env -f /opt/sibi-bisindo-ai/current/compose.yaml ps
docker compose --env-file /opt/sibi-bisindo-ai/.env -f /opt/sibi-bisindo-ai/current/compose.yaml logs --tail=100
curl -f http://127.0.0.1:8000/health/ready
```

Script menyimpan image sehat di `/opt/sibi-bisindo-ai/current-image` dan symlink Compose sehat di `current`. Nilai `AI_IMAGE` di `.env` adalah placeholder/bootstrap; selama deployment script menggunakan tag yang diterimanya dari CI. Untuk perintah `up` manual, ekspor image aktif terlebih dahulu:

```bash
export AI_IMAGE=$(cat /opt/sibi-bisindo-ai/current-image)
```

Rollback manual ke commit tertentu dapat dilakukan dengan menjalankan script release yang sesuai dan tag image immutable sebelumnya. Tidak mengubah bundle model otomatis. Saat policy/fitur model berubah, buat bundle baru dan sesuaikan `MODEL_BUNDLE_PATH` secara sengaja; ketidakcocokan hash akan menggagalkan startup.

Detail kontrak frontend: [API_CONTRACT.md](API_CONTRACT.md). Pemilahan Git: [GITHUB_PUBLISH_CHECKLIST.md](GITHUB_PUBLISH_CHECKLIST.md). Pola satu proses per container dan GHCR mengikuti [panduan FastAPI](https://fastapi.tiangolo.com/deployment/docker/) serta [dokumentasi GitHub](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images).

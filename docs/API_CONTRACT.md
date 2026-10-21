# Kontrak API untuk developer web

Service mengenali satu pose alfabet statis per frame. Pilih bahasa secara eksplisit: `sibi` atau `bisindo`. Tidak merangkai kata/kalimat. J/Z SIBI masih memakai pose statis; D/F/P BISINDO memiliki keterbatasan pada webcam.

## Endpoint

| Endpoint | Fungsi |
|---|---|
| `GET /health/live` | Proses hidup |
| `GET /health/ready` | Dua model sudah dimuat dan diverifikasi |
| `GET /v1/models` | ID model, label, dan batasan |
| `POST /v1/predict/{language}` | Satu frame JPEG/PNG menjadi prediksi |
| `WS /v1/ws/{language}` | Frame webcam biner, satu respons JSON per frame |
| `GET /docs` | Swagger/OpenAPI interaktif |

API memakai dua checkpoint terpilih: SIBI `sibi_20261020_130607_889055` dan BISINDO `bisindo_20261021_133454_756637`. Classifier dan praproses sesuai demo webcam. Tidak ada threshold atau voting lintas frame tambahan di server.

## Format input

Frame dikirim sebagai **byte JPEG atau PNG**, bukan JSON/base64 atau multipart. HTTP harus menggunakan `Content-Type: image/jpeg` atau `image/png`. Batas default 2.000.000 byte dan 2.073.600 piksel. Rekomendasi awal: 640x480, JPEG quality 0,8.

Input normal adalah gambar **tidak dimirror**. Untuk kamera browser, gunakan `drawImage(video,...)` tanpa flip. Video preview boleh memakai CSS `transform: scaleX(-1)`; CSS tidak mengubah gambar yang dikirim. Bila input memang sudah dimirror, tambahkan `?mirrored=true`; server membaliknya satu kali sebelum inferensi. Landmark respons selalu dalam koordinat gambar yang sudah dikembalikan ke orientasi tidak dimirror.

Query opsional pada HTTP dan WebSocket:

- `include_landmarks=true`: sertakan koordinat xyz (default false).
- `mirrored=true`: perbaiki input yang sudah terlanjur dimirror (default false).

Contoh HTTP:

```javascript
const canvas = document.createElement('canvas');
canvas.width = 640;
canvas.height = 480;
canvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
const frame = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
const response = await fetch('https://ai.example.com/v1/predict/bisindo', {
  method: 'POST', headers: {'Content-Type': 'image/jpeg'}, body: frame
});
if (!response.ok) throw new Error(await response.text());
const prediction = await response.json();
```

Pertahankan rasio aspek kamera pada aplikasi sebenarnya; contoh 640x480 mengasumsikan video 4:3. Tunggu respons sebelum mengirim frame berikutnya. Jangan memakai `setInterval` yang terus menumpuk request ketika inferensi belum selesai.

## Respons

```json
{
  "language": "bisindo",
  "model_id": "bisindo_20261021_133454_756637",
  "status": "predicted",
  "detection_status": "two_hands_actor_unverified",
  "label": "A",
  "score": 0.9988,
  "score_type": "softmax_uncalibrated",
  "hand_count": 2,
  "width": 640,
  "height": 480,
  "preprocessing": {
    "method": "rotate45",
    "attempts": 9,
    "recovered": true,
    "input_was_mirrored": false,
    "coordinates": "unmirrored_normalized_xyz"
  },
  "timings_ms": {"detector": 230.0, "classifier": 0.7, "total": 240.0},
  "landmarks": null
}
```

`label` dan `score` bernilai null saat ekstraksi tidak valid. `status` dapat berupa `no_hand`, `unexpected_extra_hand` (SIBI membutuhkan satu tangan), atau `invalid_geometry`. Jika valid, `status=predicted`; `detection_status` tetap menerangkan keluaran detektor. Satu tangan valid pada BISINDO belum membuktikan pose dua tangan lengkap.

`score` adalah softmax yang belum dikalibrasi, bukan probabilitas kebenaran yang terjamin. Belum ada kelas unknown/background. Huruf yang terbaca konsisten belum tentu pose benar.

Jika diminta, `landmarks` berbentuk `[jumlah_tangan][21][3]`; slot diurutkan menurut posisi pergelangan pada gambar, bukan label tangan kanan/kiri anatomis. x/y dinormalisasi terhadap lebar/tinggi, z relatif terhadap tangan masing-masing. Respons tidak mengandung gambar.

## WebSocket

URL: `wss://ai.example.com/v1/ws/bisindo`. Kirim Blob/ArrayBuffer JPEG/PNG; tunggu event `message`, lalu kirim frame selanjutnya. Browser mengirim header Origin otomatis dan origin tersebut harus ada pada konfigurasi server. WebSocket tidak dilindungi middleware CORS HTTP, sehingga handshake diperiksa tersendiri.

Respons sukses sama dengan respons HTTP dengan tambahan `"type":"prediction"`. Error berupa `{"type":"error","code":"503","detail":"..."}` atau `binary_frame_required`/`empty_frame`. Frame terlalu besar menutup koneksi dengan kode 1009; origin atau bahasa tidak sah ditolak dengan kode 1008. Koneksi tanpa frame selama 60 detik ditutup.

Contoh lengkap ada pada [examples/webcam.html](../examples/webcam.html). Untuk uji lokal, jalankan `& '.\.venv-service\Scripts\python.exe' scripts/run_local_demo.py` lalu buka `http://localhost:8000`. Browser memerlukan localhost atau HTTPS untuk `getUserMedia`. Halaman memakai satu frame aktif; opsi cermin hanya membalik preview, sedangkan piksel yang dikirim ke API tetap sesuai kamera. Halaman tidak merekam atau menyimpan frame.

## Error HTTP dan kapasitas

| Status | Arti/tindakan |
|---|---|
| 403 | Origin browser tidak diizinkan |
| 413 | Byte frame terlalu besar; perkecil |
| 415 | Content-Type tidak sesuai |
| 422 | Bahasa/query/gambar tidak valid atau piksel terlalu banyak |
| 503 | Service sibuk/tidak siap; HTTP menyertakan `Retry-After: 1` |
| 500 | Kegagalan inferensi; simpan error_id untuk pemeriksaan server |

Respons HTTP memiliki `X-Request-ID`. Server tidak mencatat atau menyimpan frame, dan tidak menyimpan riwayat prediksi pengguna. CORS mengatur origin browser, bukan autentikasi pengguna; service MVP ini belum memvalidasi login/JWT aplikasi web.

Satu proses menggunakan satu jalur inferensi CPU agar graph MediaPipe tidak diakses bersamaan. Request berlebih segera mendapat 503 setelah menunggu maksimal 100 ms. Mulai dari 3–5 frame/detik yang menyesuaikan respons; fallback BISINDO bisa lebih lambat. Untuk banyak pengguna, kapasitas perlu diukur dan ditambah dengan beberapa container.

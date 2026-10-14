# Baseline klasifikasi gambar langsung BISINDO

Eksperimen ini mengklasifikasikan gambar penuh menjadi 26 huruf A-Z tanpa ekstraksi landmark MediaPipe. Kandidat awal adalah MobileNetV3-Small dengan bobot ImageNet dan fine-tuning dua tahap. Pemilihan checkpoint memakai macro-F1 validation. Skrip hanya memuat gambar split train dan validation dari manifest BISINDO v2; split test tidak dibaca.

## Environment

PyTorch dan TorchVision harus tersedia. Google Colab biasanya sudah menyediakannya. Di Colab, jangan memasang ulang PyTorch karena bisa mengganti build CUDA bawaan. Periksa perangkat dan dependensi:

```python
import torch
print(torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Jika `scikit-learn` tidak tersedia di Colab, pasang hanya paket itu dengan `%pip install scikit-learn`. Untuk training lokal, pasang PyTorch/TorchVision yang sesuai dengan perangkat dan CUDA melalui pemilih instalasi resmi PyTorch, kemudian pasang dependensi proyek:

```powershell
python -m pip install -r requirements-image-training.txt
```

Environment Python yang tersedia saat baseline disiapkan memakai PyTorch CPU, sehingga training lokal dapat jauh lebih lambat. Gunakan Colab GPU bila tersedia. Pemanggilan pertama dengan `--weights imagenet` mengunduh bobot MobileNetV3 jika belum ada di cache; gunakan `--weights none` jika harus berlatih tanpa unduhan, dengan konsekuensi model mulai dari bobot acak.

## Menyiapkan Google Colab

1. Di Colab pilih **Runtime → Change runtime type → GPU**, lalu jalankan pemeriksaan CUDA di atas. Jenis GPU dan ketersediaannya dapat berubah.
2. Dari root repository, buat arsip train/validation yang dikunci manifest:

```powershell
python scripts/package_bisindo_train_val.py
```

Skrip menghasilkan `bisindo_train_val_images.zip` di root. Arsip hanya memuat 8.341 gambar train dan 1.593 validation sesuai manifest v2, memeriksa hash gambar yang dimasukkan, serta tidak membaca atau memasukkan gambar test. Unggah ZIP itu dan empat berkas kecil berikut ke folder `My Drive/sibi-bisindo/`: `configs/bisindo_kaggle_v2.json`, `data/manifests/bisindo-kaggle-v2/manifest.csv`, `data/manifests/bisindo-kaggle-v2/split_lock.json`, serta `scripts/train_image_classifier.py`. Jangan unggah folder dataset penuh.
3. Mount Drive, salin berkas kecil ke disk runtime, lalu ekstrak ZIP ke susunan folder yang dibutuhkan skrip:

```python
from google.colab import drive
from pathlib import Path
import shutil

drive.mount('/content/drive')
drive_project = Path('/content/drive/MyDrive/sibi-bisindo')
runtime_project = Path('/content/sibi-bisindo')
runtime_project.mkdir(parents=True, exist_ok=True)
for folder in ('configs', 'data', 'scripts'):
    shutil.copytree(drive_project / folder, runtime_project / folder, dirs_exist_ok=True)
```

```python
from pathlib import Path
import zipfile

archive = Path('/content/drive/MyDrive/sibi-bisindo/bisindo_train_val_images.zip')
destination = Path('/content/sibi-bisindo/dataset/Indonesian Sign Language - BISINDO')
destination.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as images_zip:
    images_zip.extractall(destination)
```

4. Jalankan training dari salinan lokal. Simpan hasil ke Drive agar checkpoint bertahan jika runtime Colab berakhir atau terputus:

```python
%cd /content/sibi-bisindo
!python scripts/train_image_classifier.py --device cuda --batch-size 32 --num-workers 2 --output-dir /content/drive/MyDrive/sibi-bisindo/artifacts/bisindo-image-run-01
```

Jika CUDA out-of-memory, gunakan `--batch-size 16`. Untuk run berikutnya, ganti nama `bisindo-image-run-01` karena skrip menolak menimpa direktori hasil yang sudah berisi file. Ketersediaan GPU tidak dijamin dan runtime Colab dapat berakhir; simpan checkpoint di Drive. Menyalin satu arsip lalu mengekstraknya ke disk runtime juga menghindari membaca ribuan file satu per satu dari Drive.
## Menjalankan training

Jalankan dari root repository setelah dataset dan manifest v2 tersedia:

```powershell
python scripts/train_image_classifier.py
```

Default: 30 epoch maksimum, 3 epoch warm-up head, early stopping patience 7, batch size 32, dan pemilihan perangkat otomatis. Gunakan `--device cuda` atau `--device cpu` untuk menentukan perangkat secara eksplisit. Training menampilkan loss, accuracy, macro-F1 validation, dan durasi setiap epoch. `history.csv`, `last.pt`, `best.pt`, `best_validation_metrics.json`, dan `training_state.json` disimpan di direktori unik bawah `artifacts/`.

Contoh untuk mengurangi batch size jika memori GPU terbatas:

```powershell
python scripts/train_image_classifier.py --batch-size 16
```

Data input diubah menjadi RGB dan 224x224. Training memakai horizontal flip, rotasi ringan, serta variasi brightness/contrast/saturation. Validation tidak diaugmentasi. Manifest, split lock, dan hash gambar train/validation diverifikasi sebelum training.

## Membaca hasil

`best_validation_metrics.json` berisi macro-F1, accuracy, confusion matrix, dan metrik tiap kelas dari checkpoint dengan macro-F1 validation terbaik. Metrik ini hanya mengukur klasifikasi pada gambar berlabel yang memiliki pose; baseline ini belum memiliki kelas `no_hand`/`unknown`, jadi belum mengukur perilaku tanpa tangan atau prediksi palsu di latar belakang. Perlu kumpulkan contoh webcam untuk kalibrasi penolakan dan evaluasi domain sebelum klaim performa realtime.

Jangan membuka test untuk memilih model atau epoch. Setelah model, preprocessing, dan kebijakan penolakan dibekukan, test hanya dipakai untuk evaluasi akhir satu kali sesuai [protokol evaluasi](EVALUATION_PROTOCOL.md). Split v2 mengelompokkan kandidat duplikat berdasarkan pHash, bukan identitas peserta/sesi; evaluasi ini bukan signer-independent.



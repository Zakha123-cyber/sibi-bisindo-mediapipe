# Review BISINDO recovery v2 training run 01

**Pembaruan setelah perbandingan:** pengguna kemudian menilai model baru bagus secara umum. Kendala terpusat pada D/F/P; F/P masih kadang berhasil, tetapi D belum dikenali dalam percobaan pengguna. Laporan sebelumnya tentang regresi umum bukan kesimpulan akhir. Script perbandingan tidak menyimpan frame atau log prediksi, sehingga sesi tersebut belum dapat ditinjau ulang secara langsung.

Pemeriksaan validation v2: D benar 52/54 (96,30%), salah ke C dan S masing-masing satu; F benar 57/59 (96,61%), salah ke B dan G; P benar 63/67 (94,03%), salah ke D/K/S/X masing-masing satu. Tidak ada reject pada ketiga kelas ini. Angka ini tidak membuktikan pose webcam pengguna cocok dengan data, dan belum mendukung klaim bahwa ketiganya terutama saling tertukar. D yang gagal di webcam meskipun kuat pada validation menunjukkan keterwakilan pose/orientasi/relasi tangan perlu diperiksa dari contoh aktual sebelum menentukan perbaikan.

Tiga baris tampilan perbandingan berasal dari dua checkpoint. NEW/baseline dan NEW/recovery memakai bobot yang sama dengan praproses berbeda. Seluruh input baris tersebut berasal dari frame kamera yang sama.

**Pembaruan webcam:** pengguna melaporkan prediksi recovery v2 lebih buruk daripada model sebelumnya, sedangkan rangka landmark mengikuti tangan. Hasil validation di bawah tetap benar untuk split dataset, tetapi belum menunjukkan perbaikan pada webcam pengguna. Penyebab belum dipastikan; pola koordinat yang dipelajari classifier dapat berbeda dari input webcam, dan recovery juga dapat mengubah koordinat yang tampak masuk akal secara visual.

Untuk sementara, model lama dapat dipakai kembali:

```powershell
& 'C:\Program Files\Python312\python.exe' scripts/webcam_coordinates.py --run-dir artifacts/bisindo-transfer-v1/bisindo_20261020_152850_281288
```

Perbandingan tanpa rekaman atau pengumpulan sampel tambahan:

```powershell
& 'C:\Program Files\Python312\python.exe' scripts/webcam_compare_bisindo.py
```

Tampilan menunjukkan OLD/baseline, NEW pada baseline yang identik, dan NEW/recovery pada frame yang sama. Bila OLD benar tetapi NEW/baseline salah, bobot baru memburuk pada input yang sama. Bila NEW/baseline benar dan NEW/recovery salah, recovery mengubah input menjadi kurang cocok. Keduanya dapat terjadi bersamaan. NEW/baseline adalah ablation diagnostik yang berbeda dari policy training v2, bukan mode produksi baru. Bila baris Recovery menunjukkan baseline, kedua input baru identik dan selisih dengan model lama berasal dari bobot. Tidak ada klaim akurasi otomatis karena label sebenarnya tidak dimasukkan.

Jangan menambah epoch, mengubah confidence, atau menyatakan perlu dataset baru sebelum pemisahan pengaruh ini. Hash preprocessing dan model diperiksa sebelum kamera dibuka; tidak mengubah checkpoint atau policy versi sebelumnya.

Run: `artifacts/bisindo-transfer-recovery-v2/bisindo_20261021_133454_756637`. Training finished 10 head epochs plus 60 fine-tuning epochs. Best checkpoint is **epoch 69** (fine-tuning epoch 59); epoch 70 was slightly lower. Early stopping did not fire because the configured 60 fine-tuning epochs completed. The 10-epoch head stage kept the SIBI encoder unchanged.

## Validation results

| Measure | Transfer v1 | Recovery v2 |
|---|---:|---:|
| Validation images | 1.593 | 1.593 |
| Valid landmark tensors | 1.401 (87,95%) | 1.537 (96,48%) |
| Classifier accuracy on valid tensors | 92,79% | **94,53%** |
| Classifier macro-F1 on valid tensors | 92,54% | **94,38%** |
| Correct predictions across all images | 1.300 | **1.453** |
| System accuracy across all images | 81,61% | **91,21%** |
| System macro-F1 across all images | 86,41% | **92,62%** |
| Rejected for missing/invalid features | 192 | **56** |

The train/validation split and image order match the previous run. On the same 1.401 images that had valid features in both versions, the new classifier gets 1.337 correct (95,43%) versus 1.300 (92,79%) for v1. It also correctly classifies 116 of the 136 additional validation images with recovered features (85,29%). These are validation measurements used during checkpoint selection; they are not an independent test result.

## Per-letter observations

The weakest end-to-end system recall is A: 39/66 (59,09%); 17 A images are still rejected by feature extraction, and 10 valid ones are classified as another letter. Y is next at 52/71 (73,24%). Among valid features, Y has 77,61% recall and 83,87% F1. T system recall is 41/54 (75,93%). I and O classify all of their valid validation examples correctly; this does not establish performance on new signers.

The two-hand-count subsets are similar: 728/772 correct for one detected hand (94,43%) and 725/765 for two detected hands (94,64%). The extractor cannot certify that a one-hand pose is complete, so this grouping is only by detected landmark count.

## Integrity and runtime

The selected checkpoint hash, prepared dataset hash, training script hash, and both source SIBI checkpoint hashes match their recorded metadata. The head checkpoint reports the encoder unchanged. No test arrays were loaded. The local webcam `--check-only` command succeeds and selects `bisindo-runtime-recovery-v2`, with unmirrored RGB inference and mirrored preview.

The classifier alone takes a median 0,158 ms per sample on CPU. This excludes MediaPipe and the fallback views. A 13-image runtime parity check showed cache/runtime tensors agree; observed full preprocessing time was 32–408 ms per image on this machine. That is not a webcam FPS benchmark and may feel slow, especially when fallback views are needed.

## Assessment and next step

This run improves the previous transfer run on the same validation split, in both common-image classification and system coverage. It is a good candidate for an initial webcam check, while A/Y and detector latency remain clear limitations. There is no need to add epochs: epoch 69 already edges epoch 70, and validation is not independent. Test remains unopened.

Use:

```powershell
& 'C:\Program Files\Python312\python.exe' scripts/webcam_coordinates.py --run-dir artifacts/bisindo-transfer-recovery-v2/bisindo_20261021_133454_756637 --check-only
& 'C:\Program Files\Python312\python.exe' scripts/webcam_coordinates.py --run-dir artifacts/bisindo-transfer-recovery-v2/bisindo_20261021_133454_756637
```

Then assess the camera behavior, particularly A, Y, and recovery delay. Do not evaluate on the test split until model/policy choices are frozen.

Detailed machine-readable comparison: [run review JSON](../data/reports/bisindo-recovery-v2/training_review_run01.json). [Epoch log](../artifacts/bisindo-transfer-recovery-v2/bisindo_20261021_133454_756637/cnn_epochs.csv), [validation metrics](../artifacts/bisindo-transfer-recovery-v2/bisindo_20261021_133454_756637/cnn_softmax_validation.json), and [learning curves](../artifacts/bisindo-transfer-recovery-v2/bisindo_20261021_133454_756637/learning_curves.png).

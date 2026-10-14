# Kaggle BISINDO dataset - audit and split proposal

Source directory: `dataset/Indonesian Sign Language - BISINDO/`. Snapshot: `bisindo-kaggle-v1`.

Inventory: **11,470 images**; **11,468 included**, **2 excluded**. Original dimensions are 640 x 640. Class folders A-Z map to YOLO class IDs 0-25.

## New split after perceptual grouping

| Split | Images | Source-image groups | From original train | From original val |
|---|---:|---:|---:|---:|
| train | 7,982 | 4,666 | 6,375 | 1,607 |
| val | 1,788 | 959 | 1,427 | 361 |
| test | 1,698 | 1,209 | 1,365 | 333 |

## Findings and limits

- The provided train/val split had exact duplicate image groups across splits. The new assignment groups files within each letter by connected components of 64-bit perceptual hashes with Hamming distance <= 2, then assigns each component to one split. Visual spot-checks of candidate pairs showed same-capture images with augmentation changes.
- This perceptual grouping reduces augmented-image leakage but is a heuristic, not a recovered capture/session or signer ID. We cannot claim signer-independent evaluation.
- Missing and empty YOLO annotation files are excluded from this classification candidate manifest; original files remain unchanged. YOLO boxes are audited but the current landmark classifier reads full images.
- The downloaded folder has no README/license metadata. Verify Kaggle attribution and license before external release.
- Full rows, hashes, assignments, and per-class summary are local under `data/manifests/bisindo-kaggle-v1/`; generated data is ignored by Git.

Per-class counts and exact configuration/manifest hashes are in `summary.json` and `split_lock.json`.

## MediaPipe landmark extraction

The existing `landmarks-v1` detector and 132-dimensional feature contract were used, with an isolated cache at `data/features/landmarks-bisindo-kaggle-v1/`:

| Split | Images | Feature-valid | Coverage | No hand detected |
|---|---:|---:|---:|---:|
| train | 7,982 | 6,283 | 78.71% | 1,699 |
| val | 1,788 | 1,360 | 76.06% | 428 |

Coverage varies sharply by class (for example, train M/N: 100%, A/Y: 58.2%/62.2%; validation M/N: 100%, Y: 37.7%). This imbalance means system-level metrics must count failed detections as rejects. Train and validation caches are complete; test has not been extracted, and no classifier has been trained on this dataset.


## Baseline classifier validation

Training ran on train/validation only. The selected candidate is MLP with one 128-unit hidden layer, chosen by validation system macro-F1:

| Candidate | Valid-only macro-F1 | System macro-F1 | System accuracy |
|---|---:|---:|---:|
| MLP 128 (selected) | 0.9486 | 0.8202 | 0.7220 |
| MLP 128x64 | 0.9490 | 0.8193 | 0.7226 |
| Linear SVM C=1 | 0.9380 | 0.8075 | 0.7075 |

For the selected model, validation recall ranges from 0.377 (Y) to 1.000 (M, O, U, Y among feature-valid images; system recall differs when extraction rejects are included). There are 428 validation images rejected due to no detected hand, so the 0.9486 valid-only macro-F1 does not represent end-to-end performance. System macro-F1 is 0.8202 and system recall for Y is 0.377. MLP 128 reached the configured 80-epoch limit; train macro-F1 reached 0.9889 while valid-only validation macro-F1 was 0.9486, indicating a modest train/validation gap.

The prior BISINDO baseline selected RBF SVM C=10 with validation system macro-F1 0.6136 on a different manifest/split. The new score is not a controlled apples-to-apples comparison because the dataset and grouping/split changed. Neither validation result establishes signer-independent or webcam performance. The Kaggle test split remains unopened (`test_accessed: false` in the selection record); do not use it until validation selection is reviewed and frozen.

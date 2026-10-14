# Kaggle BISINDO dataset - audit and split proposal

Source directory: `dataset/Indonesian Sign Language - BISINDO/`. Snapshot: `bisindo-kaggle-v2`.

Inventory: **11,470 images**; **11,468 included**, **2 excluded**. Original dimensions are 640 x 640. Class folders A-Z map to YOLO class IDs 0-25.

## New split after perceptual grouping

| Split | Images | Source-image groups | From original train | From original val |
|---|---:|---:|---:|---:|
| train | 8,341 | 4,784 | 6,681 | 1,660 |
| val | 1,593 | 1,024 | 1,261 | 332 |
| test | 1,534 | 1,026 | 1,225 | 309 |

## Findings and limits

- The provided train/val split had exact duplicate image groups across splits. The new assignment groups files within each letter by connected components of 64-bit perceptual hashes with Hamming distance <= 2, then assigns each component to one split. Visual spot-checks of candidate pairs showed same-capture images with augmentation changes.
- This perceptual grouping reduces augmented-image leakage but is a heuristic, not a recovered capture/session or signer ID. We cannot claim signer-independent evaluation.
- Missing and empty YOLO annotation files are excluded from this classification candidate manifest; original files remain unchanged. YOLO boxes are audited but the current landmark classifier reads full images.
- The downloaded folder has no README/license metadata. Verify Kaggle attribution and license before external release.
- Full rows, hashes, assignments, and per-class summary are local under `data/manifests/bisindo-kaggle-v2/`; generated data is ignored by Git.

Per-class counts and exact configuration/manifest hashes are in `summary.json` and `split_lock.json`.

V2 assigns approximately 70/15/15 of the source groups within each class, then searches deterministic assignments to keep image counts close to the requested ratios. Validation now contains 31-57 source groups per class (v1 contained only 1 group for U and 2 for Q). These are pHash components, not verified captures or signers.

## MediaPipe extraction

Using the same `landmarks-v1` detector and 132-dimensional feature contract:

| Split | Images | Feature-valid | Coverage | No hand detected |
|---|---:|---:|---:|---:|
| train | 8,341 | 6,390 | 76.61% | 1,951 |
| val | 1,593 | 1,354 | 84.99% | 239 |

Validation coverage remains uneven: Y 38.0%, A 53.0%, while several classes are above 95%. Visual inspection of v1 failures showed repeated augmented variants of some captures; examples also included two-hand signs and low-contrast/cropped views. V2 improves group representation, but the low Y/A detector coverage remains an issue to investigate. Test has not been extracted. The validation classifier result is recorded below.


## Baseline classifier validation

Training used only v2 train and validation features. The selected model is MLP with hidden layers 128x64, selected using system macro-F1 (invalid landmark rows count as rejects):

| Candidate | Valid-only macro-F1 | System macro-F1 | System accuracy |
|---|---:|---:|---:|
| MLP 128x64 (selected) | 0.9594 | 0.8770 | 0.8154 |
| RBF SVM C=10 | 0.9565 | 0.8747 | 0.8136 |
| Linear SVM C=10 | 0.9560 | 0.8744 | 0.8123 |

The selected MLP reached its best validation system score at epoch 65 and early-stopped after 77 epochs. The classifier-only score is high when landmarks are valid, but 239/1,593 validation images (15.0%) were rejected due to no hand detection. System recall is lowest for Y (0.338) and A (0.530); Y recall among feature-valid rows is 0.889. This remains below the project's initial targets of system macro-F1 >= 0.90 and per-class recall >= 0.80.

The MLP's system score is only 0.0023 above RBF SVM C=10 on this validation split, so the ranking is narrow. The MLP is selected by the documented metric, but this is not evidence of a meaningful margin. The test split is untouched (`test_accessed: false`). Results are still not signer-independent or webcam metrics because participant/session metadata and independent webcam trials are unavailable.


## Validation-only detector ablation

Compared the v2 baseline (RGB, no CLAHE, min detection confidence 0.5) with one preprocessing and two threshold alternatives. All use the same 1,593 validation images; test was not used.

| Variant | Feature-valid | Coverage | Change vs baseline |
|---|---:|---:|---:|
| Baseline, confidence 0.5 | 1,354 | 84.99% | - |
| CLAHE, confidence 0.5 | 1,217 | 76.40% | -137 net |
| Confidence 0.3 | 1,401 | 87.95% | +47 net |
| Confidence 0.7 | 1,313 | 82.42% | -41 net |

At confidence 0.3, 49 baseline failures became feature-valid and 2 previously valid rows became invalid. The 49 gains span 27 pHash groups, but 16 recovered Y images all belong to a single pHash group; Y coverage changes from 27/71 (38.0%) to 43/71 (60.6%). A changes from 35/66 (53.0%) to 37/66 (56.1%). CLAHE recovers only 2 rows while losing 139 baseline-valid rows. Raising confidence loses 41 valid rows and recovers none.

Visual overlay review found plausible one-hand landmarks on recovered I examples. Some recovered A examples show only one of the two visible hands, so `feature_valid` can still accept an incomplete sign. The BISINDO policy currently permits one or two hands because official per-letter hand-count rules have not been confirmed. Lowering confidence to 0.3 is therefore a candidate for retraining, not an adopted production change. It must be followed by train-feature extraction, classifier retraining, validation system metrics, and a review of hand-count correctness. The baseline MLP was trained on confidence 0.5 features; confidence 0.3 uses separate train/validation caches and a separately trained classifier. Train extraction at confidence 0.3 is complete: 6,680/8,341 (80.09%) feature-valid, versus 6,390/8,341 (76.61%) at 0.5. Validation at 0.3 is 1,401/1,593 (87.95%). Each pair of caches shares one config hash; no test cache exists.

Experiment configs are `configs/landmarks_bisindo_kaggle_v2_clahe.json`, `configs/landmarks_bisindo_kaggle_v2_conf03.json`, and `configs/landmarks_bisindo_kaggle_v2_conf07.json`. The validation-only overlay samples are local under `reports/bisindo_kaggle_v2/detector-review/` and are not included in Git.


To compare the classifier after the detector change, train separately on the confidence-0.3 features:

```powershell
python scripts/train_baselines.py --languages bisindo --features-dir data/features/landmarks-bisindo-kaggle-v2-conf03 --output-dir artifacts/classification-bisindo-kaggle-v2-conf03
```

This comparison must use the matching train and validation caches. Test remains reserved and untouched.


## Confidence 0.3 classifier validation

The confidence-0.3 pipeline was trained separately on its matching train/validation feature caches. Linear SVM C=1 was selected by validation system macro-F1. Its artifact checksum matches `selection.json`, the cache config and manifest hashes match the v2 snapshot, and test remains unopened (`test_accessed: false`).

| Pipeline and selected model | Feature coverage | Valid-only macro-F1 | System macro-F1 | System accuracy |
|---|---:|---:|---:|---:|
| Confidence 0.5, MLP 128x64 | 84.997% | 0.9594 | 0.8770 | 0.8154 |
| Confidence 0.3, Linear SVM C=1 | 87.947% | 0.9620 | 0.8981 | 0.8456 |

On this same validation split, system recall for Y rose from 0.338 to 0.563; A stayed at 0.530. For A, valid-only recall moved from 1.000 to 0.946, so the extra detections did not improve end-to-end recall. The new setting improves aggregate validation results, but still misses the initial 0.90 system macro-F1 target slightly and falls short of the 0.80 recall target for A and Y. This remains a validation result, not signer-independent or webcam performance. Repeated tuning on validation may also make the displayed score optimistic; reserve the untouched test for a final one-time check after the configuration and model are frozen.


## Visual comparison grid

Representative pairs from confidence-0.3 validation are generated by `scripts/render_bisindo_landmark_comparison.py`. Green/pink lines and red points show detected landmarks; the paired image is a `no_hand` example. When available, both examples come from the same pHash group. Images are local and ignored by Git under `reports/bisindo_kaggle_v2/landmark-comparison-conf03/`.

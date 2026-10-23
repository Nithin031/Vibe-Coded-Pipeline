# V4 vs V5 - people-counting benchmark (ESP32-S3, WiCount)

Generated 2026-10-23 14:36 on Kaggle. V4 = GitHub Nithin031/Vibe-Coded-Pipeline branch v4, commit `f81302dad3f213126d13aa40a1c3a7cc767ed269`, run with its own code. V5 = esp32_counter.py.

Harness check: V4 LightGBM on its own protocol and windows = exact 0.843 (published 0.84).

## Winners

| benchmark             | metric     | best                                               |   V4 LightGBM |   V4 gap-CNN |   V5 WiCount-only |   V5 final |
|:----------------------|:-----------|:---------------------------------------------------|--------------:|-------------:|------------------:|-----------:|
| B1_v4_folds           | exact      | V5 WiCount-only                                    |         0.849 |        0.85  |             0.885 |      0.881 |
| B1_v4_folds           | mae        | V5 WiCount-only                                    |         0.151 |        0.15  |             0.115 |      0.119 |
| B1_v4_folds           | f1_4       | V5 final                                           |         0.72  |        0.685 |             0.461 |      0.736 |
| B1_v4_folds           | file_exact | V4 LightGBM, V5 final                              |         0.933 |        0.867 |             0.867 |      0.933 |
| B1_v4_folds           | recall_3   | V4 LightGBM                                        |         0.086 |        0.057 |             0     |      0.083 |
| B2_new_session        | exact      | V5 final                                           |         0.755 |        0.783 |             0.825 |      0.854 |
| B2_new_session        | mae        | V5 final                                           |         0.245 |        0.217 |             0.175 |      0.146 |
| B2_new_session        | f1_4       | V5 final                                           |         0.4   |        0.414 |             0.433 |      0.687 |
| B2_new_session        | file_exact | V5 final                                           |         0.867 |        0.733 |             0.867 |      0.933 |
| B2_new_session        | recall_3   | V4 LightGBM, V4 gap-CNN, V5 WiCount-only, V5 final |         0     |        0     |             0     |      0     |
| B3_noise (mean of 22) | exact      | V4 gap-CNN                                         |         0.685 |        0.8   |             0.701 |      0.593 |
| B3_noise (mean of 22) | mae        | V4 gap-CNN                                         |         0.33  |        0.212 |             0.309 |      0.546 |
| B3_noise (mean of 22) | answered   | V4 LightGBM, V4 gap-CNN                            |         0.898 |        0.898 |             0.876 |      0.876 |
| B3_noise (mean of 22) | exact_all  | V4 gap-CNN                                         |         0.624 |        0.727 |             0.62  |      0.504 |

## B1 (V4 protocol) and B2 (new session, leave one file out)

| protocol       | system          |   windows |   exact |   mae |   within1 |   recall_0 |   recall_1 |   recall_2 |   recall_3 |   empty_false_alarm |   balanced_acc |   f1_4 |   file_exact |
|:---------------|:----------------|----------:|--------:|------:|----------:|-----------:|-----------:|-----------:|-----------:|--------------------:|---------------:|-------:|-------------:|
| B1_v4_folds    | V4 LightGBM     |       661 |   0.849 | 0.151 |         1 |          1 |      0.898 |      0.873 |      0.086 |                   0 |          0.714 |  0.72  |        0.933 |
| B1_v4_folds    | V4 gap-CNN      |       661 |   0.85  | 0.15  |         1 |          1 |      0.972 |      0.793 |      0.057 |                   0 |          0.706 |  0.685 |        0.867 |
| B1_v4_folds    | V5 WiCount-only |       269 |   0.885 | 0.115 |         1 |          0 |      0.962 |      0.965 |      0     |                   1 |          0.482 |  0.461 |        0.867 |
| B1_v4_folds    | V5 final        |       269 |   0.881 | 0.119 |         1 |          1 |      0.865 |      0.974 |      0.083 |                   0 |          0.73  |  0.736 |        0.933 |
| B2_new_session | V4 LightGBM     |       669 |   0.755 | 0.245 |         1 |          0 |      0.823 |      0.87  |      0     |                   1 |          0.423 |  0.4   |        0.867 |
| B2_new_session | V4 gap-CNN      |       669 |   0.783 | 0.217 |         1 |          0 |      0.969 |      0.768 |      0     |                   1 |          0.434 |  0.414 |        0.733 |
| B2_new_session | V5 WiCount-only |       274 |   0.825 | 0.175 |         1 |          0 |      0.902 |      0.93  |      0     |                   1 |          0.458 |  0.433 |        0.867 |
| B2_new_session | V5 final        |       274 |   0.854 | 0.146 |         1 |          1 |      0.872 |      0.939 |      0     |                   0 |          0.703 |  0.687 |        0.933 |

## B3 - noise (mean over 22 conditions)

| system          |   exact |   mae |   f1_4 |   answered |   exact_all |
|:----------------|--------:|------:|-------:|-----------:|------------:|
| V4 LightGBM     |   0.685 | 0.33  |  0.516 |      0.898 |       0.624 |
| V4 gap-CNN      |   0.8   | 0.212 |  0.638 |      0.898 |       0.727 |
| V5 WiCount-only |   0.701 | 0.309 |  0.348 |      0.876 |       0.62  |
| V5 final        |   0.593 | 0.546 |  0.525 |      0.876 |       0.504 |

### Exact count per noise condition

| noise            |   level |   V4 LightGBM |   V4 gap-CNN |   V5 WiCount-only |   V5 final |
|:-----------------|--------:|--------------:|-------------:|------------------:|-----------:|
| agc_jumps        |       1 |         0.829 |        0.843 |             0.885 |      0.881 |
| agc_jumps        |       2 |         0.778 |        0.825 |             0.885 |      0.881 |
| agc_jumps        |       3 |         0.729 |        0.806 |             0.885 |      0.881 |
| crazy            |       3 |         0.341 |        0.602 |             0.375 |      0.625 |
| dead_subcarriers |       1 |         0.778 |        0.791 |             0.799 |      0.699 |
| dead_subcarriers |       2 |         0.719 |        0.849 |             0.491 |      0.487 |
| dead_subcarriers |       3 |         0.632 |        0.838 |             0.506 |      0.569 |
| fan              |       1 |         0.837 |        0.844 |             0.881 |      0.829 |
| fan              |       2 |         0.793 |        0.843 |             0.87  |      0.576 |
| fan              |       3 |         0.734 |        0.841 |             0.71  |      0.48  |
| impulses         |       1 |         0.681 |        0.841 |             0.431 |      0.204 |
| impulses         |       2 |         0.386 |        0.764 |             0.424 |      0.089 |
| impulses         |       3 |         0.33  |        0.672 |             0.424 |      0.327 |
| jitter           |       1 |         0.84  |        0.846 |             0.885 |      0.81  |
| jitter           |       2 |         0.828 |        0.841 |             0.877 |      0.836 |
| jitter           |       3 |         0.831 |        0.85  |             0.876 |      0.85  |
| packet_loss      |       1 |         0.821 |        0.847 |             0.874 |      0.892 |
| packet_loss      |       2 |         0.78  |        0.812 |             0.823 |      0.726 |
| packet_loss      |       3 |         0.756 |        0.769 |             0.8   |      0.8   |
| white_noise      |       1 |         0.778 |        0.844 |             0.881 |      0.037 |
| white_noise      |       2 |         0.508 |        0.835 |             0.424 |      0.428 |
| white_noise      |       3 |         0.366 |        0.589 |             0.424 |      0.149 |

## B4 - device

| item                                          | V4                                                                                                   | V5                                                    |
|:----------------------------------------------|:-----------------------------------------------------------------------------------------------------|:------------------------------------------------------|
| input rate / window / hop                     | 20 Hz / 3 s / 1 s                                                                                    | 50 Hz / 5 s / 2.5 s                                   |
| smoothing (display)                           | median of 5 windows (~5 s)                                                                           | median of 12 windows (30 s)                           |
| first answer after start                      | ~3 s                                                                                                 | 15 s (10 s warm-up + 5 s window)                      |
| install step                                  | none for counting                                                                                    | 20 s empty room                                       |
| counting model size                           | LightGBM 3 seeds x 6 classes: 101066 nodes (~1086 KB as arrays); gap-CNN 48902 params (~191 KB fp32) | 300 trees, 8700 nodes (~77 KB)                        |
| CPU time per window (this Kaggle CPU, Python) | features 4.71 ms + LightGBM 4.36 ms                                                                  | features 2.17 ms + trees 16.79 ms                     |
| on-chip code                                  | none (Python + PyTorch)                                                                              | C header with trees + input prep; cues still Python   |
| bad-signal handling                           | window marked 'unknown' (gap > 0.5 s or > 25 % missing)                                              | window skipped (> 20 % missing); gaps <= 60 ms filled |
| presence model                                | yes, separate (4 presence datasets)                                                                  | no (count 0 = empty)                                  |

## How to read this

- Scored windows: window end >= 15 s and outside the first 30 s of the empty file, for every system.
- Each system uses its own display smoothing (V4 ~5 s, V5 30 s); a longer median is steadier but slower.
- B1 tests counts 0 and 3 on time blocks of the SAME recording the other blocks trained on (easy for those counts).
- B2 holds out whole files: an ESP32-only model has no 0- or 3-person training file when that file is tested.
- 'answered' = share of the clean scored windows that still get an answer under noise (V4 says 'unknown', V5 skips the window). exact / mae are over answered windows only; exact_all counts every unanswered window as wrong.
- V5 final also trains on mat80 + WiMANS (other rooms, other hardware); V5 WiCount-only uses exactly V4's data.
- Not compared: presence on the V4 presence datasets (V5 has no presence model), V4's WiCount -> OTH cross-room test (the OTH files are not available).
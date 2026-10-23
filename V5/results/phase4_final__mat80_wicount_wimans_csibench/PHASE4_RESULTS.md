# Phase 4 results - final ESP32-S3 people counter (all labelled sources)

Generated 2026-10-23 13:08. One link, device preprocessing, 5.0 s windows every 2.5 s, output = median of the last 30 s (WiMANS / CSI-Bench: one clip = one estimate).

## Final model

- Chosen: **all4 / empty_cal**
- Rule: sites with an empty recording; WiCount MAE within 0.1 of the best; then lowest mean relative MAE over datasets; within 0.02 prefer zero_shot, then the pool with more data
- Trained on: wicount, mat80, wimans (99992 windows; CSI-Bench and mat80 PC2a have no empty recording, so the calibrated model cannot use them - identical to pool esp32+mat80+wimans), counts 0-10
- Calibration: Leave the room empty, start the counter, call begin_calibration(); the next 7 windows (20 s after the 10 s warm-up) set the offset.
- Size: 300 trees, 8700 nodes, ~77 KB flash
- Files: `deploy/` at the V5 root (trees.json, counter_model.h, model.txt, model_card.json)

## Selection (sites with an empty recording, smoothed MAE per dataset)

| experiment            | mode      |   mae_mat80 |   mae_wicount |   mae_wimans |   rel_mae_mean |   datasets_scored | esp32_ok   |
|:----------------------|:----------|------------:|--------------:|-------------:|---------------:|------------------:|:-----------|
| all4                  | empty_cal |       2.668 |         0.294 |        1.011 |          0.744 |                 3 | True       |
| all4                  | zero_shot |       1.796 |         0.844 |        1.086 |          0.958 |                 3 | False      |
| esp32+mat80           | empty_cal |       2.761 |         0.306 |        1.449 |          0.868 |                 3 | True       |
| esp32+mat80           | zero_shot |       1.648 |         0.36  |        1.62  |          0.798 |                 3 | True       |
| esp32+mat80+wimans    | empty_cal |       2.668 |         0.294 |        1.011 |          0.744 |                 3 | True       |
| esp32+mat80+wimans    | zero_shot |       1.704 |         0.916 |        1.061 |          0.981 |                 3 | False      |
| esp32+wimans+csibench | empty_cal |       4.061 |         0.289 |        1.006 |          0.917 |                 3 | True       |
| esp32+wimans+csibench | zero_shot |       3.616 |         0.975 |        1.064 |          1.257 |                 3 | False      |
| esp32_only            | empty_cal |       3.919 |         0.34  |        1.221 |          0.979 |                 3 | True       |
| esp32_only            | zero_shot |       3.858 |         0.34  |        1.492 |          1.036 |                 3 | True       |

## Held-out results of the final model (all sites)

| experiment   | mode      | test    |     n |   mae |   rel_mae |   exact |   within1 |   f1_4 |   spearman_rec |
|:-------------|:----------|:--------|------:|------:|----------:|--------:|----------:|-------:|---------------:|
| all4         | empty_cal | mat80   | 49120 | 2.668 |     1.015 |   0.186 |     0.379 |  0.453 |          0.498 |
| all4         | empty_cal | wicount |   274 | 0.294 |     0.491 |   0.854 |     1     |  0.687 |          0.915 |
| all4         | empty_cal | wimans  | 50598 | 1.011 |     0.726 |   0.313 |     0.751 |  0.58  |          0.662 |

## Final model: mean estimate per true count

| dataset   |   count |     n |   mean_pred |   exact |
|:----------|--------:|------:|------------:|--------:|
| mat80     |       0 |  5056 |       0.105 |   0.953 |
| mat80     |       2 |  5168 |       4.084 |   0.06  |
| mat80     |       3 |  4576 |       4.358 |   0.168 |
| mat80     |       4 |  5328 |       4.541 |   0.205 |
| mat80     |       5 |  4672 |       4.698 |   0.085 |
| mat80     |       6 |  4544 |       4.838 |   0.056 |
| mat80     |       7 |  4832 |       4.903 |   0.108 |
| mat80     |       8 |  5456 |       5.019 |   0.056 |
| mat80     |       9 |  4384 |       5.074 |   0.141 |
| mat80     |      10 |  5104 |       5.124 |   0.007 |
| wicount   |       0 |    11 |       0     |   1     |
| wicount   |       1 |   133 |       1.27  |   0.872 |
| wicount   |       2 |   114 |       1.772 |   0.939 |
| wicount   |       3 |    16 |       2.006 |   0     |
| wimans    |       0 |  2484 |       0.097 |   0.933 |
| wimans    |       1 | 16038 |       1.974 |   0.309 |
| wimans    |       2 |  8019 |       2.584 |   0.313 |
| wimans    |       3 |  8019 |       2.849 |   0.47  |
| wimans    |       4 |  8019 |       3.013 |   0.263 |
| wimans    |       5 |  8019 |       3.096 |   0.021 |

## All candidates, all sites

| experiment            | mode      | test     |     n |   mae |   rel_mae |   exact |   within1 |   f1_4 |   spearman_rec |
|:----------------------|:----------|:---------|------:|------:|----------:|--------:|----------:|-------:|---------------:|
| all4                  | empty_cal | mat80    | 49120 | 2.668 |     1.015 |   0.186 |     0.379 |  0.453 |          0.498 |
| all4                  | empty_cal | wicount  |   274 | 0.294 |     0.491 |   0.854 |     1     |  0.687 |          0.915 |
| all4                  | empty_cal | wimans   | 50598 | 1.011 |     0.726 |   0.313 |     0.751 |  0.58  |          0.662 |
| all4                  | zero_shot | countfi  |  6200 | 3.577 |     1.496 |   0.089 |     0.254 |  0.155 |          0.164 |
| all4                  | zero_shot | csibench | 20193 | 0.625 |     1.556 |   0.538 |     0.93  |  0.25  |          0.168 |
| all4                  | zero_shot | mat80    | 63440 | 1.852 |     0.725 |   0.179 |     0.48  |  0.432 |          0.907 |
| all4                  | zero_shot | wicount  |   274 | 0.844 |     1.411 |   0.277 |     0.942 |  0.18  |          0.671 |
| all4                  | zero_shot | wimans   | 50598 | 1.086 |     0.78  |   0.327 |     0.708 |  0.573 |          0.58  |
| esp32+mat80           | empty_cal | mat80    | 49120 | 2.761 |     1.051 |   0.184 |     0.363 |  0.455 |          0.415 |
| esp32+mat80           | empty_cal | wicount  |   274 | 0.306 |     0.511 |   0.807 |     1     |  0.689 |          0.901 |
| esp32+mat80           | empty_cal | wimans   | 50598 | 1.449 |     1.041 |   0.249 |     0.599 |  0.363 |          0.121 |
| esp32+mat80           | zero_shot | countfi  |  6200 | 4.435 |     1.854 |   0.049 |     0.168 |  0.085 |          0.108 |
| esp32+mat80           | zero_shot | csibench | 20193 | 3.488 |     8.689 |   0.137 |     0.274 |  0.106 |          0.133 |
| esp32+mat80           | zero_shot | mat80    | 63440 | 1.75  |     0.685 |   0.206 |     0.504 |  0.506 |          0.914 |
| esp32+mat80           | zero_shot | wicount  |   274 | 0.36  |     0.602 |   0.704 |     0.993 |  0.658 |          0.839 |
| esp32+mat80           | zero_shot | wimans   | 50598 | 1.62  |     1.163 |   0.215 |     0.537 |  0.221 |          0.604 |
| esp32+mat80+wimans    | empty_cal | mat80    | 49120 | 2.668 |     1.015 |   0.186 |     0.379 |  0.453 |          0.498 |
| esp32+mat80+wimans    | empty_cal | wicount  |   274 | 0.294 |     0.491 |   0.854 |     1     |  0.687 |          0.915 |
| esp32+mat80+wimans    | empty_cal | wimans   | 50598 | 1.011 |     0.726 |   0.313 |     0.751 |  0.58  |          0.662 |
| esp32+mat80+wimans    | zero_shot | countfi  |  6200 | 3.658 |     1.53  |   0.09  |     0.244 |  0.134 |          0.262 |
| esp32+mat80+wimans    | zero_shot | csibench | 20193 | 3.567 |     8.888 |   0.087 |     0.218 |  0.07  |          0.126 |
| esp32+mat80+wimans    | zero_shot | mat80    | 63440 | 1.789 |     0.701 |   0.185 |     0.491 |  0.425 |          0.922 |
| esp32+mat80+wimans    | zero_shot | wicount  |   274 | 0.916 |     1.532 |   0.241 |     0.898 |  0.33  |          0.632 |
| esp32+mat80+wimans    | zero_shot | wimans   | 50598 | 1.061 |     0.762 |   0.332 |     0.717 |  0.581 |          0.604 |
| esp32+wimans+csibench | empty_cal | mat80    | 49120 | 4.061 |     1.545 |   0.133 |     0.245 |  0.322 |          0.422 |
| esp32+wimans+csibench | empty_cal | wicount  |   274 | 0.289 |     0.483 |   0.847 |     1     |  0.684 |          0.901 |
| esp32+wimans+csibench | empty_cal | wimans   | 50598 | 1.006 |     0.722 |   0.342 |     0.75  |  0.6   |          0.683 |
| esp32+wimans+csibench | zero_shot | countfi  |  6200 | 3.671 |     1.535 |   0.095 |     0.224 |  0.113 |          0.286 |
| esp32+wimans+csibench | zero_shot | csibench | 20193 | 0.451 |     1.125 |   0.641 |     0.984 |  0.249 |          0.239 |
| esp32+wimans+csibench | zero_shot | mat80    | 63440 | 3.625 |     1.42  |   0.092 |     0.283 |  0.176 |          0.615 |
| esp32+wimans+csibench | zero_shot | wicount  |   274 | 0.975 |     1.629 |   0.157 |     0.872 |  0.082 |          0.539 |
| esp32+wimans+csibench | zero_shot | wimans   | 50598 | 1.064 |     0.764 |   0.331 |     0.717 |  0.581 |          0.598 |
| esp32_only            | empty_cal | mat80    | 49120 | 3.919 |     1.491 |   0.098 |     0.253 |  0.211 |          0.442 |
| esp32_only            | empty_cal | wicount  |   274 | 0.34  |     0.569 |   0.814 |     1     |  0.428 |          0.81  |
| esp32_only            | empty_cal | wimans   | 50598 | 1.221 |     0.877 |   0.256 |     0.679 |  0.359 |          0.488 |
| esp32_only            | zero_shot | countfi  |  6200 | 4.052 |     1.694 |   0.086 |     0.174 |  0.05  |          0.013 |
| esp32_only            | zero_shot | csibench | 20193 | 0.67  |     1.669 |   0.462 |     0.957 |  0.238 |          0.11  |
| esp32_only            | zero_shot | mat80    | 63440 | 3.923 |     1.536 |   0.044 |     0.237 |  0.031 |          0.854 |
| esp32_only            | zero_shot | wicount  |   274 | 0.34  |     0.569 |   0.818 |     1     |  0.43  |          0.81  |
| esp32_only            | zero_shot | wimans   | 50598 | 1.492 |     1.072 |   0.352 |     0.56  |  0.357 |          0.537 |

## Clip check (mat80 + WiCount windows; causal = what the ESP32 computes)

| input           |    n |   mae |   exact |   within1 |   mean_abs_diff_vs_causal |
|:----------------|-----:|------:|--------:|----------:|--------------------------:|
| causal (device) | 1774 | 1.476 |   0.243 |     0.576 |                     0     |
| clip mean, 5 s  | 1774 | 1.469 |   0.252 |     0.578 |                     0.215 |
| clip mean, 3 s  | 1774 | 1.537 |   0.231 |     0.569 |                     0.5   |

## Replay of the raw ESP32 CSVs through the saved bundle (pipeline check)

| file                          |   true |   windows |   last_count |   exact_share |   stream_median |   batch_median |   stream_vs_batch |
|:------------------------------|-------:|----------:|-------------:|--------------:|----------------:|---------------:|------------------:|
| 2023-12-21_19-23-53-985_104_1 |      0 |        11 |            0 |          1    |            0    |           0    |                 0 |
| 2023-12-21_19-27-21-930_104_1 |      1 |        19 |            1 |          1    |            1.24 |           1.24 |                 0 |
| 2023-12-21_19-31-17-706_104_1 |      1 |        19 |            1 |          1    |            1.1  |           1.1  |                 0 |
| 2023-12-21_19-34-42-783_104_1 |      1 |        19 |            1 |          1    |            1.06 |           1.06 |                 0 |
| 2023-12-21_19-37-36-861_104_1 |      1 |        19 |            1 |          1    |            1.1  |           1.1  |                 0 |
| 2023-12-21_19-40-12-110_104_1 |      1 |        19 |            1 |          1    |            1.07 |           1.07 |                 0 |
| 2023-12-21_19-42-28-267_104_1 |      1 |        19 |            1 |          1    |            1.24 |           1.24 |                 0 |
| 2023-12-21_19-45-07-501_104_1 |      1 |        19 |            1 |          1    |            1.15 |           1.15 |                 0 |
| 2023-12-21_19-47-14-669_104_1 |      2 |        19 |            2 |          1    |            1.82 |           1.82 |                 0 |
| 2023-12-21_19-49-37-940_104_1 |      2 |        19 |            2 |          1    |            1.83 |           1.83 |                 0 |
| 2023-12-21_19-54-01-269_104_1 |      2 |        19 |            2 |          1    |            1.99 |           1.99 |                 0 |
| 2023-12-21_19-56-22-903_104_1 |      2 |        19 |            2 |          1    |            1.96 |           1.96 |                 0 |
| 2023-12-21_20-02-00-329_104_1 |      2 |        19 |            2 |          1    |            1.83 |           1.83 |                 0 |
| 2023-12-21_20-04-39-406_104_1 |      2 |        19 |            2 |          1    |            1.8  |           1.8  |                 0 |
| 2023-12-21_20-07-25-709_104_1 |      3 |        16 |            3 |          0.75 |            2.53 |           2.53 |                 0 |

## ESP32-S3 budget (estimates)

| item                         | value       | limit                    | ok   |
|:-----------------------------|:------------|:-------------------------|:-----|
| flash: model arrays          | 77 KB       | 512 KB                   | True |
| RAM: counter state           | 75 KB       | 128 KB                   | True |
| compute per 2.5 s window     | 3.0 M flops | 25 M (~10 % of one core) | True |
| packet rate (WiCount median) | 87 Hz       | >= 50 Hz                 | True |

## Caveats

- WiCount is the only ESP32-S3 data: one room, one evening, counts 0-3 (0 and 3 one file each); it was also used in the selection rule.
- WiMANS / CSI-Bench rows use clip-mean normalisation (3 s / 5 s clips), not the device's 30 s causal baseline - see clip_check for the size of that difference.
- Packet times of WiMANS (1 kHz) and CSI-Bench (100 Hz) are assumed uniform (no timestamps).
- Counts above 3 come from router / Intel 5300 data only.
- Re-validate on your own ESP32-S3 recordings before trusting the count.
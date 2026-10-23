# Phase 3 results - final ESP32-S3 people counter

Generated 2026-10-23 01:00. One link (one view), causal preprocessing, 5.0 s windows every 2.5 s, output = median of the last 30 s.

## Final model

- Chosen: **E4_mat80_plus_esp32 / empty_cal** (rule: lowest smoothed MAE on WiCount held-out files; within 0.05 prefer the earlier candidate (more data, no install step))
- Trained on: mat80, wicount (49394 windows), counts 0-10
- Calibration: Leave the room empty, start the counter, call begin_calibration(); the next 7 windows (20 s after the 10 s warm-up) set the offset.
- Size: 300 trees, 8700 nodes, ~77 KB flash
- Files: `D:/wifi/models/esp32_counter/` (trees.json, counter_model.h, model.txt, model_card.json)

## Held-out results (smoothed output)

| experiment          | mode      | test    | sites      | train                       |     n |   mae |   exact |   within1 |   spearman_rec |
|:--------------------|:----------|:--------|:-----------|:----------------------------|------:|------:|--------:|----------:|---------------:|
| E1_mat80_loso       | empty_cal | mat80   | all        | mat80 other setups          | 49120 | 2.853 |   0.186 |     0.353 |          0.438 |
| E1_mat80_loso       | zero_shot | mat80   | all        | mat80 other setups          | 63440 | 1.74  |   0.215 |     0.502 |          0.915 |
| E1_mat80_loso       | zero_shot | mat80   | with_empty | mat80 other setups          | 49120 | 1.639 |   0.238 |     0.529 |          0.915 |
| E2_esp32_only       | zero_shot | wicount | all        | wicount other files         |   274 | 0.34  |   0.818 |     1     |          0.81  |
| E3_mat80_to_esp32   | empty_cal | wicount | all        | mat80 all                   |   274 | 1.068 |   0.124 |     0.803 |          0.864 |
| E3_mat80_to_esp32   | zero_shot | wicount | all        | mat80 all                   |   274 | 0.771 |   0.365 |     0.891 |          0.341 |
| E4_mat80_plus_esp32 | empty_cal | wicount | all        | mat80 + wicount other files |   274 | 0.306 |   0.807 |     1     |          0.901 |
| E4_mat80_plus_esp32 | zero_shot | wicount | all        | mat80 + wicount other files |   274 | 0.36  |   0.704 |     0.993 |          0.839 |
| E5_mat80_to_countfi | zero_shot | countfi | all        | mat80 all                   |  6200 | 4.648 |   0.035 |     0.15  |         -0.025 |

## Candidates on the ESP32-S3 data (WiCount)

| experiment          | mode      |   n |   mae |   exact |   within1 |
|:--------------------|:----------|----:|------:|--------:|----------:|
| E4_mat80_plus_esp32 | zero_shot | 274 | 0.36  |   0.704 |     0.993 |
| E4_mat80_plus_esp32 | empty_cal | 274 | 0.306 |   0.807 |     1     |
| E3_mat80_to_esp32   | zero_shot | 274 | 0.771 |   0.365 |     0.891 |
| E3_mat80_to_esp32   | empty_cal | 274 | 1.068 |   0.124 |     0.803 |
| E2_esp32_only       | zero_shot | 274 | 0.34  |   0.818 |     1     |

## Replay of the raw ESP32 CSVs through the saved bundle (pipeline check)

| file                          |   true |   windows |   last_count |   exact_share |   stream_median |   batch_median |   stream_vs_batch |
|:------------------------------|-------:|----------:|-------------:|--------------:|----------------:|---------------:|------------------:|
| 2023-12-21_19-23-53-985_104_1 |      0 |        11 |            0 |          1    |            0    |           0    |              0    |
| 2023-12-21_19-27-21-930_104_1 |      1 |        19 |            1 |          1    |            1.24 |           1.24 |              0    |
| 2023-12-21_19-31-17-706_104_1 |      1 |        19 |            1 |          1    |            1.09 |           1.09 |              0    |
| 2023-12-21_19-34-42-783_104_1 |      1 |        19 |            1 |          1    |            1.06 |           1.06 |              0    |
| 2023-12-21_19-37-36-861_104_1 |      1 |        19 |            1 |          1    |            1.03 |           1.03 |              0    |
| 2023-12-21_19-40-12-110_104_1 |      1 |        19 |            1 |          1    |            1.07 |           1.07 |              0    |
| 2023-12-21_19-42-28-267_104_1 |      1 |        19 |            1 |          1    |            1.26 |           1.26 |              0    |
| 2023-12-21_19-45-07-501_104_1 |      1 |        19 |            1 |          1    |            1.17 |           1.17 |              0    |
| 2023-12-21_19-47-14-669_104_1 |      2 |        19 |            2 |          1    |            1.85 |           1.88 |              0.03 |
| 2023-12-21_19-49-37-940_104_1 |      2 |        19 |            2 |          1    |            1.92 |           1.92 |              0    |
| 2023-12-21_19-54-01-269_104_1 |      2 |        19 |            2 |          1    |            2    |           2    |              0    |
| 2023-12-21_19-56-22-903_104_1 |      2 |        19 |            2 |          1    |            1.95 |           1.95 |              0    |
| 2023-12-21_20-02-00-329_104_1 |      2 |        19 |            2 |          0.95 |            1.82 |           1.82 |              0    |
| 2023-12-21_20-04-39-406_104_1 |      2 |        19 |            2 |          1    |            1.84 |           1.84 |              0    |
| 2023-12-21_20-07-25-709_104_1 |      3 |        16 |            3 |          1    |            2.61 |           2.61 |              0    |

## ESP32-S3 budget (estimates)

| item                         | value       | limit                    | ok   |
|:-----------------------------|:------------|:-------------------------|:-----|
| flash: model arrays          | 77 KB       | 512 KB                   | True |
| RAM: counter state           | 75 KB       | 128 KB                   | True |
| compute per 2.5 s window     | 3.0 M flops | 25 M (~10 % of one core) | True |
| packet rate (WiCount median) | 87 Hz       | >= 50 Hz                 | True |

## Caveats

- The WiCount score of the chosen model is optimistic: the same files were used to choose it.
- WiCount = one room, one evening, counts 0-3; counts 0 and 3 have one file each.
- No ESP32 data above 3 people; higher counts come from 4-antenna 80 MHz router data only.
- 80 MHz setup PC2a has no empty recording; PC4a_n00 looked occupied in the audit.
- Re-validate on your own ESP32-S3 recordings before trusting the count.
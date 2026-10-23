# Phase 2 results - honest held-out baselines

Generated 2026-10-23 00:20 from shared_format (5.0 s windows, 50 Hz, 52 tones).

## Best model per dataset (view ensemble)

| dataset   | model   | task   | calib   |   mae |   exact |   within1 |    n |
|:----------|:--------|:-------|:--------|------:|--------:|----------:|-----:|
| wicount   | gbm     | clf    | none    | 0.32  |   0.68  |     1     |  341 |
| mat80     | gbm     | clf    | group_z | 1.111 |   0.31  |     0.695 | 4142 |
| countfi   | cnn     | reg    | none    | 2.193 |   0.141 |     0.426 | 1843 |

## All models (view ensemble)

| dataset   | model         | task   | calib   |   mae |   exact |   within1 |
|:----------|:--------------|:-------|:--------|------:|--------:|----------:|
| mat80     | gbm           | clf    | none    | 1.409 |   0.247 |     0.602 |
| mat80     | gbm           | reg    | none    | 1.548 |   0.199 |     0.536 |
| mat80     | mean_baseline | reg    | none    | 2.571 |   0.095 |     0.294 |
| mat80     | gbm           | clf    | group_z | 1.111 |   0.31  |     0.695 |
| mat80     | gbm           | reg    | group_z | 1.24  |   0.267 |     0.642 |
| countfi   | gbm           | clf    | none    | 2.351 |   0.141 |     0.373 |
| countfi   | gbm           | reg    | none    | 2.234 |   0.115 |     0.361 |
| countfi   | mean_baseline | reg    | none    | 2.415 |   0.118 |     0.4   |
| countfi   | gbm           | clf    | group_z | 2.492 |   0.097 |     0.319 |
| countfi   | gbm           | reg    | group_z | 2.302 |   0.12  |     0.339 |
| wicount   | gbm           | clf    | none    | 0.32  |   0.68  |     1     |
| wicount   | gbm           | reg    | none    | 0.393 |   0.71  |     1     |
| wicount   | mean_baseline | reg    | none    | 0.664 |   0.472 |     0.877 |
| mat80     | cnn           | clf    | none    | 1.239 |   0.261 |     0.684 |
| mat80     | cnn           | reg    | none    | 1.259 |   0.235 |     0.66  |
| countfi   | cnn           | clf    | none    | 2.457 |   0.137 |     0.339 |
| countfi   | cnn           | reg    | none    | 2.193 |   0.141 |     0.426 |
| wicount   | cnn           | clf    | none    | 0.545 |   0.481 |     0.974 |
| wicount   | cnn           | reg    | none    | 1.037 |   0.264 |     0.718 |

## Single view (what one ESP32 link sees)

| dataset   | model         | task   | calib   |   mae |   within1 |
|:----------|:--------------|:-------|:--------|------:|----------:|
| mat80     | gbm           | clf    | none    | 2.289 |     0.437 |
| mat80     | gbm           | reg    | none    | 1.849 |     0.485 |
| mat80     | mean_baseline | reg    | none    | 2.571 |     0.294 |
| mat80     | gbm           | clf    | group_z | 1.543 |     0.567 |
| mat80     | gbm           | reg    | group_z | 1.415 |     0.604 |
| countfi   | gbm           | clf    | none    | 2.927 |     0.335 |
| countfi   | gbm           | reg    | none    | 2.331 |     0.363 |
| countfi   | mean_baseline | reg    | none    | 2.415 |     0.4   |
| countfi   | gbm           | clf    | group_z | 3.059 |     0.3   |
| countfi   | gbm           | reg    | group_z | 2.401 |     0.346 |
| wicount   | gbm           | clf    | none    | 0.32  |     1     |
| wicount   | gbm           | reg    | none    | 0.393 |     1     |
| wicount   | mean_baseline | reg    | none    | 0.664 |     0.877 |
| mat80     | cnn           | clf    | none    | 1.912 |     0.487 |
| mat80     | cnn           | reg    | none    | 1.83  |     0.522 |
| countfi   | cnn           | clf    | none    | 3.241 |     0.306 |
| countfi   | cnn           | reg    | none    | 2.377 |     0.399 |
| wicount   | cnn           | clf    | none    | 0.545 |     0.974 |
| wicount   | cnn           | reg    | none    | 1.037 |     0.718 |

## Caveats

- mat80 has no 1-person recordings; PC4a_n00 looked occupied in the deep audit.
- countfi: each count is one time block per scenario, so results measure scenario transfer only.
- wicount: counts 0 and 3 have one file each; a left-out file of those counts was never seen in training.
- group_z calibration uses all unlabelled windows of the test group (slightly optimistic).
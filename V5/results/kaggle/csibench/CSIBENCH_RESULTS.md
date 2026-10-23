# CSI-Bench v3 - Localization people counting (0-2) + cross-dataset + ESP32 check

Generated 2026-10-22 19:30; 5-s windows, 50 Hz, 52 tones, 3 device links per sample.

## Localization, unseen room (best row per model, all 3 links averaged)

| setting   | model           | task   | calib   |   mae |   exact |   within1 |   f1_4 |    n |
|:----------|:----------------|:-------|:--------|------:|--------:|----------:|-------:|-----:|
| zero_shot | cnn_multi       | clf    | none    | 0.268 |   0.732 |         1 |  0.677 | 6731 |
| zero_shot | cnn_multi_int8  | clf    | none    | 0.268 |   0.732 |         1 |  0.676 | 6731 |
| zero_shot | gbm_base10      | clf    | group_z | 0.305 |   0.722 |         1 |  0.459 | 6731 |
| zero_shot | gbm_single_link | clf    | group_z | 0.306 |   0.726 |         1 |  0.482 | 6731 |
| zero_shot | gbm_multi_link  | clf    | group_z | 0.313 |   0.716 |         1 |  0.523 | 6731 |
| zero_shot | cnn_single_int8 | reg    | none    | 0.388 |   0.673 |         1 |  0.483 | 6731 |
| zero_shot | cnn_single      | reg    | none    | 0.388 |   0.673 |         1 |  0.485 | 6731 |
| zero_shot | mean_baseline   | reg    | none    | 0.403 |   0.722 |         1 |  0.419 | 6731 |

## One device link (single-link models, scored per link)

| setting   | model           | task   | calib   |   mae |   exact |   within1 |   f1_4 |     n |
|:----------|:----------------|:-------|:--------|------:|--------:|----------:|-------:|------:|
| zero_shot | gbm_base10      | clf    | group_z | 0.305 |   0.695 |         1 |  0.497 | 20193 |
| zero_shot | gbm_single_link | clf    | group_z | 0.306 |   0.694 |         1 |  0.522 | 20193 |
| zero_shot | cnn_single_int8 | clf    | none    | 0.389 |   0.611 |         1 |  0.497 | 20193 |
| zero_shot | cnn_single      | clf    | none    | 0.389 |   0.611 |         1 |  0.497 | 20193 |

## Cross-dataset (train on one dataset, test on the other; 0-2 people)

| setting          | model         | task   | calib   |   mae |   exact |   within1 |   f1_4 |    n |
|:-----------------|:--------------|:-------|:--------|------:|--------:|----------:|-------:|-----:|
| csibench->wimans | gbm_base10    | clf    | none    | 0.395 |   0.625 |     1     |  0.351 | 2970 |
| csibench->wimans | gbm_base10    | reg    | none    | 0.425 |   0.618 |     1     |  0.329 | 2970 |
| csibench->wimans | gbm_base10    | clf    | room_z  | 0.429 |   0.607 |     0.993 |  0.306 | 2970 |
| csibench->wimans | gbm_base10    | reg    | room_z  | 0.496 |   0.608 |     1     |  0.286 | 2970 |
| csibench->wimans | mean_baseline | reg    | none    | 0.511 |   0.6   |     1     |  0.25  | 2970 |
| wimans->csibench | mean_baseline | reg    | none    | 0.367 |   0.722 |     1     |  0.419 | 6731 |
| wimans->csibench | gbm_base10    | reg    | room_z  | 0.367 |   0.726 |     1     |  0.379 | 6731 |
| wimans->csibench | gbm_base10    | clf    | room_z  | 0.383 |   0.689 |     1     |  0.4   | 6731 |
| wimans->csibench | gbm_base10    | reg    | none    | 0.414 |   0.728 |     1     |  0.684 | 6731 |
| wimans->csibench | gbm_base10    | clf    | none    | 0.424 |   0.633 |     1     |  0.613 | 6731 |

## ESP32 reality check (median base cues per source)

| source                    |   cv_median |   diff_energy |   tone_corr |   n_pc_90 |
|:--------------------------|------------:|--------------:|------------:|----------:|
| CSI-Bench loc: 1 people   |      0.1844 |        0.1645 |      0.033  |        10 |
| CSI-Bench loc: 2 people   |      0.2008 |        0.1674 |      0.026  |        10 |
| ESP32 fall clips: Fall    |      0.0711 |        0.063  |      0.1276 |         8 |
| ESP32 fall clips: Nonfall |      0.0493 |        0.0419 |      0.125  |        15 |
| WiMANS 5 GHz: 0 people    |      0.0094 |        0.0122 |      0.1547 |         6 |
| WiMANS 5 GHz: 1 people    |      0.0802 |        0.0368 |      0.0624 |         5 |
| WiMANS 5 GHz: 2 people    |      0.1332 |        0.054  |      0.0438 |         5 |

## Caveats

- Only the 168-subcarrier Localization recordings (3 devices x 56) are used; the 2496-subcarrier ones are left out.
- Packet times are assumed uniform at 100 Hz (the h5 files have no timestamps).
- f1_4 here covers only the classes present (0-2).
- Cross-dataset: WiMANS windows are 3 s, CSI-Bench 5 s; room_z z-scores each room with its own unlabelled data.
- int8 rows quantise weights only; activations stay float.
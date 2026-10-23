# WiMANS - people counting v3 (5 GHz, unseen room)

Generated 2026-10-22 18:38; 3-s windows, 50 Hz, 52 tones, 9 links per sample.

## Best per model and setting (all 9 links averaged; best task + calibration by 0/1/2/3+ F1)

| setting   | model           | task   | calib   |   mae |   exact |   within1 |   acc4 |   f1_4 |    n |
|:----------|:----------------|:-------|:--------|------:|--------:|----------:|-------:|-------:|-----:|
| cal_all   | gbm_single_link | clf4   | group_z | 0.368 |   0.71  |     0.961 |  0.71  |  0.72  | 4158 |
| cal_all   | gbm_base10      | clf4   | group_z | 0.396 |   0.685 |     0.952 |  0.685 |  0.698 | 4158 |
| cal_all   | gbm_multi_link  | reg    | group_z | 0.808 |   0.411 |     0.839 |  0.662 |  0.669 | 4158 |
| cal_all   | cnn_single_int8 | clf    | none    | 0.835 |   0.399 |     0.816 |  0.64  |  0.642 | 4158 |
| cal_all   | cnn_single      | clf    | none    | 0.834 |   0.397 |     0.816 |  0.64  |  0.641 | 4158 |
| cal_all   | cnn_multi_int8  | clf    | none    | 0.95  |   0.431 |     0.763 |  0.609 |  0.63  | 4158 |
| cal_all   | cnn_multi       | clf    | none    | 0.95  |   0.431 |     0.763 |  0.609 |  0.63  | 4158 |
| cal_all   | mean_baseline   | reg    | none    | 1.433 |   0.143 |     0.429 |  0.429 |  0.15  | 4158 |
| cal_empty | gbm_single_link | clf4   | group_z | 0.368 |   0.706 |     0.963 |  0.706 |  0.721 | 4158 |
| cal_empty | gbm_base10      | clf4   | group_z | 0.398 |   0.679 |     0.952 |  0.679 |  0.694 | 4158 |
| cal_empty | gbm_multi_link  | reg    | group_z | 0.814 |   0.414 |     0.835 |  0.663 |  0.671 | 4158 |
| cal_empty | cnn_single      | clf    | none    | 0.875 |   0.393 |     0.789 |  0.627 |  0.639 | 4158 |
| cal_empty | cnn_single_int8 | clf    | none    | 0.874 |   0.392 |     0.788 |  0.626 |  0.637 | 4158 |
| cal_empty | cnn_multi_int8  | clf    | none    | 1.037 |   0.408 |     0.717 |  0.593 |  0.622 | 4158 |
| cal_empty | cnn_multi       | clf    | none    | 1.039 |   0.408 |     0.715 |  0.592 |  0.621 | 4158 |
| cal_empty | mean_baseline   | reg    | none    | 1.432 |   0.143 |     0.429 |  0.429 |  0.15  | 4158 |
| zero_shot | gbm_single_link | clf4   | group_z | 0.368 |   0.706 |     0.963 |  0.706 |  0.721 | 4158 |
| zero_shot | gbm_base10      | clf4   | group_z | 0.398 |   0.679 |     0.952 |  0.679 |  0.694 | 4158 |
| zero_shot | gbm_multi_link  | reg    | group_z | 0.814 |   0.414 |     0.835 |  0.663 |  0.671 | 4158 |
| zero_shot | cnn_single_int8 | clf    | none    | 0.878 |   0.392 |     0.789 |  0.628 |  0.639 | 4158 |
| zero_shot | cnn_single      | clf    | none    | 0.878 |   0.388 |     0.789 |  0.627 |  0.637 | 4158 |
| zero_shot | cnn_multi       | clf    | none    | 1.047 |   0.41  |     0.709 |  0.596 |  0.627 | 4158 |
| zero_shot | cnn_multi_int8  | clf    | none    | 1.046 |   0.411 |     0.711 |  0.596 |  0.626 | 4158 |
| zero_shot | mean_baseline   | reg    | none    | 1.432 |   0.143 |     0.429 |  0.429 |  0.15  | 4158 |

## One ESP32 link (single-link models, scored per link)

| setting   | model           | task   | calib   |   mae |   exact |   within1 |   acc4 |   f1_4 |     n |
|:----------|:----------------|:-------|:--------|------:|--------:|----------:|-------:|-------:|------:|
| cal_all   | gbm_single_link | reg    | group_z | 0.852 |   0.416 |     0.81  |  0.646 |  0.655 | 37422 |
| cal_all   | gbm_base10      | clf    | group_z | 0.93  |   0.463 |     0.745 |  0.65  |  0.627 | 37422 |
| cal_all   | cnn_single      | clf    | none    | 1     |   0.41  |     0.728 |  0.598 |  0.618 | 37422 |
| cal_all   | cnn_single_int8 | clf    | none    | 1.001 |   0.409 |     0.728 |  0.597 |  0.617 | 37422 |
| cal_empty | gbm_single_link | reg    | group_z | 0.858 |   0.414 |     0.805 |  0.643 |  0.654 | 37422 |
| cal_empty | gbm_base10      | clf    | group_z | 0.941 |   0.466 |     0.738 |  0.649 |  0.628 | 37422 |
| cal_empty | cnn_single_int8 | clf    | none    | 1.041 |   0.407 |     0.705 |  0.587 |  0.612 | 37422 |
| cal_empty | cnn_single      | clf    | none    | 1.042 |   0.406 |     0.705 |  0.586 |  0.611 | 37422 |
| zero_shot | gbm_single_link | reg    | group_z | 0.858 |   0.414 |     0.805 |  0.643 |  0.654 | 37422 |
| zero_shot | gbm_base10      | clf    | group_z | 0.941 |   0.466 |     0.738 |  0.649 |  0.628 | 37422 |
| zero_shot | cnn_single      | clf    | none    | 1.047 |   0.406 |     0.701 |  0.588 |  0.612 | 37422 |
| zero_shot | cnn_single_int8 | clf    | none    | 1.049 |   0.405 |     0.702 |  0.586 |  0.611 | 37422 |

## Deployment (final models trained on all 5 GHz samples)

|        | input        |   window_s |   params |   macs_per_window |   weights_fp32_kb |   weights_int8_kb | onnx   |
|:-------|:-------------|-----------:|---------:|------------------:|------------------:|------------------:|:-------|
| single | [52, 150]    |          3 |    35575 |           2891968 |               139 |              34.7 | ok     |
| multi  | [9, 52, 150] |          3 |    35575 |          26024128 |               139 |              34.7 | ok     |

## LightGBM feature importance (share of gain, zero-shot)

|                  |   base10 |   multi_link |   single_link |
|:-----------------|---------:|-------------:|--------------:|
| cv_median        |    0.061 |        0.046 |         0.062 |
| cv_p90           |    0.071 |        0.03  |         0.033 |
| cv_p10           |    0.543 |        0.403 |         0.435 |
| diff_energy      |    0.178 |        0.165 |         0.171 |
| pc1_frac         |    0.011 |        0.002 |         0.004 |
| n_pc_90          |    0.054 |        0.033 |         0.04  |
| tone_corr        |    0.024 |        0.012 |         0.015 |
| low_freq_frac    |    0.02  |        0.004 |         0.006 |
| high_freq_frac   |    0.016 |        0.001 |         0.003 |
| mean_std         |    0.021 |        0.014 |         0.015 |
| pc2_frac         |  nan     |        0.002 |         0.004 |
| pc3_frac         |  nan     |        0.003 |         0.004 |
| eig_entropy      |  nan     |        0.006 |         0.009 |
| doppler_centroid |  nan     |        0.03  |         0.033 |
| doppler_spread   |  nan     |        0.058 |         0.078 |
| psd_entropy      |  nan     |        0.031 |         0.034 |
| activity_cv      |  nan     |        0.011 |         0.028 |
| tone_energy_cv   |  nan     |        0.012 |         0.014 |
| half_band_corr   |  nan     |        0.007 |         0.014 |
| link_env_corr    |  nan     |        0.077 |       nan     |
| link_energy_cv   |  nan     |        0.053 |       nan     |

## Caveats

- Intel 5300 30 grouped subcarriers were interpolated to 52 tones; no ESP32 hardware involved.
- group_z calibration uses all unlabelled samples of the test room (slightly optimistic).
- cal_* settings add the first 30 s of one session per count (cal_empty: count 0 only); those sessions are never tested.
- int8 rows quantise weights only; activations stay float. BatchNorm would be folded into the convs on device.
- multi-link models need all 9 links of a sample at once (several receivers).
- Packet times are assumed uniform at 1 kHz (amp files have no timestamps).
# Phase 5 results - which upgrades make the ESP32 counter better

Generated 2026-10-23 14:23 on Kaggle. Leave one site out on all 5 datasets; stream datasets scored after the 30 s median, clips one estimate each.

Rule: an upgrade **helps** when it lowers the MAE by >= 0.05 people on at least 2 datasets and does not make WiCount (ESP32) worse by more than 0.05.

## Verdicts

| upgrade                                | mode      | helps   | better_on                | worse_on               |   wicount_d_mae |
|:---------------------------------------|:----------|:--------|:-------------------------|:-----------------------|----------------:|
| 2 links vs 1 link (base_reg)           | empty_cal | False   |                          |                        |         nan     |
| 2 links vs 1 link (base_reg)           | zero_shot | True    | csibench, mat80          |                        |         nan     |
| 3 links vs 1 link (base_reg)           | empty_cal | False   | mat80                    |                        |         nan     |
| 3 links vs 1 link (base_reg)           | zero_shot | True    | csibench, mat80          |                        |         nan     |
| base_clf4 vs base_reg (0/1/2/3+ scale) | empty_cal | False   | wicount                  | mat80                  |          -0.129 |
| base_clf4 vs base_reg (0/1/2/3+ scale) | zero_shot | True    | csibench, wicount        | countfi                |          -0.053 |
| extra_reg vs base_reg                  | empty_cal | False   | mat80                    |                        |          -0.029 |
| extra_reg vs base_reg                  | zero_shot | True    | csibench, mat80, wicount | countfi                |          -0.103 |
| mem_reg vs base_reg_streams            | empty_cal | False   | mat80                    | wicount                |           0.073 |
| mem_reg vs base_reg_streams            | zero_shot | False   |                          | countfi                |           0.026 |
| two-point calibration vs none          | empty_cal | False   |                          | mat80, wicount, wimans |           0.076 |
| two-point calibration vs none          | zero_shot | False   | wicount                  | mat80, wimans          |          -0.416 |

## Baseline = Phase 4 model (base_reg), one link

| variant   | mode      | dataset   | calibration       |   links |     n |   mae |   rel_mae |   exact |   within1 |   acc4 |   f1_4 |   empty_recall |
|:----------|:----------|:----------|:------------------|--------:|------:|------:|----------:|--------:|----------:|-------:|-------:|---------------:|
| base_reg  | empty_cal | mat80     | none / empty only |       1 | 48784 | 2.643 |     1.008 |   0.187 |     0.376 |  0.758 |  0.459 |          0.944 |
| base_reg  | empty_cal | mat80     | two-point         |       1 | 46752 | 3.237 |     1.234 |   0.206 |     0.336 |  0.448 |  0.432 |          0.987 |
| base_reg  | empty_cal | wicount   | none / empty only |       1 |   267 | 0.307 |     0.512 |   0.85  |     1     |  0.85  |  0.685 |          1     |
| base_reg  | empty_cal | wicount   | two-point         |       1 |   267 | 0.383 |     0.639 |   0.734 |     1     |  0.734 |  0.732 |          1     |
| base_reg  | empty_cal | wimans    | none / empty only |       1 | 50409 | 1.011 |     0.724 |   0.315 |     0.75  |  0.556 |  0.582 |          0.93  |
| base_reg  | empty_cal | wimans    | two-point         |       1 | 50409 | 1.094 |     0.784 |   0.348 |     0.701 |  0.461 |  0.57  |          0.952 |
| base_reg  | zero_shot | countfi   | none / empty only |       1 |  6172 | 3.57  |     1.493 |   0.092 |     0.249 |  0.255 |  0.158 |        nan     |
| base_reg  | zero_shot | csibench  | none / empty only |       1 | 20181 | 0.625 |     1.557 |   0.535 |     0.932 |  0.535 |  0.246 |        nan     |
| base_reg  | zero_shot | mat80     | none / empty only |       1 | 62992 | 1.848 |     0.726 |   0.182 |     0.48  |  0.836 |  0.437 |          0.378 |
| base_reg  | zero_shot | mat80     | two-point         |       1 | 45736 | 2.616 |     0.997 |   0.216 |     0.403 |  0.688 |  0.517 |          0.976 |
| base_reg  | zero_shot | wicount   | none / empty only |       1 |   267 | 0.84  |     1.402 |   0.296 |     0.933 |  0.303 |  0.194 |          0     |
| base_reg  | zero_shot | wicount   | two-point         |       1 |   267 | 0.424 |     0.707 |   0.64  |     1     |  0.64  |  0.579 |          1     |
| base_reg  | zero_shot | wimans    | none / empty only |       1 | 50409 | 1.086 |     0.778 |   0.326 |     0.706 |  0.507 |  0.574 |          0.934 |
| base_reg  | zero_shot | wimans    | two-point         |       1 | 50409 | 1.222 |     0.875 |   0.336 |     0.664 |  0.449 |  0.537 |          0.99  |

## Every upgrade against its baseline (same windows)

| upgrade                                | mode      | dataset   |     n |   mae_base |   mae_new |   d_mae |   f1_4_base |   f1_4_new |   d_f1_4 | calibration       |
|:---------------------------------------|:----------|:----------|------:|-----------:|----------:|--------:|------------:|-----------:|---------:|:------------------|
| extra_reg vs base_reg                  | empty_cal | mat80     | 48784 |      2.643 |     2.395 |  -0.248 |       0.459 |      0.461 |    0.003 | nan               |
| extra_reg vs base_reg                  | empty_cal | wicount   |   267 |      0.307 |     0.277 |  -0.029 |       0.685 |      0.693 |    0.009 | nan               |
| extra_reg vs base_reg                  | empty_cal | wimans    | 50409 |      1.011 |     1.001 |  -0.01  |       0.582 |      0.577 |   -0.004 | nan               |
| extra_reg vs base_reg                  | zero_shot | countfi   |  6172 |      3.57  |     3.758 |   0.188 |       0.158 |      0.121 |   -0.037 | nan               |
| extra_reg vs base_reg                  | zero_shot | csibench  | 20181 |      0.625 |     0.496 |  -0.129 |       0.246 |      0.251 |    0.005 | nan               |
| extra_reg vs base_reg                  | zero_shot | mat80     | 62992 |      1.848 |     1.711 |  -0.137 |       0.437 |      0.406 |   -0.032 | nan               |
| extra_reg vs base_reg                  | zero_shot | wicount   |   267 |      0.84  |     0.737 |  -0.103 |       0.194 |      0.158 |   -0.037 | nan               |
| extra_reg vs base_reg                  | zero_shot | wimans    | 50409 |      1.086 |     1.037 |  -0.049 |       0.574 |      0.594 |    0.02  | nan               |
| base_clf4 vs base_reg (0/1/2/3+ scale) | empty_cal | mat80     | 48784 |      0.359 |     0.444 |   0.084 |       0.459 |      0.479 |    0.02  | nan               |
| base_clf4 vs base_reg (0/1/2/3+ scale) | empty_cal | wicount   |   267 |      0.307 |     0.178 |  -0.129 |       0.685 |      0.674 |   -0.011 | nan               |
| base_clf4 vs base_reg (0/1/2/3+ scale) | empty_cal | wimans    | 50409 |      0.557 |     0.55  |  -0.007 |       0.582 |      0.59  |    0.009 | nan               |
| base_clf4 vs base_reg (0/1/2/3+ scale) | zero_shot | countfi   |  6172 |      1.058 |     1.406 |   0.347 |       0.158 |      0.146 |   -0.012 | nan               |
| base_clf4 vs base_reg (0/1/2/3+ scale) | zero_shot | csibench  | 20181 |      0.565 |     0.396 |  -0.169 |       0.246 |      0.238 |   -0.008 | nan               |
| base_clf4 vs base_reg (0/1/2/3+ scale) | zero_shot | mat80     | 62992 |      0.203 |     0.202 |  -0.001 |       0.437 |      0.43  |   -0.007 | nan               |
| base_clf4 vs base_reg (0/1/2/3+ scale) | zero_shot | wicount   |   267 |      0.807 |     0.755 |  -0.053 |       0.194 |      0.256 |    0.062 | nan               |
| base_clf4 vs base_reg (0/1/2/3+ scale) | zero_shot | wimans    | 50409 |      0.612 |     0.613 |   0.001 |       0.574 |      0.562 |   -0.012 | nan               |
| mem_reg vs base_reg_streams            | empty_cal | mat80     | 48784 |      2.755 |     2.253 |  -0.502 |       0.451 |      0.492 |    0.041 | nan               |
| mem_reg vs base_reg_streams            | empty_cal | wicount   |   267 |      0.303 |     0.375 |   0.073 |       0.71  |      0.636 |   -0.074 | nan               |
| mem_reg vs base_reg_streams            | zero_shot | countfi   |  6172 |      4.46  |     4.6   |   0.14  |       0.079 |      0.071 |   -0.008 | nan               |
| mem_reg vs base_reg_streams            | zero_shot | mat80     | 62992 |      1.752 |     1.761 |   0.009 |       0.497 |      0.522 |    0.025 | nan               |
| mem_reg vs base_reg_streams            | zero_shot | wicount   |   267 |      0.359 |     0.385 |   0.026 |       0.671 |      0.675 |    0.005 | nan               |
| two-point calibration vs none          | empty_cal | mat80     | 46752 |      2.548 |     3.237 |   0.689 |       0.475 |      0.432 |   -0.044 | nan               |
| two-point calibration vs none          | empty_cal | wicount   |   267 |      0.307 |     0.383 |   0.076 |       0.685 |      0.732 |    0.047 | nan               |
| two-point calibration vs none          | empty_cal | wimans    | 50409 |      1.011 |     1.094 |   0.084 |       0.582 |      0.57  |   -0.011 | nan               |
| two-point calibration vs none          | zero_shot | mat80     | 45736 |      1.789 |     2.616 |   0.827 |       0.433 |      0.517 |    0.084 | nan               |
| two-point calibration vs none          | zero_shot | wicount   |   267 |      0.84  |     0.424 |  -0.416 |       0.194 |      0.579 |    0.384 | nan               |
| two-point calibration vs none          | zero_shot | wimans    | 50409 |      1.086 |     1.222 |   0.136 |       0.574 |      0.537 |   -0.037 | nan               |
| 2 links vs 1 link (base_reg)           | empty_cal | mat80     |   nan |      2.643 |     2.595 |  -0.048 |     nan     |    nan     |  nan     | none / empty only |
| 2 links vs 1 link (base_reg)           | empty_cal | mat80     |   nan |      3.237 |     3.259 |   0.022 |     nan     |    nan     |  nan     | two-point         |
| 2 links vs 1 link (base_reg)           | empty_cal | wimans    |   nan |      1.011 |     0.979 |  -0.031 |     nan     |    nan     |  nan     | none / empty only |
| 2 links vs 1 link (base_reg)           | empty_cal | wimans    |   nan |      1.094 |     1.042 |  -0.053 |     nan     |    nan     |  nan     | two-point         |
| 2 links vs 1 link (base_reg)           | zero_shot | countfi   |   nan |      3.57  |     3.574 |   0.004 |     nan     |    nan     |  nan     | none / empty only |
| 2 links vs 1 link (base_reg)           | zero_shot | csibench  |   nan |      0.625 |     0.565 |  -0.06  |     nan     |    nan     |  nan     | none / empty only |
| 2 links vs 1 link (base_reg)           | zero_shot | mat80     |   nan |      1.848 |     1.706 |  -0.142 |     nan     |    nan     |  nan     | none / empty only |
| 2 links vs 1 link (base_reg)           | zero_shot | mat80     |   nan |      2.616 |     2.546 |  -0.071 |     nan     |    nan     |  nan     | two-point         |
| 2 links vs 1 link (base_reg)           | zero_shot | wimans    |   nan |      1.086 |     1.055 |  -0.031 |     nan     |    nan     |  nan     | none / empty only |
| 2 links vs 1 link (base_reg)           | zero_shot | wimans    |   nan |      1.222 |     1.127 |  -0.095 |     nan     |    nan     |  nan     | two-point         |
| 3 links vs 1 link (base_reg)           | empty_cal | mat80     |   nan |      2.643 |     2.58  |  -0.063 |     nan     |    nan     |  nan     | none / empty only |
| 3 links vs 1 link (base_reg)           | empty_cal | mat80     |   nan |      3.237 |     3.312 |   0.075 |     nan     |    nan     |  nan     | two-point         |
| 3 links vs 1 link (base_reg)           | empty_cal | wimans    |   nan |      1.011 |     0.968 |  -0.043 |     nan     |    nan     |  nan     | none / empty only |
| 3 links vs 1 link (base_reg)           | empty_cal | wimans    |   nan |      1.094 |     1.026 |  -0.068 |     nan     |    nan     |  nan     | two-point         |
| 3 links vs 1 link (base_reg)           | zero_shot | csibench  |   nan |      0.625 |     0.547 |  -0.077 |     nan     |    nan     |  nan     | none / empty only |
| 3 links vs 1 link (base_reg)           | zero_shot | mat80     |   nan |      1.848 |     1.658 |  -0.19  |     nan     |    nan     |  nan     | none / empty only |
| 3 links vs 1 link (base_reg)           | zero_shot | mat80     |   nan |      2.616 |     2.536 |  -0.08  |     nan     |    nan     |  nan     | two-point         |
| 3 links vs 1 link (base_reg)           | zero_shot | wimans    |   nan |      1.086 |     1.044 |  -0.042 |     nan     |    nan     |  nan     | none / empty only |
| 3 links vs 1 link (base_reg)           | zero_shot | wimans    |   nan |      1.222 |     1.092 |  -0.13  |     nan     |    nan     |  nan     | two-point         |

## Links: 1 / 2 / 3 averaged (base_reg)

| variant   | mode      | dataset   | calibration       |   links |     n |   mae |   rel_mae |   exact |   within1 |   acc4 |   f1_4 |   empty_recall |
|:----------|:----------|:----------|:------------------|--------:|------:|------:|----------:|--------:|----------:|-------:|-------:|---------------:|
| base_reg  | empty_cal | mat80     | none / empty only |       1 | 48784 | 2.643 |     1.008 |   0.187 |     0.376 |  0.758 |  0.459 |          0.944 |
| base_reg  | empty_cal | mat80     | none / empty only |       2 | 73176 | 2.595 |     0.989 |   0.192 |     0.381 |  0.755 |  0.461 |          0.947 |
| base_reg  | empty_cal | mat80     | none / empty only |       3 | 48784 | 2.58  |     0.984 |   0.193 |     0.378 |  0.764 |  0.471 |          0.971 |
| base_reg  | empty_cal | mat80     | two-point         |       1 | 46752 | 3.237 |     1.234 |   0.206 |     0.336 |  0.448 |  0.432 |          0.987 |
| base_reg  | empty_cal | mat80     | two-point         |       2 | 67080 | 3.259 |     1.242 |   0.208 |     0.323 |  0.426 |  0.422 |          0.996 |
| base_reg  | empty_cal | mat80     | two-point         |       3 | 42688 | 3.312 |     1.262 |   0.207 |     0.311 |  0.408 |  0.415 |          1     |
| base_reg  | empty_cal | wicount   | none / empty only |       1 |   267 | 0.307 |     0.512 |   0.85  |     1     |  0.85  |  0.685 |          1     |
| base_reg  | empty_cal | wicount   | two-point         |       1 |   267 | 0.383 |     0.639 |   0.734 |     1     |  0.734 |  0.732 |          1     |
| base_reg  | empty_cal | wimans    | none / empty only |       1 | 50409 | 1.011 |     0.724 |   0.315 |     0.75  |  0.556 |  0.582 |          0.93  |
| base_reg  | empty_cal | wimans    | none / empty only |       2 | 50409 | 0.979 |     0.702 |   0.313 |     0.762 |  0.563 |  0.586 |          0.931 |
| base_reg  | empty_cal | wimans    | none / empty only |       3 | 16803 | 0.968 |     0.693 |   0.317 |     0.764 |  0.572 |  0.594 |          0.947 |
| base_reg  | empty_cal | wimans    | two-point         |       1 | 50409 | 1.094 |     0.784 |   0.348 |     0.701 |  0.461 |  0.57  |          0.952 |
| base_reg  | empty_cal | wimans    | two-point         |       2 | 50409 | 1.042 |     0.746 |   0.354 |     0.724 |  0.473 |  0.589 |          0.96  |
| base_reg  | empty_cal | wimans    | two-point         |       3 | 16803 | 1.026 |     0.735 |   0.347 |     0.729 |  0.471 |  0.59  |          0.964 |
| base_reg  | zero_shot | countfi   | none / empty only |       1 |  6172 | 3.57  |     1.493 |   0.092 |     0.249 |  0.255 |  0.158 |        nan     |
| base_reg  | zero_shot | countfi   | none / empty only |       2 |  3004 | 3.574 |     1.497 |   0.096 |     0.251 |  0.207 |  0.144 |        nan     |
| base_reg  | zero_shot | csibench  | none / empty only |       1 | 20181 | 0.625 |     1.557 |   0.535 |     0.932 |  0.535 |  0.246 |        nan     |
| base_reg  | zero_shot | csibench  | none / empty only |       2 | 20181 | 0.565 |     1.408 |   0.554 |     0.949 |  0.554 |  0.262 |        nan     |
| base_reg  | zero_shot | csibench  | none / empty only |       3 |  6727 | 0.547 |     1.365 |   0.548 |     0.965 |  0.548 |  0.348 |        nan     |
| base_reg  | zero_shot | mat80     | none / empty only |       1 | 62992 | 1.848 |     0.726 |   0.182 |     0.48  |  0.836 |  0.437 |          0.378 |
| base_reg  | zero_shot | mat80     | none / empty only |       2 | 94488 | 1.706 |     0.67  |   0.183 |     0.505 |  0.85  |  0.413 |          0.399 |
| base_reg  | zero_shot | mat80     | none / empty only |       3 | 62992 | 1.658 |     0.651 |   0.18  |     0.514 |  0.853 |  0.4   |          0.417 |
| base_reg  | zero_shot | mat80     | two-point         |       1 | 45736 | 2.616 |     0.997 |   0.216 |     0.403 |  0.688 |  0.517 |          0.976 |
| base_reg  | zero_shot | mat80     | two-point         |       2 | 65048 | 2.546 |     0.97  |   0.219 |     0.4   |  0.778 |  0.564 |          0.997 |
| base_reg  | zero_shot | mat80     | two-point         |       3 | 41672 | 2.536 |     0.966 |   0.217 |     0.393 |  0.812 |  0.585 |          1     |
| base_reg  | zero_shot | wicount   | none / empty only |       1 |   267 | 0.84  |     1.402 |   0.296 |     0.933 |  0.303 |  0.194 |          0     |
| base_reg  | zero_shot | wicount   | two-point         |       1 |   267 | 0.424 |     0.707 |   0.64  |     1     |  0.64  |  0.579 |          1     |
| base_reg  | zero_shot | wimans    | none / empty only |       1 | 50409 | 1.086 |     0.778 |   0.326 |     0.706 |  0.507 |  0.574 |          0.934 |
| base_reg  | zero_shot | wimans    | none / empty only |       2 | 50409 | 1.055 |     0.756 |   0.323 |     0.724 |  0.514 |  0.59  |          0.986 |
| base_reg  | zero_shot | wimans    | none / empty only |       3 | 16803 | 1.044 |     0.748 |   0.327 |     0.727 |  0.521 |  0.596 |          0.998 |
| base_reg  | zero_shot | wimans    | two-point         |       1 | 50409 | 1.222 |     0.875 |   0.336 |     0.664 |  0.449 |  0.537 |          0.99  |
| base_reg  | zero_shot | wimans    | two-point         |       2 | 50409 | 1.127 |     0.807 |   0.346 |     0.699 |  0.46  |  0.558 |          0.999 |
| base_reg  | zero_shot | wimans    | two-point         |       3 | 16803 | 1.092 |     0.782 |   0.347 |     0.718 |  0.467 |  0.567 |          1     |

## Mean estimate per true count, empty_cal, one link: empty room only vs two-point

| dataset   |   count |     n |   mean_none |   mean_two_point |
|:----------|--------:|------:|------------:|-----------------:|
| mat80     |       0 |  5056 |        0.11 |             0.04 |
| mat80     |       2 |  4832 |        4.15 |             1.94 |
| mat80     |       3 |  4576 |        4.42 |             2.18 |
| mat80     |       4 |  5328 |        4.58 |             2.35 |
| mat80     |       5 |  4672 |        4.75 |             2.41 |
| mat80     |       6 |  4544 |        4.89 |             2.57 |
| mat80     |       7 |  4832 |        4.97 |             2.76 |
| mat80     |       8 |  5456 |        5.09 |             2.77 |
| mat80     |       9 |  4384 |        5.15 |             2.76 |
| mat80     |      10 |  5104 |        5.2  |             2.78 |
| wicount   |       0 |    11 |        0    |             0    |
| wicount   |       1 |   133 |        1.28 |             1.5  |
| wicount   |       2 |   107 |        1.76 |             2.19 |
| wicount   |       3 |    16 |        2.03 |             2.49 |
| wimans    |       0 |  2484 |        0.1  |             0.07 |
| wimans    |       1 | 16038 |        1.98 |             1.59 |
| wimans    |       2 |  7830 |        2.58 |             2.02 |
| wimans    |       3 |  8019 |        2.85 |             2.3  |
| wimans    |       4 |  8019 |        3.02 |             2.43 |
| wimans    |       5 |  8019 |        3.1  |             2.48 |

## All variants, one link

| variant          | mode      | dataset   | calibration       |   links |     n |   mae |   rel_mae |   exact |   within1 |   acc4 |   f1_4 |   empty_recall |
|:-----------------|:----------|:----------|:------------------|--------:|------:|------:|----------:|--------:|----------:|-------:|-------:|---------------:|
| base_clf4        | empty_cal | mat80     | none / empty only |       1 | 48784 | 3.251 |     1.239 |   0.187 |     0.351 |  0.672 |  0.479 |          0.991 |
| base_clf4        | empty_cal | wicount   | none / empty only |       1 |   267 | 0.178 |     0.297 |   0.831 |     1     |  0.831 |  0.674 |          1     |
| base_clf4        | empty_cal | wimans    | none / empty only |       1 | 50409 | 1.027 |     0.736 |   0.383 |     0.675 |  0.639 |  0.59  |          0.983 |
| base_clf4        | zero_shot | countfi   | none / empty only |       1 |  6172 | 3.924 |     1.641 |   0.103 |     0.206 |  0.26  |  0.146 |        nan     |
| base_clf4        | zero_shot | csibench  | none / empty only |       1 | 20181 | 0.396 |     0.989 |   0.638 |     0.966 |  0.638 |  0.238 |        nan     |
| base_clf4        | zero_shot | mat80     | none / empty only |       1 | 62992 | 3.071 |     1.206 |   0.136 |     0.372 |  0.833 |  0.43  |          0.459 |
| base_clf4        | zero_shot | wicount   | none / empty only |       1 |   267 | 0.755 |     1.259 |   0.423 |     0.839 |  0.423 |  0.256 |          0     |
| base_clf4        | zero_shot | wimans    | none / empty only |       1 | 50409 | 1.09  |     0.781 |   0.383 |     0.656 |  0.593 |  0.562 |          0.94  |
| base_reg         | empty_cal | mat80     | none / empty only |       1 | 48784 | 2.643 |     1.008 |   0.187 |     0.376 |  0.758 |  0.459 |          0.944 |
| base_reg         | empty_cal | mat80     | two-point         |       1 | 46752 | 3.237 |     1.234 |   0.206 |     0.336 |  0.448 |  0.432 |          0.987 |
| base_reg         | empty_cal | wicount   | none / empty only |       1 |   267 | 0.307 |     0.512 |   0.85  |     1     |  0.85  |  0.685 |          1     |
| base_reg         | empty_cal | wicount   | two-point         |       1 |   267 | 0.383 |     0.639 |   0.734 |     1     |  0.734 |  0.732 |          1     |
| base_reg         | empty_cal | wimans    | none / empty only |       1 | 50409 | 1.011 |     0.724 |   0.315 |     0.75  |  0.556 |  0.582 |          0.93  |
| base_reg         | empty_cal | wimans    | two-point         |       1 | 50409 | 1.094 |     0.784 |   0.348 |     0.701 |  0.461 |  0.57  |          0.952 |
| base_reg         | zero_shot | countfi   | none / empty only |       1 |  6172 | 3.57  |     1.493 |   0.092 |     0.249 |  0.255 |  0.158 |        nan     |
| base_reg         | zero_shot | csibench  | none / empty only |       1 | 20181 | 0.625 |     1.557 |   0.535 |     0.932 |  0.535 |  0.246 |        nan     |
| base_reg         | zero_shot | mat80     | none / empty only |       1 | 62992 | 1.848 |     0.726 |   0.182 |     0.48  |  0.836 |  0.437 |          0.378 |
| base_reg         | zero_shot | mat80     | two-point         |       1 | 45736 | 2.616 |     0.997 |   0.216 |     0.403 |  0.688 |  0.517 |          0.976 |
| base_reg         | zero_shot | wicount   | none / empty only |       1 |   267 | 0.84  |     1.402 |   0.296 |     0.933 |  0.303 |  0.194 |          0     |
| base_reg         | zero_shot | wicount   | two-point         |       1 |   267 | 0.424 |     0.707 |   0.64  |     1     |  0.64  |  0.579 |          1     |
| base_reg         | zero_shot | wimans    | none / empty only |       1 | 50409 | 1.086 |     0.778 |   0.326 |     0.706 |  0.507 |  0.574 |          0.934 |
| base_reg         | zero_shot | wimans    | two-point         |       1 | 50409 | 1.222 |     0.875 |   0.336 |     0.664 |  0.449 |  0.537 |          0.99  |
| base_reg_streams | empty_cal | mat80     | none / empty only |       1 | 48784 | 2.755 |     1.05  |   0.184 |     0.361 |  0.744 |  0.451 |          0.931 |
| base_reg_streams | empty_cal | mat80     | two-point         |       1 | 45736 | 3.247 |     1.238 |   0.204 |     0.33  |  0.448 |  0.429 |          0.971 |
| base_reg_streams | empty_cal | wicount   | none / empty only |       1 |   267 | 0.303 |     0.505 |   0.805 |     1     |  0.805 |  0.71  |          1     |
| base_reg_streams | empty_cal | wicount   | two-point         |       1 |   267 | 0.321 |     0.536 |   0.801 |     1     |  0.801 |  0.707 |          1     |
| base_reg_streams | zero_shot | countfi   | none / empty only |       1 |  6172 | 4.46  |     1.865 |   0.045 |     0.164 |  0.1   |  0.079 |        nan     |
| base_reg_streams | zero_shot | mat80     | none / empty only |       1 | 62992 | 1.752 |     0.688 |   0.203 |     0.5   |  0.863 |  0.497 |          0.68  |
| base_reg_streams | zero_shot | mat80     | two-point         |       1 | 48784 | 2.684 |     1.023 |   0.215 |     0.409 |  0.677 |  0.513 |          0.962 |
| base_reg_streams | zero_shot | wicount   | none / empty only |       1 |   267 | 0.359 |     0.6   |   0.727 |     0.993 |  0.734 |  0.671 |          1     |
| base_reg_streams | zero_shot | wicount   | two-point         |       1 |   267 | 0.392 |     0.655 |   0.727 |     0.985 |  0.734 |  0.732 |          1     |
| extra_reg        | empty_cal | mat80     | none / empty only |       1 | 48784 | 2.395 |     0.913 |   0.198 |     0.418 |  0.764 |  0.461 |          0.858 |
| extra_reg        | empty_cal | mat80     | two-point         |       1 | 46752 | 2.987 |     1.139 |   0.211 |     0.353 |  0.672 |  0.516 |          0.952 |
| extra_reg        | empty_cal | wicount   | none / empty only |       1 |   267 | 0.277 |     0.463 |   0.869 |     1     |  0.869 |  0.693 |          1     |
| extra_reg        | empty_cal | wicount   | two-point         |       1 |   267 | 0.281 |     0.469 |   0.865 |     1     |  0.865 |  0.7   |          1     |
| extra_reg        | empty_cal | wimans    | none / empty only |       1 | 50409 | 1.001 |     0.717 |   0.322 |     0.757 |  0.553 |  0.577 |          0.891 |
| extra_reg        | empty_cal | wimans    | two-point         |       1 | 50409 | 1.074 |     0.77  |   0.357 |     0.71  |  0.504 |  0.59  |          0.929 |
| extra_reg        | zero_shot | countfi   | none / empty only |       1 |  6172 | 3.758 |     1.572 |   0.085 |     0.225 |  0.18  |  0.121 |        nan     |
| extra_reg        | zero_shot | csibench  | none / empty only |       1 | 20181 | 0.496 |     1.236 |   0.574 |     0.969 |  0.574 |  0.251 |        nan     |
| extra_reg        | zero_shot | mat80     | none / empty only |       1 | 62992 | 1.711 |     0.672 |   0.187 |     0.527 |  0.835 |  0.406 |          0.305 |
| extra_reg        | zero_shot | mat80     | two-point         |       1 | 46752 | 2.609 |     0.995 |   0.213 |     0.389 |  0.743 |  0.541 |          0.996 |
| extra_reg        | zero_shot | wicount   | none / empty only |       1 |   267 | 0.737 |     1.23  |   0.322 |     0.951 |  0.322 |  0.158 |          0     |
| extra_reg        | zero_shot | wicount   | two-point         |       1 |   267 | 0.412 |     0.687 |   0.708 |     0.989 |  0.708 |  0.602 |          1     |
| extra_reg        | zero_shot | wimans    | none / empty only |       1 | 50409 | 1.037 |     0.743 |   0.337 |     0.731 |  0.547 |  0.594 |          0.933 |
| extra_reg        | zero_shot | wimans    | two-point         |       1 | 50409 | 1.175 |     0.842 |   0.347 |     0.675 |  0.452 |  0.54  |          0.996 |
| mem_reg          | empty_cal | mat80     | none / empty only |       1 | 48784 | 2.253 |     0.859 |   0.217 |     0.448 |  0.789 |  0.492 |          0.972 |
| mem_reg          | empty_cal | mat80     | two-point         |       1 | 47768 | 3.19  |     1.216 |   0.181 |     0.322 |  0.536 |  0.467 |          1     |
| mem_reg          | empty_cal | wicount   | none / empty only |       1 |   267 | 0.375 |     0.626 |   0.685 |     1     |  0.689 |  0.636 |          1     |
| mem_reg          | empty_cal | wicount   | two-point         |       1 |   267 | 0.301 |     0.501 |   0.813 |     1     |  0.816 |  0.734 |          1     |
| mem_reg          | zero_shot | countfi   | none / empty only |       1 |  6172 | 4.6   |     1.924 |   0.027 |     0.155 |  0.071 |  0.071 |        nan     |
| mem_reg          | zero_shot | mat80     | none / empty only |       1 | 62992 | 1.761 |     0.691 |   0.211 |     0.505 |  0.869 |  0.522 |          0.817 |
| mem_reg          | zero_shot | mat80     | two-point         |       1 | 48784 | 2.604 |     0.993 |   0.21  |     0.412 |  0.678 |  0.507 |          0.964 |
| mem_reg          | zero_shot | wicount   | none / empty only |       1 |   267 | 0.385 |     0.642 |   0.73  |     0.989 |  0.738 |  0.675 |          1     |
| mem_reg          | zero_shot | wicount   | two-point         |       1 |   267 | 0.282 |     0.471 |   0.768 |     0.996 |  0.772 |  0.715 |          1     |

## Notes

- Links: mat80 = antennas of one router, WiMANS = receive antennas of one card (both close together, so a lower bound for 2 boards on different walls); CSI-Bench = 3 devices; CountFi = 2 receivers matched by window index.
- Two-point calibration uses 20 s of the empty room and 20 s with 2 people of the same site; those windows are never trained on or scored. Not possible on CountFi (no empty) or CSI-Bench (no empty).
- CountFi is never trained on; each count there is one time block (stress test only).
- WiMANS / CSI-Bench clips are clip-mean normalised; they have no 30 s memory, so mem_reg is streams only.
- No model is exported: each upgrade needs esp32_counter.py / firmware changes first.
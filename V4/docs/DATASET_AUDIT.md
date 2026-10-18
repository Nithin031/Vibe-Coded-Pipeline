# V4 dataset audit

All raw data is in `<project>/datasets/` (one folder per source; set `CSI_DATASETS` to use another folder).
V4 never copies raw data; it writes processed windows to `V4/cache/` (rebuildable
with `python V4/src/data.py --build all`). Measured numbers below come from
`V4/cache/*_parse.json`.

## Used in V4

| Source | Folder | Hardware | Labels used | Recordings | Length | Packet rate | Valid windows | Role |
|---|---|---|---|---|---|---|---|---|
| ESPectre | `datasets/espectre` | 5 ESP32 chip types, real homes | empty / occupied (+ static / moving) | 120 (8 excluded by the dataset) | 337 min | 91–499 Hz | 100 % | presence train / val / test (dataset's own `train` / `selection` / `holdout`) |
| esp32-csi-pipeline | `datasets/esp32_csi_pipeline` | 2 ESP32, router link | empty / occupied | 9 | 40 min | 100 Hz (packet counter, no real clock) | 100 % | presence train (room 1 quiet), test (room 1 **congested channel**, room 2) |
| HomeOccupancy | `datasets/homeoccupancy` | 2 ESP32-C6, one home | empty / occupied (sleep, work kept only as activity) | 9 | 150 min | 119–193 Hz | 94 % | presence train (_1), val (_2), development test (_3, already inspected) |
| OfficeLocalization | `datasets/officelocalization` | 2 ESP32-C6, one office (same authors as Home) | empty / occupied (zones not used) | 11 | 173 min | 197–237 Hz | 100 % | presence train (_1), **untouched external test (_2)**, val (empty_3), extra empty test (empty_4, empty_5) |
| WiCount | `datasets/wicount` | ESP32-S3, one meeting room | people count 0–3 | 15 (0: 1, 1: 7, 2: 6, 3: 1) | 15 min | 80–94 Hz | 99.5 % | counting (3 folds) |
| OTH room 2 samples | `archive/datasets/esp32_reference` | ESP32, ESP-NOW HT40 | people count 0–3 | 4 (one per count) | 4 min | 24–29 Hz | 99 % | counting cross-room check |

## Problems found in the raw data (and what V4 does)

| Problem | Where | Effect if ignored | V4 fix |
|---|---|---|---|
| A few timestamps are huge garbage numbers (about 9e17) | HomeOccupancy (150 rows) | the clock "unwraps" wrongly and each recording looks 3–5x longer; **50–83 % of windows became pure interpolation** in the preliminary V4 | out-of-range values are removed **before** unwrapping (`data.clean_timestamps`) |
| The same clock is printed as a signed 32-bit number (negative values) | OfficeLocalization empty_4, empty_5 (448 551 rows) | whole files would be dropped | converted back to the unsigned clock |
| Rows with shifted columns, wrong `len`, impossible RSSI, garbled I/Q text | HomeOccupancy (526), OfficeLocalization (45) | wrong CSI values | rejected and counted |
| Isolated timestamp spikes | HomeOccupancy (3) | fake gaps | dropped if > 1 s away from the median of neighbours |
| Real long packet gaps | HomeOccupancy work_1 (only 56 % valid windows) | interpolated values would be scored as real | windows with a gap > 0.5 s or > 25 % missing steps are marked **invalid** and reported as "unknown", never scored |
| Packet counter instead of a clock | pipeline | timing jitter cannot be measured | documented; packet rate assumed 100 Hz as the dataset README says |

## 20 Hz grid check (WiCount, native ~87 Hz)

In occupied WiCount recordings, 81–91 % of the amplitude fluctuation power is below 10 Hz
(`results/metrics/wicount_20hz_check.json`). In the empty recording 73 % lies above 10 Hz,
which is what flat sensor noise gives (75 %). So the 20 Hz grid keeps the human
signal and averages away mostly noise. Caveat: the high-to-low frequency ratio is itself
an empty-room cue that 20 Hz features can only partly see.

## Audited but NOT used

| Source | Why not |
|---|---|
| RF_ESP32 (`datasets/rf_esp32`) | each session stores only 200 CSI frames, but its event labels point to ~33 000 frame indices, so most labels have no CSI. Not usable for 3 s windows. |
| OpenCSI (`datasets/opencsi`) | **corrected reason**: packets arrive in bursts (2–4 bursts/s per link, ~0.5 ms apart inside a burst, 7.6–24.9 packets/s per link on average), and the firmware timestamps cannot separate packets inside a burst. Spreading them onto a 20 Hz grid would invent timing. (V3's write-up said "about 1 snapshot/s per link", which is wrong.) |
| Intel 5300 crowd counting (`datasets/intel5300`) | counts 0–8 in 3 rooms, but Intel laptop Wi-Fi card, not ESP32 |
| EHUNAM subset (`datasets/ehunam`) | counts 0–5 (MC3), but Raspberry Pi / Nexmon, not ESP32 |
| WiGesture, WiFall | always exactly one person: no presence or count contrast |
| "Less is More" apartment mesh (Zenodo 22028063) | 45.7 GB; labels are hourly diary entries (empty / sleep / active), not counts. Only its 180 KB label files were checked (moved to `archive/`). |

## Not available (needs a person to request access)

| Source | What it has | How to get it |
|---|---|---|
| WiFiVision | ESP32, **0–7 people**, one lab room, 1 000 + 200 samples per class | "Request Access" on the Google Drive link in github.com/nguyen-phan-duc-minh/WiFiVision-Counting-Tools (academic, non-commercial) |
| OTH thesis full set | ESP32, 0–3 people, ~120 sessions, 3 rooms | ask the thesis authors (repo in `archive/reference_repos/wifi-csi-occupancy`) |
| WiFi-Wild (Zenodo 18431784) | ESP32 under everyday environment changes | Zenodo "request access" |
| IIT (BHU) multi-human | ESP32, 1–3 users doing activities | IEEE DataPort subscription |

## Related papers (no code or weights to load)

- CSI-JEPA (arXiv 2605.14171): self-supervised masked prediction over time x subcarrier. No code or weights were found; using it would mean re-implementing it.
- WiFi-JEPA (ECCV 2026, arXiv 2607.11064): 3D pose from 9 antenna links. Its link masking needs many links; our setup has one link.

The full search record is `manifests/dataset_search.json`.

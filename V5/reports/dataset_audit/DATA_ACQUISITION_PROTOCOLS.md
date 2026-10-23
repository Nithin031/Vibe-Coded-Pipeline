# Data Acquisition Protocols

Audit date: 2026-10-19. Part of the [dataset audit](DATASET_AUDIT_REPORT.md). Field-level details are in the [feature dictionary](FEATURE_DICTIONARY.md).

Every specification carries a source class:

- **[P] Confirmed from a primary source:** the official dataset page or the authors' paper or code.
- **[S] Reported by a secondary source:** `D:/wifi/csi_counting_dataset_survey.pdf`, a compiled survey in your workspace, or a re-hosted dataset card.
- **[F] Inferred / measured from the files:** audit scripts in `scripts/`.
- **[U] Unknown:** not found.

A value marked [F] that disagrees with [P] is flagged ⚠.

---

## 1. WiCount

| Item | Value | Source |
|---|---|---|
| Identity | WiCount, Hugging Face `RS2002/WiCount`. Used in *CSI-BERT2* (Zhao et al., IEEE TMC 2025, doi:10.1109/TMC.2025.3640420, arXiv 2412.06861) | [P] local README + paper |
| Collectors / org | Zijian Zhao, Tingwei Chen; AI-RAN Lab, SRIBD, CUHK-Shenzhen | [P] README |
| Transmitter | README: "single transmitter … using ESP32-S3 devices". Paper: "a home router as TX" | [P] ⚠ the two primary sources conflict |
| Receiver | ESP32-S3, 1 antenna. README: "multiple receivers"; paper: one RX | [P] ⚠ conflict |
| Receivers in local files | **One** stream: TX MAC `d4:da:21:98:07:d9`, every filename ends `_104_1` | [F] |
| CSI tool | ESP-IDF CSI callback (`CSI_DATA` format, ESP-CSI style columns) | [F] (column names) |
| Standard / mode | README: 802.11n. Files: **all frames legacy non-HT** (sig_mode = 0, MCS 0, 20 MHz) | [P] vs [F] ⚠ |
| Band / channel | 2.4 GHz [P]; channel 12, secondary flag 2 [F] | [P] [F] |
| Bandwidth / tones | 20 MHz, 52 subcarriers (104 ints) | [P] [F] |
| Packet rate | Nominal ~100 Hz [P]. Measured: median spacing 11.99 ms on the ESP32 clock (≈ 83 Hz); file means 80–94 Hz; 38–281 gaps > 20 ms per file | [F] ⚠ |
| Recording | 15 files × 60 s continuous = 15 min total (paper Table III says 15 min) | [P] [F] |
| Environment | Small indoor meeting room with walls and a soft pad; layout sketch in the paper (Fig. 5); no dimensions given | [P] |
| Subjects / behaviour | 0–3 volunteers who "moved naturally, performing … walking, squatting, jumping, sitting …" | [P] paper |
| Labels | Folder = number of people, set by the experimenters from the setup; no further annotation | [P] README |
| Collection timeline | One evening, 2023-12-21, 19:23 → 20:08 (UTC+8), recorded in order 0 → 1 → 2 → 3 people | [F] |
| Pre-distribution processing | None to the CSV (raw). The provided scripts (not applied) fill lost packets with −1000 and window into 1 s (100-packet) samples assuming 10 ms spacing | [D] |
| Licence | Not declared in README | [D] |

---

## 2. 80 MHz Human Sensing (`Dataset/`) — Meneghello et al.

| Item | Value | Source |
|---|---|---|
| Identity | "CSI dataset for wireless human sensing on 80 MHz Wi-Fi channels", IEEE DataPort, doi:10.21227/xbhv-f125. Paper: Meneghello et al., IEEE Communications Magazine 2023 | [P] https://ieee-dataport.org/documents/csi-dataset-wireless-human-sensing-80-mhz-wi-fi-channels |
| Identity check against files | File grammar `PCxy_nN.mat` and per-trace length (≈ 40–52 k packets ≈ 4–5 min at 173 Hz, against the published "40 s to 5 min, 6,000–50,000 samples") agree | [F] + [P] |
| TX / RX | Asus RT-AC86U or Netgear X4S AC2600, one antenna enabled (single spatial stream); iPerf3 traffic TX → RX | [P] |
| Monitor (CSI sniffer) | Asus RT-AC86U, 4 antennas enabled. Chipset not named on the page (the RT-AC86U is commonly a BCM4366C0 — not verified) | [P] / [U] |
| CSI tool | Nexmon CSI → pcap → Nexmon MATLAB script → `.mat` | [P] |
| Standard / channel | IEEE 802.11ac, channel 42, centre 5,210 MHz, 80 MHz | [P] |
| Tones | 256 FFT bins stored; 242 usable (−122…−2, 2…122) | [P] + [F] (13 low-energy bins observed) |
| Antennas | 1 TX stream × 4 RX chains, rows interleaved | [P] + [F] |
| Packet rate | 173 pkt/s (6 ms) | [P] (not verifiable: no timestamps in files) |
| Environments (whole dataset) | Bedroom, living room, kitchen, university lab, university office, semi-anechoic chamber, meeting room | [P] |
| PC setups 1a–4a | Defined in `set_names.pdf` (not downloaded) | [U] |
| People-counting protocol | N concurrent people "moving in the environment"; n00 = empty room; n01 traces live in the PI folders | [S] survey (paper paywalled) |
| Labels | Per-trace count from the experiment design | [P] |
| Pre-distribution processing | None (raw CFR). Users must remove 14 null bins, invert the sign of the upper part of the spectrum (hardware artefact) and sanitise phase (SHARP repo) | [P] DataPort page + SHARP code |
| Licence | Not stated on the page; IEEE DataPort subscription access | [P] |

---

## 3. CountFi

| Item | Value | Source |
|---|---|---|
| Identity | CountFi — "RSSI-Assisted CSI-Based Passenger Counting with Multiple Wi-Fi Receivers", Guo, Zhuang, Mao, Ho (PolyU), IEEE WCNC 2025, arXiv 2410.11400 | [P] https://guojingtao-1997.github.io/CountFi/ |
| Which part is local | The **2025-04-08 "mixed (moving + stationary)"** set: the dataset page documents 1,000 CSI samples per pcap for the 2025 sets, and file timestamps are 2025-04-08 06:37–09:43 UTC (14:37–17:43 HKT) | [P] + [F] |
| Environment | Upper deck of double-decker buses in Hong Kong; front / middle / end sections. The bus was moving or stopped on this date | [P] |
| Transmitters | Two single-antenna "pingers" that make the AP generate traffic (paper, 2023 set) | [P] |
| Receivers | Two single-antenna Raspberry Pis, Nexmon CSI + tcpdump, decoded with Nexcsi (paper). Pi model not stated | [P]; model [U] |
| Receivers in local files | Two (`head`, `end`), capturing simultaneously (start times within ~20 ms); core / stream always 0 | [F] |
| Channel | 5 GHz, ch 42, 80 MHz (chanspec 0xe02a in 100% of packets) | [F] (not stated on page or paper) |
| Tones | 256 bins; the paper removes 14 nulls → 242 | [P] + [F] |
| Sniffed frames | Single source MAC; 53% Block-Ack, 45% QoS-Data, ~2% beacon / probe-response | [F] |
| Packet rate | Local files ≈ 1,000 captured packets/s (≈ 470/s per frame type) | [F] — the paper's 2023 set was ≈ 90/s ⚠ (different set) |
| Max count | 20 passengers (whole dataset) | [P] |
| Ground truth | "Passenger counts were manually recorded during data collection … synchronised with CSI timestamps" | [P] — the log itself is **not in the download** [F] |
| Local archive | The page's April 2025 download is `48.zip`, matching local `CountFi/48/`; Feb 2025 = `26_27_data.zip`; Jun 2023 = `613_20People.zip`; "Full dataset" Drive link is dead | [P] (link targets) + [F] |
| Local label proxy | Numbered subfolders 1–8, 10 in contiguous time blocks. The page says each release includes ground-truth counts, and the download has nothing else → counts are encoded in the folders (not stated in words) | [F] + [P] → inferred, medium–high |
| Passenger behaviour | Static activities in the 2023 set (sitting, talking, sleeping); the April 2025 set is "mixed" moving + stationary | [P] |
| Paper's processing | Windows of 300 samples with stride 150; RSSI-rescaled amplitude; 790 segments per class | [P] (not applied to local files) |
| Licence | CC BY-NC 4.0, cite the paper | [P] |

---

## 4. WiMANS (`Wiman/`)

| Item | Value | Source |
|---|---|---|
| Identity | WiMANS: multi-user activity sensing benchmark (Huang et al., ECCV 2024, arXiv 2402.09430), github.com/huangshk/WiMANS | [P] |
| TX / RX | Two HP EliteDesk 800 G2 PCs, Intel 5300 NIC, Linux 802.11n CSI Tool, monitor mode | [P] paper |
| Antennas / tones | 3 TX × 3 RX × 30 subcarriers (Intel 5300 grouping) | [P] + [F] |
| Bands | 2.4 GHz (ch 12) and 5 GHz (ch 64) | [P] |
| Packet rate | 1,000 packets/s; 3 s samples (3,000 packets nominal); mean packet loss 4.52% (2.4 GHz) / 2.31% (5 GHz) | [P]; observed T = 2,772–2,986 [F] |
| Environments | Classroom, meeting room, empty room; 5 locations each | [P] |
| Subjects | 6 volunteers; 0–5 simultaneous users per sample | [P] |
| Activities | Nothing, walking, rotation, jumping, waving, lying down, picking up, sitting down, standing up | [P] |
| Labelling | Scripts announced before each capture, plus synchronised video | [P] |
| Count distribution (all 11,286) | 0 users: 594; 1: 3,564; 2–5: 1,782 each | [P] |
| Local | 3,057 amplitude `.npy` files (31 groups); `annotation.csv`, raw `.mat` and video absent | [F] |
| Processing applied | Amplitude computed from raw CSI by the authors' `preprocess.py` (unnormalised; max 179.6 = 8-bit I/Q saturation) | [P] + [F] |
| Licence | Paper page CC BY-NC-SA 4.0; data for academic research | [P] |

---

## 5. IIT-BHU multi-human activity recognition

| Item | Value | Source |
|---|---|---|
| Identity | "Channel State Information Dataset for Multi-Human Activity Recognition in Indoor Environments", Gupta, Jahnavi, Bhavikbhai (IIT BHU), Mishra (IIT Patna), IEEE DataPort, doi:10.21227/e6c6-aa21 | [P] via DataPort search result; folder `wifi_dataset_Collected/2_users/User_25_26_bend_run …` matches the published listing |
| Hardware | ESP32 NodeMCU transmitter–receiver pair; ESP32-CSI-Tool | [P] abstract |
| Receiver mode | Tool's station firmware (`STA` column) | [F] |
| Transmitters seen | 3 different dominant TX MACs across sessions (Espressif OUIs) → at least 3 TX boards or configurations | [F] |
| Channel / bandwidth | Ch 6, HT40 (secondary above) on ~96% of rows; legacy 20 MHz on the rest | [F] |
| Packet rate | Per-file median 27.7 Hz (4.1–51.6 Hz) | [F] |
| Duration | 12–277 s per file (median 37 s); 576.8 min total | [F] |
| Dates | 7 days between 2024-08-31 and 2024-09-22 | [F] |
| Subjects | "80+ participants"; local user IDs 1–82 | [P] + [F] |
| Activities | "Six activities, subtle hand movements to full-body actions"; local filenames: bend, clap, move, run, stretch, wave (+ 5 files "pen") | [P] + [F] |
| Scenarios | Single-user, two-user, three-user; controlled indoor room | [P] |
| Room layout / distance | Not documented | [U] |
| Labelling | Scenario and activity encoded in folder and filename; method not documented | [F] / [U] |
| Licence / access | IEEE DataPort subscription | [P] |

---

## 6. "CSI HAR" — Zhuravchak & Kapshii

| Item | Value | Source |
|---|---|---|
| Identity | "Dataset for Human Activity Recognition using Wi-Fi Channel State Information (CSI) data", figshare doi:10.6084/m9.figshare.14386892 (CC BY 4.0). Paper: Zhuravchak et al. (full author list not extracted), *Procedia Computer Science* 198 (2022) 59–66, doi:10.1016/j.procs.2021.12.211. Code: github.com/retsediv/WIFI_CSI_based_HAR | [P] |
| Identity check against files | Folder grammar (`room_1/2/3`, sessions, `data.csv` / `label.csv` / `label_boxes.csv`), 4 streams × 114 tones, the 7 activities, and size 1.1 GB vs "1.2 Gb (no images)" all agree | [F] + [P] |
| TX / RX | Two TP-Link TL-WDR4300 routers (AP + client), Atheros CSI Tool; CSI streamed over UDP to a laptop | [P] |
| Channel | 5 GHz, ch 60, 40 MHz | [P] |
| Antennas / tones | 2 TX × 2 RX, 114 subcarriers | [P] + [F] |
| Packet rate | Not stated; no timestamps in the files | [U] |
| Camera | A photo taken per packet for scene parsing; label and bounding box per packet | [P] |
| Subject / rooms | 1 person (the author); 3 rooms; "lab-like over-controlled environment" | [P] |
| Activities | Walking, sitting, standing, lying, getting up, getting down, no activity (= no person in the room) | [P] |
| Sessions | 10 (room_1: 4, room_2: 1, room_3: 5); 67,449 packets | [F] |
| Processing applied | CSI converted to amplitude + phase (polar); phase not sanitised | [F] |

---

## 7. Haron98 WiFi CSI Router Reception Dataset

| Item | Value | Source |
|---|---|---|
| Identity | HF `Haron98/WIFI-CSI-Dataset`, a copy of `ilyakolosov/WIFI-CSI-Dataset` | [P] HF card |
| Hardware | 3 ESP32 receivers (model not stated) + one router / AP (MAC `50:ff:20:e5:a8:f3`) | [D] + [F] |
| Capture method | Passive CSI from the router's own traffic: HT frames at MCS 7 with A-MPDU / STBC flags set → ordinary data traffic, not dedicated sounding | [F] / [I] |
| Channel | 2.4 GHz ch 6, 20 MHz | [F] |
| Tones | LLTF 64 (52 usable) | [F] + ESP-IDF docs [P] |
| Rate | ~25 Hz (median 39 ms) | [F] |
| Trials | 4 subjects × 4 labels × 100 trials × 3 receivers; ~4 s each; 2026-04-18 20:17–23:21 (UTC+3) | [F] |
| Order of recording | Subjects 1–4 each did label_01 → 02 → 03; then label_00 for subjects 1–4 back-to-back at the end | [F] |
| Labels | `label_00…03`, meaning **not documented** | [D] |
| Room / layout / distance | Not documented | [U] |
| Licence | Contradictory (YAML "mit" vs body "no licence") | [D] |

---

## 8. Shared caveats

1. **ESP32 I/Q order.** Espressif documents each CSI item as *imaginary part followed by real part*. The WiCount and Haron98 readmes describe the opposite order. Only phase-based features are affected.
   - Source: [ESP-IDF Wi-Fi vendor features](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-guides/wifi-driver/wifi-vendor-features.html).
2. **Nexmon CSI formats.** CountFi's int16 decode was chosen by a controlled test, not by documentation (see `scripts/08b_countfi_decode_test.py`). The 80 MHz `.mat` files are already decoded.
3. **No dataset ships an empty-room calibration session per day** apart from the per-setup `n00` traces in the 80 MHz set.

# Feature Dictionary — Wi-Fi CSI datasets in `D:/wifi/Datasets`

Audit date: 2026-10-19. Part of the [dataset audit](DATASET_AUDIT_REPORT.md). See also [acquisition protocols](DATA_ACQUISITION_PROTOCOLS.md) and [dataset comparison](DATASET_COMPARISON.md).

**Evidence tags used throughout**

| Tag | Meaning |
|---|---|
| **[F]** | Verified from file contents by an audit script in `scripts/` |
| **[D]** | Verified from local documentation (README / scripts shipped with the data) |
| **[E]** | Supported by an external publication or official dataset page (URL given) |
| **[I]** | Inferred — reasoning stated |
| **[U]** | Unknown / requires further investigation |

Confidence: **High** = verified by two independent kinds of evidence, or by an unambiguous file test. **Medium** = one source, consistent with files. **Low** = inference only.

---

## 0. Plain-language primer

| Term | Plain meaning | Technical meaning |
|---|---|---|
| CSI (Channel State Information) | A per-packet "fingerprint" of how the radio signal was changed on its way from transmitter to receiver. People in the room change it. | The complex channel frequency response H(f) estimated by the receiver from training symbols in the Wi-Fi preamble, one complex number per subcarrier per antenna pair. |
| Subcarrier | One narrow frequency slice of the Wi-Fi channel. A 20 MHz channel is split into 64 slices, 80 MHz into 256. | OFDM tone; index k ↔ frequency f_c + k·312.5 kHz. |
| Null / guard / DC subcarrier | Slices that carry no signal (edges and centre), so their CSI is zero or junk. | Unused OFDM tones; must be removed before modelling. |
| Amplitude | How strong the signal is on a subcarrier. | \|H\| = sqrt(I² + Q²). |
| Phase | The timing/rotation of the signal on a subcarrier. Raw phase is dominated by clock offsets and is not directly usable. | ∠H = atan2(Q, I); corrupted by CFO/SFO/packet-detection delay, needs sanitisation. |
| I/Q | The two integer components the chip reports for each subcarrier. | Real (in-phase) and imaginary (quadrature) parts. |
| RSSI | Total received power of the packet, one number. | Received Signal Strength Indicator, dBm. |
| Spatial stream / RX core / antenna pair | Each TX-antenna → RX-antenna path has its own CSI. | MIMO channel matrix entries. |
| LLTF / HT-LTF | Two different training fields in a Wi-Fi packet; ESP32 can report CSI from either. | Legacy Long Training Field (20 MHz, 64 tones) and High-Throughput LTF (802.11n). |
| Non-HT duplicate | An old-style 20 MHz frame copied four times across an 80 MHz channel (used for control frames such as Block-Acks). Its CSI looks like four separate 20 MHz spectra. | 802.11ac non-HT duplicate PPDU. |

---

## 1. WiCount (`WiCount/`)

**What it is:** 15 CSV recordings (60 s each) from one ESP32-S3 receiver in one meeting room. The folder name gives the number of people (0–3) [D].

**Row = one received Wi-Fi packet** [F] (each row has a distinct `local_timestamp`; rows ≈ 80–94 per second).

**Example row** (from `data/0/2023-12-21_19-23-53-985_104_1.csv`, first data line; CSI truncated):
```
type=CSI_DATA, seq=5608, timestamp=2023-12-21 19:23:53.985, taget_seq=1, taget=test,
mac=d4:da:21:98:07:d9, rssi=-27, rate=11, sig_mode=0, mcs=0, cwb=0, ..., noise_floor=-99,
channel_primary=12, channel_secondary=2, local_timestamp=67951197, ant=0, sig_len=37,
rx_state=0, len=104, first_word_invalid=0,
data="[-3, 9, -5, 7, -7, 6, ... , 11, -11]"   (104 integers)
```

| Field | Raw / derived / meta | Shape | Type | Meaning (plain → technical) | Unit & observed range | How collected | Label link | Quality issues | Evidence | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| `data` | **Raw** | 104 ints per row | int (text list) | The CSI: 52 subcarriers × (2 numbers each). | Arbitrary ADC units, [-36, 36] | ESP-IDF CSI callback on ESP32-S3 | none | **No always-zero positions** → null/guard tones already removed, but *which* 52 tone indices are kept is not documented [U]. Pair order disputed: README says `a[2i] + j·a[2i+1]`; Espressif docs say each pair is **imaginary then real** [E]. Amplitude unaffected; phase sign/quadrant affected. The shipped `process1.py` uses `x[2i] + j·x[2i−1]` — an **indexing bug** (for i=0 it pairs with the last element). | [F] [D] [E] | High (length, range); Low (pair order) |
| `rssi` | Raw | scalar | int | Total packet power | dBm, −90 … −23 (mean ≈ −28) | ESP32 rx_ctrl | none | Rare −90 outliers | [F] | High |
| `local_timestamp` | Raw meta | scalar | int | ESP32 receive clock | µs (ESP-IDF rx_ctrl), monotonic in all files | ESP32 | none | Median spacing 11.99 ms; gaps >20 ms: 38–281 per file | [F] [E] | High |
| `timestamp` | Meta | scalar | string | Host PC wall-clock when the line was logged | ms resolution, UTC+8 [D] | logging PC | none | Host spacing jittery (median 11–15 ms, max 2.0 s) — use `local_timestamp` for timing | [F] [D] | High |
| `seq` | Meta | scalar | int | Line counter of the logger | increments by exactly 1 on every row in all files | logger | none | Cannot reveal packet loss (it counts received packets) | [F] | High |
| `mac` | Meta | scalar | string | Transmitter MAC | one value `d4:da:21:98:07:d9` | — | — | Single TX | [F] | High |
| `sig_mode`, `mcs`, `rate`, `cwb`, `smoothing`, `not_sounding`, `aggregation`, `stbc`, `fec_coding`, `sgi`, `ampdu_cnt`, `ant`, `rx_state`, `first_word_invalid` | Meta | scalar | int | PHY descriptors of the received frame | all **constant**: sig_mode=0 (legacy/non-HT), mcs=0, rate=11, cwb=0 (20 MHz) | rx_ctrl | — | Constant → zero information for ML. Frames are legacy although README says 802.11n | [F] | High |
| `noise_floor` | Meta | scalar | int | Noise floor | constant −99 | rx_ctrl | — | Constant → placeholder value, not a measurement [I] | [F] | Medium |
| `channel_primary`, `channel_secondary` | Meta | scalar | int | Wi-Fi channel | 12 and 2 (constant) | rx_ctrl | — | — | [F] | High |
| `sig_len` | Meta | scalar | int | PPDU length | 37 (one row 82) | rx_ctrl | — | Same frame type throughout | [F] | High |
| `len` | Meta | scalar | int | CSI vector length | 104 | — | — | — | [F] | High |
| `type`, `taget`, `taget_seq` | Meta | scalar | string/int | Logger tags | constant `CSI_DATA`, `test`, 1 | logger | — | No information | [F] | High |
| folder `data/<N>/` | **Label** | per file | int | Number of people present | 0, 1, 2, 3 | Assigned by experimenters from setup [D] | — | 1 / 7 / 6 / 1 files per class | [F] [D] | High |

**Usable ML features:** 52 complex CSI values (or 52 amplitudes + 52 phases) + RSSI + timing. The other 22 columns are constant.
**Cannot be reconstructed:** the original 64-tone layout (which tones were dropped), and any receiver other than the one logged (the README mentions multiple receivers; only one MAC/file stream is present [F]).

---

## 2. 80 MHz Human Sensing — people-counting folders (`Dataset/PC-*a/`)

**What it is:** Nexmon-extracted 802.11ac CSI from the Meneghello et al. dataset. Each file = one trace with N concurrent people, `PCxy_nN.mat` [E]. Only the **PC** (people-counting) folders are present locally.

**Variable:** each MAT-v5 file contains exactly one variable, `csi_buff` [F].

| Field | Raw/derived | Shape | Type | Meaning | Unit & range | How collected | Label | Quality | Evidence | Conf. |
|---|---|---|---|---|---|---|---|---|---|---|
| `csi_buff` rows | **Raw** | N_rows = 159,648 – 207,496 | complex, stored as int16 real + int16 imag | One row = one CFR snapshot from **one receive chain**; rows cycle through the 4 RX chains: row 4p+c = packet p, chain c | ADC units, real & imag in [−2047, 2047] | Nexmon CSI on monitor router, converted to MAT by Nexmon MATLAB script [E] | file-level | Lag-4 amplitude correlation 0.82–0.9998 vs lag-1 0.05–0.44 → 4-way interleave **verified** [F]. Matches SHARP code `csi_buff[stream::n_tot]` [E]. `PC1a_n05` has 178,197 rows (not ÷4) → one partial packet [F]. Saturation at ±2047: 0.5–4.5 × 10⁻⁶ of values [F] | [F] [E] | High |
| `csi_buff` columns | Raw | 256 | — | The 256 FFT bins of the 80 MHz channel, **natural FFT order** (bin 0 = DC) | — | — | — | Low-energy bins: 1, 123–133, 255 (13 bins) [F]. SHARP applies `fftshift`, then deletes 14 bins [0–5, 127–129, 251–255] = 11 guard + 3 DC → **242 usable tones** (−122…−2, 2…122) [E]. SHARP also **negates columns 64: (after fftshift)** — a hardware sign artefact documented on the DataPort page [E] | [F] [E] | High |
| filename `nNN` | **Label** | per file | int | Number of concurrent people | 0, 2–10 (n01 absent) | Experiment design [E] | — | `PC2a_n00.mat` = a single all-zero 1×256 uint8 row → no data [F] | [F] [E] | High |
| folder `PC-ka` | Meta | — | — | Setup k, campaign a | k = 1…4 | — | — | Meaning of each setup (room / device positions) is in `set_names.pdf`, **not present locally** [U] | [E] | — |

**Not stored:** timestamps, RSSI, MAC, sequence numbers, frame type. The packet rate (173 pkt/s, 6 ms) is a published value [E] and cannot be checked from the files.
**Example tensor (row 0 of `PC1a_n02.mat`, bins 0–7, real/imag):** `[-13,-47] [-6,5] [394,-93] [384,-174] [347,-256] [299,-319] [229,-331] [158,-347]`. Bin 1 is near zero (DC neighbour); bins 2+ carry signal.
**Model-ready view:** `(packets, 4 RX, 242 tones)` complex → amplitude and sanitised phase.

---

## 3. CountFi (`CountFi/48/...`)

**What it is:** raw Nexmon-CSI packet captures from a Raspberry-Pi-class receiver on a Hong Kong double-decker bus [E]. Locally only the **2025-04-08 "mixed"** capture day is present [F]+[E].

**File = 1,000 captured CSI packets ≈ 1 s** [F]. Path grammar: `48/48_<rx>_<cond>/<n>/<k>.pcap`.

| Field | Raw/derived | Shape | Type | Meaning | Unit & range | How collected | Label | Quality | Evidence | Conf. |
|---|---|---|---|---|---|---|---|---|---|---|
| pcap record | Container | 1 per packet | Ethernet/IPv4/UDP | Nexmon forwards each CSI to UDP port 5500 (192.168.1.24 → 192.168.1.50) | 1,084-byte frames, payload 1,042 B | tcpdump on the Pi [E] | — | 100% of 4,740,000 packets are Nexmon-CSI (magic 0x1111) [F] | [F] | High |
| `magic` | Meta | 2 B | uint16 | Nexmon marker | 0x1111 | — | — | — | [F] | High |
| `rssi` | **Raw** | 1 B | int8 | Packet power | dBm, file means −58.5 … −49.9 | Nexmon header | — | — | [F] | High |
| `fctl` | Meta | 1 B | uint8 | Frame-control byte of the sniffed frame → frame type | 0x94 Block-Ack 52.9%, 0x88 QoS-Data 45.2%, 0x50 Probe-Resp 1.1%, 0x80 Beacon 0.7%, 0x08 Data 0.04% | Nexmon header | — | **Different frame types carry structurally different CSI** (next row). | [F] | High |
| `src_mac` | Meta | 6 B | — | Transmitter of the sniffed frame | single MAC `80:cc:9c:bf:ab:de` | Nexmon MAC filter | — | — | [F] | High |
| `seq` | Meta | 2 B | uint16 | 802.11 sequence number | ~42% unique per file | — | — | Block-Acks repeat values; not a loss counter | [F] | Medium |
| `core/ss` | Meta | 2 B | uint16 | RX core and spatial stream | always 0 → **1 RX antenna, 1 stream** | — | — | — | [F] | High |
| `chanspec` | Meta | 2 B | uint16 | Channel spec | 0xe02a = **5 GHz, channel 42, 80 MHz** in 100% of packets | — | — | — | [F] | High |
| `chip` | Meta | 2 B | uint16 | Chip ID field | 0x0065 | — | — | Mapping 0x0065 → chip model not confirmed [U] | [F] | — |
| CSI | **Raw** | 256 complex | int16 LE (real, imag) pairs | Channel response per 80 MHz FFT bin | int16 | Nexmon | — | Decode format **chosen by experiment**: int16 pairs give a smooth spectrum (median adjacent-bin step 2–3%) with VHT80 guards 123–133 on QoS-Data; Nexmon packed-float decode gives noise (step 200%) [F]. **Block-Ack frames show 4×20 MHz non-HT-duplicate structure** (nulls at bins 3–5, 32, 59–69, 96, 123–133, 160, 187–197, 224, 251–255) → must be filtered out or modelled separately [F] | [F] [I] | High (decode); Medium (frame-type interpretation) |
| pcap timestamp | Meta | per packet | µs | Capture time | median spacing 0.76–0.93 ms → **~1 kHz overall**, ~470 Hz per frame type | host | — | No non-monotonic timestamps, no gap >1 s within files [F] | [F] | High |
| `<rx>` = `head` / `end` | Meta | folder | — | Two receivers recording simultaneously | start-time difference median 0.01 ms, 90% within ±20 ms [F] | — | — | Paper: one receiver at front, one at rear of the upper deck [E]; mapping head/end → front/rear is [I] | [F] [E] [I] | Medium |
| `<cond>` = `fm, mm, mm_v2, mm_v3, ms, rm` | Meta | folder | — | Scenario code | — | — | — | Dataset page: suffix `_xm` = "mixed" with x = section, `_fm` = front section [E] → mm = middle, rm = rear [I]; `ms`, `_v2/_v3` undocumented [U]. "48" undocumented (possibly bus route) [U] | [E] [I] | Low |
| `<n>` subfolder | **Label?** | folder | int | Numbers 1–8 and 10 (9 and 0 absent) | — | — | — | Each subfolder is a **contiguous time block**, and the number changes along the trip (e.g. `mm`: 6→7→5→4→3→4→7→8→10→5→1→4→2) [F] → consistent with a passenger count as people board and alight [I]. **Not documented anywhere** [U] | [F] [I] | Low |

**Full decode of all 4,740,000 packets** (`scripts/08c_countfi_full_decode.py`, 2026-10-19) [F]:
- **Container integrity:** 0 truncated records, 0 non-IPv4/UDP-5500 packets, 0 non-Nexmon payloads, 0 non-monotonic timestamps. Every packet's payload is 18 + 1,024 bytes. 10 pcaps are empty and 4 are header-only.
- **No dead CSI:** no all-zero CSI packets, no consecutive identical payloads, no int16 saturation.
- **Junk bins to drop:**
  - Bins 0–1 (DC region) hold large non-channel values in every packet (~2 values > 8,192 per packet, e.g. `−13392−32640j`).
  - Bins 125–131 (inside the guard band) hold **constant** values that never change over time, in both receivers.
  - Removing the standard VHT80 nulls (bins 255, 0, 1 and 123–133) removes all of these and leaves **242 usable tones**.
- **`head` vs `end`:** the constant guard-bin values happen to be zero for `end` but non-zero for `head`. That is why a naive "guard must be ~0" test flags all 2,370 `head` files. It is an extractor artefact, not a faulty receiver. The two receivers also have different gain/frequency profiles (receiver-specific), so per-receiver normalisation is needed.
- **Frame-type smoothness:**
  - QoS-data frames are smooth: median adjacent-bin step 1.2–4.5% per file.
  - Block-Ack frames are rougher at 7–21%, as expected for non-HT duplicate CSI.
  - Every file has both types.

**Dataset-page re-check (2026-10-19, full page text + download-link targets)** [E]:
- **Your copy is the complete April 2025 release.** The page's "Download CSI Data (Apr 2025)" button resolves to **`48.zip`** on the authors' OneDrive, which matches the local top folder `CountFi/48/`. "48" is just the archive name; its meaning is not stated.
- **The labels are almost certainly the subfolder numbers.** The page says "All three datasets include CSI data with corresponding ground truth passenger counts" and that it provides the "raw dataset". The April 2025 download contains only pcaps, so the only place a count can live is the folder structure. Together with the contiguous-time-block evidence above, the subfolder number = passenger count is now **Inferred, medium–high confidence**. It is still not stated in words.
- **Condition codes:** `_xm` means "mixed moving + stationary", where x = upper-deck section (front / middle / end). So `fm` = front, `mm` = middle, and `rm` = rear/end [I]. `ms`, `mm_v2` and `mm_v3` are not explained.
- **The page gives no hardware, channel or bandwidth details.** The 5 GHz / ch 42 / 80 MHz / single-chain values come from the files [F].
- **Other releases (not downloaded):**
  - Feb 2025 stationary set: `26_27_data.zip`.
  - Jun 2023 engine-off benchmark set: `613_20People.zip`, 12,000 samples per pcap. This is the set used in the paper (counts up to 20).
  - The "Download Full Dataset" Google Drive link returns **Page not found**.

**Not reconstructable locally:** the ground-truth log that the page says was "manually recorded and synchronised with CSI timestamps" [E] is not in the download.

---

## 4. WiMANS amplitude subset (`Wiman/amp/act_<g>_<s>.npy`)

**What it is:** preprocessed amplitude tensors from WiMANS [E]; **3,057 of the 11,286 published samples**, amplitude only, **no labels** [F].

| Field | Raw/derived | Shape | Type | Meaning | Unit & range | How collected | Label | Quality | Evidence | Conf. |
|---|---|---|---|---|---|---|---|---|---|---|
| array axis 0 | Derived (from raw CSI) | T = 2,772 – 2,986 (sample of 77) | float32 | **Time (packets)**, nominally 3,000 = 3 s × 1 kHz; shortfall = packet loss | — | Intel 5300 CSI Tool, monitor mode [E] | — | Lag-1 autocorrelation along axis 0 = 0.37–0.95 for groups 19–24, 100–108 but **≈ 0 (−0.02 … 0.16) for groups 1, 10–18, 201–206** [F]: in those groups, consecutive rows are uncorrelated. Cause unknown [U] (band, environment, or packet ordering) | [F] [E] | High (axis); Low (cause of anomaly) |
| axis 1 | — | 3 | — | TX antenna | — | — | — | Paper order is time × TX × RX × subcarrier [E]; axis 1 vs axis 2 cannot be separated from the values alone | [E] | Medium |
| axis 2 | — | 3 | — | RX antenna | — | — | — | — | [E] | Medium |
| axis 3 | — | 30 | — | Intel 5300 grouped subcarriers | — | — | — | Smooth along this axis (lag-1 0.81–0.87) [F] | [F] [E] | High |
| values | **Derived amplitude** | — | float32 | \|H\| of raw int8 I/Q | [0, 179.605] | `preprocess.py` from `.mat` [E] | — | Maximum exactly 179.605 = √(127²+127²) → **un-normalised magnitude of 8-bit I/Q, clipping visible** [F]+[I]. Small number of exact zeros (13–41 per file). No NaN/Inf in 77 sampled files [F] | [F] | High |
| filename `act_<g>_<s>` | ID | — | — | `<group>_<sample>` per paper [E]; 31 groups × 99 samples (act_24: 87) | g ∈ {1, 10–24, 100–108, 201–206} | — | — | `act_24_89.npy` listed as failed in `validation_summary.txt` and now **absent** [F] | [F] [E] | High |
| `annotation.csv` | **Labels** | — | — | number_of_users (0–5), environment, band, user identities, locations, activities [E] | — | — | — | **Not downloaded** → zero labels available locally [F] | [F] [E] | High |

**Not reconstructable:** phase (only amplitude stored), raw complex CSI (`wifi_csi/mat/` absent), labels.

---

## 5. IIT-BHU multi-human activity (`IIT-BHU/wifi_dataset_Collected/`)

**What it is:** ESP32 (NodeMCU) CSI logged with the ESP32-CSI-Tool [E]. Folders give the number of users in the scenario (1/2/3) [E]; filenames give user IDs and activities [F].

**Row = one received packet** [F]. **Example** (`single_user/User_29_run (1).csv`, line 1, CSI truncated):
```
CSI_DATA,STA,3C:71:BF:FD:03:79,-55,11,0,0,0,0,0,0,0,0,0,-96,0,6,1,26926,0,34,0,0,0.211047,128,
[34 32 2 0 0 0 0 0 0 0 0 0 -19 0 -20 -1 ... 0 0 ],1725261757.795135
```

| Pos | Field | Raw/derived | Type | Meaning | Observed values | Evidence | Conf. |
|---|---|---|---|---|---|---|---|
| 0 | type | meta | str | Tag `CSI_DATA` | constant (one garbled `7CSI_DATA`) | [F] | High |
| 1 | role / project type | meta | str | `STA` → receiver ran the tool's station firmware | constant | [F] [E] | High |
| 2 | mac | meta | str | Transmitter MAC of the frame | 3 dominant TX MACs across files (Espressif OUIs 3C:71:BF, A4:CF:12); per file the dominant MAC holds median 98.5% (min 55.7%) of rows; other MACs are stray frames | [F] | High |
| 3 | rssi | **raw** | int | dBm | ≈ −61 … −52 typical | [F] | High |
| 4–13 | rate, sig_mode, mcs, bandwidth, smoothing, not_sounding, aggregation, stbc, fec_coding, sgi | meta | int | PHY descriptors | sig_mode = 1 (HT) in ~96%, 0 (legacy) ~4%; bandwidth = 1 (40 MHz) on HT rows; mcs 0–7 varies | [F] [E] | High |
| 14 | noise_floor | raw meta | int | dBm | −96 … −92 | [F] | High |
| 15 | ampdu_cnt | meta | int | A-MPDU count | 0–6 | [F] | High |
| 16–17 | channel, secondary_channel | meta | int | Channel 6, secondary = 1 (above) → **HT40 on ch 6** | some rows report ch 1/2/11 (stray frames) | [F] | High |
| 18 | local_timestamp | meta | int | ESP32 µs clock | 4 files have resets | [F] | High |
| 19–21 | ant, sig_len, rx_state | meta | int | — | ant = 0 | [F] | High |
| 22–23 | real_time_set, real_timestamp | meta | int/float | Tool's steady-clock fields | — | [F] [E] | Medium |
| 24 | len | meta | int | CSI byte count | **384** (96.8% of rows), **128** (3.2%), + ~40 odd lengths (corrupted lines) | [F] | High |
| 25 | `[...]` CSI | **raw** | int8 list (space-separated) | 384 = LLTF 64 + HT-LTF 128 tones (HT40, above) × 2 bytes; 128 = LLTF only [E] | Valid range int8; corrupted lines contain values up to 3,619 | [F] [E] | High |
| 26 | host time | meta | float | **Unix epoch seconds appended by the logging PC** (not part of the tool's format) | 2024-08-31 … 2024-09-22 | [F] [I] | High |
| folder | **label** | — | int | Users in the scenario | 1 / 2 / 3 | [E] [F] | High |
| filename | **label** | — | str | `User_<ids>_<activities> (<rep>).csv`; activities ∈ {bend, clap, move, pen, run, stretch, wave}; multi-user files list one activity per user (e.g. `bend_run`) [I] | — | [F] [I] | Medium |

CSI positions that are always zero (384-value rows): 3–11, 64–65, 118–127 [F]. These are the ESP32's invalid first word plus guard/DC tones in the ESP-IDF ordering [E]+[I]. Pair order inside each tone is imaginary-then-real per Espressif [E].
**Not stored:** per-person positions; no 0-person recordings.

---

## 6. "CSI HAR" (`CSI HAR/room_<r>/<s>/`) = Zhuravchak & Kapshii HAR dataset

**What it is:** one person performing scripted activities in 3 rooms. CSI comes from a TP-Link TL-WDR4300 router pair running the Atheros CSI Tool: 5 GHz, channel 60, 40 MHz, 2 TX × 2 RX, 114 subcarriers [E]. Every packet is labelled with an activity and a camera bounding box [E].

**`data.csv` — headerless, 1,026 numbers per row, row = one CSI packet** [F]+[E]

| Columns | Raw/derived | Shape | Meaning | Range | Evidence | Conf. |
|---|---|---|---|---|---|---|
| 0–113 | meta | 114 | Constant `0,1,…,113` in **every** row (67,449 rows checked) → subcarrier index vector, carries no information | exact | [F] | High |
| 114–569 | **derived amplitude** | 4 × 114 | \|H\| for 4 TX-RX antenna pairs × 114 tones | 0 … 577.7 | [F] | High |
| 570–1025 | **derived phase** | 4 × 114 | ∠H in radians, raw (unsanitised) | [−π, π] | [F] | High |

- Amplitude and phase are the exact polar form of **integer I/Q**: amplitude² is an integer, and I = A·cos φ and Q = A·sin φ round to integers in 100% of sampled values [F]. The original complex CSI is therefore **losslessly recoverable**.
- \|I\|, \|Q\| ≤ ~404 [F]. That is consistent with Atheros 10-bit signed CSI (±511) [I].
- Which block is which antenna pair (tx1rx1, tx1rx2, …) is **not documented** [U].
- The four blocks have different mean levels (e.g. 105 / 115 / 195 / 84 in `room_1/1`) [F].
- **No timestamps, RSSI or packet counter** [F] → the sampling rate is unknown [U].

**`label.csv`** — `index,label` [F]. Labels: walking, sitting, standing, lying, get_down, get_up, no_person.
- It is **one row shorter than `data.csv` in all 10 sessions** [F], so the alignment of the last row (or an off-by-one at the start) is ambiguous [U].
- Labels come in contiguous segments; lengths are multiples of 5, and the session medians are 110–330 rows [F].

**`label_boxes.csv`** — `(index), x, y, width, height` [F]: the person's bounding box in camera-image pixels [E].
- It is missing for `room_2/1` [F].
- Its row count equals data rows +0 … +8 depending on the session [F], so alignment is unverified.
- `no_person` rows still carry box values [F] (meaning unclear [U]).

---

## 7. Haron98 "WiFi CSI Router Reception Dataset" (`Haron98_WIFI-CSI-Dataset/`)

**What it is:** three ESP32 receivers (model not stated) passively logged CSI from one router's traffic. There are 4 subjects × 4 unnamed labels × 100 trials of ~4 s each [D].

**Line = one received frame** [F]. **Example** (`id_person_01/label_00/test_01/test1__dev1_64_E8_33_57_AA_F4.data`, line 2, CSI truncated):
```
18.04.2026 22:44:41.143 +03:00 CSI_DATA,328252,50:ff:20:e5:a8:f3,-43,11,1,7,0,1,1,1,1,0,0,-90,0,6,0,669572834,0,83,0,128,0,"[0,0,-39,16,-37,13,...]"
```

| Field | Raw/derived | Meaning | Observed | Evidence | Conf. |
|---|---|---|---|---|---|
| host timestamp | meta | PC log time, UTC+3, ms | 2026-04-18 20:17:59 → 23:21:01 | [F] | High |
| seq | meta | logger counter | step 1 (474,598×), 3 rare jumps | [F] | High |
| mac | meta | transmitting router | constant `50:ff:20:e5:a8:f3` | [F] | High |
| rssi | **raw** | dBm | mean −48 … −50 by label | [F] | High |
| rate, sig_mode, mcs, bandwidth, smoothing, not_sounding, aggregation, stbc, fec_coding, sgi | meta | PHY descriptors | sig_mode = 1 (HT) 99.9%; MCS 7 (99.4%); bandwidth 0 (20 MHz); stbc = 1 | [F] | High |
| noise_floor | raw meta | dBm | −98 … −90 | [F] | High |
| channel / secondary | meta | — | 6 / 0 | [F] | High |
| local_timestamp | meta | ESP32 µs clock | wraps negative (32-bit) [D] | [D] | High |
| sig_len | meta | — | 83 (99.8%), 24, 118 | [F] | High |
| len | meta | CSI length | 128 | [F] | High |
| CSI list | **raw** | 64 tones × (imag, real) [E] | int, [−92, 87]; pairs 0 and 27–37 always zero (DC and guard band in ESP-IDF LLTF order `0…31, −32…−1`) → **52 usable tones** | [F] [E] | High |
| folder `id_person_XX` | meta | subject ID | 01–04 | [D] | High |
| folder `label_XX` | **label (undefined)** | class 00–03 | README: "TODO" | [D] | High (that it is undefined) |
| filename `devK_<MAC>` | meta | receiver 1–3 | 3 receiver MACs 64:E8:33:… | [F] | High |

**Single-trial walk-through** (`id_person_01/label_00/test_01`, all 3 receivers, plus `label_01/test_01/dev1`; re-downloaded from Hugging Face on 2026-10-19 after the local deletion; `scripts/11_haron_file_walkthrough.py`) [F]:

| | dev1 | dev2 | dev3 | label_01 dev1 (comparison) |
|---|---|---|---|---|
| Valid lines | 100/100 | 99/100 (line 1 has two log lines spliced together) | 99/100 (line 1 has 26 fields, spliced) | 100/100 |
| Start time (UTC+3) | 22:44:41.10 | 22:44:41.07 | 22:44:41.09 | 20:17:59.60 (2 h 27 min earlier) |
| Duration / rate | 4.06 s / 24.6 fps | 4.02 s / 24.6 fps | 3.30 s / 30.0 fps | 3.88 s / 25.8 fps |
| RSSI / noise floor | −44 / −90 dBm | −43 / −90 | **−62 / −98** | −49 / −90 |
| Raw CSI range | ±43 | −50…44 | **−9…8** | −26…23 |
| Mean amplitude | 23.7 | 23.5 | **2.3** | 13.6 |
| Mean rel. temporal std | 0.12 | 0.18 | **0.64** | 0.21 |

- **Constant metadata:** every metadata field is constant within each file (HT, MCS 7 except 3 MCS-5 frames on dev3, 20 MHz, ch 6, STBC, sig_len 83), so these fields carry no per-frame information.
- **Subcarrier layout:** decoded with the ESP-IDF layout (64 tones, order 0…31, −32…−1, imaginary first). The always-zero tones are exactly **−32…−27, 0, 27…31** (guards + DC), leaving **52 usable tones (−26…−1, 1…26)**. This confirms the 802.11 20 MHz layout.
- **Receivers are not equal:**
  - In this trial, **dev3 is weak and heavily quantised.** Its I/Q values span only ±9, giving amplitudes of 1–6 units, so its "variation" is mostly quantisation noise.
  - dev1 and dev2 have similar strength but different frequency profiles. Their profile correlation is only +0.59, because each receiver sits at a different position.
  - Treat each receiver as its own channel, and check signal strength before using a receiver's data.
- **Line-1 corruption:** the malformed first lines are two serial-log lines spliced together, consistent with logging that started mid-line.

**Data quality:**
- **Malformed lines:** 598 of 480,000 lines (0.125%) are malformed, always line 1 of a file [F].
- **Frame interval:** median 39 ms (≈ 25 Hz), p95 44 ms, maximum 1.36 s [F].
- **Trial length:** median 3.98 s, range 3.26–7.33 s [F].

---

## 8. Feature-count summary

| Dataset | Raw complex CSI available? | Tones per stream (usable) | Streams | Extra per-packet features | Constant/uninformative fields |
|---|---|---|---|---|---|
| WiCount | Yes (int) | 52 (nulls already removed) | 1 | RSSI, local µs timestamp | 22 of 28 columns |
| 80 MHz (Dataset/) | Yes (int16) | 256 bins → 242 usable | 4 RX chains (1 TX) | none | — |
| CountFi | Yes (int16, in pcap) | 256 bins → 242 usable (QoS frames) | 1 | RSSI, frame type, seq, µs time | core, chanspec, MAC constant |
| WiMANS (local) | **No** — amplitude only | 30 | 3 × 3 = 9 | none | — |
| IIT-BHU | Yes (int8) | LLTF 52 + HT-LTF 114 usable (HT40) | 1 | RSSI, noise floor, PHY mode, µs time | several |
| CSI HAR | Yes, recoverable from amp/phase | 114 | 4 (2 × 2) | none (no time, no RSSI) | 114 index columns |
| Haron98 | Yes (int8) | 52 usable of 64 | 1 per receiver, 3 receivers | RSSI, noise floor, µs time | many PHY fields |

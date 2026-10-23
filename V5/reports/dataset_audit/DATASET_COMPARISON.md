# Dataset Comparison — features, acquisition, labels, compatibility

Audit date: 2026-10-19. Part of the [dataset audit](DATASET_AUDIT_REPORT.md). Evidence tags follow the [feature dictionary](FEATURE_DICTIONARY.md): [F] files, [D] local docs, [E] external, [I] inferred, [U] unknown.

## 1. Side-by-side

| | WiCount | 80 MHz (Dataset/) | CountFi (local) | WiMANS (Wiman/) | IIT-BHU | CSI HAR | Haron98 |
|---|---|---|---|---|---|---|---|
| **CSI hardware** | ESP32-S3 RX; TX router/ESP32 (sources conflict) | Asus/Netgear routers; Nexmon monitor (4 RX) | Raspberry Pi + Nexmon (×2 RX) | Intel 5300 (3×3) | ESP32 NodeMCU (ESP32-CSI-Tool) | TP-Link TL-WDR4300, Atheros CSI Tool (2×2) | 3 × ESP32 (model unknown) |
| **Raw complex CSI** | Yes | Yes | Yes (in pcap) | **No** (amplitude only) | Yes | Yes (recoverable from amp+phase) | Yes |
| **Band / BW / channel** | 2.4 GHz / 20 MHz / ch 12 | 5 GHz / 80 MHz / ch 42 | 5 GHz / 80 MHz / ch 42 | 2.4 + 5 GHz / 20 MHz / ch 12, 64 [E] | 2.4 GHz / 40 MHz (HT40) / ch 6 | 5 GHz / 40 MHz / ch 60 | 2.4 GHz / 20 MHz / ch 6 |
| **PHY of measured frames** | Legacy (non-HT) | VHT (iPerf data) | VHT QoS-data **mixed with** non-HT-duplicate Block-Acks | HT | HT40 (96%) + legacy (4%) | HT | HT MCS 7 |
| **Tones per stream** | 52 (nulls removed, indices undocumented) | 242 of 256 | 242 of 256 (QoS) | 30 (grouped) | LLTF 52 + HT-LTF 114 | 114 | 52 of 64 |
| **Streams** | 1 | 4 RX | 1 | 9 | 1 | 4 | 1 per receiver (3 receivers) |
| **Tensor per sample (as stored)** | rows × 104 int | (4·P) × 256 complex int16 | packets × 256 complex int16 | T × 3 × 3 × 30 float32 | rows × 384 (or 128) int8 | rows × (114 idx + 4×114 amp + 4×114 phase) | 100 × 128 int |
| **Packet rate (observed)** | ~83 Hz (median spacing 12 ms) | 173 Hz [E] (no timestamps) | ~1 kHz total, ~470 Hz per frame type | 1 kHz nominal; ≈955–977 Hz effective after 2.3–4.5% loss [E]; T = 2,772–2,986 per 3 s [F] | 4–52 Hz, median 28 Hz | **unknown** | ~25 Hz |
| **Timestamps** | ESP32 µs + host ms | none | pcap µs | none (implicit index) | ESP32 µs + host epoch | none | host ms + ESP32 µs |
| **RSSI** | yes | no | yes | no | yes | no | yes |
| **Label type** | People count, per recording | People count, per recording | Subfolder number (count?, unverified) | none locally (published: users 0–5 + activities) | Users in scenario (1–3) + activity combo, per file | Activity per packet incl. `no_person` | 4 undefined classes |
| **People-count range present** | 0–3 | 0, 2–10 | 1–8, 10 (if counts) | — | 1–3 | 0–1 (implicit) | — |
| **Missing count classes** | none in 0–3, but 0 and 3 have one file each | **1** | 0, 9, 11–20 | all (no labels) | 0 | ≥ 2 | — |
| **Recording structure** | 15 × 60 s continuous | 39 × 4–5 min continuous (4 setups) | 4,740 × 1 s segments of continuous trips, 2 receivers | 3,057 × 3 s isolated samples | 925 isolated files, 12–277 s | 10 continuous sessions, scripted activity segments | 1,600 × 4 s trials × 3 receivers |
| **Environments** | 1 room, 1 evening | 4 setups (identities undocumented locally) | 1 bus, 1 afternoon, 6 scenario codes | 3 rooms (published) | 1 room, 7 days | 3 rooms | 1 room, 1 evening |
| **Pre-applied processing** | none | none | none | amplitude extraction | none | polar conversion | none |
| **ESP32-S3 reproducible?** | **Yes**: same chip family, same CSI format | No: 5 GHz, 80 MHz, 4 RX | No: 5 GHz, 80 MHz | No: Intel 5300, 3×3, 1 kHz | **Largely**: ESP32 CSI format; LLTF part directly, HT40 HT-LTF supported on S3 [E][I] | No: 5 GHz, 2×2, Atheros | **Yes**: ESP32 LLTF 20 MHz |
| **Main unknowns** | 52-tone index map; TX hardware; 0/3 classes = single file each | `set_names.pdf`; why n01 missing | subfolder semantics; condition codes; chip ID 0x0065 | labels; low temporal autocorrelation in 16 groups | room layout; duplicate files | sampling rate; stream order; label off-by-one | label meanings; room |

## 2. Task suitability (evidence-backed)

| Dataset | Direct people count | Presence (0 vs ≥1) | Motion | Entry/exit events | Activity recognition | Reason |
|---|---|---|---|---|---|---|
| WiCount | **Yes (0–3)**, but statistically weak | Yes, from 1 empty file only | No labels | No | No | Explicit count per recording [D]. Only 15 recordings, 1 room, 1 evening; counts recorded in time order 0→1→2→3, so **count is confounded with elapsed time** [F] |
| 80 MHz | **Yes (0, 2–10)**, strongest labels | Yes (n00 in 3 setups) | No | No | No (AR folders not downloaded) | Documented `nN` naming [E]; 4 setups give a natural setup-held-out evaluation. One trace per count per setup; n01 and PC2a_n00 missing |
| CountFi | **Conditional**: only if the subfolder numbers are confirmed as passenger counts | Not testable (no 0 class) | No | Board/alight boundaries **could** be derived from label changes [I] | No | Ground-truth log is not in the download [F]; the folder meaning is undocumented [U] |
| WiMANS | **Blocked locally**. With `annotation.csv`: count 0–5 + per-user activity | Blocked | Blocked | No | Blocked | No labels present [F] |
| IIT-BHU | **Indirect**: 1/2/3-user scenario labels; no empty class | No | No | No | Yes (scenario-level activity combos) | An activity dataset whose folder gives group size [E]. Usable for a 1–3 count *only as a reformulation*; duplicate files and multi-TX logging need cleaning |
| CSI HAR | No (single subject) | **Yes**: per-packet `no_person` vs activity | Yes (walking vs static classes) | Room entry/exit only as label transitions [I] | **Yes** (7 classes) | Per-packet labels [F][E]. Occupancy never exceeds 1 |
| Haron98 | No | Unknown (label_00 is possibly empty — unverified) | Unknown | No | Unknown | Labels undefined [D] |

## 3. Pairwise compatibility (for combining raw inputs)

Classes: **DC** = directly compatible · **CP** = compatible after justified preprocessing · **RN** = related, not directly combinable · **II** = insufficient information.

| | WiCount | 80 MHz | CountFi | WiMANS | IIT-BHU | CSI HAR | Haron98 |
|---|---|---|---|---|---|---|---|
| **WiCount** | — | RN | RN | RN | CP | RN | CP |
| **80 MHz** | RN | — | CP | RN | RN | RN | RN |
| **CountFi** | RN | CP | — | RN | RN | RN | RN |
| **WiMANS** | RN | RN | RN | — | RN | RN | RN / II (labels) |
| **IIT-BHU** | CP | RN | RN | RN | — | RN | CP |
| **CSI HAR** | RN | RN | RN | RN | RN | — | RN |
| **Haron98** | CP | RN | RN | RN / II | CP | RN | — |

No pair is **directly compatible**. Every pair differs in at least one of: tone set, channel, packet rate, PHY frame type, or label semantics.

### Evidence for each non-trivial classification

- **WiCount ↔ IIT-BHU — CP.**
  - Both are ESP32-family `CSI_DATA` logs at 2.4 GHz, and the first 128 bytes of IIT-BHU rows are the same LLTF field that WiCount and Haron98 report [F][E].
  - Required preprocessing:
    - Map WiCount's 52 values to tone indices. This is currently undocumented, so it is a blocker until verified, e.g. against an ESP32-S3 capture.
    - Drop IIT-BHU's HT-LTF part, or vice versa.
    - Unify the I/Q order.
    - Resample or window by time to cover 28 Hz vs 83 Hz.
  - Labels overlap only for counts 1–3, and the activity mix differs (IIT-BHU scripted gestures vs WiCount free movement).
  - Channels differ (6 vs 12).
- **WiCount ↔ Haron98 — CP.** Same chip family, both legacy/LLTF 64-tone layout at 20 MHz, 2.4 GHz [F]. Same 52-tone mapping blocker. Haron98's labels are undefined, so combining serves representation learning only.
- **IIT-BHU ↔ Haron98 — CP.** Same ESP32 CSI layout on channel 6 at 2.4 GHz [F]. Rate 28 vs 25 Hz. Haron98's labels are undefined.
- **80 MHz ↔ CountFi — CP.**
  - Both are Nexmon VHT80 captures on channel 42 with 256 bins and 242 usable tones [F][E].
  - Required preprocessing:
    - Keep only CountFi QoS-data frames.
    - Choose one of the 80 MHz set's 4 RX chains, or use CountFi's single chain.
    - Harmonise rate (173 vs ~470 Hz per frame type) by time-based windowing.
    - Handle the documented sign-inversion artefact of the 80 MHz set's chip, which may not apply to CountFi's (chip ID 0x0065 unresolved).
    - Normalise per-device gain (amplitudes are not calibrated across chips).
  - Environments differ completely (rooms vs bus).
- **ESP32 group ↔ {80 MHz, CountFi, WiMANS, CSI HAR} — RN.** Different band (2.4 vs 5 GHz), bandwidth, tone grid and antenna count; no ESP32-S3 can measure 80 MHz / 5 GHz or 3×3 / 2×2 MIMO. Only coarse summaries (e.g. RSSI or temporal-variance statistics) are conceptually comparable.
- **WiMANS ↔ CSI HAR — RN.** Intel 5300's 30 grouped tones vs Atheros's 114 tones; different rates (1 kHz vs unknown).
- **WiMANS ↔ anything for counting — II** until `annotation.csv` is obtained.

## 4. Label quality ranking (for counting)

1. **80 MHz:** documented, experiment-controlled counts with 4 setups, but only one trace per (setup, count).
2. **WiCount:** documented, but tiny and time-confounded.
3. **WiMANS:** high quality and balanced 2–5 (published), but **absent locally**.
4. **IIT-BHU:** a scenario size rather than an occupancy measurement; contains duplicates.
5. **CountFi:** potentially rich, but **unverified** locally.
6. **CSI HAR:** presence only.
7. **Haron98:** undefined.

## 5. ESP32-S3 relevance ranking

1. **WiCount:** native ESP32-S3; same CSV schema as ESP-IDF `csi_recv`.
2. **Haron98:** ESP32 LLTF 20 MHz; labels unusable.
3. **IIT-BHU:** ESP32 classic, HT40; the LLTF subset maps to S3 output.
4. **All others:** not reproducible on ESP32-S3 hardware. They are useful only for method development or pre-training on a different representation.

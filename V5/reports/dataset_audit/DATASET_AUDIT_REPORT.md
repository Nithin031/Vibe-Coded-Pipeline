# Wi-Fi CSI Dataset Audit — People Counting / ESP32-S3

**Audit date:** 2026-10-19 · **Root:** `D:/wifi/Datasets` · **Scope:** 7 local dataset folders (EHUNAM exists only in Colab notebooks and is excluded).

**Detailed reports:**
- [FEATURE_DICTIONARY.md](FEATURE_DICTIONARY.md): every field, shape, unit and meaning
- [DATA_ACQUISITION_PROTOCOLS.md](DATA_ACQUISITION_PROTOCOLS.md): hardware, channel, rate, environment, labelling
- [DATASET_COMPARISON.md](DATASET_COMPARISON.md): side-by-side comparison and pairwise compatibility

**Tables:**
- [dataset_inventory.csv](dataset_inventory.csv)
- [label_distribution.csv](label_distribution.csv)
- [compatibility_matrix.csv](compatibility_matrix.csv)
- [audit_findings.json](audit_findings.json)
- [audit_errors.log](audit_errors.log)

**Evidence tags:**
- **[F]** verified from file contents
- **[D]** verified from local documentation
- **[E]** external publication or official page
- **[I]** inferred, with reasoning
- **[U]** unknown

No original file was modified, moved or extracted. Nothing was resampled, normalised or trained.

> **Update 2026-10-19:** at your request, `Wiman/` (WiMANS, 9.0 GB) and `Haron98_WIFI-CSI-Dataset/` (229 MB) were **permanently deleted** from `D:/wifi/Datasets`, because you work with them in Google Colab. Their findings below are kept as a record of what the local copies contained. Any further work on them happens in Colab. D: free space rose from 8.3 to 17.8 GB.

---

## 1. Executive summary

- **Datasets discovered: 7.**
  - Fully audited: 6 (WiCount, 80 MHz PC, IIT-BHU, CSI HAR, Haron98, CountFi). CountFi was upgraded after a full decode of all 4.74 M packets; its **labels remain unverified**.
  - Partially audited: 1 (WiMANS).
  - Blocked: 0. WiMANS is blocked *for labels only*.
- **Only two local datasets carry documented, direct people-count labels:**
  - **80 MHz Human Sensing PC folders** (`Dataset/`): counts 0 and 2–10, 4 setups, Nexmon 5 GHz / 80 MHz / 4 RX chains. Best label quality, but **not reproducible on an ESP32-S3**, and **n01 is missing**.
  - **WiCount:** counts 0–3, **native ESP32-S3**, but only 15 one-minute recordings in one room, one evening. Classes 0 and 3 have **one file each**, and the counts were recorded **in time order 0→1→2→3**, so a classifier can learn elapsed time instead of occupancy.
- **CountFi** (5 GHz Nexmon, bus): the local folder is the **complete April 2025 release (`48.zip`)**. The dataset page says every release ships with ground-truth passenger counts, and this download contains only pcaps in numbered folders. So the subfolder numbers (1–8, 10) are almost certainly the counts (medium–high confidence), but the page never says so in words. Confirm with the authors before publishing results.
- **WiMANS** (counts 0–5 published) is unusable locally:
  - Its `annotation.csv` was never downloaded, so the 3,057 amplitude files have **no labels**.
  - Only ~27% of the samples are present.
- **Related-task only:**
  - **IIT-BHU** is an activity dataset with 1/2/3-user scenarios and no empty class. It also contains duplicates.
  - **CSI HAR** is single-person HAR with per-packet `no_person` labels, so it supports presence only.
  - **Haron98** has four undefined labels.
- **ESP32-S3 relevance:**
  - Only WiCount, Haron98 and IIT-BHU use ESP32-family CSI (2.4 GHz LLTF/HT-LTF).
  - The four other datasets use 5 GHz, 40/80 MHz, MIMO hardware (Nexmon, Intel 5300, Atheros) whose feature space an ESP32-S3 cannot produce.
- **Most important quality risks:**
  1. Time/count confounding (WiCount; Haron98 `label_00`).
  2. Simultaneous or overlapping recordings: CountFi head/end receivers and 1-s files cut from continuous trips; Haron98's 3 receivers per trial.
  3. Mixed CSI types within one file: CountFi Block-Ack vs data frames; IIT-BHU HT40 vs legacy rows and stray transmitters.
  4. Missing labels or metadata: WiMANS annotations, CountFi ground truth, 80 MHz setup descriptions, CSI HAR timestamps.
  5. Byte-identical IIT-BHU files under different user IDs.

---

## 2. Dataset inventory (measured)

| Folder | Identity (evidence) | Files | Size | Format | Extraction | Integrity |
|---|---|---|---|---|---|---|
| `CountFi/` | CountFi, Apr-2025 subset [E]+[F] | 4,754 pcap | 4.86 GiB | Nexmon-CSI pcap | complete, no archives | 10 empty + 4 header-only pcaps (mirrored head/end) |
| `Dataset/` | 80 MHz Human Sensing, PC folders [E]+[F] | 40 mat | 5.55 GiB | MAT v5 compressed | complete | `PC2a_n00.mat` holds no data; n01 absent |
| `Wiman/` | WiMANS amplitude subset [E]+[F] | 3,057 npy + 1 txt | 8.93 GiB | NPY float32 | complete (partial download) | `act_24_89.npy` missing; no labels |
| `WiCount/` | WiCount [D]+[E] | 15 csv + 4 py + 2 md | 0.04 GiB | CSV | complete | clean |
| `IIT-BHU/` | IIT-BHU multi-human HAR [E]+[F] | 928 csv | 1.14 GiB | CSV (ESP32-CSI-Tool) | complete | 3 empty files; 11 duplicate pairs; 2,452 garbled lines |
| `CSI HAR/` | Zhuravchak & Kapshii HAR (figshare 14386892) [E]+[F] | 29 csv + 3 `.DS_Store` | 1.10 GiB | headerless CSV | complete | label off-by-one in all sessions |
| `Haron98_WIFI-CSI-Dataset/` | WiFi CSI Router Reception (HF) [D] | 4,800 .data + README | 0.21 GiB | text log | complete | 598 malformed first lines |

Licences and conditions:

| Dataset | Licence / conditions |
|---|---|
| CountFi | CC BY-NC 4.0, cite the WCNC 2025 paper [E] |
| CSI HAR | CC BY 4.0 [E] |
| WiMANS | Academic research only; paper CC BY-NC-SA [E] |
| 80 MHz, IIT-BHU | IEEE DataPort subscription, no licence stated [E] |
| WiCount | None declared [D] |
| Haron98 | Contradictory (YAML says MIT, body says no licence) [D] |

---

## 3. Per-dataset answers to the central question

> *Exactly what information do I have, how was it measured, what does each feature mean, and what can I legitimately learn or predict from it?*

### 3.1 WiCount
- **What you have:**
  - 15 × 60 s CSV logs, 78,618 packets.
  - Per packet: 52 complex subcarriers (104 ints, nulls already stripped), RSSI, ESP32 µs timestamp and host timestamp [F].
- **How it was measured:**
  - Receiver: one ESP32-S3 with 1 antenna, 2.4 GHz channel 12, 20 MHz.
  - Frames: all legacy (non-HT).
  - Rate: ~83 Hz observed against ~100 Hz claimed. The transmitter was a router (paper) or an ESP32-S3 (README) [F][E][D].
- **What the features mean:** CSI = per-subcarrier channel response. Twenty-two metadata columns are constant and carry no information [F].
- **What you can learn:** a 0–3 people classifier. With one recording each for classes 0 and 3 and strict time ordering, any recording-level held-out evaluation is either impossible or measures drift. It is suitable as an **ESP32-S3 target-domain sanity set**, not as a training corpus on its own.
- **Model input / target:** `(window_len × 52)` amplitude (± phase, ± RSSI) → count ∈ {0,1,2,3}. Window from the continuous 60-s file, not resampled.

### 3.2 80 MHz Human Sensing — PC folders (`Dataset/`)
- **What you have:**
  - 39 usable traces: 4 setups × counts {0, 2–10}, minus `PC2a_n00`.
  - Each trace is `csi_buff` with 159k–207k rows × 256 complex int16 bins [F].
- **How it was measured:**
  - TX→RX: iPerf on 802.11ac, ch 42, 80 MHz, single stream.
  - Monitor: Asus RT-AC86U with Nexmon, 4 RX chains, 173 pkt/s [E].
  - Rows cycle over the 4 chains; verified with lag-4 amplitude correlation 0.82–0.9998 vs lag-1 0.05–0.44 [F].
- **What the features mean:**
  - 256 FFT bins, of which 13–14 are null/guard → 242 usable tones [F][E].
  - The upper half of the spectrum needs a sign flip (documented hardware artefact) [E].
  - No timestamps or RSSI.
- **What you can learn:** a people-count classifier/regressor for 0 and 2–10 moving people, evaluated leave-one-setup-out. One person (n01) is missing, so an empty / 2–10 model cannot be checked on single-occupant traffic.
- **Model input / target:** `(window × 4 × 242)` amplitude and sanitised phase → count.

### 3.3 CountFi (local Apr-2025 subset)
- **What you have:**
  - 4,740 one-second pcaps with 1,000 Nexmon CSI packets each.
  - Two simultaneous receivers (`head` / `end`), 6 scenario codes, subfolders numbered 1–8 and 10 [F].
- **How it was measured:**
  - Raspberry-Pi Nexmon receivers on a double-decker bus; 5 GHz ch 42, 80 MHz; single core and stream [E][F].
  - Packet types: 53% Block-Ack (non-HT duplicate CSI) and 45% QoS-data (VHT80 CSI) [F].
- **What the features mean:**
  - CSI: 256 complex int16 bins. The decode was verified by a spectral test [F].
  - Also per packet: RSSI, frame type and µs time.
- **What you can learn:**
  - If subfolders are counts, passenger counting over 1–10 people with RSSI-assisted CSI.
  - Until that is confirmed, only unsupervised or representation learning.
  - Any supervised split must be made by trip block, not by file, and must keep `head` and `end` together.

### 3.4 WiMANS (`Wiman/`)
- **What you have:** 3,057 three-second amplitude tensors `(T≈2,900 × 3 TX × 3 RX × 30)` [F][E], from 31 sample groups.
- **How it was measured:** Intel 5300, 1 kHz, 2.4 and 5 GHz, 3 rooms, 0–5 users doing 9 activities [E].
- **What the features mean:** raw amplitude of 8-bit I/Q, clipped at 179.6 [F]. Phase and raw CSI are absent.
- **What you can learn:** **nothing supervised until `annotation.csv` (small) is downloaded.** After that: user count 0–5 plus per-user activity and location.
- **Unexplained:** 16 of the 31 groups show near-zero temporal autocorrelation [F]. This must be explained before use.

### 3.5 IIT-BHU
- **What you have:** 925 non-empty CSV files (single 300, 2-user 257, 3-user 368), 961k packets, 577 min [F].
- **How it was measured:**
  - ESP32 NodeMCU with the ESP32-CSI-Tool; ch 6, HT40.
  - CSI per row: 384 int8 values (LLTF + HT-LTF) or 128 (legacy).
  - Irregular rate: 4–52 Hz.
  - 3 different transmitter boards across sessions; 7 days [F][E].
- **What the features mean:** per-packet CSI, RSSI, noise floor, PHY flags, ESP32 and host timestamps [F].
- **What you can learn:**
  - Scenario-level activity recognition.
  - A **1–3 group-size** classifier (no empty class). This needs dedup, a dominant-MAC filter and an HT/legacy split.
- **Leakage warning:** the same people appear in several groups, and 11 recordings are duplicated under different user IDs.

### 3.6 CSI HAR (Zhuravchak & Kapshii)
- **What you have:**
  - 10 continuous sessions in 3 rooms, 67,449 packets.
  - Each packet: 4 antenna pairs × 114 tones of amplitude + phase, with an activity label [F][E].
- **How it was measured:** TP-Link TL-WDR4300 routers running the Atheros CSI Tool; 5 GHz ch 60, 40 MHz, 2×2. Labels came from a camera photo per packet [E].
- **What the features mean:**
  - Amplitude/phase are the exact polar form of integer I/Q, so the complex CSI is recoverable [F].
  - The first 114 columns are a constant index.
  - No timestamps, so the sampling rate is unknown [F].
- **What you can learn:**
  - Single-person HAR (7 classes).
  - **Presence detection** (`no_person` vs occupied, 0 vs 1).
  - It cannot train a counter.

### 3.7 Haron98
- **What you have:** 1,600 trials × 3 ESP32 receivers × 100 frames of 64-tone LLTF CSI, plus RSSI and metadata [F].
- **How it was measured:** passive capture of one router's HT MCS7 traffic; ch 6, 20 MHz, ~25 Hz, one evening [F].
- **What the features mean:** standard ESP-IDF CSI. Pairs 0 and 27–37 are always zero (DC and guard tones) [F].
- **What you can learn:** nothing label-supervised, because `label_00…03` are undefined [D].
  - `label_00` behaves like a quieter baseline: higher mean amplitude, lower temporal variance.
  - It was recorded last [F], so this pattern cannot define its meaning.
  - Useful only for ESP32 representation and pre-processing development.

---

## 4. Label and class distribution (summary)

All counts are taken from [label_distribution.csv](label_distribution.csv).

| Dataset | Classes present → recordings | Missing in range | Granularity | Imbalance |
|---|---|---|---|---|
| WiCount | 0→1, 1→7, 2→6, 3→1 files (5,218 / 36,559 / 31,935 / 4,906 packets) | — | per 60-s recording | 7:1 between extremes |
| 80 MHz PC | 0→3 usable, 2…10→4 each (≈ 1,000 s per class at 173 Hz) | **1** | per trace | balanced except 0 |
| CountFi (if counts) | 1: 286 s · 2: 115 s · 3: 334 s · 4: 457 s · 5: 292 s · 6: 212 s · 7: 63 s · 8: 290 s · 10: 379 s (per receiver) | 0, 9, 11–20 | 1-s file within contiguous trip blocks | 7:1 |
| IIT-BHU | 1 user→300 files, 2→257, 3→368 | 0 | per file | mild |
| CSI HAR | 0 (no_person) 8,620 rows · 1 person 58,829 rows | ≥2 | per packet | 7:1 |
| WiMANS | none locally (published: 0:594, 1:3,564, 2–5:1,782 each) | all | per 3-s sample | — |
| Haron98 | 4 undefined classes × 400 trials | n/a | per trial | balanced |

Distinguishing labels:
- **Direct people-count labels:** WiCount and 80 MHz only. CountFi is pending verification.
- **Scenario size:** IIT-BHU.
- **Presence:** CSI HAR.
- **Activity labels:** CSI HAR, IIT-BHU, and WiMANS (published).
- **No entry/exit event labels exist.** CountFi label boundaries could approximate them [I].

---

## 5. Signal quality and temporal consistency

| Dataset | Timing source | Observed rate / regularity | Gaps / duplicates | Values | Phase |
|---|---|---|---|---|---|
| WiCount | ESP32 µs | 12.0 ms median; 38–281 gaps > 20 ms per file; host clock max gap 2 s | timestamps monotonic | int [−36, 36]; no NaN | raw; I/Q order disputed |
| 80 MHz | none | n/a (173 Hz published) | 1 file with an incomplete 4-row group | int16 ±2047, saturation ≤ 4.5e-6 | raw, wrapped; sign artefact |
| CountFi | pcap µs | 0.76–0.93 ms median; no gap > 1 s; monotonic | Block-Ack/QoS interleaved | int16; Block-Ack CSI has 4× sub-band nulls | raw |
| WiMANS | implicit | 2,772–2,986 of 3,000 packets (loss) | 16 groups with ~0 lag-1 correlation | float32 [0, 179.6], clipping | absent |
| IIT-BHU | ESP32 µs + host epoch | 4–52 Hz per file; 15 files host non-monotonic; 4 ESP32 clock resets | 2,452 garbled lines; 40 bad lengths | int8 (corrupt lines up to 3,619) | raw |
| CSI HAR | none | unknown | no duplicate consecutive rows; no NaN | 10-bit-range ints in polar form | raw [−π, π] |
| Haron98 | host ms + ESP32 µs | 39 ms median (33–44 ms p5–p95), max 1.36 s | 598 malformed first lines | int [−92, 87] | raw |

**Proposed resampling requirements (not performed):**
- The rates span roughly 25 Hz to 1 kHz. Any cross-dataset model must define windows in **seconds**, not packets.
- Gaps should be handled with masks rather than interpolation.
- The 80 MHz set and CSI HAR lack timestamps. For the 80 MHz set, assume the published 173 Hz. For CSI HAR the rate is unknown, so it should only be windowed by packet count.

---

## 6. Compatibility matrix (summary)

Full matrix and reasoning: [DATASET_COMPARISON.md §3](DATASET_COMPARISON.md).

- **No pair is directly compatible.**
- **Compatible after justified preprocessing:**
  - WiCount ↔ IIT-BHU ↔ Haron98, the ESP32 LLTF family. This is blocked on recovering WiCount's 52-tone index map.
  - 80 MHz ↔ CountFi, the Nexmon VHT80 ch 42 family. This needs frame-type filtering, chain selection, sign and gain handling.
- **Every ESP32 ↔ 5 GHz/MIMO pair is related but not directly combinable.**
- **ESP32-S3 deployability:**
  - WiCount: yes.
  - Haron98: yes (format).
  - IIT-BHU: largely (LLTF subset).
  - Others: no.

---

## 7. Leakage, overlap and duplication

| Issue | Dataset | Evidence | Effect on evaluation |
|---|---|---|---|
| Two receivers record the same seconds | CountFi (`head`/`end`) | start times within ~20 ms [F] | Splitting head and end of the same second across train/test leaks |
| 1-s files are consecutive slices of continuous trip blocks | CountFi | contiguous timestamps per subfolder [F] | Random file split = near-duplicate leakage; split by block or condition |
| Three receivers per trial | Haron98 | identical trial IDs in dev1–3 [F] | Keep all 3 devices of a trial on the same side |
| Byte-identical files under different user IDs | IIT-BHU | 11 SHA-1 duplicate pairs [F] | Inflates the data; may place the same recording in train and test |
| Same users in several groups | IIT-BHU | user IDs 31–33, 73–74, 59–61 [F] | Subject leakage across 1/2/3-user classes |
| Count order = time order | WiCount; Haron98 (label_00 last) | timestamps [F] | Models can learn drift, not occupancy |
| Rows from multiple TX devices / PHY modes | IIT-BHU | 1.5% stray MACs, 2.5% legacy rows [F] | Feature distribution shift inside one file |
| Scripted segments within continuous sessions | CSI HAR | label run lengths [F] | Window-level random split leaks |
| Metadata mistaken for data | CSI HAR (114 index columns), WiCount (22 constant columns), `validation_summary.txt`, `.DS_Store` | [F] | Must be dropped before modelling |
| Synthetic / augmented samples | none found | — | — |

---

## 8. Resource and storage audit

- **Disk:** 8.3 GiB free on D: at start, 95% used.
- **RAM:** 15.6 GiB total, ~2 GiB free at start.
- **Temporary disk:** no archives exist, so no extraction space is needed.

| Dataset | On disk (measured) | RAM per representative unit (estimate) | Full in-RAM (estimate) | Local OK? |
|---|---|---|---|---|
| WiCount | 0.04 GiB | <50 MB whole set | <50 MB | yes |
| 80 MHz PC | 5.55 GiB | 1 file complex128 ≈ 850 MB (complex64 ≈ 425 MB); streaming reader used <150 MB | ≈ 30 GB complex128 | yes, with per-file streaming |
| CountFi | 4.86 GiB | 1 MB per pcap | ≈ 4.9 GB int16 | yes, streaming; float32 conversion would double it |
| WiMANS | 8.93 GiB | 3.3 MB per sample | 9 GB | yes, via mmap. Completing the full dataset (raw + video) needs tens of GB → **Colab or cloud** |
| IIT-BHU | 1.14 GiB | <50 MB per file | ≈ 0.75 GB int16 | yes |
| CSI HAR | 1.10 GiB | ≤107 MB per session | ≈ 0.55 GB | yes |
| Haron98 | 0.21 GiB | trivial | ≈ 0.13 GB | yes |

Sampled or metadata-only inspection was sufficient for structure. The full-signal passes that were actually run:
- every row of WiCount, Haron98, IIT-BHU, CSI HAR and the 80 MHz set (streamed);
- every packet header of CountFi;
- every `.npy` header of WiMANS, plus 77 files read in full.

---

## 9. Limitations and unresolved questions

1. **CountFi:** subfolder numbers are very likely passenger counts (page states ground truth is included; the download has nothing else) but not stated explicitly; codes `ms`, `mm_v2`, `mm_v3` and `48` are undocumented; chip ID `0x0065` is unmapped. The signal side is now fully verified: every packet was decoded; the pcaps are clean; bins 0–1 and 125–131 hold non-channel values to drop.
2. **80 MHz:** what setups PC-1a…4a are (`set_names.pdf`); whether people moved during PC traces (secondary source says yes); why n01 is absent; which chip the monitor used.
3. **WiMANS:** labels are absent. The near-zero temporal autocorrelation in groups 1, 10–18 and 201–206 is unexplained.
4. **WiCount:** which 52 tones are stored; TX hardware (the paper and README disagree); whether "multiple receivers" data exist elsewhere.
5. **CSI HAR:** packet rate; order of the 4 antenna-pair blocks; exact label/box alignment (off-by-one).
6. **Haron98:** label semantics.
7. **IIT-BHU:** room geometry; which user performed which activity in multi-user files.
8. **ESP32 I/Q order:** Espressif documents imaginary-then-real, which conflicts with two READMEs. Phase features should be checked on a known capture from your own ESP32-S3.
9. **Paywalled papers not read in full:** the 80 MHz Comms Mag paper and IIT-BHU. Details marked [S]/[U].

---

## 10. Evidence-based recommendations

1. **Fix label availability first (small downloads, with your permission):**
   - WiMANS `annotation.csv` (one CSV) unlocks 3,057 labelled 0–5-user samples.
   - The CountFi ground-truth / naming documentation (ask the authors) decides whether 4.7 M packets are counting data.
   - The 80 MHz `set_names.pdf`.
2. **Treat WiCount as an ESP32-S3 validation set, not a training set.** Its size and time ordering cannot support a trustworthy held-out estimate. Collecting your own ESP32-S3 recordings with **interleaved** count order, several sessions per count, and an explicit 0-person session each day would address both weaknesses.
3. **Use the 80 MHz PC set (and CountFi, if confirmed) for method development:** window design, count-vs-variance relations, leave-one-setup-out protocols. Do not expect weight transfer to the ESP32-S3 input space: bands, bandwidths, tone counts and antenna counts all differ.
4. **For ESP32-family pooling** (WiCount + IIT-BHU + Haron98), first establish a common 52-tone LLTF representation.
   - Verify WiCount's tone map and the imag/real order on a reference ESP32-S3 capture.
   - Filter IIT-BHU to the dominant transmitter and to HT40 rows, then take the LLTF part.
   - Deduplicate IIT-BHU.
5. **Define every split at recording/block/setup level.** Keep CountFi `head`/`end` and Haron98's three receivers together.
6. **Reframe tasks honestly:**
   - IIT-BHU: "1/2/3-person scenario" classification.
   - CSI HAR: presence detection.
   - Haron98: unlabeled.

---

## 11. Reproducibility

All scripts are in `D:/wifi/reports/dataset_audit/scripts/` and only read the datasets. Intermediate JSON goes to `_work/`. Run with `py -P -E <script> <args>`; `-I` hides the user-installed `dpkt`/`h5py`.

| Script | Purpose |
|---|---|
| `01_inventory.py` | File walk, extensions, small files, SHA-1 duplicates |
| `02_wicount.py` | Every WiCount column, CSI, timing |
| `03_haron.py` | Full parse of Haron98 logs |
| `04_iitbhu.py`, `04b_iitbhu_macs.py` | Full parse of IIT-BHU; transmitter / PHY mix |
| `05_csihar.py` | Streamed parse of CSI HAR sessions and labels |
| `06_wiman.py` | WiMANS headers + memory-mapped sample statistics |
| `07_mat80.py` | Streaming MAT-v5 reader for the 80 MHz files (no full load) |
| `08_countfi.py`, `08b_countfi_decode_test.py` | Header scan of all pcaps; CSI decode test |
| `09_build_tables.py`, `10_findings_json.py` | CSV / log / JSON deliverables |

External reference copies stored: `_work/external/` (figshare readme, CSI HAR paper text). Packages added: `h5py 3.16.0`, `dpkt 1.9.8` (pip `--user`).

### Sources
- WiCount: [CSI-BERT2 paper](https://arxiv.org/abs/2412.06861); local `WiCount/README.md`
- 80 MHz: [IEEE DataPort](https://ieee-dataport.org/documents/csi-dataset-wireless-human-sensing-80-mhz-wi-fi-channels); [SHARP preprocessing code](https://github.com/francescamen/SHARP)
- CountFi: [dataset page](https://guojingtao-1997.github.io/CountFi/); [paper](https://arxiv.org/abs/2410.11400)
- WiMANS: [repo](https://github.com/huangshk/WiMANS); [paper](https://arxiv.org/abs/2402.09430)
- IIT-BHU: [IEEE DataPort record](https://ieee-dataport.org/comment/9122) (DOI 10.21227/e6c6-aa21)
- CSI HAR: [figshare 14386892](https://figshare.com/articles/dataset/Dataset_for_Human_Activity_Recognition_using_Wi-Fi_Channel_State_Information_CSI_data/14386892); [paper](https://eprints.whiterose.ac.uk/id/eprint/184207/)
- Haron98: [HF card](https://huggingface.co/datasets/Haron98/WIFI-CSI-Dataset)
- ESP32 CSI format: [ESP-IDF Wi-Fi vendor features](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-guides/wifi-driver/wifi-vendor-features.html); [ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool)
- Secondary: `D:/wifi/csi_counting_dataset_survey.pdf` (used only where marked [S])

# Vibe-Coded Pipeline - Wi-Fi CSI sensing on the ESP32-S3

| Version | What | Where |
|---|---|---|
| **V5** (2026-10-23, latest) | People counting 0 / 1 / 2 / 3+ from one ESP32-S3 link; trained on WiCount + 80 MHz set + WiMANS, rooms held out; C header for the trees | [`V5/`](V5/README.md) |
| V4 | Presence (empty / occupied) + counting 0-2 in one room; noise suite, "unknown" gate | this folder (deploy) + [`V4/`](V4/) |

V5 does not replace V4's presence model. Head-to-head counting benchmark (V4's own code vs V5, same data):
V5 is better on a clean signal in a new session (exact 0.85 vs 0.78, and it still finds the empty room);
V4's gap-CNN is clearly better under noise (0.73 vs 0.50). Details: [`V5/results/README.md`](V5/results/README.md#v4-vs-v5-benchmark-kaggle-2026-10-23).

---

# V4 deploy folder

The model files and what the ESP32 must send. Everything here is **experimental**: it has
only been tested on recorded files, not on a live ESP32.

```
deploy/
  model_card.json            thresholds, classes, preprocessing, SHA-256 of every file
  infer.py                   run the models on a recorded CSV
  presence/
    lgbm_cal_s0..s2.txt      MAIN presence model (LightGBM, 3 seeds averaged), threshold 0.90
    cnn_cal.pt               presence as a PyTorch .pt (CNN), threshold 0.45: the second choice
  counting/
    clean/count_gapcnn.pt    people count, normal room (standard preprocessing)
    noisy/count_gapcnn.pt    people count, noisy room (robust preprocessing + noise training)
    */count_lgbm_s*.txt      LightGBM versions of the same counters
```

The `.pt` files are PyTorch `state_dict`s. Each has a `.json` file next to it naming its
architecture (`cnn` = V3 CNN1D, `gapcnn` = V4 gap-aware CNN, defined in `V4/src/models.py`).
Use `infer.py` (or `models.load_net`) instead of loading them by hand, because the input must be
prepared exactly as in training.

## What it can detect

| What | Can it? | How good (tested) |
|---|---|---|
| **Room empty or occupied** | **Yes**, the main feature | **0.98** balanced accuracy in a new home and a new office it never saw. In other homes / chips: 0.78–0.85, rarely false alarms (0–3 %) but **misses 30–43 %** of occupied windows |
| **Someone sitting or sleeping (not moving)** | **Yes**, counted as "occupied" | in the home test, sleep and desk work were both detected (about 1 % missed) |
| **Count 0 / 1 / 2 people** | **Yes, only in the trained room** | about 89 % exact, never off by more than one person |
| **Count 3 people** | **Not really** | almost always says "2" (recall 0–4 %) |
| **Count 4–5 people** | **No** | no training data exists |
| **Bad signal / broken stream** | **Yes**: answers "unknown" | e.g. at 50 % packet loss most windows are reported "unknown" instead of guessed |
| Sleeping vs working, standing vs walking, where in the room | **No** | V4 does not classify activity or location |
| Who the person is | **No** | – |

### Counting 0–2 people in detail

Tested in the WiCount meeting room, on recordings the model never trained on:

| True people | Correctly counted | Wrong answers go to |
|---|---|---|
| 0 | **100 %** | – |
| 1 | **90–98 %** | "2" (rarely "0") |
| 2 | **74–86 %** | "1" |

- **Exact count for 0–2 overall:** about **89 %** (LightGBM 0.886, gap-CNN noisy version 0.895).
- **Mistakes:** only 1 ↔ 2 mix-ups.
- **Empty room:** never counted as people.
- **Evidence for 0 is weak.** There is only one empty recording, so it was tested on time blocks of that same recording.
- **Other rooms:** accuracy drops to **26–53 %**. The model learned "this room", not "people in general".

### In short

- **Ready to try in your room:** empty vs occupied, after a 30 s empty-room reference.
- **Counting 0–2:** works, but only after it is retrained on recordings from **your** room.
- **Beyond 2–3 people:** you need your own recordings with 3–5 people (`V4/docs/COLLECTION_PROTOCOL.md`) or access to the WiFiVision dataset.

## Training data

All of it is public recordings made by other groups; none of it comes from our own ESP32 setup yet.

### Presence models (`presence/lgbm_cal_s*.txt`, `presence/cnn_cal.pt`)

**Training: 9,347 windows of 3 s.**

| Dataset | Where / hardware | Recordings used | Windows (empty / occupied) |
|---|---|---|---|
| ESPectre | real homes (bedroom, living room, hobby room), 5 ESP32 chip types | 7 empty + 10 occupied | 826 / 1,330 |
| HomeOccupancy | one home, 2 x ESP32-C6 | empty_1, sleep_1, work_1 | 979 / 1,551 |
| OfficeLocalization | one office, 2 x ESP32-C6 | empty_1, one_1, two_1, five_1 | 997 / 2,991 |
| esp32-csi-pipeline | room 1, quiet Wi-Fi channel, 2 x ESP32 | empty, stationary, moving | 222 / 451 |

**Validation: 7,968 windows**, used only to pick the threshold (and the CNN's epochs):
- ESPectre's own "selection" recordings
- HomeOccupancy recording 2
- OfficeLocalization empty_3

**Kept for testing only (never trained on):**
- ESPectre holdout sessions
- pipeline on the congested channel and in room 2
- HomeOccupancy recording 3
- OfficeLocalization `_2` files, empty_4 and empty_5

**How the data was weighted and trained:**
- Every dataset gets equal weight, so one large dataset does not dominate.
- LightGBM averages 3 seeds.
- The CNN was trained from scratch.

**Known weakness.** A calibrated model can only learn from a recording whose exact device also
has an empty recording in the training split. Because of this, only 10 of ESPectre's 54 occupied
training recordings could be used. That likely explains why presence misses people in the ESPectre homes.

### Counting models (`counting/*/count_gapcnn.pt`, `counting/*/count_lgbm_s*.txt`)

- **clean:** all of **WiCount**: one meeting room, ESP32-S3, about 87 packets/s. That is 864 windows from 15 one-minute recordings:

  | People | Recordings |
  |---|---|
  | 0 | 1 |
  | 1 | 7 |
  | 2 | 6 |
  | 3 | 1 |

- **noisy:** the same WiCount recordings with robust preprocessing, plus one noisy copy of every recording (random noise type and strength).
- **No data with 4 or 5 people.** Only one room was seen, which is why counting fails in other rooms.

## 1. What the ESP32 must send

The esp-csi `CSI_DATA` output, saved as CSV **with a header row**. The columns used are:

| Column | Needed | Why |
|---|---|---|
| `type` | yes | must be `CSI_DATA` |
| `local_timestamp` | **yes** | ESP32 microsecond clock; it builds the time grid and finds packet gaps |
| `rssi` | yes | signal strength, also used as a feature |
| `len` | yes | CSI buffer length; it must stay the same during a recording |
| `data` | **yes** | the **raw interleaved I/Q integers**, e.g. `"[0,0,17,-6,...]"`. Not amplitudes, not a plot. |
| `seq`, `mac`, `channel`, `noise_floor`, `agc_gain`, `fft_gain`, `rate`, `sig_len`, `rx_state` | keep them | not used by the model now, but needed to diagnose problems |

Accepted buffer lengths:
- `len` = 128 (64 subcarriers, HT20);
- 104 (52 subcarriers);
- 384 (the first 128 values are used).

Run it:

```powershell
$py = "python"          # Python 3.12
& $py V4/deploy/infer.py --reference empty_30s.csv --input room.csv --out result.csv
& $py V4/deploy/infer.py --reference empty_30s.csv --input room.csv --env noisy --out result.csv
```

You get one line per second: `t_end_s, quality (ok/unknown), p_occupied, presence, count`.

## 2. Setup conditions (clean and noisy rooms)

| Condition | Required | What the training data covered |
|---|---|---|
| Link | **one** link: ESP32 to ESP32 (ESP-NOW) or ESP32 to router | single links only |
| Band | 2.4 GHz, 20 MHz (HT20) | HT20 everywhere except one small HT40 set |
| Chip | any ESP32 (classic, S2, S3, C3, C5, C6) | 5 chip types; *new chips are the weakest point* |
| Packet rate | **≥ 100 packets/s** is best. **≥ 50/s** is the minimum. Below 20/s, windows become "unknown" | 87–499 packets/s |
| RSSI | between about **-30 and -65 dBm** | -27 to -64 dBm |
| Distance | 3–6 m apart, line of sight across the room, about 1–1.5 m high | 3–8 m |
| Fixed setup | boards taped down; channel, bandwidth and firmware fixed | – |
| **Empty reference** | **30 s (better 60 s) with nobody in the room**, made after installing. **Redo it** after moving a board, changing channel or firmware, or moving furniture | the presence model needs it |
| Warm-up | 3 s for the first window, then about 5 s until the smoothed answer is stable | measured 3–5 s |

## 3. What happens with noise (measured on the test data)

| Noise in the room | Presence model | What to do |
|---|---|---|
| Busy Wi-Fi channel (real congested channel tested) | 0 % false alarms, but misses ~30 % of people | pick the quietest channel if you can |
| Packet loss up to 50 % | windows become **"unknown"** instead of wrong | keep the rate ≥ 100 packets/s; check the USB cable / baud rate |
| ESP32 gain (AGC) jumps, USB timing jitter | almost no effect | – |
| Short interference spikes (microwave, Bluetooth) | small effect (0.88 → 0.81) | – |
| Dead or stuck subcarriers | no effect after calibration | – |
| **Fans, AC, machines** (periodic vibration) | light ripple is fine. **Strong ripple causes errors** (0.88 → 0.68) | calibrate **with the fans running as they normally do**; keep boards away from fans |
| **Weak signal** (low SNR) | medium is OK (0.71). **Very weak signal fails** (0.59) | keep RSSI above about -65 dBm; move the boards closer |
| Everything at once | fails (0.55) | – |

Clean or noisy room:
- **Presence:** use the same model for both. It is calibrated to your room's own empty reference, which includes the room's normal background noise.
- **Counting:** `--env clean` for a quiet room, `--env noisy` if there are fans, machines or interference. The noisy model holds up better under noise: 0.56 vs 0.53 exact with everything at once, and 0.84 vs 0.68 with spikes.

## 4. Limits you must know

- **Counting only works in the room it was trained in.** It was trained in the WiCount meeting room (0–3 people) and fails in other rooms. In a test on an occupied office it said "0". **It must be retrained on your own recordings** (`V4/docs/COLLECTION_PROTOCOL.md`).
- **The count stops at 3.** It almost never says 3, and outputs 4 and 5 were never trained.
- **Presence was strongest in rooms similar to its training rooms.** It got 0.98 in a new home and a new office, but missed 30–43 % of people in some other homes. Check it in your room with an empty and an occupied recording before trusting it.
- **"unknown" is a real answer.** It means the packets were too broken to decide. Do not treat it as empty.

# ESP32-S3 people counter - deployment guide

Everything needed to install, calibrate and use the final model (Phase 4, 2026-10-23).
Results behind it: `../PHASE4_RESULTS.md`. Code: `esp32_counter/esp32_counter.py`.

```
model_esp32_counter/
├── README.md          <- this guide
├── trees.json         the model for Python (no LightGBM needed)
├── counter_model.h    the same model for ESP32 firmware (C arrays + cc_prepare() + cc_predict())
├── model.txt          LightGBM native file (for retraining / inspection only)
└── model_card.json    settings, calibration, training data, held-out results, caveats
```

---

## 1. What it does and what to expect

One ESP32-S3 link (a sender and a receiver) measures how much the Wi-Fi channel "wobbles" when people move.
Every 2.5 s the counter turns the last 5 s of CSI into 10 numbers (cues), the model turns them into an estimate,
and the display shows the median of the last 30 s.

| | |
|---|---|
| Output | people count 0-10, new value every 2.5 s |
| **Honest output** | **empty / 1 / 2 / 3+** - show anything >= 3 as "3+" |
| Delay | ~10 s warm-up, then changes appear after ~15-30 s (30 s median) |
| Install step | **required:** 20 s of empty room (section 4) |
| Trained on | WiCount (ESP32-S3) + mat80 (router, 80 MHz) + WiMANS (Intel 5300, 5 GHz) |

Held-out results (rooms / files the model never saw):

| Test data | Error (MAE, people) | Exact | Within ±1 | Empty detected |
|---|---|---|---|---|
| WiCount, ESP32-S3, 0-3 people | 0.29 | 85 % | 100 % | 100 % |
| WiMANS, unseen room, 0-5 people | 1.01 | 31 % | 75 % | 93 % |
| mat80, unseen setup, 0-10 people | 2.67 | 19 % | 38 % | 95 % |

Mean estimate per true count (why "3+"):

| True people | 0 | 1 | 2 | 3 | 4 | 5 | 10 |
|---|---|---|---|---|---|---|---|
| ESP32 (WiCount) | 0.0 | 1.3 | 1.8 | 2.0 | - | - | - |
| WiMANS | 0.1 | 2.0 | 2.6 | 2.8 | 3.0 | 3.1 | - |
| mat80 | 0.1 | - | 4.1 | 4.4 | 4.5 | 4.7 | 5.1 |

Empty vs occupied is reliable. Above 2-3 people the estimates bunch together.

---

## 2. Hardware setups

### Option A - 2 x ESP32-S3 (one link) - **the validated setup**

```
   [ESP32 #1 sender] ~~~~~~~~ room / where people are ~~~~~~~~ [ESP32 #2 receiver] --USB--> laptop / Raspberry Pi
```

- **#1 sender:** sends packets non-stop (ESP-NOW or ping), **>= 100 packets/s** (WiCount ran at ~87/s; 50/s is the minimum).
- **#2 receiver:** CSI enabled, prints one `CSI_DATA` line per packet over serial (the `esp-csi` `csi_recv` format,
  same as WiCount), 20 MHz channel (HT20 / legacy), one antenna.
- A 1 ESP32 + Wi-Fi router variant also works: the ESP32 pings the router at >= 100 Hz and reads CSI from the replies.

### Option B - 3 x ESP32-S3 (two links) - **planned, not yet validated**

```
                      [ESP32 #2 receiver A] --USB--> host
   [ESP32 #1 sender]
                      [ESP32 #3 receiver B] --USB--> host
```

- One sender, two receivers at **different positions** (e.g. two different walls), so the two links cross the room differently.
- Each receiver runs **its own** counter with **its own** empty-room calibration (the offset is per link).
- Final count = mean of the two smoothed estimates, then round and show empty / 1 / 2 / 3+.
- Why: averaging several links cut the error from 1.42 to 1.11 on the 80 MHz data in Phase 2. The gain for this
  exact model on 2 ESP32 links is **not measured yet** (planned Phase 5).

### Placement (recommendations, not tested values)

- Sender and receiver(s) at the same height (~1-1.5 m), 3-6 m apart, with the area where people sit / walk between them.
- **Nothing may move after calibration** - not the boards, not their cables, not big furniture. If anything moves, recalibrate.
- Keep away from fans, AC units, swinging doors and moving machines: they look like people
  (in earlier noise tests fans broke every model).
- Same Wi-Fi channel all the time; a busy channel is tolerable, a fixed channel is required.

---

## 3. What the receiver must deliver (input format)

Per packet: a **time** (seconds, increasing) and the **amplitude of 52 tones**.

| Item | Rule |
|---|---|
| Tones | the 52 used tones of a 20 MHz channel (subcarriers -26...-1 and +1...+26; no DC, no guard tones) |
| Tone order | **does not matter** - none of the 10 cues depend on it - but keep it the same every packet |
| Amplitude | \|CSI\| = sqrt(I² + Q²) per tone; the scale does not matter (each packet is normalised) |
| Time | from the ESP32 `local_timestamp` (µs, 32-bit, wraps every ~71 min - add 2³² on wrap) |
| Rate | >= 50 packets/s, aim for >= 100; gaps up to 60 ms are filled, longer gaps skip windows |

If your firmware prints 128 values (64 subcarriers, `esp-csi` order 0...31, -32...-1), keep subcarriers at positions
1-26 and 38-63. If it prints 104 values (like WiCount), they are already the 52 tones.

Parsing one `CSI_DATA` line (WiCount / `esp-csi` format):

```python
import json
import numpy as np

WRAP = 2 ** 32

def parse_csi_line(fields, last_ts, t_offset):
    """fields: the CSV columns of one CSI_DATA line. Returns (t_seconds, amp52, last_ts, t_offset)."""
    ts = int(fields["local_timestamp"])             # microseconds since boot, 32-bit
    if last_ts is not None and ts < last_ts:
        t_offset += WRAP                            # counter wrapped
    iq = np.array(json.loads(fields["data"]), dtype=np.float32)
    amp = np.hypot(iq[0::2], iq[1::2])              # 52 amplitudes from 104 numbers
    return (ts + t_offset) * 1e-6, amp, ts, t_offset
```

---

## 4. Calibration (required)

**What it does:** measures the empty room for 20 s and stores 10 numbers (the offset) that describe
"no people" for *this* link in *this* room. All later estimates are relative to it.

**How:**
1. Install and switch on everything. **Leave the room - nobody inside, doors closed, fans / AC as they will normally be.**
2. Start the counter and call `begin_calibration()` immediately.
3. Wait 30 s: 10 s warm-up + 20 s calibration (7 windows). The result with `status == "calibrated"` marks the end.
4. Save the offset (`counter.offset`, 10 floats) to a file, so a restart does not need a new calibration.
5. People may enter. Counts appear from now on.

**Recalibrate when:** a board, its antenna or cable moved; big furniture moved; the Wi-Fi channel changed;
the empty room starts showing 1 for a long time. Once a day while testing is a good habit.

**If someone was in the room during calibration:** that person becomes part of "empty", so every count reads too low.
Recalibrate.

**Option B (3 boards):** calibrate both receivers at the same time (same empty 30 s), save one offset per receiver.

---

## 5. Using the model in Python (works today)

The host (laptop / Raspberry Pi) reads the receiver's serial lines and runs the counter.

```python
import sys
import numpy as np

sys.path.insert(0, "esp32_counter")
import esp32_counter as ec

MODEL_DIR = "results/phase4_final__mat80_wicount_wimans_csibench/model_esp32_counter"
model, card = ec.load_bundle(MODEL_DIR)           # checks the settings match esp32_counter.py

# ---- first install: calibrate (room empty) ----
counter = ec.StreamingCounter(model)
counter.begin_calibration()
for t, amp52 in packets():                         # your serial reader: (seconds, 52 amplitudes)
    for r in counter.push(t, amp52):
        if r["status"] == "calibrated":
            np.save("offset_link_A.npy", counter.offset)
        elif r["status"] == "ok":
            n = r["count"]
            print("empty" if n == 0 else ("3+" if n >= 3 else n), f"(raw {r['smoothed']:.2f})")

# ---- later restarts: reuse the saved offset, no calibration ----
counter = ec.StreamingCounter(model, offset=np.load("offset_link_A.npy"))
```

`push(t, amp52)` returns a list (usually empty, one item every 2.5 s). Each item:

| Key | Meaning |
|---|---|
| `status` | `"calibrating"`, `"calibrated"` (last calibration window), or `"ok"` |
| `t_start_s`, `t_end_s` | window time since the first packet |
| `features` | the 10 raw cues |
| `raw` | this window's estimate (0-10, not rounded) |
| `smoothed` | median of the last 30 s of estimates - **use this** |
| `count` | `smoothed` rounded and clipped |

Other calls: `counter.calibrating` (True during calibration), `counter.flush()` at the end of a recording.
Packets must arrive in time order; late packets are dropped.

**Option B:** one `StreamingCounter` per receiver, each with its own offset; average their `smoothed` values.

---

## 6. Running it on the ESP32 itself (firmware) - partly done

| Part | Status |
|---|---|
| Tree model (`cc_predict`) + input preparation (`cc_prepare`) | **done** - `counter_model.h`, checked against Python by the unit tests |
| Settings (50 Hz, 5 s window, 2.5 s hop, EMA 30 s, warm-up, calibration, smoothing) | `#define`s in `counter_model.h` |
| Preprocessing (gain normalisation, 50 Hz bins, gap fill, causal baseline) | **to port to C** (Python reference: `StreamingCounter`) |
| The 10 cues per window | **to port to C** (Python reference: `window_features`) |
| 30 s median + count | **to port to C** (simple) |

Porting rules, so the device gives the same numbers as Python:
- Window = 250 bins x 52 tones, float32. Missing bins count as 0 in the cues.
- `n_pc_90` / `pc1_frac`: eigenvalues of the 52 x 52 covariance of the centred window (Jacobi is fine) =
  squared singular values in Python.
- `low_freq_frac` / `high_freq_frac`: spectrum of the tone-averaged series over **250 points** (bins below 2 Hz /
  above 5 Hz). Do **not** zero-pad to 256 - that changes the numbers. A direct DFT of the needed bins is cheap.
- Check: replay a WiCount CSV through the C code and compare with `StreamingCounter` (should match within ~0.01 people).
- Budget estimate: ~77 KB flash for the trees, ~75 KB RAM, ~3 M operations every 2.5 s (fine for the S3).

In C, per window:

```c
#include "counter_model.h"

float raw[CC_N_FEATURES];      /* the 10 cues of this window, in model_card order */
float x[CC_N_FEATURES];
cc_prepare(raw, offset, x);    /* offset = the 10 calibration floats (NULL = not calibrated) */
float estimate = cc_predict(x);
/* clip to 0..CC_MAX_COUNT, push into the last-12 median, round -> count */
```

The calibration offset on the device = mean of `cc_prepare(raw, NULL, ...)` over the 7 calibration windows.

---

## 7. Known limits

- Only one room of real ESP32-S3 data (WiCount: one evening, 0-3 people; 0 and 3 one file each).
- Counts above 3 come from router / Intel 5300 data only and saturate.
- The cues measure motion: people sitting very still are probably under-counted (not tested).
- Validate on your own recordings before trusting it: several rooms and days, 0-5 people, sitting and walking,
  one empty recording per room, >= 100 packets/s.

## 8. Where everything comes from

| File | What |
|---|---|
| `esp32_counter/esp32_counter.py` | counter code (training + device reference), export, bundle load |
| `esp32_counter/test_esp32_counter.py` | tests: stream == batch, no future leakage, C header == Python |
| `notebooks/kaggle/kaggle_phase4_features.ipynb` | Kaggle: WiMANS + CSI-Bench -> cue tables |
| `notebooks/local/phase4_final_local.ipynb` | pools everything, picks and exports this model |
| `notebook_builders/build_phase4_nbs.py` | generates both notebooks |

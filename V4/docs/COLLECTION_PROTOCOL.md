# How to record ESP32 data for counting 0 to 5 people

No public ESP32 dataset with 4 or 5 people is open to download (see
`manifests/dataset_search.json`). To train and test a real 0–5 counter, we
must record it ourselves. This page says exactly how.

## Equipment

- 2 ESP32 boards (one sends, one receives), the same boards for the whole
  study, or 1 ESP32 receiving from a Wi-Fi router.
- A laptop on the receiving board's USB port, logging raw `CSI_DATA` lines
  (esp-csi format, with `local_timestamp`, `rssi`, `len` and the raw I/Q `data`).
- A phone timer or a video for ground truth (optional but strongly advised).

## Fix the setup first

1. Put both boards at fixed places (tape them down) about 1–1.5 m high,
   3–6 m apart, across the middle of the room.
2. Fix the Wi-Fi channel and bandwidth (for example channel 6, 20 MHz).
3. Write the setup into the manifest (`configs/recording_manifest_schema.json`).
4. Do not move boards, furniture or the channel during a session. If anything
   moves, it is a new session and needs a new empty reference.

## One session

A session is one sitting with the same boards, layout and channel. In each session:

| Step | People in room | Length | Notes |
|---|---|---|---|
| 1 | 0 (empty reference) | 60 s | everybody outside, door closed. The first 30 s are used for calibration. |
| 2 | 0 | 2 min | the empty test recording |
| 3 | 1 | 2 min | |
| 4 | 2 | 2 min | |
| 5 | 3 | 2 min | |
| 6 | 4 | 2 min | |
| 7 | 5 | 2 min | |
| 8 | 0 | 1 min | closing empty check (shows drift during the session) |

- Shuffle the order of steps 3–7 between sessions, so the count is not tied
  to the time of day or to drift.
- Inside each recording, people should mix sitting, standing and walking. Write
  what they did in `people_activity` (one entry per person).
- Nobody enters or leaves during a recording. Stop, change the people, start a new file.

## How many sessions

- At least **6 sessions on at least 3 different days**. Then whole sessions
  can be held out for testing (for example 4 for training, 1 for validation,
  1 for testing, and rotate).
- Better: also 2 sessions in a second room, to measure cross-room behaviour.
- Record at least one session with fans/AC on and one at a busy Wi-Fi time.
  Write it in `traffic_condition` and `notes`.

## File names and manifest

`<room>_<date>_s<session>_c<count>.csv` plus `<same name>.json`, for example
`lab_2026-11-02_s03_c4.csv` and `lab_2026-11-02_s03_c4.json`.

## Check every file

```
python V4/src/validate_recording.py path/to/*.csv
```

A file passes only if:

- its manifest is complete;
- it is at least 120 s long (30 s for the empty reference);
- its packet rate is at least 50 Hz;
- at least 95 % of its 3 s windows are valid.

Fix or re-record files that fail.

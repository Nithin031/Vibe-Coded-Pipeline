"""
"CSI HAR" folder (room_<r>/<session>/data.csv + label.csv [+ label_boxes.csv]).

Verified row layout (see report):  114 ints (0..113, constant index)
                                   + 4 x 114 amplitudes + 4 x 114 phases (rad)
This pass streams every data.csv in chunks and reports per session:
  - rows, columns, NaN / inf counts, index-block constancy
  - amplitude / phase ranges per stream, I/Q integer check on a sample
  - row counts of data vs label vs label_boxes (alignment)
  - label distribution per session and per room
  - run-lengths of labels (how many consecutive rows per activity segment)

Usage:  py -I 05_csihar.py <csihar_root> <out_json>
"""

import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

N_SC = 114
N_STREAM = 4
CHUNK = 500


def label_runs(labels):
    runs = []
    prev, n = None, 0
    for v in labels:
        if v == prev:
            n += 1
        else:
            if prev is not None:
                runs.append((prev, n))
            prev, n = v, 1
    runs.append((prev, n))
    return runs


def main():
    root, out_path = sys.argv[1], sys.argv[2]
    sessions = sorted(glob.glob(os.path.join(root, "room_*", "*", "data.csv")))
    out = []
    room_labels = {}

    for path in sessions:
        sdir = os.path.dirname(path)
        name = os.path.relpath(sdir, root).replace("\\", "/")
        room = name.split("/")[0]
        print("session", name, flush=True)

        n_rows, n_nan, n_inf, idx_bad = 0, 0, 0, 0
        ncols = set()
        amp_min = np.full(N_STREAM, np.inf)
        amp_max = np.full(N_STREAM, -np.inf)
        amp_sum = np.zeros(N_STREAM)
        ph_min, ph_max = np.inf, -np.inf
        iq_int_ok, iq_checked = 0, 0
        iq_absmax = 0.0
        zero_amp_rows = 0
        dup_consecutive = 0
        prev_last = None

        for chunk in pd.read_csv(path, header=None, chunksize=CHUNK, dtype=np.float64):
            a = chunk.values
            ncols.add(a.shape[1])
            n_rows += len(a)
            n_nan += int(np.isnan(a).sum())
            n_inf += int(np.isinf(a).sum())
            idx_bad += int((a[:, :N_SC] != np.arange(N_SC)).any(axis=1).sum())

            b = a[:, N_SC:].reshape(len(a), 2 * N_STREAM, N_SC)
            amp, ph = b[:, :N_STREAM], b[:, N_STREAM:]
            amp_min = np.minimum(amp_min, amp.min(axis=(0, 2)))
            amp_max = np.maximum(amp_max, amp.max(axis=(0, 2)))
            amp_sum += amp.mean(axis=2).sum(axis=0)
            ph_min = min(ph_min, float(ph.min()))
            ph_max = max(ph_max, float(ph.max()))
            zero_amp_rows += int((amp.reshape(len(a), -1) == 0).all(axis=1).sum())

            # exact duplicate consecutive rows (frozen / repeated packets)
            body = a[:, N_SC:]
            if prev_last is not None:
                dup_consecutive += int(np.array_equal(prev_last, body[0]))
            dup_consecutive += int((body[1:] == body[:-1]).all(axis=1).sum())
            prev_last = body[-1]

            s = amp[::25]
            I = s * np.cos(ph[::25])
            Q = s * np.sin(ph[::25])
            ok = np.isclose(I, np.round(I), atol=1e-3) & np.isclose(Q, np.round(Q), atol=1e-3)
            iq_int_ok += int(ok.sum())
            iq_checked += ok.size
            iq_absmax = max(iq_absmax, float(np.abs(I).max()), float(np.abs(Q).max()))

        lab = pd.read_csv(os.path.join(sdir, "label.csv"), header=None, names=["idx", "label"])
        box_path = os.path.join(sdir, "label_boxes.csv")
        n_box = len(pd.read_csv(box_path)) if os.path.exists(box_path) else None
        dist = lab["label"].value_counts().to_dict()
        runs = label_runs(lab["label"].tolist())
        room_labels.setdefault(room, Counter()).update(dist)

        out.append({
            "session": name,
            "data_rows": n_rows,
            "columns": sorted(ncols),
            "label_rows": len(lab),
            "label_idx_contiguous": bool((lab["idx"].values == np.arange(len(lab))).all()),
            "label_boxes_rows": n_box,
            "rows_minus_labels": n_rows - len(lab),
            "nan_values": n_nan,
            "inf_values": n_inf,
            "rows_with_bad_index_block": idx_bad,
            "all_zero_amplitude_rows": zero_amp_rows,
            "duplicate_consecutive_rows": dup_consecutive,
            "amp_min_per_stream": amp_min.round(3).tolist(),
            "amp_max_per_stream": amp_max.round(3).tolist(),
            "amp_mean_per_stream": (amp_sum / n_rows).round(2).tolist(),
            "phase_range": [round(ph_min, 4), round(ph_max, 4)],
            "iq_integer_fraction_sampled": round(iq_int_ok / iq_checked, 5),
            "iq_absmax_sampled": iq_absmax,
            "label_distribution": dist,
            "n_label_segments": len(runs),
            "segment_len_median": float(np.median([r[1] for r in runs])),
            "segment_sequence_head": [f"{l}:{n}" for l, n in runs[:15]],
        })

    result = {"sessions": out,
              "label_distribution_by_room": {r: dict(c) for r, c in room_labels.items()}}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    for s in out:
        print(json.dumps(s)[:1200])
    print(json.dumps(result["label_distribution_by_room"]))


if __name__ == "__main__":
    main()

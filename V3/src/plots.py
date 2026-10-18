"""V3 figures from results/N*.json and the cached real-noise tables.

    fig1_presence_raw_vs_cal.png     balanced accuracy per test, raw vs calibrated, per model
    fig2_false_alarms.png            share of EMPTY-room windows called occupied
    fig3_congested_channel.png       N2 confusion matrices: transformer on the congested channel
    fig4_espectre_rooms.png          N1 per held-out home room, raw vs calibrated
    fig5_empty_vs_occupied.png       why it is hard: empty rooms fluctuate as much as occupied ones,
                                     and what calibration does to it

Usage:  python plots.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"

# ---------------------------------------------------------------------------
# Style (same palette as the Midsem report; fixed order, gray = baseline)
# ---------------------------------------------------------------------------

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GRAY, INK, INK2 = "#a3a29c", "#0b0b0b", "#52514e"
MODEL_COLOR = {"lgbm": BLUE, "cnn": ORANGE, "tf": AQUA, "ensemble": VIOLET}
MODEL_LABEL = {"lgbm": "LightGBM", "cnn": "CNN", "tf": "Transformer", "ensemble": "Ensemble"}
MODELS = ["lgbm", "cnn", "tf", "ensemble"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


def mean_n1(d, task, model, key):
    vals = [d[k][model]["smoothed"][key] for k in d if k.endswith(task)]
    return float(np.mean(vals)), float(np.std(vals))


# ---------------------------------------------------------------------------
# Collect presence results: test -> (raw, cal) -> model -> (bal, std, false_alarm)
# ---------------------------------------------------------------------------

TESTS = [
    ("N2", "Same room,\ncongested WiFi channel", "N2_pipeline_quiet_to_congested", "N2c_pipeline_quiet_to_congested_calibrated"),
    ("N3", "Unseen room\n(pipeline)", "N3_pipeline_room1_to_room2", "N3c_pipeline_room1_to_room2_calibrated"),
    ("N1", "Unseen home rooms\n(ESPectre, mean of 3)", "N1_espectre_leave_room_out", "N1c_espectre_leave_room_out_calibrated"),
]


def presence_table(task="presence"):
    out = {}
    for key, label, raw_f, cal_f in TESTS:
        out[key] = {"label": label}
        for variant, f in [("raw", raw_f), ("cal", cal_f)]:
            d = load(f)
            out[key][variant] = {}
            for m in MODELS:
                if key == "N1":
                    b, s = mean_n1(d, task, m, "bal_acc")
                    fa, _ = mean_n1(d, task, m, "false_alarm")
                else:
                    r = d[task][m]["smoothed"]
                    b, s, fa = r["bal_acc"], 0.0, r["false_alarm"]
                out[key][variant][m] = (b, s, fa)
    return out


# ---------------------------------------------------------------------------
# Figure 1 + 2: balanced accuracy and false alarms, raw vs calibrated
# ---------------------------------------------------------------------------

def fig_bars(table, value_idx, ylabel, title, path, chance=None, lower_is_better=False):
    fig, axes = plt.subplots(1, len(TESTS), figsize=(11, 3.4), sharey=True)
    width = 0.2
    for ax, (key, *_rest) in zip(axes, TESTS):
        t = table[key]
        x = np.arange(2)                                       # raw, cal
        for j, m in enumerate(MODELS):
            vals = [t["raw"][m][value_idx], t["cal"][m][value_idx]]
            errs = [t["raw"][m][1], t["cal"][m][1]] if value_idx == 0 else [0, 0]
            xs = x + (j - 1.5) * width
            ax.bar(xs, vals, width * 0.9, color=MODEL_COLOR[m], label=MODEL_LABEL[m],
                   yerr=errs, error_kw=dict(lw=0.7, capsize=2, ecolor=INK2))
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 0.02, f"{v:.2f}" if value_idx == 0 else f"{v:.0%}",
                        ha="center", fontsize=6, color=INK, rotation=90)
        if chance is not None:
            ax.axhline(chance, color=GRAY, lw=1, ls="--")
            ax.text(1.55, chance + 0.01, "chance", color=INK2, fontsize=7, ha="right")
        ax.set_xticks(x, ["No calibration", "30 s empty-room\ncalibration"])
        ax.set_title(t["label"].replace("\n", " "), fontsize=9)
        ax.set_ylim(0, 1.12)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.32))
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", y=1.13)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: N2 confusion matrices (transformer, congested channel)
# ---------------------------------------------------------------------------

def confusion(ax, cm, labels, title):
    cm = np.asarray(cm, float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.grid(False)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)


def fig_congested(path):
    raw, cal = load("N2_pipeline_quiet_to_congested"), load("N2c_pipeline_quiet_to_congested_calibrated")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0))
    confusion(axes[0], raw["presence"]["lgbm"]["smoothed"]["confusion"], ["empty", "occupied"],
              "LightGBM, presence")
    confusion(axes[1], raw["presence"]["tf"]["smoothed"]["confusion"], ["empty", "occupied"],
              "Transformer, presence")
    confusion(axes[2], raw["motion"]["tf"]["smoothed"]["confusion"], ["empty", "static", "moving"],
              "Transformer, motion state")
    confusion(axes[3], cal["presence"]["ensemble"]["smoothed"]["confusion"], ["empty", "occupied"],
              "Calibrated ensemble, presence")
    fig.suptitle("Trained on the QUIET channel, tested on the MOST CONGESTED channel (same room, real recordings)",
                 x=0.01, ha="left", fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: ESPectre per held-out room
# ---------------------------------------------------------------------------

def fig_rooms(path):
    raw, cal = load("N1_espectre_leave_room_out"), load("N1c_espectre_leave_room_out_calibrated")
    rooms = ["bedroom", "living_room", "hobby_room"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, (variant, d, title) in zip(axes, [("raw", raw, "No calibration"),
                                               ("cal", cal, "30 s empty-room calibration")]):
        x = np.arange(len(rooms))
        for j, m in enumerate(MODELS):
            vals = [d[f"{r}|presence"][m]["smoothed"]["bal_acc"] for r in rooms]
            ax.bar(x + (j - 1.5) * 0.2, vals, 0.18, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
        ax.axhline(0.5, color=GRAY, lw=1, ls="--")
        ax.set_xticks(x, [r.replace("_", " ") for r in rooms])
        ax.set_title(f"Held-out room, presence - {title}", fontsize=9)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.25))
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: the physics problem, and what calibration does
# ---------------------------------------------------------------------------

def fig_levels(path):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from build import feature_cols
    from calibrate_v3 import calibrate_all
    from experiments_noise import load as load_tables

    df, seq, _ = load_tables(["pipeline"])
    cols = feature_cols(df.drop(columns=["motion_id"]))
    dfc, seqc, ref, _ = calibrate_all(df, seq, cols)
    df["level"] = np.median(np.abs(seq.astype(np.float32)), axis=(1, 2))
    dfc["level"] = np.nanmedian(np.abs(seqc), axis=(1, 2))

    sites = [("room1", "quiet", "Room 1, quiet channel"), ("room1", "noisy", "Room 1, congested channel"),
             ("room2", "quiet", "Room 2 (other room)")]
    states = [("empty", GRAY), ("static", BLUE), ("moving", ORANGE)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
    for ax, (d, title, mask) in zip(axes, [(df, "Raw: fluctuation level per window", np.ones(len(df), bool)),
                                           (dfc, "After 30 s empty-room calibration", ~ref.to_numpy())]):
        pos = 0
        ticks, labels = [], []
        for room, chan, name in sites:
            for k, (state, col) in enumerate(states):
                sel = mask & (d["room"] == room).to_numpy() & (d["channel"] == chan).to_numpy() \
                    & (d["motion"] == state).to_numpy()
                vals = d.loc[sel, "level"].dropna().to_numpy()
                bp = ax.boxplot(vals, positions=[pos], widths=0.7, patch_artist=True, showfliers=False,
                                medianprops=dict(color=INK), whiskerprops=dict(color=INK2),
                                capprops=dict(color=INK2), boxprops=dict(edgecolor=INK2))
                bp["boxes"][0].set_facecolor(col)
                pos += 1
            ticks.append(pos - 2)
            labels.append(name)
            pos += 1
        ax.set_xticks(ticks, labels, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Median |relative fluctuation|")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in states]
    axes[0].legend(handles, ["empty room", "person static", "person moving"], ncol=3, loc="upper left",
                   bbox_to_anchor=(0, 1.25))
    fig.suptitle("Real ESP32 data: an EMPTY room can fluctuate as much as an occupied one",
                 x=0.01, ha="left", fontweight="bold", y=1.12)
    fig.savefig(path)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    table = presence_table("presence")
    fig_bars(table, 0, "Balanced accuracy (presence)",
             "Presence detection on real noisy ESP32 recordings: no calibration vs 30 s empty-room calibration",
             FIG / "fig1_presence_raw_vs_cal.png", chance=0.5)
    fig_bars(table, 2, "False alarms on empty room (lower is better)",
             "How often an EMPTY room is reported as occupied",
             FIG / "fig2_false_alarms.png", lower_is_better=True)
    fig_congested(FIG / "fig3_congested_channel.png")
    fig_rooms(FIG / "fig4_espectre_rooms.png")
    fig_levels(FIG / "fig5_empty_vs_occupied.png")
    for p in sorted(FIG.glob("*.png")):
        print("wrote", p)


if __name__ == "__main__":
    main()

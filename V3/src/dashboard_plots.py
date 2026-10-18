"""Additional V3 dashboard figures from results/metrics/v3_metrics.csv."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics" / "v3_metrics.csv"
FIG = ROOT / "results" / "figures"
COLORS = {"lgbm": "#2a78d6", "cnn": "#eb6834", "tf": "#1baf7a", "ensemble": "#6250d6"}
LABELS = {"lgbm": "LightGBM", "cnn": "CNN", "tf": "Transformer", "ensemble": "Ensemble"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e4e3df",
    "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight",
})


def save(fig, name):
    path = FIG / name
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def filter_rows(df, experiments, split_contains, smoothing="smoothed"):
    return df[df["experiment"].isin(experiments)
              & df["split"].str.contains(split_contains, regex=False)
              & (df["smoothing"] == smoothing)]


def fig6(df):
    raw = filter_rows(df, ["N3_pipeline_room1_to_room2"], "presence")
    cal = filter_rows(df, ["N3c_pipeline_room1_to_room2_calibrated"], "presence")
    raw = raw[raw["variant"].isin(COLORS)]
    cal = cal[cal["variant"].isin(COLORS)]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, key, title in zip(axes, ["bal_acc", "false_alarm"], ["Balanced accuracy", "Empty-room false alarms"]):
        for i, (label, part) in enumerate([("Raw", raw), ("30 s calibration", cal)]):
            values = [part.loc[part["variant"] == m, key].mean() for m in COLORS]
            x = range(len(values))
            axes[0 if key == "bal_acc" else 1].bar([v + i * 0.35 for v in x], values, 0.32,
                label=label, color="#a3a29c" if i == 0 else "#1baf7a")
        ax.set_xticks([v + 0.175 for v in range(4)], [LABELS[m] for m in COLORS], rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_ylabel("Score")
    axes[0].legend()
    fig.suptitle("N3 unseen-room presence: raw versus calibrated", x=0.01, y=0.99,
                 ha="left", fontweight="bold")
    fig.subplots_adjust(top=0.78, wspace=0.25)
    save(fig, "fig6_n3_cross_room.png")


def fig7(df):
    part = filter_rows(df, ["N4_all_to_pipeline"], "presence")
    part = part[part["variant"].isin(COLORS)]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, m in enumerate(COLORS):
        row = part[part["variant"] == m]
        ax.bar(i - 0.18, row["bal_acc"].mean(), 0.35, color=COLORS[m], label=LABELS[m])
        ax.text(i - 0.18, row["bal_acc"].mean() + 0.02, f"{row['bal_acc'].mean():.2f}", ha="center", fontsize=8)
        ax.bar(i + 0.18, row["false_alarm"].mean(), 0.35, color=COLORS[m], alpha=0.35)
        ax.text(i + 0.18, row["false_alarm"].mean() + 0.02, f"{row['false_alarm'].mean():.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(4), [LABELS[m] for m in COLORS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("N4 new setup: balanced accuracy (solid) and false alarms (faded)")
    ax.legend(handles=[Patch(facecolor="#777777", label="Balanced accuracy"),
                       Patch(facecolor="#777777", alpha=0.35, label="False-alarm rate")],
              loc="upper right")
    save(fig, "fig7_n4_cross_dataset.png")


def fig8(df):
    part = filter_rows(df, ["N2_pipeline_quiet_to_congested"], "presence")
    part = part[part["variant"].isin(COLORS)]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, m in enumerate(COLORS):
        value = part.loc[part["variant"] == m, "bal_acc"].mean()
        ax.bar(i, value, color=COLORS[m])
        ax.text(i, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="#a3a29c", ls="--", lw=1, label="chance")
    ax.set_xticks(range(4), [LABELS[m] for m in COLORS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Smoothed balanced accuracy")
    ax.set_title("N2 model-family comparison: quiet training to congested channel")
    ax.legend()
    save(fig, "fig8_model_family_n2.png")


def fig9(df):
    names = ["N1_espectre_leave_room_out", "N2_pipeline_quiet_to_congested", "N3_pipeline_room1_to_room2"]
    part = filter_rows(df, names, "presence")
    part = part[part["variant"].isin(COLORS)]
    grouped = part.groupby("variant")[["class_0_recall", "class_1_recall"]].mean().reindex(COLORS)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = range(len(grouped))
    ax.bar([v - 0.18 for v in x], grouped["class_0_recall"], 0.35, label="Empty specificity", color="#a3a29c")
    ax.bar([v + 0.18 for v in x], grouped["class_1_recall"], 0.35, label="Occupied sensitivity", color="#1baf7a")
    ax.set_xticks(list(x), [LABELS[m] for m in grouped.index])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean smoothed recall")
    ax.set_title("Presence per-class metrics across N1-N3")
    ax.legend()
    save(fig, "fig9_presence_per_class.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METRICS)
    fig6(df)
    fig7(df)
    fig8(df)
    fig9(df)


if __name__ == "__main__":
    main()
"""Make every V4 figure and the numbers table from saved result files only.

Reads the newest run of each kind in V4/results/metrics/ and writes:

    V4/results/figures/fig*.png        figures used by docs/REPORT.md
    V4/results/figures/captions.json   caption + data source of each figure
    V4/results/metrics/summary.json    the numbers quoted in docs/REPORT.md

Usage:  python V4/src/report.py
"""
from __future__ import annotations

import argparse
import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES, METRICS, ROOT, to_jsonable

# validated categorical palette (dataviz reference, light mode), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
GOOD_LINE = "#52514e"

GROUP_LABELS = {
    "espectre/test": "ESPectre\nunseen sessions",
    "pipeline/congested": "Pipeline\ncongested channel",
    "pipeline/unseen_room": "Pipeline\nunseen room",
    "homeoccupancy/test_dev": "Home\nrecording 3",
    "officelocalization/test_external": "Office\ntest files",
    "officelocalization/test_extra_empty": "Office\nextra empty",
}
V3_SEEN = {"espectre/test", "pipeline/congested", "pipeline/unseen_room"}
captions = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def newest(prefix):
    runs = sorted(p for p in METRICS.glob(f"{prefix}_2*/result.json"))
    return (json.loads(runs[-1].read_text()), runs[-1].parent.name) if runs else (None, None)


def style(ax, title=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def save(fig, name, caption, source):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    captions[name] = {"caption": caption, "data": source}


def bars(ax, groups, series, values, ylim=(0, 1.05), fmt="{:.2f}"):
    """Grouped bars: values[s][g] (None -> no bar). 2px white gap between bars."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        v = [values[s].get(g) for g in groups]
        pos = x - 0.4 + width * (i + 0.5)
        h = [np.nan if a is None else a for a in v]
        ax.bar(pos, h, width * 0.92, color=C[i], label=s, edgecolor="white", linewidth=1)
        for p, a in zip(pos, v):
            if a is not None and a >= 0.005:
                ax.text(p, a + 0.015, fmt.format(a), ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x, [GROUP_LABELS.get(g, g) for g in groups], fontsize=8)
    ax.set_ylim(*ylim)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_data_fix(summary):
    """Share of HomeOccupancy windows with NO real packet, before vs after the parser fix."""
    old = ROOT / "archive" / "V4_removed_2026-10-18" / "cache" / "homeoccupancy"
    if not old.exists():                      # fresh Git clone: the archived buggy cache is not there
        print("fig1 skipped: archived preliminary cache not found (existing figure kept)")
        return
    from data import build
    new, _, _ = build("homeoccupancy")
    before = {}
    for split in ("train", "test"):
        t = pickle.loads((old / f"{split}.pkl").read_bytes())
        for rec, m in t.groupby("recording")["missing"]:
            before[rec.replace(".csv", "")] = float((m >= 1).mean())
    after = {rec: float((m >= 1).mean()) for rec, m in new.groupby("recording")["missing"]}
    recs = sorted(before)
    summary["data_fix"] = {"before": before, "after": after}

    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(recs))
    ax.bar(x - 0.2, [before[r] for r in recs], 0.38, color=C[1], label="preliminary V4 parser", edgecolor="white")
    ax.bar(x + 0.2, [after[r] for r in recs], 0.38, color=C[0], label="fixed V4 parser", edgecolor="white")
    for i, r in enumerate(recs):
        ax.text(i - 0.2, before[r] + 0.02, f"{before[r]:.0%}", ha="center", fontsize=7, color=INK2)
        ax.text(i + 0.2, after[r] + 0.02, f"{after[r]:.0%}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, recs, fontsize=8)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    style(ax, "Fake windows in HomeOccupancy: before and after the timestamp fix", "windows with no real packet")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "fig1_data_fix.png",
         "Share of 3 s windows that contained no real packet (pure interpolation). A few corrupted timestamps "
         "stretched each recording 3-5x; after removing them before unwrapping, almost every window is real.",
         "archived preliminary cache vs V4/cache/homeoccupancy_standard.pkl")


def fig_presence(presence, summary, final, name="fig2_presence_groups.png", variant="standard"):
    models_ = {"V3 CNN (frozen)": "v3_cnn_raw", "V3 transformer (frozen)": "v3_tf_raw",
               "V4 final (LightGBM calibrated)": final, "V4 gap-CNN (calibrated)": "gapcnn_cal"}
    models_ = {k: v for k, v in models_.items() if v in presence["models"]}
    groups = list(GROUP_LABELS)
    bal = {k: {g: presence["models"][m]["common"].get(g, {}).get("bal_acc") for g in groups} for k, m in models_.items()}
    fa = {k: {g: presence["models"][m]["common"].get(g, {}).get("false_alarm") for g in groups} for k, m in models_.items()}
    miss = {k: {g: presence["models"][m]["common"].get(g, {}).get("miss") for g in groups} for k, m in models_.items()}
    summary[f"presence_{variant}"] = {"run_selected": presence["selected_on_validation"], "balanced_accuracy": bal,
                                      "false_alarm": fa, "miss": miss}

    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True)
    bars(axes[0], groups, list(models_), bal)
    style(axes[0], "Balanced accuracy (higher is better)", "balanced accuracy")
    bars(axes[1], groups, list(models_), fa)
    axes[1].axhline(0.05, color=GOOD_LINE, linestyle="--", linewidth=1)
    axes[1].text(-0.45, 0.07, "target ≤ 5 %", ha="left", fontsize=7, color=INK2)
    style(axes[1], "False alarms: empty windows called occupied (lower is better)", "false-alarm rate")
    bars(axes[2], groups, list(models_), miss)
    axes[2].axhline(0.15, color=GOOD_LINE, linestyle="--", linewidth=1)
    axes[2].text(-0.45, 0.17, "target ≤ 15 %", ha="left", fontsize=7, color=INK2)
    style(axes[2], "Missed people: occupied windows called empty (lower is better)", "miss rate")
    for ax in axes:
        for i, g in enumerate(groups):
            if g in V3_SEEN:
                ax.axvspan(i - 0.45, i + 0.45, color="#f3f2ee", zorder=0)
    axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.12))
    fig.text(0.01, -0.01, "Shaded groups: V3 was trained on these sources, so V3 bars there are not a fair test.",
             fontsize=8, color=INK2)
    save(fig, name,
         f"Presence on every test group ({variant} preprocessing), same windows for every model, 5-window smoothing, "
         "thresholds chosen on validation. Shaded groups were part of V3's training data.",
         "results/metrics/presence_*/result.json")


def fig_loso(loso, summary, name="fig3_loso.png"):
    models_ = {"V3 CNN (frozen)": "v3_cnn_raw", "V3 transformer (frozen)": "v3_tf_raw",
               "V4 final (LightGBM calibrated)": "lgbm_cal", "V4 gap-CNN calibrated": "gapcnn_cal"}
    models_ = {k: v for k, v in models_.items() if v in next(iter(loso["targets"].values()))}
    targets = list(loso["targets"])
    labels = {"espectre": "ESPectre", "pipeline": "Pipeline", "homeoccupancy": "Home", "officelocalization": "Office"}
    summary["loso"] = {t: {k: {x: loso["targets"][t][m][x] for x in ("bal_acc", "false_alarm", "miss")}
                           for k, m in models_.items()} for t in targets}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, metric, title in zip(axes, ["bal_acc", "false_alarm", "miss"],
                                 ["Balanced accuracy", "False-alarm rate", "Miss rate"]):
        vals = {k: {t: loso["targets"][t][m][metric] for t in targets} for k, m in models_.items()}
        n = len(models_)
        x = np.arange(len(targets))
        for i, k in enumerate(models_):
            v = [vals[k][t] for t in targets]
            ax.bar(x - 0.4 + 0.8 / n * (i + 0.5), v, 0.8 / n * 0.92, color=C[i], label=k, edgecolor="white")
        ax.set_xticks(x, [labels[t] + ("*" if t in ("espectre", "pipeline") else "") for t in targets], fontsize=8)
        ax.set_ylim(0, 1.05)
        style(ax, title)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.1), ncol=4)
    fig.text(0.01, -0.04, "* V3 was trained on ESPectre and pipeline, so its bars there are not a fair test. "
             "Home and Office are unseen by every model.", fontsize=8, color=INK2)
    save(fig, name,
         "Leave-one-source-out: V4 is trained on the other three sources and tested on the left-out one; "
         "threshold fixed at 0.5. Home and Office are the fair V3 vs V4 comparison.",
         "results/metrics/loso_*/result.json")


def fig_noise(noise_runs, summary, name="fig4_noise_heatmap.png"):
    """Rows: models; columns: noise condition; colour: mean balanced accuracy over test groups."""
    rows, data = [], []
    for variant, (res, picks) in noise_runs.items():
        for label, m in picks.items():
            rows.append(label)
            data.append([res["conditions"][c][m]["summary"]["mean_bal_acc"] for c in res["conditions"]])
            cols = list(res["conditions"])
    data = np.array(data, dtype=float)
    summary["noise"] = {r: dict(zip(cols, d)) for r, d in zip(rows, data.tolist())}

    fig, ax = plt.subplots(figsize=(13, 0.55 * len(rows) + 1.6))
    im = ax.imshow(data, cmap="Blues", vmin=0.4, vmax=1.0, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if data[i, j] > 0.78 else INK)
    ax.set_xticks(range(len(cols)), [c.replace("_", " ").replace("subcarriers", "subc.") for c in cols],
                  rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.spines[:].set_visible(False)
    ax.set_title("Synthetic noise stress test: mean balanced accuracy over the test groups", loc="left", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.ax.tick_params(labelsize=7)
    save(fig, name,
         "Each column adds one synthetic noise type to the raw test recordings (levels 1-3; 'crazy' = all at level 3). "
         "Darker = better. Noise is synthetic: this is a stress test, not real-world evidence.",
         "results/metrics/noise_*/result.json")


def fig_noise_lines(noise_runs, summary, name="fig5_noise_fa_miss.png"):
    """False alarms and misses at the strongest level of each noise type, per model."""
    kinds = ["clean_0", "packet_loss_3", "agc_jumps_3", "impulses_3", "white_noise_3", "fan_3", "jitter_3",
             "dead_subcarriers_3", "crazy_3"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    keep = {"V3 CNN (frozen)", "V3 transformer (frozen)", "V4 final: LightGBM cal (standard)",
            "V4 LightGBM cal (robust+aug)"}
    i = 0
    for variant, (res, picks) in noise_runs.items():
        for label, m in picks.items():
            if label not in keep:
                continue
            fa = [res["conditions"][k][m]["summary"]["mean_false_alarm"] for k in kinds]
            miss = [res["conditions"][k][m]["summary"]["mean_miss"] for k in kinds]
            axes[0].plot(range(len(kinds)), fa, marker="o", markersize=4, linewidth=2, color=C[i % len(C)], label=label)
            axes[1].plot(range(len(kinds)), miss, marker="o", markersize=4, linewidth=2, color=C[i % len(C)], label=label)
            i += 1
    for ax, title, target in zip(axes, ["Mean false-alarm rate", "Mean miss rate"], [0.05, 0.15]):
        ax.axhline(target, color=GOOD_LINE, linestyle="--", linewidth=1)
        ax.set_xticks(range(len(kinds)), [k.rsplit("_", 1)[0].replace("_", " ") for k in kinds], rotation=40,
                      ha="right", fontsize=8)
        style(ax, title + " (strongest level)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    save(fig, name, "False alarms and misses at the strongest level of each noise type; dashed lines are the targets.",
         "results/metrics/noise_*/result.json")


def fig_gap(presence, noise_std, summary, name="fig6_gap_ablation.png"):
    pairs = [("cnn_raw", "gapcnn_raw"), ("cnn_cal", "gapcnn_cal")]
    conds = ["clean_0", "packet_loss_1", "packet_loss_2", "packet_loss_3", "jitter_3", "crazy_3"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    out = {}
    for i, (a, b) in enumerate(pairs):
        for j, m in enumerate((a, b)):
            v = [noise_std["conditions"][c][m]["summary"]["mean_bal_acc"] for c in conds]
            out[m] = dict(zip(conds, v))
            ax.plot(range(len(conds)), v, marker="o", markersize=5, linewidth=2,
                    color=C[2 * i + j], linestyle="-" if j else "--",
                    label=("gap-CNN" if j else "plain CNN (ignores gaps)") + (" calibrated" if i else " raw"))
    ax.set_xticks(range(len(conds)), [c.replace("_", " ") for c in conds], fontsize=8)
    ax.set_ylim(0.4, 1.0)
    style(ax, "Does giving the CNN the packet-gap mask help?", "mean balanced accuracy")
    ax.legend(frameon=False, fontsize=8)
    summary["gap_ablation"] = out
    save(fig, name, "Plain V3-style CNN vs the V4 gap-aware CNN, same data and training, clean and with packet loss.",
         "results/metrics/noise_standard_*/result.json")


def fig_counting(counting, summary, name="fig7_counting.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    summary["counting"] = {}
    for ax, kind, title in zip(axes, ["lgbm", "gapcnn"], ["LightGBM", "gap-CNN"]):
        m = counting["pooled"][kind]
        cm = np.array(m["confusion"], dtype=float)
        rows = cm.sum(1, keepdims=True)
        share = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                if rows[i, 0] > 0:
                    ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=8,
                            color="white" if share[i, j] > 0.6 else INK)
        for i in (4, 5):
            ax.text(2.5, i, "no data for this count", ha="center", va="center", fontsize=7.5, color=INK2)
        ax.set_xticks(range(6), range(6))
        ax.set_yticks(range(6), range(6))
        ax.set_xlabel("predicted people", fontsize=9, color=INK2)
        ax.set_ylabel("true people", fontsize=9, color=INK2)
        ax.set_title(f"{title}: exact {m['exact_accuracy']:.2f}, MAE {m['mae']:.2f}", loc="left", fontsize=10)
        summary["counting"][kind] = {k: m[k] for k in ("exact_accuracy", "mae", "within_one", "overcount_rate",
                                                       "undercount_rate", "empty_false_alarm", "balanced_accuracy")}
        summary["counting"][kind]["recall_per_count"] = {c: m["per_class"][c]["recall"] for c in "0123"}
    save(fig, name,
         "WiCount people counting (standard pipeline), pooled over 3 folds (each window tested once). Counts 1-2: held-out recordings; "
         "counts 0 and 3: held-out time blocks of their single recording. Rows 4-5: no data exists.",
         "results/metrics/counting_*/result.json")


def fig_counting_noise(count_runs, summary, name="fig8_counting_noise.png"):
    kinds = ["clean", "packet_loss_3", "agc_jumps_3", "impulses_3", "white_noise_3", "fan_3", "jitter_3",
             "dead_subcarriers_3", "crazy_3"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    i = 0
    out = {}
    for variant, res in count_runs.items():
        for kind in ["lgbm", "gapcnn"]:
            v = [res["pooled"][kind]["exact_accuracy"]] + [res["noise"][kind][k]["exact_accuracy"] for k in kinds[1:]]
            out[f"{kind} {variant}"] = dict(zip(kinds, v))
            ax.plot(range(len(kinds)), v, marker="o", markersize=5, linewidth=2, color=C[i],
                    linestyle="-" if variant != "standard" else "--", label=f"{kind} ({variant})")
            i += 1
    ax.set_xticks(range(len(kinds)), [k.replace("_3", "").replace("_", " ") for k in kinds], rotation=30,
                  ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    style(ax, "People counting under synthetic noise (strongest level)", "exact-count accuracy")
    ax.legend(frameon=False, fontsize=8)
    summary["counting_noise"] = out
    save(fig, name, "Exact-count accuracy on the same WiCount test windows when noise is added.",
         "results/metrics/counting_*/result.json")


def fig_tradeoff(presence, summary, m, name="fig9_tradeoff.png"):
    curves = presence["models"][m]["tradeoff_common"]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    groups = [g for g in GROUP_LABELS if g in curves and g != "officelocalization/test_extra_empty"]
    for i, g in enumerate(groups):
        c = [p for p in curves[g] if p["false_alarm"] is not None and p["miss"] is not None]
        ax.plot([p["false_alarm"] for p in c], [p["miss"] for p in c], linewidth=2, color=C[i],
                label=GROUP_LABELS[g].replace("\n", " "))
    ax.axvline(0.05, color=GOOD_LINE, linestyle="--", linewidth=1)
    ax.axhline(0.15, color=GOOD_LINE, linestyle="--", linewidth=1)
    ax.fill_between([0, 0.05], 0, 0.15, color="#e9f2fb", zorder=0)
    ax.text(0.055, 0.155, "target box", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("false-alarm rate", color=INK2, fontsize=9)
    style(ax, f"Trade-off of the selected model ({m})", "miss rate")
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, name, "Every possible threshold of the selected model: lower-left is better. "
                    "The blue box is the provisional target (≤ 5 % false alarms and ≤ 15 % misses).",
         "results/metrics/presence_*/result.json")


# ---------------------------------------------------------------------------
# Final selection (validation only) and model card
# ---------------------------------------------------------------------------

def select_final(noise_std, noise_rob, summary):
    """Predeclared rule: best mean VALIDATION balanced accuracy over clean + all noise
    conditions, across both variants. Test numbers are not looked at."""
    table = {}
    for variant, res in (("standard", noise_std), ("robust_aug", noise_rob)):
        if not res:
            continue
        for m in res["conditions"]["clean_0"]:
            if m == "rejected_rate" or m.startswith("v3_"):
                continue
            vals = [res["conditions"][c][m]["val_bal_acc"] or 0 for c in res["conditions"]]
            table[f"{variant}/{m}"] = float(np.mean(vals))
    best = max(table, key=table.get)
    variant, model = best.split("/")
    summary["final_selection"] = {"rule": "max mean validation balanced accuracy over clean + 22 noise conditions",
                                  "validation_scores": dict(sorted(table.items(), key=lambda x: -x[1])),
                                  "selected": best}
    return {"variant": variant, "model": model}


def write_model_card(final, presence, summary):
    from common import MODELS, sha256
    folder = MODELS / "presence" / final["variant"]
    files = sorted(folder.glob(f"{final['model']}_s*.txt")) + sorted(folder.glob(f"{final['model']}.pt"))
    entry = presence["models"][final["model"]]
    card = {
        "name": "V4 presence candidate (experimental, NOT promoted)",
        "task": "room presence: empty (0) / occupied (1); windows failing the quality gate -> 'unknown'",
        "variant": final["variant"], "model": final["model"],
        "files": [{"file": f.relative_to(MODELS).as_posix(), "sha256": sha256(f)} for f in files],
        "features": json.loads((folder / f"{final['model']}.json").read_text()).get("features"),
        "needs_calibration": final["model"].endswith("_cal"),
        "calibration": "30 s of empty room after installation (see configs/v4_protocol.json)",
        "threshold": entry["threshold_from_val"],
        "smoothing": "causal median over 5 one-second windows",
        "input": "esp-csi raw CSI_DATA, L-LTF amplitude, 20 Hz grid, 3 s windows, 1 s hop",
        "trained_on": "train split of ESPectre, pipeline (room 1 quiet), HomeOccupancy (_1), OfficeLocalization (_1)",
        "selection": summary["final_selection"],
        "status": "experimental: promotion needs a live capture in the target room (configs/v4_protocol.json)",
    }
    (MODELS / "presence" / "MODEL_CARD.json").write_text(json.dumps(card, indent=2))
    summary["model_card"] = card


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    summary = {}
    pres_std, id_ps = newest("presence_standard")
    pres_rob, id_pr = newest("presence_robust_aug")
    noise_std, id_ns = newest("noise_standard")
    noise_rob, id_nr = newest("noise_robust_aug")
    loso_std, id_ls = newest("loso_standard")
    loso_rob, id_lr = newest("loso_robust_aug")
    cnt_std, id_cs = newest("counting_standard")
    cnt_rob, id_cr = newest("counting_robust_aug")
    summary["runs"] = {"presence_standard": id_ps, "presence_robust_aug": id_pr, "noise_standard": id_ns,
                       "noise_robust_aug": id_nr, "loso_standard": id_ls, "loso_robust_aug": id_lr,
                       "counting_standard": id_cs, "counting_robust_aug": id_cr}

    fig_data_fix(summary)
    final = select_final(noise_std, noise_rob, summary)
    write_model_card(final, pres_std if final["variant"] == "standard" else pres_rob, summary)
    if pres_std:
        fig_presence(pres_std, summary, final["model"])
        fig_tradeoff(pres_std if final["variant"] == "standard" else pres_rob, summary, final["model"])
    if pres_rob:
        fig_presence(pres_rob, summary, final["model"], name="fig2b_presence_groups_robust.png", variant="robust_aug")
    if loso_std:
        fig_loso(loso_std, summary)
    if loso_rob:
        summary["loso_robust_aug"] = {t: {m: {x: v[x] for x in ("bal_acc", "false_alarm", "miss")} for m, v in r.items()}
                                      for t, r in loso_rob["targets"].items()}
    if noise_std:
        runs = {"standard": (noise_std, {"V3 CNN (frozen)": "v3_cnn_raw", "V3 transformer (frozen)": "v3_tf_raw",
                                         "V4 final: LightGBM cal (standard)": "lgbm_cal",
                                         "V4 gap-CNN cal (standard)": "gapcnn_cal",
                                         "V4 gap-CNN raw (standard)": "gapcnn_raw"})}
        if noise_rob:
            runs["robust"] = (noise_rob, {"V4 LightGBM cal (robust+aug)": "lgbm_cal",
                                          "V4 gap-CNN cal (robust+aug)": "gapcnn_cal",
                                          "V4 gap-CNN raw (robust+aug)": "gapcnn_raw"})
        fig_noise(runs, summary)
        fig_noise_lines(runs, summary)
        fig_gap(pres_std, noise_std, summary)
    if cnt_std:
        fig_counting(cnt_std, summary, name="fig7_counting.png")
        if cnt_rob:
            summary["counting_robust_aug"] = {k: {x: cnt_rob["pooled"][k][x] for x in
                                                  ("exact_accuracy", "mae", "balanced_accuracy", "within_one",
                                                   "overcount_rate", "undercount_rate", "empty_false_alarm")}
                                              for k in ("lgbm", "gapcnn")}
        fig_counting_noise({k: v for k, v in {"standard": cnt_std, "robust+aug": cnt_rob}.items() if v}, summary)

    (FIGURES / "captions.json").write_text(json.dumps(captions, indent=2))
    (METRICS / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2))
    print(f"wrote {len(captions)} figures to {FIGURES}")


if __name__ == "__main__":
    main()

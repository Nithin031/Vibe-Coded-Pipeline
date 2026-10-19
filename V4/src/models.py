"""V4 models: LightGBM, plain CNN (V3 architecture), gap-aware CNN, frozen V3 loaders.

    lgbm      LightGBM on the V3 window features, trained with sample weights
    cnn       V3's CNN1D: it IGNORES the packet-gap mask (kept as the ablation baseline)
    gapcnn    V4 fix: interpolated steps are zeroed, the gap mask is an extra input
              channel, and pooling uses only real (non-gap) time steps

All models return class probabilities with columns in class order (0, 1, ...).
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import V3, log
from data import feature_cols

import deep                                            # V3, read-only
from experiments_noise import LGB_PARAMS               # V3 LightGBM settings

DEVICE = deep.DEVICE
INPUT_SCALE = deep.INPUT_SCALE


# ---------------------------------------------------------------------------
# Sample weights: every source equal, every class within a source equal,
# every recording within a source-class equal
# ---------------------------------------------------------------------------

def balanced_weights(table: pd.DataFrame, label: str) -> np.ndarray:
    w = np.zeros(len(table))
    sources = table["source"].unique()
    for source in sources:
        s = table["source"] == source
        classes = table.loc[s, label].unique()
        for c in classes:
            sc = s & (table[label] == c)
            recs = table.loc[sc, "recording"].unique()
            for r in recs:
                m = (sc & (table["recording"] == r)).to_numpy()
                w[m] = 1.0 / len(sources) / len(classes) / len(recs) / m.sum()
    return w * len(table) / w.sum()


# ---------------------------------------------------------------------------
# Gap-aware CNN
# ---------------------------------------------------------------------------

class GapCNN(nn.Module):
    """CNN that knows which time steps are real packets and which are interpolated."""

    def __init__(self, n_classes=2, n_bins=32, ch=64, dropout=0.3):
        super().__init__()

        def block(cin, cout, k):
            return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.GELU())

        self.b1, self.b2, self.b3 = block(n_bins + 1, ch, 7), block(ch, ch, 5), block(ch, ch, 3)
        self.pool = nn.MaxPool1d(2)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * ch, n_classes))

    def forward(self, x, gap):
        real = (~gap).float()[:, None, :]                      # [B, 1, T]
        h = torch.cat([x * real, 1 - real], dim=1)             # zero fake steps, add mask channel
        h = self.b3(self.pool(self.b2(self.pool(self.b1(h)))))

        # masked mean / std pooling: a pooled step is real if any input step under it was real
        m = nn.functional.max_pool1d(nn.functional.max_pool1d(real, 2), 2)
        m = m[:, :, : h.shape[2]]
        n = m.sum(dim=2).clamp(min=1.0)
        mean = (h * m).sum(dim=2) / n
        std = (((h - mean[..., None]) ** 2 * m).sum(dim=2) / n).clamp(min=1e-6).sqrt()
        return self.head(torch.cat([mean, std], dim=1))


def make_net(kind: str, n_classes: int) -> nn.Module:
    if kind == "gapcnn":
        return GapCNN(n_classes=n_classes).to(DEVICE)
    if kind == "cnn":
        return deep.CNN1D(n_classes=n_classes).to(DEVICE)
    raise KeyError(kind)


def forward(net, x, g):
    return net(x, g)                                           # CNN1D ignores g by design


def tensors(seq, gap):
    x = torch.from_numpy(seq.astype(np.float32) * INPUT_SCALE)
    return x, torch.from_numpy(np.asarray(gap, dtype=bool))


@torch.no_grad()
def predict_net(net, seq, gap, bs=1024) -> np.ndarray:
    net.eval()
    X, G = tensors(seq, gap)
    out = []
    for b in range(0, len(X), bs):
        out.append(torch.softmax(forward(net, X[b:b + bs].to(DEVICE), G[b:b + bs].to(DEVICE)), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2))


def train_net(kind, seq, gap, y, weights, n_classes, epochs, seed=0, val=None, patience=5, tag=""):
    """Weighted-sampling training with V3's augmentation (gain + circular shift).

    val = (seq, gap, y) -> keep the epoch with the best validation balanced accuracy.
    Returns (net, history).
    """
    from sklearn.metrics import balanced_accuracy_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    X, G = tensors(seq, gap)
    X, G = X.to(DEVICE), G.to(DEVICE)
    Y = torch.as_tensor(y, device=DEVICE, dtype=torch.long)
    probs = torch.as_tensor(weights / weights.sum(), device=DEVICE, dtype=torch.float32)

    net = make_net(kind, n_classes)
    lr = 2e-3
    bs = 64
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(X) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    history, best, best_state, since = [], -1.0, None, 0
    for ep in range(epochs):
        net.train()
        idx = torch.multinomial(probs, len(X), replacement=True, generator=gen)
        total = 0.0
        for b in range(0, len(X), bs):
            j = idx[b: b + bs]
            if len(j) < 2:
                continue
            x, g = X[j], G[j]
            x = x * torch.empty(len(j), 1, 1, device=DEVICE).uniform_(0.8, 1.25, generator=gen)
            shift = int(torch.randint(0, x.shape[2], (1,), generator=gen, device=DEVICE))
            x, g = torch.roll(x, shift, dims=2), torch.roll(g, shift, dims=1)
            loss = loss_fn(forward(net, x, g), Y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(j)

        entry = {"epoch": ep, "train_loss": total / len(X)}
        if val is not None:
            pv = predict_net(net, val[0], val[1]).argmax(1)
            entry["val_bal_acc"] = float(balanced_accuracy_score(val[2], pv))
            if entry["val_bal_acc"] > best:
                best, since = entry["val_bal_acc"], 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                since += 1
        history.append(entry)
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"    {tag} {kind} epoch {ep:2d} loss {entry['train_loss']:.4f}"
                + (f" val bal_acc {entry['val_bal_acc']:.3f}" if val is not None else ""))
        if val is not None and since >= patience:
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    return net, history


def save_net(net, kind, n_classes, path: Path, info: dict):
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, path)
    path.with_suffix(".json").write_text(json.dumps({"kind": kind, "n_classes": n_classes, **info}, indent=2))


def load_net(path: Path):
    info = json.loads(path.with_suffix(".json").read_text())
    net = make_net(info["kind"], info["n_classes"])
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    return net, info


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def train_lgbm(table, y, weights, seed=0, cols=None):
    cols = cols or feature_cols(table)
    model = lgb.LGBMClassifier(random_state=seed, **{k: v for k, v in LGB_PARAMS.items() if k != "class_weight"})
    model.fit(table[cols], y, sample_weight=weights)
    return model, cols


def predict_lgbm(model, cols, table) -> np.ndarray:
    return model.predict_proba(table[cols])


# ---------------------------------------------------------------------------
# Frozen V3 presence models (read-only)
# ---------------------------------------------------------------------------

def v3_presence(kind: str):
    """Return predict(table, seq, gap) -> P(occupied) for a frozen V3 presence model."""
    folder = V3 / "models"
    if kind == "lgbm":
        booster = lgb.Booster(model_file=str(folder / "presence_lgbm.txt"))
        cols = booster.feature_name()

        # V3 bug: the saved booster was trained with the label column `motion_id` as a
        # feature. At real inference that label does not exist, so it is passed as NaN.
        def predict(table, seq, gap):
            x = table.reindex(columns=cols).copy()
            x["motion_id"] = np.nan
            return booster.predict(x)
        return predict

    net = deep.make_model(kind, torch.load(folder / f"presence_{kind}.pt", map_location="cpu"), n_classes=2)
    return lambda table, seq, gap: deep.predict_proba(net, seq, gap)[:, 1]

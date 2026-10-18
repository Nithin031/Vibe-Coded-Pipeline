"""Deep models for ESP32 windows: a 1D-CNN and a CSI-BERT2-style transformer.

Input of both: one window = relative fluctuation [32 subcarrier bins, 60 steps]
(3 s at 20 Hz) plus a gap mask [60] marking steps without a real packet.

CSITransformer follows CSI-BERT2 (Zhao et al., IEEE TMC 2025):
    - one token per time step (32 bins -> d_model) with an Adaptive Re-weighting
      Layer (a learned per-bin gate)
    - sinusoidal time embedding of the real step time, passed through an MLP
    - a learned [PAD] token replaces steps whose packet was lost
    - a learned [MASK] token; random masking (15-70 %) for self-supervised
      pre-training (reconstruct the masked steps), light masking (0-30 %)
      during fine-tuning so the classifier tolerates packet loss
    - attention pooling over tokens -> class logits

Calibration (V2): every window is divided by its site's empty-room level
(`ref_level`), so 1.0 means "as much fluctuation as the empty room".
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainlog import log as tlog

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = [0, 1, 2, 3]

# global scale so typical inputs are O(1); relative fluctuation is ~0.01-0.3
INPUT_SCALE = 10.0


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CNN1D(nn.Module):
    """Small temporal CNN; mean + std pooling over time."""

    def __init__(self, n_classes=4, n_bins=32, ch=64, dropout=0.3):
        super().__init__()

        def block(cin, cout, k):
            return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.GELU())

        self.features = nn.Sequential(
            block(n_bins, ch, 7), nn.MaxPool1d(2),
            block(ch, ch, 5), nn.MaxPool1d(2),
            block(ch, ch, 3),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * ch, n_classes))

    def forward(self, x, gap=None):
        h = self.features(x)
        return self.head(torch.cat([h.mean(dim=2), h.std(dim=2)], dim=1))


class ARL(nn.Module):
    """Adaptive Re-weighting Layer (CSI-BERT2): x * sigmoid(MLP(x))."""

    def __init__(self, n):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(n, n), nn.GELU(), nn.Linear(n, n))

    def forward(self, x):
        return x * torch.sigmoid(self.gate(x))


def sinusoid(t, dim):
    """Sinusoidal embedding of real-valued times t [B, T] -> [B, T, dim]."""
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t[..., None] * freq * 20.0                        # 20 = steps per second
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class CSITransformer(nn.Module):
    def __init__(self, n_classes=4, n_bins=32, d=64, layers=4, heads=4, ff=128, dropout=0.1):
        super().__init__()
        self.d = d
        self.arl = ARL(n_bins)
        self.embed = nn.Sequential(nn.Linear(n_bins, d), nn.GELU(), nn.Linear(d, d))
        self.time = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.pad_tok = nn.Parameter(torch.zeros(d))
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.norm = nn.LayerNorm(d)

        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, activation="gelu",
                                         norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)

        self.recon = nn.Linear(d, n_bins)                      # pre-training head
        self.pool = nn.Linear(d, 1)                            # attention pooling
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, n_classes))

        nn.init.normal_(self.pad_tok, std=0.02)
        nn.init.normal_(self.mask_tok, std=0.02)

    def encode(self, x, gap, mask=None):
        """x [B, bins, T], gap [B, T] bool, mask [B, T] bool -> tokens [B, T, d]."""
        tok = self.embed(self.arl(x.transpose(1, 2)))          # [B, T, d]
        tok = torch.where(gap[..., None], self.pad_tok.expand_as(tok), tok)
        if mask is not None:
            tok = torch.where(mask[..., None], self.mask_tok.expand_as(tok), tok)

        t = torch.arange(x.shape[2], device=x.device, dtype=torch.float32)[None].expand(x.shape[0], -1) / 20.0
        tok = tok + self.time(sinusoid(t, self.d))
        return self.encoder(self.norm(tok))

    def forward(self, x, gap, mask=None):
        h = self.encode(x, gap, mask)
        w = torch.softmax(self.pool(h).squeeze(-1), dim=1)     # [B, T]
        return self.head((w[..., None] * h).sum(dim=1))

    def reconstruct(self, x, gap, mask):
        return self.recon(self.encode(x, gap, mask)).transpose(1, 2)   # [B, bins, T]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_tensor(seq, gap, ref=None):
    """float16 [N, 32, T] (+ per-window empty-room level) -> scaled float32 tensors."""
    x = seq.astype(np.float32) * INPUT_SCALE
    if ref is not None:
        x = x / np.asarray(ref, dtype=np.float32)[:, None, None]
    return torch.from_numpy(x), torch.from_numpy(np.asarray(gap, dtype=bool))


def random_mask(gap, lo, hi, gen):
    """Mask a random fraction in [lo, hi] of the real (non-gap) steps per window."""
    b, t = gap.shape
    ratio = torch.empty(b, 1, device=gap.device).uniform_(lo, hi, generator=gen)
    m = torch.rand(b, t, device=gap.device, generator=gen) < ratio
    return m & ~gap


# ---------------------------------------------------------------------------
# Self-supervised pre-training (transformer)
# ---------------------------------------------------------------------------

def pretrain(seq, gap, epochs=40, seed=0, lr=1e-3, bs=128, log=tlog):
    """Masked reconstruction on unlabelled windows; returns the encoder."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    X, G = to_tensor(seq, gap)
    X, G = X.to(DEVICE), G.to(DEVICE)

    model = CSITransformer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(X) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    for ep in range(epochs):
        perm = torch.randperm(len(X), device=DEVICE)
        tot = 0.0
        model.train()
        for b in range(0, len(X), bs):
            j = perm[b: b + bs]
            x, g = X[j], G[j]
            m = random_mask(g, 0.15, 0.70, gen)
            rec = model.reconstruct(x, g, m)
            loss = F.mse_loss(rec.transpose(1, 2)[m], x.transpose(1, 2)[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(j)
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  pretrain epoch {ep:3d}  masked MSE {tot / len(X):.4f}")
    return model


# ---------------------------------------------------------------------------
# Supervised training / prediction
# ---------------------------------------------------------------------------

def make_model(kind, init_state=None, n_classes=4):
    model = CNN1D(n_classes=n_classes) if kind == "cnn" else CSITransformer(n_classes=n_classes)
    if init_state is not None:
        current = model.state_dict()
        compatible = {k: v for k, v in init_state.items()
                      if k in current and current[k].shape == v.shape}
        model.load_state_dict(compatible, strict=False)
    return model.to(DEVICE)


def fit(kind, seq, gap, y, ref=None, epochs=60, seed=0, lr=None, bs=64,
        init_state=None, tag="", n_classes=4):
    """Train a CNN or transformer classifier on windows (class-balanced)."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    X, G = to_tensor(seq, gap, ref)
    X, G = X.to(DEVICE), G.to(DEVICE)
    Y = torch.as_tensor(np.searchsorted(CLASSES, y), device=DEVICE)

    freq = torch.bincount(Y, minlength=n_classes).float()
    w = torch.where(freq > 0, freq.sum() / freq.clamp(min=1) / (freq > 0).sum(), torch.zeros_like(freq))

    model = make_model(kind, init_state, n_classes=n_classes)
    lr = lr or (2e-3 if kind == "cnn" else 5e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(X) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

    for ep in range(epochs):
        perm = torch.randperm(len(X), device=DEVICE)
        model.train()
        tot_loss, correct, seen = 0.0, 0, 0
        for b in range(0, len(X), bs):
            j = perm[b: b + bs]
            if len(j) < 2:
                continue
            x, g = X[j], G[j]

            # augmentation: fluctuation gain, circular time shift, packet drop-out
            x = x * torch.empty(len(j), 1, 1, device=DEVICE).uniform_(0.8, 1.25, generator=gen)
            shift = int(torch.randint(0, x.shape[2], (1,), generator=gen, device=DEVICE))
            x, g = torch.roll(x, shift, dims=2), torch.roll(g, shift, dims=1)

            if kind == "cnn":
                logits = model(x)
            else:
                logits = model(x, g, random_mask(g, 0.0, 0.3, gen))
            loss = loss_fn(logits, Y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            tot_loss += loss.item() * len(j)
            correct += (logits.argmax(1) == Y[j]).sum().item()
            seen += len(j)

        if ep % 10 == 0 or ep == epochs - 1:
            tlog(f"  {tag:28s} epoch {ep:3d}/{epochs}  loss {tot_loss / max(seen, 1):.4f}  "
                 f"train acc {correct / max(seen, 1):.3f}")
    return model


@torch.no_grad()
def predict_proba(model, seq, gap, ref=None, bs=512) -> np.ndarray:
    model.eval()
    X, G = to_tensor(seq, gap, ref)
    out = []
    for b in range(0, len(X), bs):
        x, g = X[b: b + bs].to(DEVICE), G[b: b + bs].to(DEVICE)
        logits = model(x) if isinstance(model, CNN1D) else model(x, g)
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out)

#!/usr/bin/env python3
"""
Train a U-Net to predict DRV density maps from placement feature maps.

Input  : (4, 256, 256) – [pin_density, AP, stripe, cell_type_mixture]
Output : (1, 256, 256) – predicted DRV map

Architecture choice: U-Net
  • Dense prediction (pixel → pixel) with skip connections
  • Multi-scale feature capture through encoder-decoder
  • Naturally handles spatial correlation of placement → DRV
  • Works well on small datasets compared to plain FCN

Loss: Weighted MSE + MAE (weight = 1 + α·gt)
  • DRV maps are ~88% near-zero; pure MSE would reward predicting all-zeros
  • Hot spots are up-weighted by factor (1 + alpha) to counteract sparsity

Usage
-----
    python train.py                          # defaults
    python train.py --epochs 300 --lr 3e-4
    python train.py --processed_dir /path/to/processed --out_dir /path/to/results
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from map_models import PlacementMapDataset


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Augmented dataset wrapper
# ──────────────────────────────────────────────────────────────────────────────

class AugmentedDataset(Dataset):
    """
    Wraps PlacementMapDataset with random spatial augmentations.

    Applied transforms (all preserve placement-map semantics):
      • Random horizontal flip
      • Random vertical flip
      • Random 90°/180°/270° rotation
      • Light Gaussian noise on input channels (σ = 0.005)

    Effective dataset size is the same (no synthetic oversampling needed;
    the random transforms provide variety every epoch).
    """

    def __init__(self, base_dataset, augment=True, noise_std=0.005, resolution=256):
        self.ds         = base_dataset
        self.augment    = augment
        self.noise_std  = noise_std
        self.resolution = resolution

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]       # x: (C,H,W)  y: (1,H,W)  — always 256×256 from disk
        if self.augment:
            # Spatial augmentation at full resolution for best quality
            if random.random() > 0.5:
                x = torch.flip(x, dims=[-1])
                y = torch.flip(y, dims=[-1])
            if random.random() > 0.5:
                x = torch.flip(x, dims=[-2])
                y = torch.flip(y, dims=[-2])
            k = random.randint(0, 3)
            if k:
                x = torch.rot90(x, k, dims=[-2, -1])
                y = torch.rot90(y, k, dims=[-2, -1])
            if self.noise_std > 0:
                x = (x + torch.randn_like(x) * self.noise_std).clamp(0.0, 1.0)
        # Downsample after augmentation so flips/rotations stay clean
        if self.resolution != 256:
            x, y = _resize(x, y, self.resolution)
        return x, y


# ──────────────────────────────────────────────────────────────────────────────
# Resize helper
# ──────────────────────────────────────────────────────────────────────────────

def _resize(x, y, resolution):
    """
    Bilinear downsample (x, y) tensors to (resolution × resolution).

    Augment first at full 256×256, then call this — bilinear downsampling
    of continuous density maps is cleaner than upscaling low-res augmentations.
    DRV target is clipped to [0, 1] after downsampling.
    """
    size = (resolution, resolution)
    x = F.interpolate(x.unsqueeze(0), size=size, mode='bilinear',
                      align_corners=False).squeeze(0)
    y = F.interpolate(y.unsqueeze(0), size=size, mode='bilinear',
                      align_corners=False).squeeze(0).clamp(0.0, 1.0)
    return x, y


# ──────────────────────────────────────────────────────────────────────────────
# U-Net
# ──────────────────────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for 256×256 → 256×256 sparse regression.

    Encoder:     in_ch → C → 2C → 4C → 8C      (4 max-pool levels)
    Bottleneck:  8C → 16C  (with dropout)
    Decoder:     mirrors encoder, skip connections at each level
    Output head: Conv1×1 → Sigmoid  (keeps prediction in [0, 1])

    Default base_ch=16 → ~1.9M parameters.
    Suitable for ~34 training samples with aggressive augmentation.
    """

    def __init__(self, in_ch=3, base_ch=16, dropout=0.3):
        super().__init__()
        C = base_ch

        # Encoder
        self.enc1 = _ConvBlock(in_ch, C)
        self.enc2 = _ConvBlock(C,     C*2)
        self.enc3 = _ConvBlock(C*2,   C*4)
        self.enc4 = _ConvBlock(C*4,   C*8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _ConvBlock(C*8, C*16, dropout=dropout)

        # Decoder
        self.up4  = nn.ConvTranspose2d(C*16, C*8,  2, stride=2)
        self.dec4 = _ConvBlock(C*16, C*8)
        self.up3  = nn.ConvTranspose2d(C*8,  C*4,  2, stride=2)
        self.dec3 = _ConvBlock(C*8,  C*4)
        self.up2  = nn.ConvTranspose2d(C*4,  C*2,  2, stride=2)
        self.dec2 = _ConvBlock(C*4,  C*2)
        self.up1  = nn.ConvTranspose2d(C*2,  C,    2, stride=2)
        self.dec1 = _ConvBlock(C*2,  C)

        # Output head
        self.head = nn.Sequential(nn.Conv2d(C, 1, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def drv_loss(pred, target, alpha=20.0):
    """
    Weighted MSE + Weighted MAE.
    Pixel weight = 1 + alpha * target, so DRV hotspots drive the loss.
    alpha=20 means a fully-lit DRV bin counts 21× more than an empty bin.
    """
    w    = 1.0 + alpha * target
    mse  = (w * (pred - target) ** 2).mean()
    mae  = (w * (pred - target).abs()).mean()
    return 0.6 * mse + 0.4 * mae


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def batch_metrics(pred, target, drv_threshold=0.1):
    """
    Compute per-batch metrics. All tensors are CPU numpy after this call.

    Returns dict with:
      mse, rmse, mae          – full-map regression errors
      corr                    – Pearson correlation (spatial pattern match)
      drv_mae, drv_mse        – errors restricted to gt > drv_threshold bins
    """
    p = pred.detach().cpu().float().numpy().ravel()
    t = target.detach().cpu().float().numpy().ravel()

    mse  = float(np.mean((p - t) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(p - t)))

    if p.std() > 1e-8 and t.std() > 1e-8:
        corr = float(np.corrcoef(p, t)[0, 1])
    else:
        corr = 0.0

    mask = t > drv_threshold
    drv_mse = float(np.mean((p[mask] - t[mask]) ** 2)) if mask.any() else 0.0
    drv_mae = float(np.mean(np.abs(p[mask] - t[mask]))) if mask.any() else 0.0

    return dict(mse=mse, rmse=rmse, mae=mae, corr=corr,
                drv_mse=drv_mse, drv_mae=drv_mae)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def _pick_test_indices(dataset):
    """Return [last_aes_idx, last_jpeg_idx] as held-out test samples."""
    aes_idx = jpeg_idx = None
    for i, f in enumerate(dataset.files):
        name = os.path.basename(f)
        if 'aes' in name:
            aes_idx = i
        elif 'jpeg' in name:
            jpeg_idx = i
    return [idx for idx in [aes_idx, jpeg_idx] if idx is not None]


def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── data ─────────────────────────────────────────────────────────────────
    base_ds = PlacementMapDataset(args.processed_dir)
    in_ch   = len(base_ds.input_idx)

    # Hold out 1 aes + 1 jpeg for testing
    test_indices  = _pick_test_indices(base_ds)
    train_indices = [i for i in range(len(base_ds)) if i not in set(test_indices)]
    print(f"Test samples  ({len(test_indices)}): "
          f"{[base_ds.sample_name(i) for i in test_indices]}")
    print(f"Train samples ({len(train_indices)})")

    train_subset = Subset(base_ds, train_indices)
    train_ds = AugmentedDataset(train_subset, augment=True, noise_std=0.005,
                                resolution=args.resolution)
    loader   = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0, pin_memory=False)
    print(f"Dataset: {len(train_subset)} train samples  "
          f"(batch_size={args.batch_size}, {len(loader)} batches/epoch)  "
          f"in_ch={in_ch}  resolution={args.resolution}×{args.resolution}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = UNet(in_ch=in_ch, base_ch=args.base_ch, dropout=args.dropout).to(device)
    print(f"U-Net parameters: {count_parameters(model):,}")

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── history ───────────────────────────────────────────────────────────────
    history = {k: [] for k in ('loss', 'mse', 'rmse', 'mae', 'corr', 'drv_mse', 'drv_mae')}
    best_loss = float('inf')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── training epochs ───────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs …")
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_met  = {k: 0.0 for k in ('mse', 'rmse', 'mae', 'corr', 'drv_mse', 'drv_mae')}

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = drv_loss(pred, y, alpha=args.loss_alpha)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss += loss.item()
            met = batch_metrics(pred, y)
            for k in ep_met:
                ep_met[k] += met[k]

        scheduler.step()

        n = len(loader)
        ep_loss /= n
        for k in ep_met:
            ep_met[k] /= n

        history['loss'].append(ep_loss)
        for k in ep_met:
            history[k].append(ep_met[k])

        # Save best model
        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'loss': best_loss}, os.path.join(args.out_dir, 'best_model.pt'))

        # Logging
        if epoch % args.log_every == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  epoch {epoch:4d}/{args.epochs}  "
                  f"loss={ep_loss:.5f}  mse={ep_met['mse']:.5f}  "
                  f"rmse={ep_met['rmse']:.5f}  mae={ep_met['mae']:.5f}  "
                  f"corr={ep_met['corr']:.4f}  "
                  f"drv_mse={ep_met['drv_mse']:.5f}  "
                  f"drv_mae={ep_met['drv_mae']:.5f}  "
                  f"lr={lr_now:.2e}")

    print(f"\nBest loss: {best_loss:.5f}")

    # ── load best weights before evaluation ──────────────────────────────────
    ckpt = torch.load(os.path.join(args.out_dir, 'best_model.pt'),
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])

    # ── save training curves ──────────────────────────────────────────────────
    _plot_curves(history, os.path.join(args.out_dir, 'training_curves.png'))

    # ── prediction plots for held-out test samples ────────────────────────────
    for test_idx in test_indices:
        name   = base_ds.sample_name(test_idx)
        design = 'aes' if 'aes' in name else 'jpeg'
        _plot_prediction(model, base_ds, device, args.out_dir,
                         sample_idx=test_idx, resolution=args.resolution,
                         filename=f'prediction_test_{design}.png')

    # ── metrics: train samples ────────────────────────────────────────────────
    _eval_all(model, base_ds, device, args.out_dir, resolution=args.resolution,
              indices=train_indices, label='train')

    # ── metrics: test samples ─────────────────────────────────────────────────
    _eval_all(model, base_ds, device, args.out_dir, resolution=args.resolution,
              indices=test_indices, label='test')

    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_all(model, dataset, device, out_dir, resolution=256, indices=None, label=''):
    """Compute and print per-sample + aggregate metrics.

    indices: list of dataset indices to evaluate; defaults to all samples.
    label:   suffix for the output filename ('train' → eval_metrics_train.txt).
    """
    if indices is None:
        indices = list(range(len(dataset)))
    model.eval()
    rows = []
    for idx in indices:
        x, y = dataset[idx]
        if resolution != 256:
            x, y = _resize(x, y, resolution)
        pred = model(x.unsqueeze(0).to(device)).squeeze(0)
        m = batch_metrics(pred, y)
        m['sample'] = dataset.sample_name(idx)
        rows.append(m)

    keys = ('mse', 'rmse', 'mae', 'corr', 'drv_mse', 'drv_mae')
    header = f"{'sample':45s}" + "".join(f"  {k:>10s}" for k in keys)
    lines  = [header, "-" * len(header)]
    for r in rows:
        lines.append(f"{r['sample']:45s}" +
                     "".join(f"  {r[k]:10.5f}" for k in keys))
    lines.append("-" * len(header))
    lines.append(f"{'AVERAGE':45s}" +
                 "".join(f"  {np.mean([r[k] for r in rows]):10.5f}" for k in keys))

    txt = "\n".join(lines)
    section = f" [{label.upper()}]" if label else ""
    print(f"\nMetrics{section}\n" + txt)
    fname = f'eval_metrics_{label}.txt' if label else 'eval_metrics.txt'
    path  = os.path.join(out_dir, fname)
    with open(path, 'w') as f:
        f.write(txt + "\n")
    print(f"Metrics table → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

_DARK = '#12121e'
_DARK2 = '#1a1a2e'
_GRAY = '#888888'
_LGRAY = '#aaaaaa'
_BORDER = '#333355'


def _ax_style(ax, title, xlabel='bin (x)', ylabel='bin (y)'):
    ax.set_facecolor(_DARK)
    ax.set_title(title, color='white', fontsize=11, pad=6)
    ax.tick_params(colors=_LGRAY, labelsize=7)
    ax.set_xlabel(xlabel, color=_GRAY, fontsize=8)
    ax.set_ylabel(ylabel, color=_GRAY, fontsize=8)
    for sp in ax.spines.values():
        sp.set_color(_BORDER)


def _colorbar(fig, im, ax):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=_LGRAY, labelsize=7)
    plt.setp(cb.ax.get_yticklabels(), color=_LGRAY)


@torch.no_grad()
def _plot_prediction(model, dataset, device, out_dir, sample_idx=0, resolution=256,
                     filename='prediction_sample.png'):
    """
    2-row figure, ncols = max(n_input_channels, 3):
      Row 0: one panel per input channel
      Row 1: ground truth DRV | predicted DRV | absolute error  (rest hidden)
    """
    _INPUT_CFG = [
        ('Input: Pin Density', 'viridis'),
        ('Input: AP',          'plasma'),
        ('Input: Stripe',      'Blues'),
        ('Input: Cell-Type',   'RdPu'),
    ]

    model.eval()
    x, y = dataset[sample_idx]
    if resolution != 256:
        x, y = _resize(x, y, resolution)
    pred = model(x.unsqueeze(0).to(device)).squeeze(0)  # (1,H,W)

    x_np    = x.cpu().numpy()       # (C,H,W)
    y_np    = y.cpu().numpy()[0]    # (H,W)
    pred_np = pred.cpu().numpy()[0]

    n_input = x_np.shape[0]
    ncols   = max(n_input, 3)

    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 12))
    fig.patch.set_facecolor(_DARK)

    # ── Row 0: inputs ────────────────────────────────────────────────────────
    for i, (data, (title, cmap)) in enumerate(zip(x_np, _INPUT_CFG)):
        ax = axes[0][i]
        im = ax.imshow(data, origin='lower', cmap=cmap, interpolation='nearest')
        _ax_style(ax, title)
        _colorbar(fig, im, ax)
    for i in range(n_input, ncols):
        axes[0][i].set_visible(False)

    # ── Row 1: ground truth / prediction / error ──────────────────────────────
    vmax = max(y_np.max(), pred_np.max(), 1e-6)

    ax_gt = axes[1][0]
    im = ax_gt.imshow(y_np, origin='lower', cmap='hot', vmin=0, vmax=vmax,
                      interpolation='nearest')
    _ax_style(ax_gt, 'Ground Truth DRV')
    _colorbar(fig, im, ax_gt)

    ax_pr = axes[1][1]
    im = ax_pr.imshow(pred_np, origin='lower', cmap='hot', vmin=0, vmax=vmax,
                      interpolation='nearest')
    _ax_style(ax_pr, 'Predicted DRV')
    _colorbar(fig, im, ax_pr)

    ax_er = axes[1][2]
    err = np.abs(pred_np - y_np)
    im = ax_er.imshow(err, origin='lower', cmap='RdYlGn_r', vmin=0, vmax=vmax,
                      interpolation='nearest')
    m = batch_metrics(pred.cpu(), y.cpu())
    _ax_style(ax_er, f'|Error|  (MAE={m["mae"]:.4f})')
    _colorbar(fig, im, ax_er)

    for i in range(3, ncols):
        axes[1][i].set_visible(False)

    name = dataset.sample_name(sample_idx)
    fig.suptitle(
        f'{name}\n'
        f'MSE={m["mse"]:.5f}  RMSE={m["rmse"]:.5f}  MAE={m["mae"]:.5f}  '
        f'Corr={m["corr"]:.4f}  DRV-MSE={m["drv_mse"]:.5f}  DRV-MAE={m["drv_mae"]:.5f}',
        color='white', fontsize=11, y=1.02, linespacing=1.5)
    plt.tight_layout()

    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Prediction plot → {path}")


def _plot_curves(history, out_path):
    """2×3 grid of training curves."""
    keys   = ['loss',  'mse',  'rmse', 'mae',  'corr',    'drv_mse']
    titles = ['Loss',  'MSE',  'RMSE', 'MAE',  'Pearson Corr', 'DRV-region MSE']
    colors = ['#00e5ff', '#ff6d00', '#76ff03', '#e040fb', '#ffea00', '#ff1744']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor(_DARK)

    for ax, key, title, color in zip(axes.ravel(), keys, titles, colors):
        vals = history.get(key, [])
        ax.set_facecolor(_DARK2)
        ax.plot(vals, color=color, linewidth=1.5)
        ax.set_title(title, color='white', fontsize=11)
        ax.tick_params(colors=_LGRAY, labelsize=8)
        ax.set_xlabel('epoch', color=_GRAY, fontsize=8)
        for sp in ax.spines.values():
            sp.set_color(_BORDER)
        if vals:
            ax.set_ylabel(f'final = {vals[-1]:.5f}', color=_GRAY, fontsize=8)

    fig.suptitle('Training Curves', color='white', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Training curves → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parser():
    base = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument('--processed_dir', default=os.path.join(base, 'data', 'processed'))
    p.add_argument('--out_dir',       default=None,
                   help='Output directory (default: results/res<N>)')
    p.add_argument('--resolution', type=int, default=256, choices=[32, 64, 128, 256],
                   help='Training resolution. 256×256 .pt maps are downsampled on-the-fly.')
    p.add_argument('--epochs',     type=int,   default=300)
    p.add_argument('--batch_size', type=int,   default=8)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--base_ch',    type=int,   default=16,
                   help='U-Net base channel count (default 16 → ~1.9M params)')
    p.add_argument('--dropout',    type=float, default=0.3)
    p.add_argument('--loss_alpha', type=float, default=20.0,
                   help='Weight multiplier for DRV hotspot bins in the loss')
    p.add_argument('--log_every',  type=int,   default=10)
    p.add_argument('--seed',       type=int,   default=42)
    return p


if __name__ == '__main__':
    args = _parser().parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    if args.out_dir is None:
        args.out_dir = os.path.join(base, 'results', f'res{args.resolution}')

    print("=" * 65)
    print("  DRV Map Predictor  –  U-Net Training")
    print("=" * 65)
    print(f"  processed_dir : {args.processed_dir}")
    print(f"  out_dir       : {args.out_dir}")
    print(f"  resolution    : {args.resolution}×{args.resolution}")
    print(f"  epochs        : {args.epochs}")
    print(f"  batch_size    : {args.batch_size}")
    print(f"  lr            : {args.lr}  weight_decay={args.weight_decay}")
    print(f"  base_ch       : {args.base_ch}  dropout={args.dropout}")
    print(f"  loss_alpha    : {args.loss_alpha}")
    print("=" * 65 + "\n")

    train(args)

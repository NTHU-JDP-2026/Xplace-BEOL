#!/usr/bin/env python3
"""
Four differentiable/smooth feature maps for deep-learning placement training.

Maps (all 256×256 float32 tensors):
  1. DensityMapModel    – Gaussian-smeared cell density  (differentiable w.r.t. cx, cy)
  2. APMapModel         – Gaussian-smeared pin AP map    (differentiable w.r.t. cx, cy)
  3. PowerStripeMapModel– Smooth M3/M4 power-stripe map (fixed buffer)
  4. DRVMapModel        – Gaussian-smeared DRV map       (fixed buffer, ground truth)

Smoothing is fully parameterised — see build_maps() kwargs.
map[i, j]: row i → y-direction, col j → x-direction, origin at bottom-left.
"""

import re
import json
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────────────

def parse_lef(lef_paths):
    """Return {macro_name: (width_um, height_um)} from one or more LEF files."""
    macros = {}
    pat = re.compile(r'^MACRO (\S+).*?SIZE ([\d.]+) BY ([\d.]+)', re.DOTALL | re.MULTILINE)
    for path in lef_paths:
        with open(path) as f:
            text = f.read()
        for m in pat.finditer(text):
            macros[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return macros


def parse_def(def_path, macro_sizes=None):
    """
    Parse DEF and return a dict with:
      dbu         – database units per micron
      die         – (x0, y0, x1, y1) in DBU
      components  – list of (inst, cell_type, xl, yl, w, h) in DBU
      m3_stripes  – list of (xc, y0, y1, width) in DBU  [vertical]
      m4_stripes  – list of (x0, x1, yc, height) in DBU [horizontal]
    """
    with open(def_path) as f:
        content = f.read()

    m = re.search(r'UNITS DISTANCE MICRONS (\d+)', content)
    dbu = int(m.group(1)) if m else 4000

    m = re.search(r'DIEAREA\s+\(\s*(\d+)\s+(\d+)\s*\)\s+\(\s*(\d+)\s+(\d+)\s*\)', content)
    die = tuple(int(m.group(i)) for i in range(1, 5))

    fallback_w = round(1.728 * dbu)
    fallback_h = round(1.08  * dbu)

    comp_section = re.search(r'COMPONENTS \d+ ;(.*?)END COMPONENTS', content, re.DOTALL)
    comp_pat = re.compile(
        r'-\s+(\S+)\s+(\S+)\s*\n\s+\+\s+PLACED\s+\(\s*(\d+)\s+(\d+)\s*\)')
    components = []
    if comp_section:
        for m in comp_pat.finditer(comp_section.group(1)):
            inst, cell_type = m.group(1), m.group(2)
            x, y = int(m.group(3)), int(m.group(4))
            if macro_sizes and cell_type in macro_sizes:
                w = round(macro_sizes[cell_type][0] * dbu)
                h = round(macro_sizes[cell_type][1] * dbu)
            else:
                w, h = fallback_w, fallback_h
            components.append((inst, cell_type, x, y, w, h))

    spec_section = re.search(r'SPECIALNETS \d+ ;(.*?)END SPECIALNETS', content, re.DOTALL)
    m3_stripes, m4_stripes = [], []
    if spec_section:
        sc = spec_section.group(1)
        p3 = re.compile(r'M3\s+(\d+)\s+\+\s+SHAPE\s+STRIPE\s+\(\s*(\d+)\s+(\d+)\s*\)'
                        r'(?:\s+MASK\s+\d+)?\s+\(\s*\*\s+(\d+)\s*\)')
        for m in p3.finditer(sc):
            sw = int(m.group(1))
            if sw > 0:
                m3_stripes.append((int(m.group(2)), int(m.group(3)), int(m.group(4)), sw))
        p4 = re.compile(r'M4\s+(\d+)\s+\+\s+SHAPE\s+STRIPE\s+\(\s*(\d+)\s+(\d+)\s*\)'
                        r'(?:\s+MASK\s+\d+)?\s+\(\s*(\d+)\s+\*\s*\)')
        for m in p4.finditer(sc):
            sw = int(m.group(1))
            if sw > 0:
                m4_stripes.append((int(m.group(2)), int(m.group(4)), int(m.group(3)), sw))

    return dict(dbu=dbu, die=die, components=components,
                m3_stripes=m3_stripes, m4_stripes=m4_stripes)


def load_ap_data(json_paths):
    """Return {cell_type: ap_median} merged from one or more AP JSON files."""
    ap_map = {}
    for path in json_paths:
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            ap_map[entry['cell']] = float(entry['ap_median'])
    return ap_map


def parse_drc(drc_path, dbu):
    """Return list of (cx_dbu, cy_dbu) for each DRV bounding-box centre."""
    violations = []
    pat = re.compile(
        r'Bounds\s*:\s*\(\s*([\d.]+),\s*([\d.]+)\s*\)\s*\(\s*([\d.]+),\s*([\d.]+)\s*\)')
    with open(drc_path) as f:
        for line in f:
            bm = pat.search(line)
            if bm:
                x1 = float(bm.group(1)) * dbu
                y1 = float(bm.group(2)) * dbu
                x2 = float(bm.group(3)) * dbu
                y2 = float(bm.group(4)) * dbu
                violations.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
    return violations


# ──────────────────────────────────────────────────────────────────────────────
# Shared helper
# ──────────────────────────────────────────────────────────────────────────────

def _separable_gaussian_map(cx, cy, sigma_x, sigma_y, weights, bins):
    """
    256×256 density map via separable 1-D Gaussians.

    map[i, j] = Σ_n  weights[n] * Gy_n[i] * Gx_n[j]

    Gx_n and Gy_n are discrete normalised (sum = 1), so total weight is
    conserved and gradients flow cleanly through the softmax-like normalisation.

    Cost: O(N·G) + one (G×N)@(N×G) matmul  instead of O(N·G²).
    """
    Gx = torch.exp(-0.5 * ((bins.unsqueeze(0) - cx.unsqueeze(1)) / sigma_x.unsqueeze(1)) ** 2)
    Gy = torch.exp(-0.5 * ((bins.unsqueeze(0) - cy.unsqueeze(1)) / sigma_y.unsqueeze(1)) ** 2)
    Gx = Gx / (Gx.sum(dim=1, keepdim=True) + 1e-9)
    Gy = Gy / (Gy.sum(dim=1, keepdim=True) + 1e-9)
    return (weights.unsqueeze(1) * Gy).T @ Gx   # (G, G)


# ──────────────────────────────────────────────────────────────────────────────
# Model 1 – Differentiable Density Map
# ──────────────────────────────────────────────────────────────────────────────

class DensityMapModel(nn.Module):
    """
    Differentiable cell-density map.

    Each cell n contributes a 2-D Gaussian weighted by normalised cell area.
    σ per cell = max(physical_half_size × smooth_factor, min_sigma_bins / G).

    cx, cy are nn.Parameters → gradients flow back to placement positions.
    """

    def __init__(self, cx_norm, cy_norm, sigma_x, sigma_y, area_weights, grid_size=256):
        super().__init__()
        self.grid_size = grid_size
        self.cx = nn.Parameter(cx_norm.clone())
        self.cy = nn.Parameter(cy_norm.clone())
        self.register_buffer('sigma_x',      sigma_x)
        self.register_buffer('sigma_y',      sigma_y)
        self.register_buffer('area_weights', area_weights)
        bins = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        self.register_buffer('bins', bins)

    def forward(self):
        return _separable_gaussian_map(
            self.cx, self.cy, self.sigma_x, self.sigma_y,
            self.area_weights, self.bins)


# ──────────────────────────────────────────────────────────────────────────────
# Model 2 – Differentiable AP Map
# ──────────────────────────────────────────────────────────────────────────────

class APMapModel(nn.Module):
    """
    Differentiable pin access-point density map.

    Same Gaussian kernel as DensityMapModel; weight = ap_median instead of area.
    A cell with more pin APs exerts greater influence on its neighbouring bins.
    """

    def __init__(self, cx_norm, cy_norm, sigma_x, sigma_y, ap_weights, grid_size=256):
        super().__init__()
        self.grid_size = grid_size
        self.cx = nn.Parameter(cx_norm.clone())
        self.cy = nn.Parameter(cy_norm.clone())
        self.register_buffer('sigma_x',    sigma_x)
        self.register_buffer('sigma_y',    sigma_y)
        self.register_buffer('ap_weights', ap_weights)
        bins = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        self.register_buffer('bins', bins)

    def forward(self):
        return _separable_gaussian_map(
            self.cx, self.cy, self.sigma_x, self.sigma_y,
            self.ap_weights, self.bins)


# ──────────────────────────────────────────────────────────────────────────────
# Model 3 – Power Stripe Map
# ──────────────────────────────────────────────────────────────────────────────

class PowerStripeMapModel(nn.Module):
    """
    Smooth power-stripe map (precomputed fixed buffer).

    M3 vertical   : Gaussian in x (σ = m3_sigma_bins / G), box in y.
    M4 horizontal : Gaussian in y (σ = m4_sigma_bins / G), box in x.

    M3 and M4 use **separate** sigma parameters because their inter-stripe
    spacings differ significantly (M3 ~14 bins, M4 ~23 bins on this design).
    sigma must stay below ~half the inter-stripe spacing to keep individual
    stripes distinguishable.  Normalised to [0, 1].

    Args:
        m3_sigma_bins : Gaussian σ for M3 vertical stripes (bins). Safe range
                        for this design: 1 – 6.  Default 3.
        m4_sigma_bins : Gaussian σ for M4 horizontal stripes (bins). Safe range
                        for this design: 1 – 10.  Default 5.
    """

    def __init__(self, m3_stripes, m4_stripes, die, grid_size=256,
                 m3_sigma_bins=3.0, m4_sigma_bins=5.0):
        super().__init__()
        x0, y0, x1, y1 = die
        die_w = x1 - x0
        die_h = y1 - y0
        G = grid_size
        sigma_m3 = m3_sigma_bins / G
        sigma_m4 = m4_sigma_bins / G
        bins = (torch.arange(G, dtype=torch.float32) + 0.5) / G

        stripe_map = torch.zeros(G, G)

        # M3 vertical: (xc, sy0, sy1, width_dbu)
        for xc_dbu, sy0_dbu, sy1_dbu, _sw in m3_stripes:
            xc_n  = (xc_dbu  - x0) / die_w
            sy0_n = (sy0_dbu - y0) / die_h
            sy1_n = (sy1_dbu - y0) / die_h
            gx = torch.exp(-0.5 * ((bins - xc_n) / sigma_m3) ** 2)
            gy = ((bins >= sy0_n) & (bins <= sy1_n)).float()
            if gy.sum() == 0:
                gy = torch.ones(G)
            stripe_map += torch.outer(gy, gx)

        # M4 horizontal: (sx0, sx1, yc, height_dbu)
        for sx0_dbu, sx1_dbu, yc_dbu, _sh in m4_stripes:
            yc_n  = (yc_dbu  - y0) / die_h
            sx0_n = (sx0_dbu - x0) / die_w
            sx1_n = (sx1_dbu - x0) / die_w
            gy = torch.exp(-0.5 * ((bins - yc_n) / sigma_m4) ** 2)
            gx = ((bins >= sx0_n) & (bins <= sx1_n)).float()
            if gx.sum() == 0:
                gx = torch.ones(G)
            stripe_map += torch.outer(gy, gx)

        if stripe_map.max() > 0:
            stripe_map = stripe_map / stripe_map.max()

        self.register_buffer('stripe_map', stripe_map)

    def forward(self):
        return self.stripe_map


# ──────────────────────────────────────────────────────────────────────────────
# Model 4 – DRV Map
# ──────────────────────────────────────────────────────────────────────────────

class DRVMapModel(nn.Module):
    """
    DRV (Design Rule Violation) density map – ground truth for training.

    Each DRV centre is spread with a symmetric 2-D Gaussian
    (σ = drv_sigma_bins / G).  Clusters of violations accumulate.
    Normalised to [0, 1].

    Args:
        drv_sigma_bins : spread radius in bins (default 4). Increase for
                         broader smoothing.
    """

    def __init__(self, violations, die, grid_size=256, drv_sigma_bins=4.0):
        super().__init__()
        x0, y0, x1, y1 = die
        die_w = x1 - x0
        die_h = y1 - y0
        G = grid_size
        sigma = drv_sigma_bins / G
        bins = (torch.arange(G, dtype=torch.float32) + 0.5) / G

        drv_map = torch.zeros(G, G)
        for vx, vy in violations:
            cx_n = (vx - x0) / die_w
            cy_n = (vy - y0) / die_h
            gx = torch.exp(-0.5 * ((bins - cx_n) / sigma) ** 2)
            gy = torch.exp(-0.5 * ((bins - cy_n) / sigma) ** 2)
            drv_map += torch.outer(gy, gx)

        if drv_map.max() > 0:
            drv_map = drv_map / drv_map.max()

        self.register_buffer('drv_map', drv_map)

    def forward(self):
        return self.drv_map


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_maps(
    def_path, lef_paths, ap_json_paths, drc_path,
    grid_size=256,
    # ── smoothing knobs ───────────────────────────────────────────────────────
    cell_smooth_factor=3.0,   # multiply physical cell half-size to get sigma
    cell_min_sigma_bins=2.0,  # floor: sigma never narrower than this many bins
    m3_sigma_bins=3.0,        # M3 vertical stripe σ (bins); keep < 7 on this design
    m4_sigma_bins=5.0,        # M4 horizontal stripe σ (bins); keep < 11 on this design
    drv_sigma_bins=5.0,       # DRV Gaussian spread (bins)
):
    """
    Parse all input files and construct the four map models.

    Smoothing parameters
    --------------------
    cell_smooth_factor   : float, default 3.0
        Multiplier on the physical cell half-size sigma.
        Larger → cells bleed further into neighbours.
    cell_min_sigma_bins  : float, default 2.0
        Hard floor sigma (bins) for density/AP cell Gaussians.
        Prevents tiny cells from becoming single-pixel spikes.
    m3_sigma_bins        : float, default 3.0
        Gaussian σ (bins) for M3 vertical stripe narrow axis.
        M3 inter-stripe spacing on this design ≈ 14 bins → keep < 7.
    m4_sigma_bins        : float, default 5.0
        Gaussian σ (bins) for M4 horizontal stripe narrow axis.
        M4 inter-stripe spacing on this design ≈ 23 bins → keep < 11.
    drv_sigma_bins       : float, default 5.0
        Gaussian σ (bins) for each DRV violation blob.

    Returns
    -------
    (density_model, ap_model, stripe_model, drv_model)
    """
    print("Parsing LEF …")
    macros = parse_lef(lef_paths)
    print(f"  {len(macros)} macros")

    print("Parsing DEF …")
    dd = parse_def(def_path, macro_sizes=macros)
    dbu   = dd['dbu']
    die   = dd['die']
    comps = dd['components']
    x0, y0, x1, y1 = die
    die_w, die_h = x1 - x0, y1 - y0
    print(f"  {len(comps):,} cells,  die {die_w/dbu:.2f} × {die_h/dbu:.2f} µm")
    print(f"  M3 stripes: {len(dd['m3_stripes'])},  M4 stripes: {len(dd['m4_stripes'])}")

    print("Loading AP data …")
    ap_lookup = load_ap_data(ap_json_paths)
    print(f"  {len(ap_lookup)} cell types with AP info")

    print("Parsing DRC …")
    violations = parse_drc(drc_path, dbu)
    print(f"  {len(violations):,} violations")

    G = grid_size
    min_sigma = cell_min_sigma_bins / G
    default_ap = float(np.median(list(ap_lookup.values()))) if ap_lookup else 1.0

    print(f"\nSmoothing config:")
    print(f"  cell_smooth_factor  = {cell_smooth_factor}")
    print(f"  cell_min_sigma_bins = {cell_min_sigma_bins}  ({min_sigma:.5f} normalised)")
    print(f"  m3_sigma_bins       = {m3_sigma_bins}  ({m3_sigma_bins/G:.5f} normalised)")
    print(f"  m4_sigma_bins       = {m4_sigma_bins}  ({m4_sigma_bins/G:.5f} normalised)")
    print(f"  drv_sigma_bins      = {drv_sigma_bins}  ({drv_sigma_bins/G:.5f} normalised)")

    cx_list, cy_list, sx_list, sy_list, area_list, ap_list = [], [], [], [], [], []
    for _inst, cell_type, xl, yl, w, h in comps:
        cx = (xl + w * 0.5 - x0) / die_w
        cy = (yl + h * 0.5 - y0) / die_h
        sx = max(w / (2.0 * die_w) * cell_smooth_factor, min_sigma)
        sy = max(h / (2.0 * die_h) * cell_smooth_factor, min_sigma)
        area = (w / die_w) * (h / die_h)
        ap   = ap_lookup.get(cell_type, default_ap)
        cx_list.append(cx);    cy_list.append(cy)
        sx_list.append(sx);    sy_list.append(sy)
        area_list.append(area); ap_list.append(ap)

    cx_t   = torch.tensor(cx_list,   dtype=torch.float32)
    cy_t   = torch.tensor(cy_list,   dtype=torch.float32)
    sx_t   = torch.tensor(sx_list,   dtype=torch.float32)
    sy_t   = torch.tensor(sy_list,   dtype=torch.float32)
    area_t = torch.tensor(area_list, dtype=torch.float32)
    ap_t   = torch.tensor(ap_list,   dtype=torch.float32)

    area_t = area_t / area_t.sum()
    ap_t   = ap_t   / ap_t.sum()

    print("\nConstructing models …")
    density_model = DensityMapModel(cx_t, cy_t, sx_t, sy_t, area_t, G)
    ap_model      = APMapModel     (cx_t, cy_t, sx_t, sy_t, ap_t,   G)
    stripe_model  = PowerStripeMapModel(
        dd['m3_stripes'], dd['m4_stripes'], die, G,
        m3_sigma_bins=m3_sigma_bins, m4_sigma_bins=m4_sigma_bins)
    drv_model     = DRVMapModel(violations, die, G, drv_sigma_bins=drv_sigma_bins)

    return density_model, ap_model, stripe_model, drv_model


# ──────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────────────────────

_MAP_CFGS = [
    ('density_map', 'Density Map\n(cell area)',        'viridis'),
    ('ap_map',      'AP Map\n(pin access points)',     'plasma'),
    ('stripe_map',  'Power Stripe Map\n(M3 / M4)',     'Blues'),
    ('drv_map',     'DRV Map\n(ground truth)',         'hot'),
]


def _stat_label(data):
    return f'min={data.min():.4f}   max={data.max():.4f}   mean={data.mean():.5f}'


def visualize_maps(density_map, ap_map, stripe_map, drv_map, out_path):
    """Save a combined 1×4 panel at 150 dpi."""
    tensors = [density_map, ap_map, stripe_map, drv_map]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5))
    fig.patch.set_facecolor('#12121e')

    for ax, tensor, (fname, title, cmap) in zip(axes, tensors, _MAP_CFGS):
        data = tensor.detach().cpu().numpy()
        ax.set_facecolor('#12121e')
        im = ax.imshow(data, origin='lower', cmap=cmap,
                       interpolation='nearest', aspect='equal')
        ax.set_title(title, color='white', fontsize=11, pad=6, linespacing=1.4)
        ax.tick_params(colors='#888888', labelsize=6)
        for spine in ax.spines.values():
            spine.set_color('#333355')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors='#aaaaaa', labelsize=6)
        plt.setp(cb.ax.get_yticklabels(), color='#aaaaaa')
        ax.set_xlabel(_stat_label(data), color='#888888', fontsize=6)

    fig.suptitle('aes_cipher_top  –  Feature Maps (256 × 256)',
                 color='white', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Combined  → {out_path}")


def save_individual_maps(density_map, ap_map, stripe_map, drv_map, out_dir, dpi=300):
    """Save four separate high-resolution PNG files."""
    tensors = [density_map, ap_map, stripe_map, drv_map]
    os.makedirs(out_dir, exist_ok=True)

    for tensor, (fname, title, cmap) in zip(tensors, _MAP_CFGS):
        data = tensor.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor('#12121e')
        ax.set_facecolor('#12121e')

        im = ax.imshow(data, origin='lower', cmap=cmap,
                       interpolation='nearest', aspect='equal')
        ax.set_title(title.replace('\n', ' – '), color='white', fontsize=14, pad=10)
        ax.tick_params(colors='#aaaaaa', labelsize=9)
        ax.set_xlabel('bin (x)', color='#888888', fontsize=9)
        ax.set_ylabel('bin (y)', color='#888888', fontsize=9)
        for spine in ax.spines.values():
            spine.set_color('#333355')

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors='#aaaaaa', labelsize=9)
        plt.setp(cb.ax.get_yticklabels(), color='#aaaaaa')

        fig.text(0.5, 0.01, _stat_label(data),
                 ha='center', color='#888888', fontsize=8)

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        path = os.path.join(out_dir, f'{fname}.png')
        plt.savefig(path, dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  {fname:12s} → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))

    def_path  = os.path.join(base, 'data', 'contest.gp.def')
    drc_path  = os.path.join(base, 'data', 'aes_cipher_top.drc.rpt')
    lef_paths = [os.path.join(base, 'SL_modified.lef'),
                 os.path.join(base, 'L_modified.lef')]
    ap_paths  = [os.path.join(base, 'SL_original_m2_ap.json'),
                 os.path.join(base, 'L_original_m2_ap.json')]

    # ── tune smoothing here ────────────────────────────────────────────────────
    density_model, ap_model, stripe_model, drv_model = build_maps(
        def_path, lef_paths, ap_paths, drc_path,
        cell_smooth_factor  = 3.0,  # ↑ larger → cells bleed more into neighbours
        cell_min_sigma_bins = 2.0,  # ↑ larger → even small cells stay wide
        m3_sigma_bins       = 3.0,  # M3 spacing ≈ 14 bins → keep < 7
        m4_sigma_bins       = 5.0,  # M4 spacing ≈ 23 bins → keep < 11
        drv_sigma_bins      = 5.0,  # ↑ larger → DRV violations spread wider
    )

    print("\nComputing maps …")
    with torch.no_grad():
        density_map = density_model()
        ap_map_out  = ap_model()
        stripe_map  = stripe_model()
        drv_map_out = drv_model()

    for name, t in [('density', density_map), ('ap', ap_map_out),
                    ('stripe',  stripe_map),   ('drv', drv_map_out)]:
        print(f"  {name:8s}: shape={list(t.shape)}  "
              f"min={t.min():.5f}  max={t.max():.5f}  mean={t.mean():.6f}")

    # Differentiability smoke-test (partial region loss; total sum is invariant)
    print("\nDifferentiability check (loss = upper-right quadrant) …")
    density_model()[128:, 128:].sum().backward()
    print(f"  density  ∂loss/∂cx : norm = {density_model.cx.grad.norm():.4e}  ✓")
    density_model.cx.grad = None
    ap_model()[128:, 128:].sum().backward()
    print(f"  ap       ∂loss/∂cx : norm = {ap_model.cx.grad.norm():.4e}  ✓")

    print("\nSaving outputs …")
    visualize_maps(density_map, ap_map_out, stripe_map, drv_map_out,
                   os.path.join(base, 'data', 'feature_maps.png'))
    save_individual_maps(density_map, ap_map_out, stripe_map, drv_map_out,
                         out_dir=os.path.join(base, 'data', 'maps'),
                         dpi=300)

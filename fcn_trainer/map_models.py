#!/usr/bin/env python3
"""
Four differentiable/smooth feature maps for deep-learning placement training.

Maps (all 256×256 float32 tensors):
  1. PinDensityMapModel – Gaussian-smeared pin-count density (differentiable w.r.t. cx, cy)
  2. APMapModel         – Gaussian-smeared pin AP map        (differentiable w.r.t. cx, cy)
  3. PowerStripeMapModel– Smooth M3/M4 power-stripe map     (fixed buffer)
  4. DRVMapModel        – Gaussian-smeared DRV map           (fixed buffer, ground truth)

Batch processing
----------------
  process_batch(data_dir, out_dir, lef_paths, ap_json_paths)
    Scans every subdirectory of data_dir for contest.gp.def + *.drc.rpt,
    skips samples with ≥ 1000 DRV violations, and saves a .pt file per sample.

PyTorch Dataset
---------------
  PlacementMapDataset(processed_dir)
    Loads all *.pt files produced by process_batch and returns (4, H, W) tensors.

map[i, j]: row i → y-direction, col j → x-direction, origin at bottom-left.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────────────

def count_drv(drc_path):
    """Fast DRV count: count 'Bounds' lines without building violation list."""
    n = 0
    with open(drc_path) as f:
        for line in f:
            if line.startswith('Bounds'):
                n += 1
    return n


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
    """Return {cell_type: {'ap_median': float, 'num_pins': int}} merged from one or more AP JSON files."""
    ap_map = {}
    for path in json_paths:
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            ap_map[entry['cell']] = {
                'ap_median': float(entry['ap_median']),
                'num_pins':  int(entry['num_pins']),
            }
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

    Each 1-D Gaussian is discrete-normalised (sum over bins = 1), so total
    weight is conserved and gradients flow cleanly through the normalisation.
    Cost: O(N·G) + one (G×N)@(N×G) matmul instead of O(N·G²).
    """
    Gx = torch.exp(-0.5 * ((bins.unsqueeze(0) - cx.unsqueeze(1)) / sigma_x.unsqueeze(1)) ** 2)
    Gy = torch.exp(-0.5 * ((bins.unsqueeze(0) - cy.unsqueeze(1)) / sigma_y.unsqueeze(1)) ** 2)
    Gx = Gx / (Gx.sum(dim=1, keepdim=True) + 1e-9)
    Gy = Gy / (Gy.sum(dim=1, keepdim=True) + 1e-9)
    return (weights.unsqueeze(1) * Gy).T @ Gx   # (G, G)


# ──────────────────────────────────────────────────────────────────────────────
# Model 1 – Differentiable Density Map
# ──────────────────────────────────────────────────────────────────────────────

class PinDensityMapModel(nn.Module):
    """
    Differentiable pin-count density map.

    Each cell n contributes a 2-D Gaussian weighted by its normalised pin count.
    σ per cell = max(physical_half_size × smooth_factor, min_sigma_bins / G).

    cx, cy are nn.Parameters → gradients flow back to placement positions.
    """

    def __init__(self, cx_norm, cy_norm, sigma_x, sigma_y, pin_weights, grid_size=256):
        super().__init__()
        self.grid_size = grid_size
        self.cx = nn.Parameter(cx_norm.clone())
        self.cy = nn.Parameter(cy_norm.clone())
        self.register_buffer('sigma_x',      sigma_x)
        self.register_buffer('sigma_y',      sigma_y)
        self.register_buffer('pin_weights',  pin_weights)
        bins = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        self.register_buffer('bins', bins)

    def forward(self):
        return _separable_gaussian_map(
            self.cx, self.cy, self.sigma_x, self.sigma_y,
            self.pin_weights, self.bins)


# ──────────────────────────────────────────────────────────────────────────────
# Model 2 – Differentiable AP Map
# ──────────────────────────────────────────────────────────────────────────────

class APMapModel(nn.Module):
    """
    Differentiable pin access-point density map.

    Same Gaussian kernel as PinDensityMapModel; weight = ap_median instead of pin count.
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

    M3 and M4 use separate sigma parameters because their inter-stripe spacings
    differ (M3 ≈ 14 bins, M4 ≈ 23 bins on aes_cipher_top / jpeg_encoder).
    Keep sigma < half the inter-stripe spacing to preserve individual stripes.
    Normalised to [0, 1].

    Args:
        m3_sigma_bins : σ for M3 vertical stripes (bins). Safe < 7 on this design.
        m4_sigma_bins : σ for M4 horizontal stripes (bins). Safe < 11 on this design.
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

        for xc_dbu, sy0_dbu, sy1_dbu, _sw in m3_stripes:
            xc_n  = (xc_dbu  - x0) / die_w
            sy0_n = (sy0_dbu - y0) / die_h
            sy1_n = (sy1_dbu - y0) / die_h
            gx = torch.exp(-0.5 * ((bins - xc_n) / sigma_m3) ** 2)
            gy = ((bins >= sy0_n) & (bins <= sy1_n)).float()
            if gy.sum() == 0:
                gy = torch.ones(G)
            stripe_map += torch.outer(gy, gx)

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
# Model 4 – Cell-Type Mixture Map
# ──────────────────────────────────────────────────────────────────────────────

def _cell_category(cell_type: str) -> int:
    """
    Map a cell-type name to one of four routing-complexity categories.

    0 – simple       INV*, BUF*                      (light routing)
    1 – sequential   DFF*, SDFF*, SDF*, LATCH*        (clock + data, high demand)
    2 – complex-comb AOI*, OAI*, FA*, MAJ*, AO*, OA*  (multi-input complex logic)
    3 – std-comb     everything else (NAND, NOR, AND, OR, XOR, MUX, …)
    """
    n = cell_type.upper()
    if re.match(r'(INV|BUF)', n):
        return 0
    if re.match(r'(DFF|SDFF|SDF|LATCH)', n):
        return 1
    if re.match(r'(AOI|OAI|FA|MAJ|AO[^I]|OA[^I]|HB)', n):
        return 2
    return 3


class CellTypeMapModel(nn.Module):
    """
    Differentiable cell-type mixture / diversity map.

    Cells are grouped into K=4 categories by name-pattern matching
    (simple, sequential, complex-combinational, standard-combinational).
    For each category g a density map D_g is built with separable Gaussians
    weighted by normalised pin count.

    The single output map encodes WHERE routing-intensive cell types cluster:

        D_g[i,j]  = Σ_{n∈cat g} pin_n · Gy_n[i] · Gx_n[j]   # category density
        M[i,j]    = Σ_g w_g · D_g[i,j]                        # routing-complexity sum
    then normalised to [0, 1].

    Fixed routing-complexity weights per category (higher = more routing demand):
        simple (INV/BUF)  : w = 1
        std_comb          : w = 2
        complex_comb      : w = 3
        sequential (DFF)  : w = 5

    High values → regions dominated by sequential / complex-combinational cells →
    high routing demand → correlated with DRV hotspots.

    Differentiable w.r.t. cx, cy through the Gaussian kernels.
    """

    _K = 4

    def __init__(self, cx_norm, cy_norm, sigma_x, sigma_y, pin_weights,
                 cell_categories, grid_size=256):
        """
        cell_categories : (N,) int64 tensor with values in {0,1,2,3}
        """
        super().__init__()
        self.grid_size = grid_size
        self.cx = nn.Parameter(cx_norm.clone())
        self.cy = nn.Parameter(cy_norm.clone())
        self.register_buffer('sigma_x',         sigma_x)
        self.register_buffer('sigma_y',         sigma_y)
        self.register_buffer('pin_weights',     pin_weights)
        self.register_buffer('cell_categories', cell_categories)
        bins = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        self.register_buffer('bins', bins)

    # routing-complexity weights per category (simple, std_comb, complex_comb, seq)
    _WEIGHTS = [1.0, 2.0, 3.0, 5.0]

    def forward(self):
        G   = self.grid_size
        eps = 1e-9

        M = torch.zeros(G, G, device=self.cx.device)
        for g, w_cat in enumerate(self._WEIGHTS):
            mask = (self.cell_categories == g)
            if not mask.any():
                continue
            w = self.pin_weights[mask]
            w = w / (w.sum() + eps)
            D_g = _separable_gaussian_map(
                self.cx[mask], self.cy[mask],
                self.sigma_x[mask], self.sigma_y[mask],
                w, self.bins)
            M = M + w_cat * D_g

        if M.max() > 0:
            M = M / M.max()
        return M


# ──────────────────────────────────────────────────────────────────────────────
# Model 7 – DRV Map
# ──────────────────────────────────────────────────────────────────────────────

class DRVMapModel(nn.Module):
    """
    DRV (Design Rule Violation) density map – ground truth for training.

    Each DRV centre is spread with a symmetric 2-D Gaussian
    (σ = drv_sigma_bins / G).  Clusters accumulate.  Normalised to [0, 1].

    Args:
        drv_sigma_bins : spread radius in bins (default 5).
    """

    def __init__(self, violations, die, grid_size=256, drv_sigma_bins=5.0):
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
    cell_smooth_factor=1.5,
    cell_min_sigma_bins=1.0,
    m3_sigma_bins=0,
    m4_sigma_bins=0,
    drv_sigma_bins=1.5,
    verbose=True,
):
    """
    Parse all input files and construct the five map models.

    Smoothing parameters
    --------------------
    cell_smooth_factor    : multiplier on physical cell half-size sigma.
    cell_min_sigma_bins   : floor sigma (bins) for pin-density/AP Gaussians.
    m3_sigma_bins         : σ (bins) for M3 vertical stripe narrow axis  (< 7).
    m4_sigma_bins         : σ (bins) for M4 horizontal stripe narrow axis (< 11).
    drv_sigma_bins        : σ (bins) for each DRV violation blob.

    Returns
    -------
    (pin_density_model, ap_model, stripe_model, cell_type_model, drv_model)
    """
    def log(msg):
        if verbose:
            print(msg)

    macros    = parse_lef(lef_paths)
    dd        = parse_def(def_path, macro_sizes=macros)
    dbu       = dd['dbu']
    die       = dd['die']
    comps     = dd['components']
    x0, y0, x1, y1 = die
    die_w, die_h = x1 - x0, y1 - y0
    ap_lookup = load_ap_data(ap_json_paths)
    violations = parse_drc(drc_path, dbu)

    log(f"  cells={len(comps):,}  die={die_w/dbu:.1f}×{die_h/dbu:.1f}µm  "
        f"M3={len(dd['m3_stripes'])}  M4={len(dd['m4_stripes'])}  "
        f"drv={len(violations)}")

    G = grid_size
    min_sigma = cell_min_sigma_bins / G
    default_ap   = float(np.median([v['ap_median'] for v in ap_lookup.values()])) if ap_lookup else 1.0
    default_pins = float(np.median([v['num_pins']  for v in ap_lookup.values()])) if ap_lookup else 1.0

    cx_list, cy_list, sx_list, sy_list, pin_list, ap_list, cat_list = [], [], [], [], [], [], []
    for _inst, cell_type, xl, yl, w, h in comps:
        cx = (xl + w * 0.5 - x0) / die_w
        cy = (yl + h * 0.5 - y0) / die_h
        sx = max(w / (2.0 * die_w) * cell_smooth_factor, min_sigma)
        sy = max(h / (2.0 * die_h) * cell_smooth_factor, min_sigma)
        cx_list.append(cx);    cy_list.append(cy)
        sx_list.append(sx);    sy_list.append(sy)
        cell_data = ap_lookup.get(cell_type)
        pin_list.append(cell_data['num_pins']  if cell_data else default_pins)
        ap_list.append( cell_data['ap_median'] if cell_data else default_ap)
        cat_list.append(_cell_category(cell_type))

    cx_t   = torch.tensor(cx_list,  dtype=torch.float32)
    cy_t   = torch.tensor(cy_list,  dtype=torch.float32)
    sx_t   = torch.tensor(sx_list,  dtype=torch.float32)
    sy_t   = torch.tensor(sy_list,  dtype=torch.float32)
    pin_t  = torch.tensor(pin_list, dtype=torch.float32)
    ap_t   = torch.tensor(ap_list,  dtype=torch.float32)
    cat_t  = torch.tensor(cat_list, dtype=torch.int64)
    pin_t  = pin_t / pin_t.sum()
    ap_t   = ap_t  / ap_t.sum()

    pin_density_model = PinDensityMapModel(cx_t, cy_t, sx_t, sy_t, pin_t, G)
    ap_model          = APMapModel        (cx_t, cy_t, sx_t, sy_t, ap_t,  G)
    stripe_model      = PowerStripeMapModel(
        dd['m3_stripes'], dd['m4_stripes'], die, G,
        m3_sigma_bins=m3_sigma_bins, m4_sigma_bins=m4_sigma_bins)
    cell_type_model   = CellTypeMapModel  (cx_t, cy_t, sx_t, sy_t, pin_t, cat_t, G)
    drv_model         = DRVMapModel(violations, die, G, drv_sigma_bins=drv_sigma_bins)

    return (pin_density_model, ap_model, stripe_model, cell_type_model, drv_model)


# ──────────────────────────────────────────────────────────────────────────────
# Single-sample processing
# ──────────────────────────────────────────────────────────────────────────────

def process_sample(
    sample_dir, lef_paths, ap_json_paths, out_dir,
    grid_size=256,
    drv_limit=1000,
    save_plots=False,
    plot_dpi=200,
    verbose=True,
    **smooth_kwargs,
):
    """
    Process one sample directory → save a .pt file.

    Expected sample_dir contents:
      contest.gp.def
      *.drc.rpt

    The .pt file contains:
      maps   – float32 tensor (5, G, G): [pin_density, ap, stripe, cell_type, drv]
      meta   – dict: sample name, drv count, grid size, smoothing config

    Returns the output path on success, None if skipped (missing files or
    DRV count ≥ drv_limit).
    """
    def_path  = os.path.join(sample_dir, 'contest.gp.def')
    drc_files = sorted(glob.glob(os.path.join(sample_dir, '*.drc.rpt')))

    if not os.path.isfile(def_path) or not drc_files:
        return None

    drc_path = drc_files[0]
    n_drv    = count_drv(drc_path)

    if n_drv >= drv_limit:
        if verbose:
            print(f"  SKIP  {os.path.basename(sample_dir):40s}  drv={n_drv} ≥ {drv_limit}")
        return None

    if verbose:
        print(f"  proc  {os.path.basename(sample_dir):40s}  drv={n_drv}")

    (pin_density_model, ap_model, stripe_model,
     cell_type_model, drv_model) = build_maps(
        def_path, lef_paths, ap_json_paths, drc_path,
        grid_size=grid_size, verbose=verbose, **smooth_kwargs)

    with torch.no_grad():
        pin_density_map = pin_density_model()
        ap_map          = ap_model()
        stripe_map      = stripe_model()
        cell_type_map   = cell_type_model()
        drv_map         = drv_model()

    maps_tensor = torch.stack(
        [pin_density_map, ap_map, stripe_map, cell_type_map, drv_map], dim=0)  # (5,G,G)

    os.makedirs(out_dir, exist_ok=True)
    sample_name = os.path.basename(sample_dir)
    out_path    = os.path.join(out_dir, f'{sample_name}.pt')

    torch.save({
        'maps': maps_tensor.cpu(),
        'meta': {
            'sample':      sample_name,
            'n_drv':       n_drv,
            'grid_size':   grid_size,
            'smooth':      smooth_kwargs,
            'map_keys':    ['pin_density', 'ap', 'stripe', 'cell_type', 'drv'],
        },
    }, out_path)

    if save_plots:
        plot_dir = os.path.join(out_dir, 'plots', sample_name)
        visualize_maps(pin_density_map, ap_map, stripe_map, cell_type_map, drv_map,
                       os.path.join(plot_dir, 'feature_maps.png'))
        save_individual_maps(pin_density_map, ap_map, stripe_map, cell_type_map, drv_map,
                             plot_dir, dpi=plot_dpi)

    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────────────────────────────────────

def process_batch(
    data_dir, out_dir, lef_paths, ap_json_paths,
    grid_size=256,
    drv_limit=1000,
    save_plots=False,
    plot_dpi=200,
    verbose=True,
    **smooth_kwargs,
):
    """
    Process all valid sample subdirectories in data_dir.

    A subdirectory is valid when it contains both contest.gp.def and at least
    one *.drc.rpt file.  Samples with ≥ drv_limit violations are skipped.

    Each accepted sample is saved as {out_dir}/{sample_name}.pt.
    A manifest JSON is written to {out_dir}/manifest.json.

    Returns
    -------
    list[str]  paths of all saved .pt files
    """
    candidates = sorted([
        e.path for e in os.scandir(data_dir)
        if e.is_dir()
    ])

    print(f"Batch processing: {len(candidates)} subdirectories in {data_dir}")
    print(f"  drv_limit={drv_limit}  save_plots={save_plots}  out={out_dir}\n")

    saved = []
    skipped_no_files = 0
    skipped_drv = 0

    for sample_dir in candidates:
        result = process_sample(
            sample_dir, lef_paths, ap_json_paths, out_dir,
            grid_size=grid_size, drv_limit=drv_limit,
            save_plots=save_plots, plot_dpi=plot_dpi,
            verbose=verbose, **smooth_kwargs)

        if result is None:
            def_exists = os.path.isfile(os.path.join(sample_dir, 'contest.gp.def'))
            drc_files  = glob.glob(os.path.join(sample_dir, '*.drc.rpt'))
            if not def_exists or not drc_files:
                skipped_no_files += 1
            else:
                skipped_drv += 1
        else:
            saved.append(result)

    # Write manifest
    manifest = {
        'data_dir':   data_dir,
        'out_dir':    out_dir,
        'grid_size':  grid_size,
        'drv_limit':  drv_limit,
        'n_saved':    len(saved),
        'n_skipped_no_files': skipped_no_files,
        'n_skipped_drv':      skipped_drv,
        'smooth':     smooth_kwargs,
        'samples':    [os.path.basename(p) for p in saved],
    }
    manifest_path = os.path.join(out_dir, 'manifest.json')
    os.makedirs(out_dir, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone.  saved={len(saved)}  "
          f"skipped(no files)={skipped_no_files}  "
          f"skipped(drv≥{drv_limit})={skipped_drv}")
    print(f"Manifest → {manifest_path}")
    return saved


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PlacementMapDataset(Dataset):
    """
    Loads all *.pt files produced by process_batch.

    Each item is a float32 tensor of shape (5, H, W):
      channel 0 – pin density map
      channel 1 – AP map
      channel 2 – power stripe map
      channel 3 – cell-type mixture map
      channel 4 – DRV map  (ground truth)

    Args:
        processed_dir : directory containing *.pt files (and manifest.json).
        input_keys    : which channels to use as input  (default: all 4 feature maps).
        target_key    : which channel is the prediction target (default: drv).
        transform     : optional callable applied to the input tensor.
    """

    _KEY_IDX = {'pin_density': 0, 'ap': 1, 'stripe': 2, 'cell_type': 3, 'drv': 4}

    def __init__(
        self,
        processed_dir,
        input_keys=('pin_density', 'ap', 'stripe', 'cell_type'),
        target_key='drv',
        transform=None,
    ):
        self.files      = sorted(glob.glob(os.path.join(processed_dir, '*.pt')))
        self.input_idx  = [self._KEY_IDX[k] for k in input_keys]
        self.target_idx = self._KEY_IDX[target_key]
        self.transform  = transform

        if not self.files:
            raise FileNotFoundError(
                f"No .pt files found in {processed_dir}. "
                "Run process_batch() first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data  = torch.load(self.files[idx], weights_only=True)
        maps  = data['maps']                              # (4, H, W)
        x     = maps[self.input_idx]                     # (C_in, H, W)
        y     = maps[self.target_idx].unsqueeze(0)       # (1, H, W)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def sample_name(self, idx):
        return os.path.splitext(os.path.basename(self.files[idx]))[0]


# ──────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ──────────────────────────────────────────────────────────────────────────────

_MAP_CFGS = [
    ('pin_density_map', 'Pin Density Map\n(pin count)',               'viridis'),
    ('ap_map',          'AP Map\n(pin access points)',                'plasma'),
    ('stripe_map',      'Power Stripe Map\n(M3 / M4)',                'Blues'),
    ('cell_type_map',   'Cell-Type Mixture\n(diversity × density)',   'RdPu'),
    ('drv_map',         'DRV Map\n(ground truth)',                    'hot'),
]


def _stat_label(data):
    return f'min={data.min():.4f}   max={data.max():.4f}   mean={data.mean():.5f}'


def visualize_maps(density_map, ap_map, stripe_map, cell_type_map, drv_map, out_path):
    """Save a combined 1×5 panel at 150 dpi."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    tensors = [density_map, ap_map, stripe_map, cell_type_map, drv_map]
    fig, axes = plt.subplots(1, 5, figsize=(30, 6.5))
    fig.patch.set_facecolor('#12121e')

    for ax, tensor, (_fname, title, cmap) in zip(axes, tensors, _MAP_CFGS):
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

    fig.suptitle(f'Feature Maps  –  {os.path.basename(os.path.dirname(out_path))}',
                 color='white', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()


def save_individual_maps(density_map, ap_map, stripe_map, cell_type_map, drv_map,
                         out_dir, dpi=300):
    """Save five separate high-resolution PNG files into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    tensors = [density_map, ap_map, stripe_map, cell_type_map, drv_map]

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
        plt.savefig(os.path.join(out_dir, f'{fname}.png'), dpi=dpi,
                    bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _make_parser():
    p = argparse.ArgumentParser(description='Build placement feature maps for training.')
    p.add_argument('--data_dir',  default=None,
                   help='Batch mode: root directory containing sample subdirectories.')
    p.add_argument('--sample_dir', default=None,
                   help='Single mode: one sample directory (contest.gp.def + *.drc.rpt).')
    p.add_argument('--out_dir',   default=None,
                   help='Output directory for .pt files (and optional plots).')
    p.add_argument('--drv_limit', type=int, default=1000,
                   help='Skip samples with ≥ this many DRV violations (default 1000).')
    p.add_argument('--save_plots', action='store_true',
                   help='Also save PNG visualisations alongside each .pt file.')
    p.add_argument('--plot_dpi', type=int, default=200)
    p.add_argument('--grid_size', type=int, default=256)
    # smoothing
    p.add_argument('--cell_smooth_factor',  type=float, default=1.0)
    p.add_argument('--cell_min_sigma_bins', type=float, default=1.0)
    p.add_argument('--m3_sigma_bins',       type=float, default=1.0)
    p.add_argument('--m4_sigma_bins',       type=float, default=1.0)
    p.add_argument('--drv_sigma_bins', type=float, default=5.0)
    return p


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))

    lef_paths = [os.path.join(base, 'SL_modified.lef'),
                 os.path.join(base, 'L_modified.lef')]
    ap_paths  = [os.path.join(base, 'SL_modified_m2_ap.json'),
                 os.path.join(base, 'L_modified_m2_ap.json')]

    args = _make_parser().parse_args()

    smooth = dict(
        cell_smooth_factor  = args.cell_smooth_factor,
        cell_min_sigma_bins = args.cell_min_sigma_bins,
        m3_sigma_bins       = args.m3_sigma_bins,
        m4_sigma_bins       = args.m4_sigma_bins,
        drv_sigma_bins      = args.drv_sigma_bins,
    )

    # ── batch mode (default) ──────────────────────────────────────────────────
    if args.data_dir or args.sample_dir is None:
        data_dir = args.data_dir or os.path.join(base, 'data', 'train_data')
        out_dir  = args.out_dir  or os.path.join(base, 'data', 'processed')
        saved = process_batch(
            data_dir, out_dir, lef_paths, ap_paths,
            grid_size=args.grid_size, drv_limit=args.drv_limit,
            save_plots=args.save_plots, plot_dpi=args.plot_dpi,
            **smooth)

    # ── single-sample mode ────────────────────────────────────────────────────
    else:
        out_dir = args.out_dir or os.path.join(base, 'data', 'processed')
        result  = process_sample(
            args.sample_dir, lef_paths, ap_paths, out_dir,
            grid_size=args.grid_size, drv_limit=args.drv_limit,
            save_plots=args.save_plots, plot_dpi=args.plot_dpi,
            **smooth)
        if result:
            print(f"Saved → {result}")
        else:
            print("Sample skipped (missing files or DRV limit exceeded).")

#!/usr/bin/env python3
"""
Feature maps for DRV prediction.

Maps (all 256×256 float32 tensors, stored as channels 0-4):
  0  PinDensityMapModel – Gaussian-smeared pin-count density
  1  APMapModel         – Gaussian-smeared pin AP map
  2  CellTypeMapModel   – routing-complexity-weighted cell mixture
  3  AOIDensityMapModel – AOI*/OAI* cell density
  4  RUDYMapModel       – RUDY routing-demand map
  5  DRVMapModel        – ground-truth DRV density  (target, not input)

Batch processing
----------------
  process_batch(data_dir, out_dir, lef_paths, ap_json_paths)

PyTorch Dataset
---------------
  PlacementMapDataset(processed_dir)
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
    """Fast DRV count: count 'Bounds' lines."""
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
      dbu        – database units per micron
      die        – (x0, y0, x1, y1) in DBU
      components – list of (inst, cell_type, xl, yl, w, h) in DBU
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

    return dict(dbu=dbu, die=die, components=components)


def load_ap_data(json_paths):
    """Return {cell_type: {'ap_median': float, 'num_pins': int}} from AP JSON files."""
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


def _parse_net_data(def_content, inst_centers):
    """
    Parse NETS section, return list of (pin_positions, xmin, xmax, ymin, ymax)
    for every signal net connecting ≥ 2 placed instances.
    inst_centers: {inst_name: (cx_dbu, cy_dbu)}
    """
    nm = re.search(r'\bNETS \d+ ;(.*?)END NETS', def_content, re.DOTALL)
    if not nm:
        return []
    ip = re.compile(r'\(\s*(\S+)\s+\S+\s*\)')
    nets = []
    for blk in re.split(r'\n\s*-\s+\S+', nm.group(1))[1:]:
        pins = []
        for m in ip.finditer(blk):
            inst = m.group(1)
            if inst != 'PIN' and inst in inst_centers:
                pins.append(inst_centers[inst])
        if len(pins) >= 2:
            xs = [p[0] for p in pins]
            ys = [p[1] for p in pins]
            nets.append((pins, min(xs), max(xs), min(ys), max(ys)))
    return nets


# ──────────────────────────────────────────────────────────────────────────────
# Shared helper
# ──────────────────────────────────────────────────────────────────────────────

def _separable_gaussian_map(cx, cy, sigma_x, sigma_y, weights, bins):
    """
    256×256 density map via separable 1-D Gaussians.

    map[i, j] = Σ_n  weights[n] * Gy_n[i] * Gx_n[j]
    """
    Gx = torch.exp(-0.5 * ((bins.unsqueeze(0) - cx.unsqueeze(1)) / sigma_x.unsqueeze(1)) ** 2)
    Gy = torch.exp(-0.5 * ((bins.unsqueeze(0) - cy.unsqueeze(1)) / sigma_y.unsqueeze(1)) ** 2)
    Gx = Gx / (Gx.sum(dim=1, keepdim=True) + 1e-9)
    Gy = Gy / (Gy.sum(dim=1, keepdim=True) + 1e-9)
    return (weights.unsqueeze(1) * Gy).T @ Gx   # (G, G)


# ──────────────────────────────────────────────────────────────────────────────
# Map models
# ──────────────────────────────────────────────────────────────────────────────

class PinDensityMapModel(nn.Module):
    """Differentiable pin-count density map (Gaussian-smeared)."""

    def __init__(self, cx_norm, cy_norm, sigma_x, sigma_y, pin_weights, grid_size=256):
        super().__init__()
        self.grid_size = grid_size
        self.cx = nn.Parameter(cx_norm.clone())
        self.cy = nn.Parameter(cy_norm.clone())
        self.register_buffer('sigma_x',     sigma_x)
        self.register_buffer('sigma_y',     sigma_y)
        self.register_buffer('pin_weights', pin_weights)
        bins = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        self.register_buffer('bins', bins)

    def forward(self):
        m = _separable_gaussian_map(
            self.cx, self.cy, self.sigma_x, self.sigma_y, self.pin_weights, self.bins)
        if m.max() > 0:
            m = m / m.max()
        return m


class APMapModel(nn.Module):
    """Differentiable pin access-point density map (weight = ap_median)."""

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
        m = _separable_gaussian_map(
            self.cx, self.cy, self.sigma_x, self.sigma_y, self.ap_weights, self.bins)
        if m.max() > 0:
            m = m / m.max()
        return m


def _cell_category(cell_type: str) -> int:
    """
    0 – simple       INV*, BUF*
    1 – sequential   DFF*, SDFF*, SDF*, LATCH*
    2 – complex-comb AOI*, OAI*, FA*, MAJ*, AO*, OA*
    3 – std-comb     everything else
    """
    n = cell_type.upper()
    if re.match(r'(INV|BUF)', n):        return 0
    if re.match(r'(DFF|SDFF|SDF|LATCH)', n): return 1
    if re.match(r'(AOI|OAI|FA|MAJ|AO[^I]|OA[^I]|HB)', n): return 2
    return 3


def _is_aoi_oai(cell_type: str) -> bool:
    return bool(re.match(r'(AOI|OAI)', cell_type.upper()))


class CellTypeMapModel(nn.Module):
    """
    Routing-complexity-weighted cell mixture map.

    Each of K=4 categories gets a density map D_g (Gaussian-smeared, pin-weighted).
    Output = Σ_g w_g · D_g, normalised to [0, 1].

    Weights per category (simple→std→complex→seq): 1, 2, 3, 5.
    """

    _K = 4
    _WEIGHTS = [1.0, 2.0, 3.0, 5.0]

    def __init__(self, cx_norm, cy_norm, sigma_x, sigma_y, pin_weights,
                 cell_categories, grid_size=256):
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

    def forward(self):
        G, eps = self.grid_size, 1e-9
        M = torch.zeros(G, G, device=self.cx.device)
        for g, w_cat in enumerate(self._WEIGHTS):
            mask = (self.cell_categories == g)
            if not mask.any():
                continue
            w = self.pin_weights[mask]
            w = w / (w.sum() + eps)
            D_g = _separable_gaussian_map(
                self.cx[mask], self.cy[mask],
                self.sigma_x[mask], self.sigma_y[mask], w, self.bins)
            M = M + w_cat * D_g
        if M.max() > 0:
            M = M / M.max()
        return M


class AOIDensityMapModel(nn.Module):
    """
    Gaussian density map for AOI*/OAI* cells only.
    Same kernel as PinDensityMapModel; returns zero map when no AOI/OAI cells exist.
    """

    def __init__(self, cx_norm, cy_norm, sigma_x, sigma_y, pin_weights, grid_size=256):
        super().__init__()
        self.grid_size = grid_size
        bins = (torch.arange(grid_size, dtype=torch.float32) + 0.5) / grid_size
        self.register_buffer('bins', bins)
        self._has_cells = cx_norm.numel() > 0
        if self._has_cells:
            self.cx = nn.Parameter(cx_norm.clone())
            self.cy = nn.Parameter(cy_norm.clone())
            self.register_buffer('sigma_x',     sigma_x)
            self.register_buffer('sigma_y',     sigma_y)
            self.register_buffer('pin_weights', pin_weights)

    def forward(self):
        if not self._has_cells:
            return torch.zeros(self.grid_size, self.grid_size, device=self.bins.device)
        m = _separable_gaussian_map(
            self.cx, self.cy, self.sigma_x, self.sigma_y, self.pin_weights, self.bins)
        if m.max() > 0:
            m = m / m.max()
        return m


class RUDYMapModel(nn.Module):
    """
    RUDY (Rectangular Uniform Wire DensitY) routing-demand map.

    For each signal net with bounding box W×H (DBU):
        demand = (W + H) / (W × H)

    Accumulated uniformly over all grid bins in the net's bounding box.
    Normalised to [0, 1].  Component centres used as pin-position proxies.
    """

    def __init__(self, net_data, die, grid_size=256):
        super().__init__()
        G = grid_size
        x0, y0, x1d, y1d = die
        die_w, die_h = x1d - x0, y1d - y0
        bins = (np.arange(G, dtype=np.float32) + 0.5) / G
        arr  = np.zeros((G, G), dtype=np.float32)
        min_w, min_h = die_w / G, die_h / G

        for _pins, xmin, xmax, ymin, ymax in net_data:
            W = max(xmax - xmin, min_w)
            H = max(ymax - ymin, min_h)
            demand = (W + H) / (W * H)
            xn0 = (xmin - x0) / die_w; xn1 = (xmax - x0) / die_w
            yn0 = (ymin - y0) / die_h; yn1 = (ymax - y0) / die_h
            cols = np.where((bins >= xn0) & (bins <= xn1))[0]
            rows = np.where((bins >= yn0) & (bins <= yn1))[0]
            if rows.size and cols.size:
                arr[np.ix_(rows, cols)] += demand

        mx = arr.max()
        if mx > 0:
            arr /= mx
        self.register_buffer('rudy_map', torch.from_numpy(arr))

    def forward(self):
        return self.rudy_map


class DRVMapModel(nn.Module):
    """
    DRV density map – ground truth target.

    Each violation centre spread with a symmetric 2-D Gaussian
    (σ = drv_sigma_bins / G).  Normalised to [0, 1].
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
    drv_sigma_bins=1.5,
    verbose=True,
):
    """
    Parse all input files and build the six map models.

    Returns
    -------
    (pin_density_model, ap_model, cell_type_model, aoi_model, rudy_model, drv_model)
    """
    def log(msg):
        if verbose:
            print(msg)

    macros     = parse_lef(lef_paths)
    dd         = parse_def(def_path, macro_sizes=macros)
    dbu        = dd['dbu']
    die        = dd['die']
    comps      = dd['components']
    x0, y0, x1, y1 = die
    die_w, die_h = x1 - x0, y1 - y0
    ap_lookup  = load_ap_data(ap_json_paths)
    violations = parse_drc(drc_path, dbu)

    with open(def_path) as _f:
        _def_content = _f.read()
    inst_centers = {inst: (xl + w * 0.5, yl + h * 0.5) for inst, _, xl, yl, w, h in comps}
    net_data = _parse_net_data(_def_content, inst_centers)

    log(f"  cells={len(comps):,}  die={die_w/dbu:.1f}×{die_h/dbu:.1f}µm  "
        f"nets={len(net_data)}  drv={len(violations)}")

    G = grid_size
    min_sigma    = cell_min_sigma_bins / G
    default_ap   = float(np.median([v['ap_median'] for v in ap_lookup.values()])) if ap_lookup else 1.0
    default_pins = float(np.median([v['num_pins']  for v in ap_lookup.values()])) if ap_lookup else 1.0

    cx_list, cy_list, sx_list, sy_list = [], [], [], []
    pin_list, ap_list, cat_list, aoi_list = [], [], [], []
    for _inst, cell_type, xl, yl, w, h in comps:
        cx = (xl + w * 0.5 - x0) / die_w
        cy = (yl + h * 0.5 - y0) / die_h
        sx = max(w / (2.0 * die_w) * cell_smooth_factor, min_sigma)
        sy = max(h / (2.0 * die_h) * cell_smooth_factor, min_sigma)
        cx_list.append(cx);  cy_list.append(cy)
        sx_list.append(sx);  sy_list.append(sy)
        cell_data = ap_lookup.get(cell_type)
        pin_list.append(cell_data['num_pins']  if cell_data else default_pins)
        ap_list.append( cell_data['ap_median'] if cell_data else default_ap)
        cat_list.append(_cell_category(cell_type))
        aoi_list.append(_is_aoi_oai(cell_type))

    cx_t  = torch.tensor(cx_list,  dtype=torch.float32)
    cy_t  = torch.tensor(cy_list,  dtype=torch.float32)
    sx_t  = torch.tensor(sx_list,  dtype=torch.float32)
    sy_t  = torch.tensor(sy_list,  dtype=torch.float32)
    pin_t = torch.tensor(pin_list, dtype=torch.float32)
    ap_t  = torch.tensor(ap_list,  dtype=torch.float32)
    cat_t = torch.tensor(cat_list, dtype=torch.int64)
    pin_t = pin_t / pin_t.sum()
    ap_t  = ap_t  / ap_t.sum()

    pin_density_model = PinDensityMapModel(cx_t, cy_t, sx_t, sy_t, pin_t, G)
    ap_model          = APMapModel        (cx_t, cy_t, sx_t, sy_t, ap_t,  G)
    cell_type_model   = CellTypeMapModel  (cx_t, cy_t, sx_t, sy_t, pin_t, cat_t, G)

    aoi_t   = torch.tensor(aoi_list, dtype=torch.bool)
    aoi_pin = pin_t[aoi_t].clone()
    if aoi_pin.numel() > 0:
        aoi_pin = aoi_pin / (aoi_pin.sum() + 1e-9)
    aoi_model = AOIDensityMapModel(
        cx_t[aoi_t], cy_t[aoi_t], sx_t[aoi_t], sy_t[aoi_t], aoi_pin, G)
    log(f"  aoi/oai cells: {aoi_t.sum().item()}")

    rudy_model = RUDYMapModel(net_data, die, G)
    drv_model  = DRVMapModel(violations, die, G, drv_sigma_bins=drv_sigma_bins)

    return (pin_density_model, ap_model, cell_type_model, aoi_model, rudy_model, drv_model)


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

    The .pt file contains:
      maps     – float32 (6, G, G): [pin_density, ap, cell_type, aoi, rudy, drv]
      meta     – dict: sample name, drv count, grid size, smoothing config

    Returns the output path, or None if skipped.
    """
    def_path  = os.path.join(sample_dir, 'contest.gp.def')
    drc_files = sorted(glob.glob(os.path.join(sample_dir, '*.drc.rpt')))

    if not os.path.isfile(def_path) or not drc_files:
        return None

    drc_path = drc_files[0]
    n_drv    = count_drv(drc_path)

    if n_drv == 0:
        if verbose:
            print(f"  SKIP  {os.path.basename(sample_dir):40s}  drv=0 (no violations)")
        return None

    if n_drv >= drv_limit:
        if verbose:
            print(f"  SKIP  {os.path.basename(sample_dir):40s}  drv={n_drv} ≥ {drv_limit}")
        return None

    if verbose:
        print(f"  proc  {os.path.basename(sample_dir):40s}  drv={n_drv}")

    (pin_density_model, ap_model, cell_type_model,
     aoi_model, rudy_model, drv_model) = build_maps(
        def_path, lef_paths, ap_json_paths, drc_path,
        grid_size=grid_size, verbose=verbose, **smooth_kwargs)

    with torch.no_grad():
        pin_density_map = pin_density_model()
        ap_map          = ap_model()
        cell_type_map   = cell_type_model()
        aoi_map         = aoi_model()
        rudy_map        = rudy_model()
        drv_map         = drv_model()

    maps_tensor = torch.stack(
        [pin_density_map, ap_map, cell_type_map, aoi_map, rudy_map, drv_map], dim=0)  # (6,G,G)

    os.makedirs(out_dir, exist_ok=True)
    sample_name = os.path.basename(sample_dir)
    out_path    = os.path.join(out_dir, f'{sample_name}.pt')

    torch.save({
        'maps': maps_tensor.cpu(),
        'meta': {
            'sample':    sample_name,
            'n_drv':     n_drv,
            'grid_size': grid_size,
            'smooth':    smooth_kwargs,
            'map_keys':  ['pin_density', 'ap', 'cell_type', 'aoi', 'rudy', 'drv'],
        },
    }, out_path)

    if save_plots:
        plot_dir = os.path.join(out_dir, 'plots', sample_name)
        visualize_maps(pin_density_map, ap_map, cell_type_map, aoi_map, rudy_map, drv_map,
                       os.path.join(plot_dir, 'feature_maps.png'))
        save_individual_maps(pin_density_map, ap_map, cell_type_map, aoi_map, rudy_map, drv_map,
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
    Each accepted sample is saved as {out_dir}/{sample_name}.pt.
    A manifest JSON is written to {out_dir}/manifest.json.
    """
    candidates = sorted([e.path for e in os.scandir(data_dir) if e.is_dir()])

    print(f"Batch processing: {len(candidates)} subdirectories in {data_dir}")
    print(f"  drv_limit={drv_limit}  save_plots={save_plots}  out={out_dir}\n")

    saved, skipped_no_files, skipped_drv = [], 0, 0

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

    manifest = {
        'data_dir':  data_dir,
        'out_dir':   out_dir,
        'grid_size': grid_size,
        'drv_limit': drv_limit,
        'n_saved':   len(saved),
        'n_skipped_no_files': skipped_no_files,
        'n_skipped_drv':      skipped_drv,
        'smooth':    smooth_kwargs,
        'samples':   [os.path.basename(p) for p in saved],
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

    Each item is a float32 tensor of shape (6, H, W):
      channel 0 – pin density
      channel 1 – AP (access points)
      channel 2 – cell-type mixture
      channel 3 – AOI/OAI density
      channel 4 – RUDY routing demand
      channel 5 – DRV map (ground truth)

    Args:
        processed_dir : directory containing *.pt files.
        input_keys    : channels to use as input (default: all 5 feature maps).
        target_key    : prediction target channel (default: drv).
    """

    _KEY_IDX = {
        'pin_density': 0, 'ap': 1, 'cell_type': 2,
        'aoi': 3, 'rudy': 4, 'drv': 5,
    }

    def __init__(
        self,
        processed_dir,
        input_keys=('pin_density', 'ap', 'cell_type', 'aoi', 'rudy'),
        target_key='drv',
        transform=None,
    ):
        self.files      = sorted(glob.glob(os.path.join(processed_dir, '*.pt')))
        self.input_idx  = [self._KEY_IDX[k] for k in input_keys]
        self.target_idx = self._KEY_IDX[target_key]
        self.transform  = transform

        if not self.files:
            raise FileNotFoundError(
                f"No .pt files found in {processed_dir}. Run process_batch() first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)
        maps = data['maps']
        x    = maps[self.input_idx]          # (C_in, H, W)
        y    = maps[self.target_idx].unsqueeze(0)   # (1, H, W)
        # Normalize each input channel to [0, 1].
        # Fixes legacy .pt files where pin_density/ap/aoi were stored
        # with sum=1 (max ≈ 1e-4) instead of max=1.
        mx = x.flatten(1).max(dim=1).values  # (C_in,)
        mx = mx.clamp(min=1e-9).view(-1, 1, 1)
        x  = x / mx
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def sample_name(self, idx):
        return os.path.splitext(os.path.basename(self.files[idx]))[0]


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

_MAP_CFGS = [
    ('pin_density_map', 'Pin Density\n(pin count)',          'viridis'),
    ('ap_map',          'AP Map\n(access points)',           'plasma'),
    ('cell_type_map',   'Cell-Type Mixture\n(complexity)',   'RdPu'),
    ('aoi_map',         'AOI/OAI Density\n(aoi*/oai*)',      'YlOrRd'),
    ('rudy_map',        'RUDY\n(routing demand)',            'OrRd'),
    ('drv_map',         'DRV Map\n(ground truth)',           'hot'),
]


def _stat_label(data):
    return f'min={data.min():.4f}  max={data.max():.4f}  mean={data.mean():.5f}'


def visualize_maps(density_map, ap_map, cell_type_map, aoi_map, rudy_map, drv_map, out_path):
    """Save a 1×6 panel PNG at 150 dpi."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    tensors = [density_map, ap_map, cell_type_map, aoi_map, rudy_map, drv_map]
    fig, axes = plt.subplots(1, 6, figsize=(36, 6.5))
    fig.patch.set_facecolor('#12121e')

    for ax, tensor, (_fname, title, cmap) in zip(axes, tensors, _MAP_CFGS):
        data = tensor.detach().cpu().numpy()
        ax.set_facecolor('#12121e')
        im = ax.imshow(data, origin='lower', cmap=cmap, interpolation='nearest', aspect='equal')
        ax.set_title(title, color='white', fontsize=11, pad=6, linespacing=1.4)
        ax.tick_params(colors='#888888', labelsize=6)
        for spine in ax.spines.values():
            spine.set_color('#333355')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors='#aaaaaa', labelsize=6)
        plt.setp(cb.ax.get_yticklabels(), color='#aaaaaa')
        ax.set_xlabel(_stat_label(data), color='#888888', fontsize=6)

    fig.suptitle(f'Feature Maps – {os.path.basename(os.path.dirname(out_path))}',
                 color='white', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def save_individual_maps(density_map, ap_map, cell_type_map, aoi_map, rudy_map, drv_map,
                         out_dir, dpi=300):
    """Save six individual high-resolution PNGs into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    tensors = [density_map, ap_map, cell_type_map, aoi_map, rudy_map, drv_map]

    for tensor, (fname, title, cmap) in zip(tensors, _MAP_CFGS):
        data = tensor.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor('#12121e')
        ax.set_facecolor('#12121e')
        im = ax.imshow(data, origin='lower', cmap=cmap, interpolation='nearest', aspect='equal')
        ax.set_title(title.replace('\n', ' – '), color='white', fontsize=14, pad=10)
        ax.tick_params(colors='#aaaaaa', labelsize=9)
        ax.set_xlabel('bin (x)', color='#888888', fontsize=9)
        ax.set_ylabel('bin (y)', color='#888888', fontsize=9)
        for spine in ax.spines.values():
            spine.set_color('#333355')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors='#aaaaaa', labelsize=9)
        plt.setp(cb.ax.get_yticklabels(), color='#aaaaaa')
        fig.text(0.5, 0.01, _stat_label(data), ha='center', color='#888888', fontsize=8)
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(os.path.join(out_dir, f'{fname}.png'), dpi=dpi,
                    bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _make_parser():
    p = argparse.ArgumentParser(description='Build placement feature maps for training.')
    p.add_argument('--data_dir',   default=None)
    p.add_argument('--sample_dir', default=None)
    p.add_argument('--out_dir',    default=None)
    p.add_argument('--drv_limit',  type=int,   default=1000)
    p.add_argument('--save_plots', action='store_true')
    p.add_argument('--plot_dpi',   type=int,   default=200)
    p.add_argument('--grid_size',  type=int,   default=256)
    p.add_argument('--cell_smooth_factor',  type=float, default=1.0)
    p.add_argument('--cell_min_sigma_bins', type=float, default=1.0)
    p.add_argument('--drv_sigma_bins',      type=float, default=5.0)
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
        drv_sigma_bins      = args.drv_sigma_bins,
    )

    if args.data_dir or args.sample_dir is None:
        data_dir = args.data_dir or os.path.join(base, 'train_data')
        out_dir  = args.out_dir  or os.path.join(base, 'processed')
        process_batch(data_dir, out_dir, lef_paths, ap_paths,
                      grid_size=args.grid_size, drv_limit=args.drv_limit,
                      save_plots=args.save_plots, plot_dpi=args.plot_dpi,
                      **smooth)
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

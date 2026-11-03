#!/usr/bin/env python3
"""
Visualise a placed DEF against the M3 power stripes, highlighting a cell class
(AOI/OAI by default).

Produces, for each input DEF, a row of three panels:
  1. full die   -- every cell, M3 stripe bands overlaid
  2. zoom       -- a few band pitches, so individual cells are actually visible
  3. phase      -- cell-centre x folded modulo the band pitch. This is the
                   quantitative panel: it shows directly whether the
                   highlighted class avoids the copper in x, which the
                   full-die view cannot show at this scale.

Reuses the parsers from analyze_stripe_distance.py (same file, same repo).

Example:
  python3 tool/plot_stripe_placement.py \
    --fp_def xplace_work/sha256_unplaced.def \
    --lef xplace_work/L_modified.lef,xplace_work/SL_modified.lef \
    --def result/sha256_ab_w0/placement_sha256_gp.def=w0 \
    --def result/sha256_v2_w10/placement_sha256_gp.def=w1.0 \
    --out result/stripe_placement.png
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_stripe_distance import (
    parse_units, parse_m3_stripes, parse_lef_sizes, parse_components,
)

# copper / cell / highlight colours
C_STRIPE = '#c1663a'
C_CELL = '#c9ced6'
C_HIT = '#d92b2b'          # highlighted cell overlapping a stripe
C_OK = '#2b6fd9'           # highlighted cell clear of the stripes


def merge_bands(stripes, gap):
    """Merge stripes whose clear gap is <= gap into single obstacle bands."""
    bands = []
    for lo, hi in sorted(stripes):
        if bands and lo - bands[-1][1] <= gap:
            bands[-1][1] = max(bands[-1][1], hi)
        else:
            bands.append([lo, hi])
    return [tuple(b) for b in bands]


def overlaps(lo, hi, bands):
    return any(not (hi <= b[0] or lo >= b[1]) for b in bands)


def collect(def_path, sizes, dbu, name_re):
    """-> (rects_other, rects_hit, rects_ok, centres_hi) in microns."""
    other, hit, ok, cx_hi = [], [], [], []
    for inst, master, x, y, orient in parse_components(def_path):
        if master not in sizes:
            continue
        w, h = sizes[master]
        lo, hi = x / dbu, x / dbu + w
        r = (lo, y / dbu, w, h)
        if name_re.match(master):
            cx_hi.append(lo + w / 2.0)
            (hit if overlaps(lo, hi, BANDS) else ok).append(r)
        else:
            other.append(r)
    return other, hit, ok, np.array(cx_hi)


def add_rects(ax, rects, color, lw=0.0, alpha=1.0, zorder=2):
    if not rects:
        return
    pc = PatchCollection(
        [Rectangle((x, y), w, h) for x, y, w, h in rects],
        facecolors=color, edgecolors='none', linewidths=lw,
        alpha=alpha, zorder=zorder,
    )
    ax.add_collection(pc)


def draw_bands(ax, y0, y1, alpha=0.30, zorder=4):
    for lo, hi in BANDS:
        ax.add_patch(Rectangle((lo, y0), hi - lo, y1 - y0, facecolor=C_STRIPE,
                               edgecolor='none', alpha=alpha, zorder=zorder))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fp_def', required=True, help='floorplan DEF with SPECIALNETS')
    ap.add_argument('--lef', required=True, help='comma-separated LEF paths')
    ap.add_argument('--def', dest='defs', action='append', required=True,
                    help='placed DEF, optionally path=label (repeatable)')
    ap.add_argument('--pattern', default=r'^(AOI|OAI)',
                    help='regex on the cell master to highlight')
    ap.add_argument('--merge_gap_um', type=float, default=1.0,
                    help='merge stripes closer than this into one band')
    ap.add_argument('--zoom_pitches', type=float, default=3.0,
                    help='how many band pitches the zoom panel spans')
    ap.add_argument('--zoom_rows_um', type=float, default=30.0,
                    help='vertical extent of the zoom panel (um)')
    ap.add_argument('--out', default='result/stripe_placement.png')
    args = ap.parse_args()

    dbu = parse_units(args.fp_def)
    stripes_dbu = parse_m3_stripes(args.fp_def)
    stripes = [(lo / dbu, hi / dbu) for lo, hi in stripes_dbu]
    global BANDS
    BANDS = merge_bands(stripes, args.merge_gap_um)
    sizes = parse_lef_sizes(args.lef.split(','))
    name_re = re.compile(args.pattern)

    runs = []
    for spec in args.defs:
        path, _, label = spec.partition('=')
        runs.append((path, label or os.path.basename(os.path.dirname(path))))

    # die/core extent from the union of all placed cells
    all_x = all_y = None
    data = {}
    for path, label in runs:
        d = collect(path, sizes, dbu, name_re)
        data[label] = d
        pts = d[0] + d[1] + d[2]
        xs = [r[0] for r in pts] + [r[0] + r[2] for r in pts]
        ys = [r[1] for r in pts] + [r[1] + r[3] for r in pts]
        all_x = (min(xs), max(xs)) if all_x is None else (min(all_x[0], min(xs)), max(all_x[1], max(xs)))
        all_y = (min(ys), max(ys)) if all_y is None else (min(all_y[0], min(ys)), max(all_y[1], max(ys)))

    pitch = (BANDS[1][0] - BANDS[0][0]) if len(BANDS) > 1 else (all_x[1] - all_x[0])
    band_w = BANDS[0][1] - BANDS[0][0]

    # zoom window: centred on the die, spanning a whole number of pitches
    zx0 = BANDS[len(BANDS) // 2][0] - (pitch - band_w) / 2
    zx1 = zx0 + args.zoom_pitches * pitch
    zy0 = 0.5 * (all_y[0] + all_y[1])
    zy1 = zy0 + args.zoom_rows_um

    nrun = len(runs)
    fig, axes = plt.subplots(nrun, 3, figsize=(19, 5.4 * nrun), squeeze=False,
                             gridspec_kw={'width_ratios': [1.0, 1.0, 0.85]})

    for row, (path, label) in enumerate(runs):
        other, hit, ok, cx_hi = data[label]
        n_hi = len(hit) + len(ok)
        frac = 100.0 * len(hit) / max(1, n_hi)

        # ---------------- panel 1: full die ----------------
        ax = axes[row][0]
        add_rects(ax, other, C_CELL, zorder=2)
        add_rects(ax, ok, C_OK, zorder=3)
        add_rects(ax, hit, C_HIT, zorder=5)
        draw_bands(ax, all_y[0], all_y[1])
        ax.set_xlim(*all_x); ax.set_ylim(*all_y)
        ax.set_aspect('equal')
        ax.set_title(f"{label} — full die\n{n_hi} {args.pattern} cells, "
                     f"{len(hit)} on copper ({frac:.1f}%)", fontsize=10)
        ax.set_xlabel('x (um)'); ax.set_ylabel('y (um)')
        # mark the zoom window
        ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, facecolor='none',
                               edgecolor='#111111', lw=1.2, ls='--', zorder=6))

        # ---------------- panel 2: zoom ----------------
        ax = axes[row][1]
        add_rects(ax, other, C_CELL, lw=0.15, zorder=2)
        add_rects(ax, ok, C_OK, lw=0.2, zorder=3)
        add_rects(ax, hit, C_HIT, lw=0.2, zorder=5)
        draw_bands(ax, zy0 - 5, zy1 + 5, alpha=0.33)
        ax.set_xlim(zx0, zx1); ax.set_ylim(zy0, zy1)
        ax.set_aspect('equal')
        ax.set_title(f"{label} — zoom: {args.zoom_pitches:g} band pitches "
                     f"x {args.zoom_rows_um:g}um", fontsize=10)
        ax.set_xlabel('x (um)'); ax.set_ylabel('y (um)')

        # ---------------- panel 3: folded phase ----------------
        ax = axes[row][2]
        # fold every highlighted cell centre into one band pitch, with the
        # band centred at 0 so "0 = dead under copper"
        bc = 0.5 * (BANDS[0][0] + BANDS[0][1])
        ph = np.mod(cx_hi - bc + pitch / 2, pitch) - pitch / 2
        # same folding for the whole population, as the reference distribution
        cx_all = np.array([r[0] + r[2] / 2 for r in (other + ok + hit)])
        ph_all = np.mod(cx_all - bc + pitch / 2, pitch) - pitch / 2

        nb = 48
        ax.hist(ph_all, bins=nb, range=(-pitch / 2, pitch / 2), density=True,
                color='#9aa3ad', alpha=0.55, label=f'all cells (n={len(cx_all)})')
        ax.hist(ph, bins=nb, range=(-pitch / 2, pitch / 2), density=True,
                histtype='step', lw=2.0, color=C_OK,
                label=f'{args.pattern} (n={len(ph)})')
        ax.axvspan(-band_w / 2, band_w / 2, color=C_STRIPE, alpha=0.30,
                   label=f'M3 band ({band_w:.2f}um)')
        ax.axvline(0, color=C_STRIPE, lw=1.0, ls='--')
        ax.set_xlim(-pitch / 2, pitch / 2)
        ax.set_xlabel('cell-centre x, folded mod band pitch (um)\n'
                      '0 = centred on copper')
        ax.set_ylabel('density')
        ax.set_title(f"{label} — phase w.r.t. stripes", fontsize=10)
        ax.legend(fontsize=7.5, loc='upper right')
        ax.grid(alpha=0.25)

    for a in axes.ravel():
        a.tick_params(labelsize=8)

    # ---------------- cross-run comparison figure ----------------
    # The per-run phase panels above can't be compared by eye across rows, so
    # overlay them: one axes for the highlighted class, one for all cells, plus
    # the run-to-run delta. This is what answers "did they move away in x".
    bc = 0.5 * (BANDS[0][0] + BANDS[0][1])

    def folded(rects):
        cx = np.array([r[0] + r[2] / 2 for r in rects])
        return np.mod(cx - bc + pitch / 2, pitch) - pitch / 2

    nb = 40
    edges = np.linspace(-pitch / 2, pitch / 2, nb + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    cmap = plt.get_cmap('tab10')

    fig2, ax2 = plt.subplots(1, 3, figsize=(18, 5),
                             gridspec_kw={'width_ratios': [1, 1, 1]})
    hi_h, all_h = {}, {}
    for i, (path, label) in enumerate(runs):
        other, hit, ok, _ = data[label]
        hi_h[label], _ = np.histogram(folded(hit + ok), bins=edges, density=True)
        all_h[label], _ = np.histogram(folded(other + hit + ok), bins=edges, density=True)
        for ax, h in ((ax2[0], hi_h[label]), (ax2[1], all_h[label])):
            ax.step(ctr, h, where='mid', lw=2.0, color=cmap(i), label=label)

    for ax, ttl in ((ax2[0], f'{args.pattern} cells'), (ax2[1], 'all cells')):
        ax.axvspan(-band_w / 2, band_w / 2, color=C_STRIPE, alpha=0.30,
                   label=f'M3 band ({band_w:.2f}um)')
        ax.axvline(0, color=C_STRIPE, lw=1.0, ls='--')
        ax.set_xlim(-pitch / 2, pitch / 2)
        ax.set_xlabel('cell-centre x, folded mod band pitch (um)\n0 = centred on copper')
        ax.set_ylabel('density')
        ax.set_title(f'{ttl} — phase w.r.t. M3 stripes')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    # delta vs the first run: positive = moved INTO this phase. Coarser bins
    # than the distributions above, because the per-bin difference of two
    # ~2.8k-sample histograms is mostly counting noise at 40 bins.
    ref = runs[0][1]
    nb_d = 16
    edges_d = np.linspace(-pitch / 2, pitch / 2, nb_d + 1)
    ctr_d = 0.5 * (edges_d[:-1] + edges_d[1:])

    def fold_hist(rects, e):
        return np.histogram(folded(rects), bins=e, density=True)[0]

    for i, (path, label) in enumerate(runs[1:], start=1):
        o0, h0, k0, _ = data[ref]
        o1, h1, k1, _ = data[label]
        ax2[2].step(ctr_d, fold_hist(h1 + k1, edges_d) - fold_hist(h0 + k0, edges_d),
                    where='mid', lw=2.2, color=cmap(i),
                    label=f'{label} - {ref}  ({args.pattern})')
        ax2[2].step(ctr_d,
                    fold_hist(o1 + h1 + k1, edges_d) - fold_hist(o0 + h0 + k0, edges_d),
                    where='mid', lw=1.3, ls=':', color=cmap(i),
                    label=f'{label} - {ref}  (all cells)')

    # key stats: how much of each class actually sits on / near the copper
    lines = []
    for _, label in runs:
        other, hit, ok, _ = data[label]
        ph_hi = np.abs(folded(hit + ok))
        ph_all = np.abs(folded(other + hit + ok))
        lines.append('%-6s  on copper %4.1f%%  |  |phase|<band/2: %s %4.1f%% / all %4.1f%%'
                     % (label, 100.0 * len(hit) / max(1, len(hit) + len(ok)),
                        args.pattern, 100.0 * np.mean(ph_hi < band_w / 2),
                        100.0 * np.mean(ph_all < band_w / 2)))
    ax2[2].text(0.02, 0.02, '\n'.join(lines), transform=ax2[2].transAxes,
                fontsize=7.5, family='monospace', va='bottom',
                bbox=dict(boxstyle='round', fc='#f4f4f4', ec='#bbbbbb'))
    ax2[2].axvspan(-band_w / 2, band_w / 2, color=C_STRIPE, alpha=0.30)
    ax2[2].axhline(0, color='#444444', lw=0.9)
    ax2[2].axvline(0, color=C_STRIPE, lw=1.0, ls='--')
    ax2[2].set_xlim(-pitch / 2, pitch / 2)
    ax2[2].set_xlabel('cell-centre x, folded mod band pitch (um)\n0 = centred on copper')
    ax2[2].set_ylabel('density difference')
    ax2[2].set_title('change vs %s\n(negative near 0 = moved off the copper)' % ref)
    ax2[2].legend(fontsize=7.5)
    ax2[2].grid(alpha=0.25)

    fig2.suptitle('Phase of cell centres relative to the M3 stripe bands — '
                  'band %.2fum, pitch %.2fum' % (band_w, pitch), fontsize=11)
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    cmp_out = os.path.splitext(args.out)[0] + '_phase.png'
    fig2.savefig(cmp_out, dpi=150)
    print(f"wrote {cmp_out}")
    fig.suptitle(
        f"M3 power stripes vs placement — {len(stripes)} stripes merged into "
        f"{len(BANDS)} bands (band {band_w:.2f}um, pitch {pitch:.2f}um)   |   "
        f"grey = other cells, blue = {args.pattern} clear, red = {args.pattern} on copper",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97 if nrun > 1 else 0.94])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    # a compact text summary next to the picture
    print(f"\n{len(stripes)} M3 stripes -> {len(BANDS)} bands, "
          f"band={band_w:.3f}um pitch={pitch:.3f}um, "
          f"band x-coverage={100.0*len(BANDS)*band_w/(all_x[1]-all_x[0]):.1f}%")
    print(f"{'run':<10}{'hi-cells':>10}{'on copper':>11}{'%':>8}")
    for _, label in runs:
        other, hit, ok, _ = data[label]
        n_hi = len(hit) + len(ok)
        print(f"{label:<10}{n_hi:>10}{len(hit):>11}{100.0*len(hit)/max(1,n_hi):>8.1f}")


if __name__ == '__main__':
    main()

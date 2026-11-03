#!/usr/bin/env python3
"""Draw the pin-access pre-pass: global placement vs the frozen cells.

  python3 tool/plot_freeze.py --design seed [--zoom_um 30] [--out fig.png]

Both panels zoom into the region with the most pin-access-critical cells:

  left  : those cells where global placement left them (clustered, some sitting
          on top of a vertical M3 power stripe)
  right : after the pre-pass -- clear of the stripes, spread apart, snapped to
          site/row and marked FIXED. The outline is the cell's previous position
          and the arrow shows how it moved.

Pass --with_overview to add a whole-die row above.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pin_access_prepass import (parse_components, parse_def_header,
                                parse_lef_widths, parse_m3_stripes)

CELL_FC = "#cfd4da"      # ordinary cells
CELL_EC = "#aab1b8"
FROZEN_FC = "#e2483b"    # frozen (FIXED) cells
FROZEN_EC = "#8f1f16"
GHOST_EC = "#f0a8a0"     # where the frozen cell sat in GP
STRIPE_FC = "#4a90d9"


def cell_boxes(comps, sizes_um, units, only=None, skip=None):
    out = {}
    for c in comps:
        if c.get("status") is None or c["master"] not in sizes_um:
            continue
        if only is not None and c["name"] not in only:
            continue
        if skip is not None and c["name"] in skip:
            continue
        w, h = sizes_um[c["master"]]
        out[c["name"]] = (c["x"] / units, c["y"] / units, w, h)
    return out


def draw(ax, boxes, fc, ec, lw=0.15, alpha=1.0, fill=True):
    if not boxes:
        return
    pc = PatchCollection(
        [Rectangle((x, y), w, h) for x, y, w, h in boxes],
        facecolors=fc if fill else "none", edgecolors=ec,
        linewidths=lw, alpha=alpha,
    )
    ax.add_collection(pc)


def draw_stripes(ax, bands, units, y0, y1):
    for lo, hi in bands:
        ax.add_patch(Rectangle((lo / units, y0), (hi - lo) / units, y1 - y0,
                               facecolor=STRIPE_FC, edgecolor="none", alpha=0.30, zorder=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--gp_def", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--pdn_def", default="")
    ap.add_argument("--lefs", default="xplace_work/L_modified.lef,xplace_work/SL_modified.lef")
    ap.add_argument("--zoom_um", type=float, default=16.0, help="side of the zoom window")
    ap.add_argument("--with_overview", action="store_true",
                    help="also draw the whole-die row above the zoom")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = args.design
    gp_def = args.gp_def or f"result/{d}/placement_{d}_gp_raw.def"
    js_path = args.json or f"exp/pa_pilot/{d}_paJ.json"
    pdn = args.pdn_def or f"xplace_work/{d}_unplaced.def"
    out = args.out or f"exp/pa_pilot/{d}_freeze.png"

    sizes_um = parse_lef_widths(args.lefs.split(","))
    units, die, rows = parse_def_header(gp_def)
    bands = parse_m3_stripes(pdn)
    _, _, _, comps = parse_components(gp_def)
    frozen = json.load(open(js_path))["cells"]
    fmap = {c["name"]: c for c in frozen}

    xl, yl, xh, yh = (v / units for v in die)
    gp_all = cell_boxes(comps, sizes_um, units)
    others = [b for n, b in gp_all.items() if n not in fmap]
    gp_frozen = [gp_all[n] for n in fmap if n in gp_all]
    pa_frozen = [(c["x"] / units, c["y"] / units) + sizes_um[c["master"]]
                 for c in frozen if c["master"] in sizes_um]

    # zoom window: the region with the most frozen cells
    step = args.zoom_um
    best, bx, by = -1, xl, yl
    ys = sorted({round(c["y"] / units / step) for c in frozen})
    xs = sorted({round(c["x"] / units / step) for c in frozen})
    for gx in xs:
        for gy in ys:
            x0, y0 = gx * step, gy * step
            n = sum(1 for c in frozen
                    if x0 <= c["x"] / units < x0 + step and y0 <= c["y"] / units < y0 + step)
            if n > best:
                best, bx, by = n, x0, y0

    nrow = 2 if args.with_overview else 1
    figsize = (13, 13.6) if args.with_overview else (14, 7.6)
    kw = {"gridspec_kw": {"height_ratios": [1, 0.62]}} if args.with_overview else {}
    fig, axes = plt.subplots(nrow, 2, figsize=figsize, squeeze=False, **kw)
    over_titles = [f"Global placement  ({len(gp_all):,} cells)",
                   f"After pin-access pre-pass  ({len(pa_frozen):,} cells frozen = FIXED)"]
    zoom_titles = ["before: the pin-access-critical cells as global placement left them",
                   "after: clear of the stripes, spread apart, frozen (FIXED)"]
    for col in (0, 1):
        for row in range(nrow):
            ax = axes[row][col]
            zoom = (row == nrow - 1)
            draw_stripes(ax, bands, units, yl, yh)
            lw = 0.4 if zoom else 0.12
            draw(ax, others, CELL_FC, CELL_EC, lw=lw)
            if col == 0:
                if zoom:   # same cells, still at their GP positions
                    draw(ax, gp_frozen, FROZEN_FC, FROZEN_EC, lw=0.5, alpha=0.45)
                else:
                    draw(ax, gp_frozen, CELL_FC, CELL_EC, lw=lw)
            else:
                if zoom:   # ghost = where each frozen cell came from
                    draw(ax, gp_frozen, "none", GHOST_EC, lw=0.8, fill=False)
                draw(ax, pa_frozen, FROZEN_FC, FROZEN_EC, lw=lw if not zoom else 0.5)
                if zoom:   # arrows for the cells that actually moved
                    for c in frozen:
                        if c["master"] not in sizes_um:
                            continue
                        x0, y0 = c["gp_x"] / units, c["gp_y"] / units
                        x1, y1 = c["x"] / units, c["y"] / units
                        if not (bx <= x1 < bx + step and by <= y1 < by + step):
                            continue
                        if (x1 - x0) ** 2 + (y1 - y0) ** 2 < 0.35 ** 2:
                            continue
                        w, h = sizes_um[c["master"]]
                        ax.annotate("", xy=(x1 + w / 2, y1 + h / 2), xytext=(x0 + w / 2, y0 + h / 2),
                                    arrowprops=dict(arrowstyle="->", color="#8f1f16",
                                                    lw=0.9, alpha=0.9, shrinkA=0, shrinkB=0))
            if zoom:
                ax.set_xlim(bx, bx + step)
                ax.set_ylim(by, by + step)
                ax.set_title(zoom_titles[col], fontsize=12, pad=8)
                ax.set_xlabel("%.0f x %.0f um window at (%.0f, %.0f)" % (step, step, bx, by),
                              fontsize=9, color="#555555")
            else:
                ax.set_xlim(xl, xh)
                ax.set_ylim(yl, yh)
                ax.set_title(over_titles[col], fontsize=12, pad=8)
                if col == 1:
                    ax.add_patch(Rectangle((bx, by), step, step, facecolor="none",
                                           edgecolor="#1b1b1b", linewidth=1.2, zorder=5))
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#888888")

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=CELL_FC, edgecolor=CELL_EC, label="movable cell"),
        Rectangle((0, 0), 1, 1, facecolor=FROZEN_FC, edgecolor=FROZEN_EC,
                  label="pin-access-critical cell, frozen (FIXED)"),
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=GHOST_EC,
                  label="its position before the pre-pass (arrow = how it moved)"),
        Rectangle((0, 0), 1, 1, facecolor=STRIPE_FC, alpha=0.30, edgecolor="none",
                  label="M3 power stripe (vertical)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4 if not args.with_overview else 2,
               frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.012))
    n_moved = sum(1 for c in frozen
                  if (c["x"] - c["gp_x"]) ** 2 + (c["y"] - c["gp_y"]) ** 2 > (0.35 * units) ** 2)
    fig.suptitle("%s: pin-access pre-pass  --  %d of %d cells frozen (%d repositioned)"
                 % (d, len(pa_frozen), len(gp_all), n_moved), fontsize=15,
                 y=0.965 if args.with_overview else 0.975)
    fig.tight_layout(rect=(0, 0.075 if not args.with_overview else 0.055, 1,
                           0.93 if not args.with_overview else 0.95))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=170)
    print("wrote %s  (frozen=%d, zoom at %.0f,%.0f um with %d frozen cells)"
          % (out, len(pa_frozen), bx, by, best))


if __name__ == "__main__":
    main()

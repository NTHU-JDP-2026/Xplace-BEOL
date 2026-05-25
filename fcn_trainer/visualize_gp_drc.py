#!/usr/bin/env python3
"""Visualize GP placement + DRC violations from DEF and DRC report files."""

import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection


def parse_lef(lef_paths):
    """Return dict mapping macro name → (width_um, height_um)."""
    macros = {}
    pat = re.compile(r'^MACRO (\S+).*?SIZE ([\d.]+) BY ([\d.]+)', re.DOTALL | re.MULTILINE)
    for path in lef_paths:
        with open(path) as f:
            text = f.read()
        for m in pat.finditer(text):
            macros[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return macros


def parse_def(def_path, macro_sizes=None, dbu=4000):
    with open(def_path, 'r') as f:
        content = f.read()

    # Units
    m = re.search(r'UNITS DISTANCE MICRONS (\d+)', content)
    dbu = int(m.group(1)) if m else dbu

    # Die area
    m = re.search(r'DIEAREA\s+\(\s*(\d+)\s+(\d+)\s*\)\s+\(\s*(\d+)\s+(\d+)\s*\)', content)
    die = tuple(int(m.group(i)) for i in range(1, 5))  # (x0, y0, x1, y1)

    # Components: name, cell_type, x, y
    comp_section = re.search(r'COMPONENTS \d+ ;(.*?)END COMPONENTS', content, re.DOTALL)
    # Match: - inst_name cell_type \n + PLACED ( x y ) orient ;
    comp_pat = re.compile(
        r'-\s+\S+\s+(\S+)\s*\n\s+\+\s+PLACED\s+\(\s*(\d+)\s+(\d+)\s*\)')
    components = []
    fallback_w = 1728  # 2 sites × 864 DBU, used if macro not in LEF
    fallback_h = 4320  # 1 row height
    if comp_section:
        for m in comp_pat.finditer(comp_section.group(1)):
            cell_type = m.group(1)
            x, y = int(m.group(2)), int(m.group(3))
            if macro_sizes and cell_type in macro_sizes:
                w_dbu = round(macro_sizes[cell_type][0] * dbu)
                h_dbu = round(macro_sizes[cell_type][1] * dbu)
            else:
                w_dbu, h_dbu = fallback_w, fallback_h
            components.append((x, y, w_dbu, h_dbu))

    # Special nets: M3 and M4 STRIPE segments
    spec_section = re.search(r'SPECIALNETS \d+ ;(.*?)END SPECIALNETS', content, re.DOTALL)
    m3_stripes = []  # vertical: (x_center, y0, y1, width)
    m4_stripes = []  # horizontal: (x0, x1, y_center, width)

    if spec_section:
        sc = spec_section.group(1)

        # M3 vertical: ( x y0 ) ( * y1 )
        p = re.compile(r'M3\s+(\d+)\s+\+\s+SHAPE\s+STRIPE\s+\(\s*(\d+)\s+(\d+)\s*\)'
                       r'(?:\s+MASK\s+\d+)?\s+\(\s*\*\s+(\d+)\s*\)')
        for m in p.finditer(sc):
            m3_stripes.append((int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(1))))

        # M4 horizontal: ( x0 y ) [MASK n] ( x1 * )
        p = re.compile(r'M4\s+(\d+)\s+\+\s+SHAPE\s+STRIPE\s+\(\s*(\d+)\s+(\d+)\s*\)'
                       r'(?:\s+MASK\s+\d+)?\s+\(\s*(\d+)\s+\*\s*\)')
        for m in p.finditer(sc):
            m4_stripes.append((int(m.group(2)), int(m.group(4)), int(m.group(3)), int(m.group(1))))

    return dict(dbu=dbu, die=die, components=components,
                m3_stripes=m3_stripes, m4_stripes=m4_stripes,
                n_lef_miss=sum(1 for c in components if c[2] == fallback_w and c[3] == fallback_h))


def parse_drc(drc_path, dbu):
    """Return list of dicts with type, center x/y in DB units."""
    violations = []
    bounds_pat = re.compile(
        r'Bounds\s*:\s*\(\s*([\d.]+),\s*([\d.]+)\s*\)\s*\(\s*([\d.]+),\s*([\d.]+)\s*\)')
    current_type = 'DRC'
    with open(drc_path, 'r') as f:
        for line in f:
            tm = re.match(r'^(\w+):\s', line)
            if tm:
                current_type = tm.group(1)
            bm = bounds_pat.search(line)
            if bm:
                x1, y1 = float(bm.group(1)) * dbu, float(bm.group(2)) * dbu
                x2, y2 = float(bm.group(3)) * dbu, float(bm.group(4)) * dbu
                violations.append(dict(type=current_type,
                                       x=(x1 + x2) / 2, y=(y1 + y2) / 2))
    return violations


def visualize(def_data, violations, output_path):
    x0, y0, x1, y1 = def_data['die']
    die_w, die_h = x1 - x0, y1 - y0

    def nx(x): return (x - x0) / die_w   # normalize to [0,1]
    def ny(y): return (y - y0) / die_h

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # --- M3 vertical power stripes (blue, drawn first so cells appear on top) ---
    for (xc, sy0, sy1, sw) in def_data['m3_stripes']:
        half_w = sw / 2 / die_w
        pxc = nx(xc)
        rect = patches.Rectangle((pxc - half_w, ny(sy0)),
                                  2 * half_w, ny(sy1) - ny(sy0),
                                  linewidth=0, facecolor='#1565c0', alpha=0.25)
        ax.add_patch(rect)

    # --- M4 horizontal power stripes (orange) ---
    for (sx0, sx1, yc, sh) in def_data['m4_stripes']:
        half_h = sh / 2 / die_h
        pyc = ny(yc)
        rect = patches.Rectangle((nx(sx0), pyc - half_h),
                                  nx(sx1) - nx(sx0), 2 * half_h,
                                  linewidth=0, facecolor='#e65100', alpha=0.25)
        ax.add_patch(rect)

    # --- Cells as rectangles (real sizes from LEF) ---
    if def_data['components']:
        from matplotlib.collections import PatchCollection as PC
        rects = [patches.Rectangle((nx(x), ny(y)), w / die_w, h / die_h)
                 for x, y, w, h in def_data['components']]
        coll = PC(rects, facecolor='#888888', edgecolor='none',
                  alpha=0.55, rasterized=True, zorder=4)
        ax.add_collection(coll)

    # --- DRC violations ---
    if violations:
        dvx = np.array([nx(v['x']) for v in violations])
        dvy = np.array([ny(v['y']) for v in violations])
        ax.scatter(dvx, dvy, s=18, c='#d50000', marker='x', linewidths=1.0,
                   alpha=0.9, zorder=6, label=f'DRV ({len(violations):,})')

    # --- 256×256 bin grid ---
    n = 256
    segs_v = [[(i / n, 0), (i / n, 1)] for i in range(1, n)]
    segs_h = [[(0, i / n), (1, i / n)] for i in range(1, n)]
    lc = LineCollection(segs_v + segs_h,
                        colors='#bbbbbb', linewidths=0.3, alpha=0.7, zorder=3)
    ax.add_collection(lc)

    # --- Die boundary ---
    ax.add_patch(patches.Rectangle((0, 0), 1, 1,
                                   linewidth=1.5, edgecolor='black',
                                   facecolor='none', zorder=10))

    # --- Formatting ---
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')

    dbu = def_data['dbu']
    w_um = die_w / dbu
    h_um = die_h / dbu
    ax.set_title(
        f'aes_cipher_top  |  GP Placement & DRC Violations\n'
        f'Die: {w_um:.1f} × {h_um:.1f} µm  |  '
        f'Cells: {len(def_data["components"]):,}  |  '
        f'DRV: {len(violations):,}',
        color='black', fontsize=12, pad=10)

    ax.tick_params(colors='black', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#aaaaaa')

    # Legend
    handles = [
        patches.Patch(facecolor='#888888', alpha=0.6,
                      label=f'Cells ({len(def_data["components"]):,})'),
        patches.Patch(facecolor='#1565c0', alpha=0.5,
                      label=f'M3 power stripes ({len(def_data["m3_stripes"])})'),
        patches.Patch(facecolor='#e65100', alpha=0.5,
                      label=f'M4 power stripes ({len(def_data["m4_stripes"])})'),
        mlines.Line2D([], [], marker='x', color='#d50000', markersize=7,
                      linestyle='none', label=f'DRV ({len(violations):,})'),
        mlines.Line2D([], [], color='#aaaaaa', linewidth=0.8,
                      label='256×256 bins'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=9,
              facecolor='white', edgecolor='#aaaaaa',
              labelcolor='black', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {output_path}")


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    def_path = os.path.join(base, 'data', 'contest.gp.def')
    drc_path = os.path.join(base, 'data', 'aes_cipher_top.drc.rpt')
    out_path = os.path.join(base, 'data', 'aes_cipher.gp.drc.png')
    lef_paths = [
        os.path.join(base, 'SL_modified.lef'),
        os.path.join(base, 'L_modified.lef'),
    ]

    print("Parsing LEF ...")
    macros = parse_lef(lef_paths)
    print(f"  Macros:      {len(macros)}")

    print("Parsing DEF ...")
    dd = parse_def(def_path, macro_sizes=macros)
    print(f"  Die:         {dd['die']}")
    print(f"  DBU/µm:      {dd['dbu']}")
    print(f"  Cells:       {len(dd['components']):,}  (LEF fallback: {dd['n_lef_miss']})")
    print(f"  M3 stripes:  {len(dd['m3_stripes'])}")
    print(f"  M4 stripes:  {len(dd['m4_stripes'])}")

    print("Parsing DRC ...")
    viols = parse_drc(drc_path, dd['dbu'])
    print(f"  Violations:  {len(viols):,}")

    print("Rendering ...")
    visualize(dd, viols, out_path)

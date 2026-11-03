#!/usr/bin/env python3
"""
Measure how far AOI/OAI (or any) cells sit from vertical M3 power stripes.

M3 stripes in this PDN run the full die height, so the only thing that
matters is the x geometry. For every instance of interest this reports:

  overlap_frac : fraction of the cell's width that sits under an M3 stripe
  gap_um       : x-distance from the cell footprint to the nearest stripe
                 edge (0 when overlapping)
  conflict U   : scarcity-weighted overlap
                 int_0^1 M(u) * stripe(x_lo + u*w) du / int_0^1 M(u) du
                 where M(u) is the flip-robust intra-cell pin-scarcity
                 profile from *_pin_scarcity_lib.json -- i.e. the quantity
                 --stripe_force_weight is supposed to minimise, evaluated
                 at full resolution rather than on the coarse placement bins.

Usage:
  python3 tool/analyze_stripe_distance.py \
      --fp_def xplace_work/sha256_unplaced.def \
      --lef xplace_work/L_modified.lef,xplace_work/SL_modified.lef \
      --scarcity_json L_1x_pin_scarcity_lib.json,SL_1x_pin_scarcity_lib.json \
      --def result/sha256_ab_w0/placement_sha256_gp.def=w0 \
      --def result/sha256_ab_w01/placement_sha256_gp.def=w0.1
"""

import argparse
import json
import math
import os
import re


def parse_units(path):
    with open(path, errors='ignore') as f:
        for line in f:
            m = re.match(r'\s*UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', line)
            if m:
                return float(m.group(1))
    return 1000.0


def parse_m3_stripes(fp_def):
    """Vertical M3 stripe centrelines + widths (DBU) from SPECIALNETS."""
    stripes = []
    in_snet = False
    pat = re.compile(
        r'\bM3\s+(\d+)\s*\+\s*SHAPE\s+STRIPE\s*\(\s*(\d+)\s+[\d*]+\s*\)\s*\(\s*\*\s+[\d]+\s*\)')
    with open(fp_def, errors='ignore') as f:
        for line in f:
            if line.startswith('SPECIALNETS'):
                in_snet = True
                continue
            if line.startswith('END SPECIALNETS'):
                break
            if not in_snet:
                continue
            m = pat.search(line)
            if m:
                w, xc = float(m.group(1)), float(m.group(2))
                stripes.append((xc - w / 2.0, xc + w / 2.0))
    stripes.sort()
    # merge touching/overlapping intervals
    merged = []
    for lo, hi in stripes:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def parse_lef_sizes(lef_paths):
    """{macro_name: (width_um, height_um)} from LEF MACRO ... SIZE w BY h."""
    sizes = {}
    for path in lef_paths:
        if not os.path.isfile(path):
            continue
        cur = None
        with open(path, errors='ignore') as f:
            for line in f:
                m = re.match(r'\s*MACRO\s+(\S+)', line)
                if m:
                    cur = m.group(1)
                    continue
                if cur:
                    m = re.match(r'\s*SIZE\s+([0-9.]+)\s+BY\s+([0-9.]+)', line)
                    if m:
                        sizes[cur] = (float(m.group(1)), float(m.group(2)))
                        cur = None
    return sizes


def parse_components(def_path):
    """[(inst, master, x_dbu, y_dbu, orient)] for PLACED/FIXED components."""
    comps = []
    in_comp = False
    # "- inst master + PLACED ( x y ) N ;" possibly split across lines
    pat = re.compile(
        r'-\s+(\S+)\s+(\S+)\s*\+\s*(?:PLACED|FIXED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*(\w+)')
    with open(def_path, errors='ignore') as f:
        buf = ''
        for line in f:
            if line.startswith('COMPONENTS'):
                in_comp = True
                continue
            if line.startswith('END COMPONENTS'):
                break
            if not in_comp:
                continue
            buf += ' ' + line.strip()
            if ';' in buf:
                m = pat.search(buf)
                if m:
                    comps.append((m.group(1), m.group(2),
                                  float(m.group(3)), float(m.group(4)), m.group(5)))
                buf = ''
    return comps


def load_scarcity(json_paths):
    lookup = {}
    for path in json_paths:
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            doc = json.load(f)
        sigma = float(doc.get('sigma_default', 0.036))
        for name, e in doc.get('cells', {}).items():
            pins = [(float(p['x']), float(p['ap'])) for p in e.get('pins', [])]
            if pins and float(e.get('size_x', 0)) > 0:
                lookup[name] = {'size_x': float(e['size_x']), 'sigma': sigma, 'pins': pins}
    return lookup


def profile_M(entry, nsamp=64):
    """M(u) = max(S(u), S_flip(u)) sampled at nsamp points of u in [0,1]."""
    sx = entry['size_x']
    sig = entry['sigma'] / sx
    pins = [(p[0] / sx, 1.0 / max(p[1], 1e-6)) for p in entry['pins']]
    out = []
    for i in range(nsamp):
        u = (i + 0.5) / nsamp
        s = sum(w * math.exp(-0.5 * ((u - c) / sig) ** 2) for c, w in pins)
        sf = sum(w * math.exp(-0.5 * ((u - (1.0 - c)) / sig) ** 2) for c, w in pins)
        out.append(max(s, sf))
    return out


def stripe_overlap_len(lo, hi, stripes):
    tot = 0.0
    for slo, shi in stripes:
        if shi <= lo:
            continue
        if slo >= hi:
            break
        tot += min(hi, shi) - max(lo, slo)
    return tot


def nearest_gap(lo, hi, stripes):
    best = float('inf')
    for slo, shi in stripes:
        if hi <= slo:
            best = min(best, slo - hi)
        elif lo >= shi:
            best = min(best, lo - shi)
        else:
            return 0.0
    return best


def in_stripe(x, stripes):
    for slo, shi in stripes:
        if slo <= x < shi:
            return True
        if x < slo:
            return False
    return False


def pct(vals, p):
    if not vals:
        return float('nan')
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def analyse(def_path, label, stripes, sizes, scarcity, dbu, name_re):
    comps = parse_components(def_path)
    prof_cache = {}
    ov, gaps, conflicts = [], [], []
    n_match, n_missing_size = 0, 0
    for inst, master, x, y, orient in comps:
        if not name_re.search(master):
            continue
        if master not in sizes:
            n_missing_size += 1
            continue
        n_match += 1
        w_um = sizes[master][0]
        w = w_um * dbu
        lo, hi = x, x + w
        ov.append(stripe_overlap_len(lo, hi, stripes) / w)
        gaps.append(nearest_gap(lo, hi, stripes) / dbu)   # um
        entry = scarcity.get(master)
        if entry is not None:
            if master not in prof_cache:
                prof_cache[master] = profile_M(entry)
            M = prof_cache[master]
            n = len(M)
            num = den = 0.0
            for i, mv in enumerate(M):
                xs = lo + (i + 0.5) / n * w
                den += mv
                if in_stripe(xs, stripes):
                    num += mv
            conflicts.append(num / den if den > 0 else 0.0)
    return {
        'label': label, 'n': n_match, 'missing_size': n_missing_size,
        'overlap_mean': sum(ov) / len(ov) if ov else 0.0,
        'overlap_any': sum(1 for v in ov if v > 0) / len(ov) if ov else 0.0,
        'overlap_p50': pct(ov, 50), 'overlap_p90': pct(ov, 90),
        'gap_mean': sum(gaps) / len(gaps) if gaps else 0.0,
        'gap_p10': pct(gaps, 10), 'gap_p50': pct(gaps, 50), 'gap_p90': pct(gaps, 90),
        'conflict_mean': sum(conflicts) / len(conflicts) if conflicts else 0.0,
        'conflict_p90': pct(conflicts, 90),
        'gaps': gaps, 'ov': ov,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fp_def', required=True, help='floorplan DEF containing SPECIALNETS PDN')
    ap.add_argument('--lef', required=True, help='comma-separated LEF paths')
    ap.add_argument('--scarcity_json', default='', help='comma-separated *_pin_scarcity_lib.json')
    ap.add_argument('--def', dest='defs', action='append', required=True,
                    help='placement DEF to analyse, as path or path=label (repeatable)')
    ap.add_argument('--pattern', default=r'^(AOI|OAI)',
                    help='regex on master name (default: AOI/OAI gates)')
    ap.add_argument('--hist', action='store_true', help='print gap histogram')
    args = ap.parse_args()

    dbu = parse_units(args.fp_def)
    stripes = parse_m3_stripes(args.fp_def)
    sizes = parse_lef_sizes([p.strip() for p in args.lef.split(',') if p.strip()])
    scarcity = load_scarcity([p.strip() for p in args.scarcity_json.split(',') if p.strip()])
    name_re = re.compile(args.pattern)

    die_w = None
    with open(args.fp_def, errors='ignore') as f:
        for line in f:
            m = re.match(r'\s*DIEAREA\s*\(\s*(-?\d+)\s+-?\d+\s*\)\s*\(\s*(-?\d+)\s+-?\d+\s*\)', line)
            if m:
                die_w = (float(m.group(2)) - float(m.group(1))) / dbu
                break
    # group stripes into bands (paired VDD/VSS stripes sit within ~1 site of
    # each other and act as one obstacle for pin access)
    bands, band_gap = [], 1.0 * dbu
    for lo, hi in stripes:
        if bands and lo - bands[-1][1] <= band_gap:
            bands[-1] = (bands[-1][0], hi)
        else:
            bands.append((lo, hi))
    cov = sum(h - l for l, h in stripes) / dbu
    bpitch = ((bands[1][0] - bands[0][0]) / dbu) if len(bands) > 1 else float('nan')
    print(f"DBU/um={dbu:.0f}  die_w={die_w:.1f}um  M3 stripes={len(stripes)} "
          f"(width={(stripes[0][1]-stripes[0][0])/dbu:.3f}um) grouped into {len(bands)} bands "
          f"(band width={(bands[0][1]-bands[0][0])/dbu:.3f}um, band pitch={bpitch:.3f}um)")
    print(f"stripe x-coverage={cov:.1f}um / {die_w:.1f}um = {cov/die_w*100:.1f}%  "
          f"band x-coverage={sum(h-l for l, h in bands)/dbu/die_w*100:.1f}%")
    print(f"pattern={args.pattern!r}  masters in LEF={len(sizes)}  scarcity entries={len(scarcity)}")
    print()

    rows = []
    for spec in args.defs:
        path, _, label = spec.partition('=')
        rows.append(analyse(path, label or os.path.basename(os.path.dirname(path)),
                            stripes, sizes, scarcity, dbu, name_re))

    hdr = (f"{'run':<10}{'cells':>7}{'ovl_any%':>10}{'ovl_mean':>10}{'ovl_p90':>9}"
           f"{'gap_mean':>10}{'gap_p10':>9}{'gap_p50':>9}{'gap_p90':>9}"
           f"{'conf_mean':>11}{'conf_p90':>10}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['label']:<10}{r['n']:>7}{r['overlap_any']*100:>10.1f}"
              f"{r['overlap_mean']:>10.4f}{r['overlap_p90']:>9.3f}"
              f"{r['gap_mean']:>10.3f}{r['gap_p10']:>9.3f}{r['gap_p50']:>9.3f}{r['gap_p90']:>9.3f}"
              f"{r['conflict_mean']:>11.4f}{r['conflict_p90']:>10.4f}")

    if args.hist:
        print("\ngap-to-nearest-stripe histogram (um):")
        edges = [0, 0.001, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 1e9]
        names = ['0(ovl)', '<0.25', '<0.5', '<1.0', '<1.5', '<2.0', '<3.0', '<4.0', '<6.0', '>=6.0']
        print(f"{'run':<10}" + ''.join(f"{n:>9}" for n in names))
        for r in rows:
            counts = [0] * (len(edges) - 1)
            for g in r['gaps']:
                for i in range(len(edges) - 1):
                    if edges[i] <= g < edges[i + 1]:
                        counts[i] += 1
                        break
            tot = max(1, len(r['gaps']))
            print(f"{r['label']:<10}" + ''.join(f"{c/tot*100:>8.1f}%" for c in counts))


if __name__ == '__main__':
    main()

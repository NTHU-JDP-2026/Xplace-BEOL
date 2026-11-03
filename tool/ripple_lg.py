#!/usr/bin/env python3
"""Glue between Xplace GP output and the Ripple legalizer (ripple/ branch
pa-two-phase).

  prep   GP DEF -> Ripple input DEF (instance names unescaped: Xplace writes
         COMPONENTS without the '\\[' escapes but NETS with them, and Ripple
         segfaults on the mismatch) + the pin-access-critical instance list,
         selected exactly like tool/pin_access_prepass.py does.
  merge  Ripple output positions -> the original GP DEF (escapes kept), so the
         result is a drop-in placement_<design>_gp.def. --fix_list marks the
         listed instances FIXED.
  check  legality, displacement, HPWL and the pin-access mechanism metrics of a
         placement DEF, against the GP DEF.
"""

import argparse
import bisect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pin_access_prepass import (COMP_POS, NET_CONN, parse_components,  # noqa: E402
                                parse_def_header, parse_lef_widths,
                                parse_m3_stripes, parse_scarcity_reports)


def unescape(name):
    return name.replace("\\[", "[").replace("\\]", "]")


def select_critical(comps, scores, sizes_um, sigma, budget_frac):
    """Same rule as pin_access_prepass.main: masters with score >= mean + k*std,
    instances in score order, skip any that would overflow the area budget."""
    vals = sorted(scores.values())
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    thresh = mean + sigma * std
    movable = [c for c in comps if c.get("status") in ("PLACED", "UNPLACED")]
    total = 0.0
    for c in movable:
        w, h = sizes_um.get(c["master"], (0.0, 0.0))
        total += w * h
    cands = [(scores[c["master"]], c) for c in movable
             if c["master"] in scores and c["master"] in sizes_um
             and scores[c["master"]] >= thresh]
    cands.sort(key=lambda t: -t[0])
    chosen, used = [], 0.0
    for sc, c in cands:
        w, h = sizes_um[c["master"]]
        if used + w * h > budget_frac * total:
            continue
        used += w * h
        chosen.append((sc, c))
    return chosen, used, total, len(cands)


def cmd_prep(args):
    with open(args.gp_def) as fh:
        text = fh.read()
    with open(args.out_def, "w") as fh:
        fh.write(unescape(text))
    print("[ripple_lg] wrote %s" % args.out_def)
    if not args.critical_out:
        return
    scores = parse_scarcity_reports(args.reports.split(","))
    sizes_um = parse_lef_widths(args.lefs.split(","))
    _, _, _, comps = parse_components(args.gp_def)
    chosen, used, total, ncand = select_critical(comps, scores, sizes_um,
                                                 args.score_sigma, args.area_budget)
    with open(args.critical_out, "w") as fh:
        for sc, c in chosen:
            fh.write("%s %.9g\n" % (unescape(c["name"]), sc))
    print("[ripple_lg] candidates=%d chosen=%d area=%.1f/%.1f um2 (%.2f%%) -> %s"
          % (ncand, len(chosen), used, total, 100.0 * used / total, args.critical_out))


def cmd_merge(args):
    _, _, _, placed = parse_components(args.ripple_def)
    pos = {unescape(c["name"]): (c["x"], c["y"], c["orient"]) for c in placed if "x" in c}
    fix = set()
    if args.fix_list:
        with open(args.fix_list) as fh:
            fix = {ln.split()[0] for ln in fh if ln.strip()}
    only = None
    if args.only_list:
        with open(args.only_list) as fh:
            only = {ln.split()[0] for ln in fh if ln.strip()}
    lines, cstart, cend, comps = parse_components(args.gp_def)
    out = lines[:cstart + 1]
    n_upd = n_fix = n_miss = 0
    for c in comps:
        key = unescape(c["name"])
        if c.get("status") not in ("PLACED", "UNPLACED") or (only is not None and key not in only):
            out.extend(c["lines"])
            continue
        if key not in pos:
            n_miss += 1
            out.extend(c["lines"])
            continue
        x, y, o = pos[key]
        status = "FIXED" if key in fix else "PLACED"
        n_fix += status == "FIXED"
        out.append(COMP_POS.sub("+ %s ( %d %d ) %s" % (status, x, y, o),
                                "".join(c["lines"]), count=1))
        n_upd += 1
    out.extend(lines[cend:])
    with open(args.out_def, "w") as fh:
        fh.writelines(out)
    print("[ripple_lg] merged %d positions (%d FIXED, %d movable missing) -> %s"
          % (n_upd, n_fix, n_miss, args.out_def))
    if n_miss:
        sys.exit(2)


def net_hpwl(def_path, centre):
    """Cell-centre HPWL over component pins (IO pins ignored), DEF units."""
    total = 0.0
    in_nets = False
    cur = None
    with open(def_path) as fh:
        for line in fh:
            if not in_nets:
                in_nets = line.startswith("NETS ")
                continue
            if line.startswith("END NETS"):
                break
            ls = line.lstrip()
            if ls.startswith("- "):
                cur = []
                ls = ls[2:]
            if cur is None:
                continue
            cur.extend(unescape(m.group(1)) for m in NET_CONN.finditer(ls)
                       if m.group(1) != "PIN")
            if ";" in ls:
                pts = [centre[m] for m in cur if m in centre]
                if len(pts) > 1:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    total += max(xs) - min(xs) + max(ys) - min(ys)
                cur = None
    return total


def pct(v, p):
    return v[min(int(p * len(v)), len(v) - 1)] if v else 0.0


def cmd_check(args):
    sizes_um = parse_lef_widths(args.lefs.split(","))
    units, _, rows = parse_def_header(args.gp_def)
    site_w = rows[0]["step"]
    x0 = min(r["x0"] for r in rows)
    xh = x0 + max(r["nx"] for r in rows) * site_w
    row_y = sorted(r["y"] for r in rows)
    row_h = row_y[1] - row_y[0]
    orient_of = {r["y"]: r["orient"] for r in rows}
    bands = parse_m3_stripes(args.pdn_def)
    band_lo = [b[0] for b in bands]

    _, _, _, gp = parse_components(args.gp_def)
    _, _, _, pl = parse_components(args.def_)
    gp_status = {unescape(c["name"]): c.get("status") for c in gp}
    gpos = {unescape(c["name"]): (c["x"], c["y"]) for c in gp if "x" in c}
    crit = []
    if args.critical:
        with open(args.critical) as fh:
            crit = [ln.split()[0] for ln in fh if ln.strip()]
    crit_set = set(crit)

    off_grid = orient_bad = out_core = 0
    per_row = {}
    disp, disp_c = [], []
    centre = {}
    cpos = {}
    for c in pl:
        if "x" not in c or c["master"] not in sizes_um:
            continue
        name = unescape(c["name"])
        w = sizes_um[c["master"]][0] * units
        h = sizes_um[c["master"]][1] * units
        centre[name] = (c["x"] + w / 2.0, c["y"] + h / 2.0)
        per_row.setdefault(c["y"], []).append((c["x"], c["x"] + w))
        if gp_status.get(name) not in ("PLACED", "UNPLACED"):
            continue          # pre-placed in the input: not ours to judge
        if args.fixed_only and c["status"] != "FIXED":
            per_row[c["y"]].pop()     # unlegalized: no grid, no overlap check
            gx, gy = gpos[name]
            disp.append((abs(c["x"] - gx) + abs(c["y"] - gy)) / units)
            continue
        if (c["x"] - x0) % site_w or c["y"] not in orient_of:
            off_grid += 1
        else:
            ro = orient_of[c["y"]]
            if not ((ro == "N" and c["orient"] in ("N", "FN")) or
                    (ro == "FS" and c["orient"] in ("FS", "S"))):
                orient_bad += 1
        if c["x"] < x0 or c["x"] + w > xh:
            out_core += 1
        gx, gy = gpos[name]
        d = (abs(c["x"] - gx) + abs(c["y"] - gy)) / units
        disp.append(d)
        if name in crit_set:
            disp_c.append(d)
            cpos[name] = (c["x"], c["x"] + w, int(round((c["y"] - row_y[0]) / row_h)))
    overlap = 0
    for lst in per_row.values():
        lst.sort()
        for a, b in zip(lst, lst[1:]):
            if a[1] > b[0]:
                overlap += 1
    disp.sort()
    disp_c.sort()

    gcentre = {}
    for c in gp:
        if "x" in c and c["master"] in sizes_um:
            w, h = sizes_um[c["master"]]
            gcentre[unescape(c["name"])] = (c["x"] + w * units / 2.0,
                                            c["y"] + h * units / 2.0)
    hp0 = net_hpwl(args.gp_def, gcentre)
    hp1 = net_hpwl(args.def_, centre)

    print("[check] %s" % (args.tag or os.path.basename(args.def_)))
    print("  legality : off-grid=%d orient=%d out-of-core=%d overlap=%d"
          % (off_grid, orient_bad, out_core, overlap))
    print("  disp all : n=%d mean=%.3f p50=%.3f p90=%.3f p99=%.3f max=%.3f um (manhattan)"
          % (len(disp), sum(disp) / max(len(disp), 1), pct(disp, .5), pct(disp, .9),
             pct(disp, .99), disp[-1] if disp else 0))
    print("  HPWL     : %.1f -> %.1f um (%+.2f%%)"
          % (hp0 / units, hp1 / units, 100 * (hp1 / hp0 - 1)))
    if not crit:
        return

    on_stripe = 0
    for a, b, _ in cpos.values():
        i = bisect.bisect_right(band_lo, b + site_w) - 1
        for j in (i, i - 1):
            if 0 <= j < len(bands) and a < bands[j][1] + site_w and b > bands[j][0] - site_w:
                on_stripe += 1
                break
    by_row = {}
    for name, (a, b, r) in cpos.items():
        by_row.setdefault(r, []).append((a, b, name))
    for lst in by_row.values():
        lst.sort()
    widest = max((b - a for a, b, _ in cpos.values()), default=0)

    def near(name, sep_sites, nrows):
        a, b, r = cpos[name]
        s = sep_sites * site_w
        for rr in range(r - nrows, r + nrows + 1):
            lst = by_row.get(rr, ())
            k = bisect.bisect_left(lst, (a - s - widest,))
            for pa, pb, pn in lst[k:]:
                if pa >= b + s:
                    break
                if pn != name and pb > a - s:
                    return True
        return False

    n = len(cpos)
    c4 = sum(near(nm, 4, 1) for nm in cpos)
    cs = sum(near(nm, args.sep, args.sep_rows) for nm in cpos)
    print("  critical : n=%d disp mean=%.3f p50=%.3f p90=%.3f max=%.3f um"
          % (n, sum(disp_c) / max(n, 1), pct(disp_c, .5), pct(disp_c, .9),
             disp_c[-1] if disp_c else 0))
    print("  critical : on-M3-stripe(+1site)=%d (%.1f%%)  peer<4s/1r=%d (%.1f%%)  "
          "peer<%ds/%dr=%d (%.1f%%)"
          % (on_stripe, 100.0 * on_stripe / max(n, 1), c4, 100.0 * c4 / max(n, 1),
             args.sep, args.sep_rows, cs, 100.0 * cs / max(n, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--gp_def", required=True)
    p.add_argument("--out_def", required=True)
    p.add_argument("--critical_out", default="")
    p.add_argument("--reports", default="")
    p.add_argument("--lefs", default="")
    p.add_argument("--score_sigma", type=float, default=1.0)
    p.add_argument("--area_budget", type=float, default=0.15)
    p = sub.add_parser("merge")
    p.add_argument("--gp_def", required=True)
    p.add_argument("--ripple_def", required=True)
    p.add_argument("--out_def", required=True)
    p.add_argument("--fix_list", default="")
    p.add_argument("--only_list", default="",
                   help="take Ripple positions only for these instances; every other "
                        "cell keeps its GP position (left to the downstream legalizer)")
    p = sub.add_parser("check")
    p.add_argument("--def", dest="def_", required=True)
    p.add_argument("--gp_def", required=True)
    p.add_argument("--pdn_def", required=True)
    p.add_argument("--lefs", required=True)
    p.add_argument("--critical", default="")
    p.add_argument("--sep", type=int, default=8)
    p.add_argument("--sep_rows", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--fixed_only", action="store_true",
                   help="judge legality of FIXED cells only (the rest is not legalized yet)")
    args = ap.parse_args()
    {"prep": cmd_prep, "merge": cmd_merge, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pin-access pre-pass: pull pin-access-critical cells off vertical M3 power
stripes, spread them apart from each other, snap them to site/row, and mark
them FIXED in the output DEF.

Everything else in the DEF is untouched, so a downstream legalizer (Innovus
refinePlace, or Xplace's own LG) is free to legalize the remaining cells around
the frozen ones.

Ranking comes from the pin-scarcity reports (per cell master, key =
score_flip_max); the stripe geometry comes from the M3 SPECIALNETS of a DEF
that still carries the PDN.
"""

import argparse
import bisect
import json
import os
import re
import sys
import time


# ── parsing helpers ────────────────────────────────────────────────────────────
def parse_scarcity_reports(paths):
    """master name -> score_flip_max, from the '[Ranking by score]' tables."""
    scores = {}
    pat = re.compile(r"^\s*\d+\s+(\S+)\s+flip_max=([0-9.eE+-]+)\s+as_placed=([0-9.eE+-]+)")
    for path in paths:
        with open(path) as fh:
            for line in fh:
                m = pat.match(line)
                if m:
                    scores[m.group(1)] = float(m.group(2))
    return scores


def parse_lef_widths(paths):
    """master name -> (width, height) in microns."""
    sizes = {}
    cur = None
    for path in paths:
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("MACRO "):
                    cur = s.split()[1]
                elif cur and s.startswith("SIZE "):
                    t = s.replace(";", "").split()
                    sizes[cur] = (float(t[1]), float(t[3]))
                elif s.startswith("END ") and cur and s.split()[-1] == cur:
                    cur = None
    return sizes


def parse_def_header(path):
    """units, die box, row geometry (site origin/step/count, row y list+orient)."""
    units, die, rows = None, None, []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("UNITS DISTANCE MICRONS"):
                units = float(s.split()[3])
            elif s.startswith("DIEAREA"):
                n = [float(v) for v in re.findall(r"-?\d+", s)]
                die = (n[0], n[1], n[2], n[3])
            elif s.startswith("ROW "):
                t = s.replace(";", "").split()
                # ROW <name> <site> <x> <y> <orient> DO <nx> BY <ny> STEP <sx> <sy>
                rows.append({
                    "x0": float(t[3]), "y": float(t[4]), "orient": t[5],
                    "nx": int(t[7]), "step": float(t[11]),
                })
            elif s.startswith("COMPONENTS"):
                break
    return units, die, rows


def parse_m3_stripes(path, layer="M3"):
    """[(xl, xh)] of vertical <layer> stripes, merged, in DEF units."""
    bands = []
    inside = False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("SPECIALNETS"):
                inside = True
                continue
            if s.startswith("END SPECIALNETS"):
                break
            if not inside or layer not in s or "SHAPE STRIPE" not in s:
                continue
            t = s.split()
            try:
                w = float(t[t.index(layer) + 1])
            except (ValueError, IndexError):
                continue
            pts = re.findall(r"\(\s*([-\d*]+)\s+([-\d*]+)\s*\)", s)
            if len(pts) < 2:
                continue
            (x1, y1), (x2, y2) = pts[0], pts[1]
            # vertical stripe: x is fixed ('*' means "same as previous point")
            if x2 == "*" or (x1 != "*" and x2 != "*" and x1 == x2):
                xc = float(x1)
                bands.append((xc - w / 2.0, xc + w / 2.0))
    bands.sort()
    merged = []
    for lo, hi in bands:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [tuple(b) for b in merged]


COMP_POS = re.compile(r"\+\s*(PLACED|FIXED|COVER|UNPLACED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*(\w+)")


NET_CONN = re.compile(r"\(\s*([^\s()]+)\s+[^\s()]+\s*\)")


def parse_components(path):
    """[(entry_lines, name, master, status, x, y, orient)] + surrounding text."""
    with open(path) as fh:
        lines = fh.readlines()
    start = end = None
    for i, line in enumerate(lines):
        if line.startswith("COMPONENTS "):
            start = i
        elif line.startswith("END COMPONENTS"):
            end = i
            break
    if start is None or end is None:
        sys.exit("COMPONENTS section not found in %s" % path)

    comps, buf = [], []
    for line in lines[start + 1:end]:
        if line.lstrip().startswith("- ") and buf:
            comps.append(buf)
            buf = []
        buf.append(line)
    if buf:
        comps.append(buf)

    parsed = []
    for buf in comps:
        text = "".join(buf)
        t = text.split()
        name, master = t[1], t[2]
        m = COMP_POS.search(text)
        if m is None:
            parsed.append({"lines": buf, "name": name, "master": master, "status": None})
            continue
        parsed.append({
            "lines": buf, "name": name, "master": master, "status": m.group(1),
            "x": int(m.group(2)), "y": int(m.group(3)), "orient": m.group(4),
        })
    return lines, start, end, parsed



def parse_nets(path, wanted):
    """cell name -> [net_id], and net_id -> [cell names], for nets that touch a
    cell in `wanted`. Only COMPONENT pins matter; IO pins are ignored because
    they do not move."""
    cell2nets, net2cells = {}, []
    in_nets = False
    cur = None
    with open(path) as fh:
        for line in fh:
            if not in_nets:
                if line.startswith("NETS "):
                    in_nets = True
                continue
            if line.startswith("END NETS"):
                break
            ls = line.lstrip()
            if ls.startswith("- "):
                if cur is not None and cur:
                    nid = len(net2cells)
                    net2cells.append(cur)
                    for nm in cur:
                        cell2nets.setdefault(nm, []).append(nid)
                cur = []
                rest = ls[2:]
            else:
                rest = ls
            if cur is None:
                continue
            # "( inst pin )" -- PIN entries reference IO pins, skip them
            for m in NET_CONN.finditer(rest):
                inst = m.group(1)
                if inst != "PIN":
                    cur.append(inst)
            if ";" in rest:
                if cur:
                    nid = len(net2cells)
                    net2cells.append(cur)
                    for nm in cur:
                        cell2nets.setdefault(nm, []).append(nid)
                cur = None
    # drop nets that no candidate touches -- they can never change
    keep = set()
    for nm in wanted:
        keep.update(cell2nets.get(nm, ()))
    return cell2nets, net2cells, keep


# ── orientation handling ───────────────────────────────────────────────────────
# Rows alternate N / FS. A cell in an N row is N or FN; in an FS row it is FS or
# S. Moving a cell across row parity mirrors it about X, which keeps every pin's
# x offset identical (so the stripe-overlap decision stays valid).
# A cell's left-right mirroring (its "flip class") is what moves pin x offsets;
# the row's own orientation decides the rest. Deriving the orientation from the
# TARGET row keeps the cell legal even when the GP position sat far from the row
# its DEF orientation was written for.
FLIP_CLASS = {"N": 0, "FS": 0, "FN": 1, "S": 1}
ORIENT_FOR = {("N", 0): "N", ("N", 1): "FN", ("FS", 0): "FS", ("FS", 1): "S"}


def orient_in_row(cell_orient, row_orient):
    key = (row_orient, FLIP_CLASS.get(cell_orient))
    return ORIENT_FOR.get(key, cell_orient)



def _self_check(out_def, bands, rows, sizes_um, units, args, results=None):
    """Re-read the DEF we just wrote and verify every frozen cell is legal:
    on the site/row grid, orientation compatible with its row, clear of every M3
    stripe, and not overlapping another frozen cell in the same row."""
    _, _, _, comps = parse_components(out_def)
    site_w = rows[0]["step"]
    row_x0 = min(r["x0"] for r in rows)
    row_orient = {r["y"]: r["orient"] for r in rows}
    band_lo = [b[0] for b in bands]
    off_grid = orient_bad = on_stripe = overlap = 0
    per_row = {}
    frozen = {r["name"] for r in results} if results is not None else None
    for c in comps:
        if c.get("status") not in ("FIXED", "COVER") or c["master"] not in sizes_um:
            continue
        if frozen is not None and c["name"] not in frozen:
            # a cell the DEF already had fixed: it only takes part in the overlap
            # scan below, its grid/orientation/stripe legality is not ours to judge
            per_row.setdefault(c["y"], []).append(
                (c["x"], c["x"] + sizes_um[c["master"]][0] * units))
            continue
        if (c["x"] - row_x0) % site_w:
            off_grid += 1
        ro = row_orient.get(c["y"])
        if ro is None:
            off_grid += 1
            continue
        if not ((ro == "N" and c["orient"] in ("N", "FN")) or
                (ro == "FS" and c["orient"] in ("FS", "S"))):
            orient_bad += 1
        w = sizes_um[c["master"]][0] * units
        x0, x1 = c["x"], c["x"] + w
        i = bisect.bisect_right(band_lo, x1) - 1
        for j in (i, i - 1):
            if 0 <= j < len(bands) and x0 < bands[j][1] and x1 > bands[j][0]:
                on_stripe += 1
                break
        per_row.setdefault(c["y"], []).append((x0, x1))
    for y, lst in per_row.items():
        lst.sort()
        for a, b in zip(lst, lst[1:]):
            if a[1] > b[0]:
                overlap += 1
    return off_grid, orient_bad, on_stripe, overlap



def hpwl_delta(name, cx, cy, ctx):
    """Change in half-perimeter wirelength, in DEF units, if `name`'s centre
    moves to (cx, cy). Every other cell is held at its current position -- the
    standard assumption in detailed placement. `rest_of[name]` holds, per net on
    this cell, the bounding box of that net's OTHER cells, so each net costs
    max(rest_box, this pin) - rest_box and the whole evaluation is O(degree)."""
    total = 0.0
    for rlx, rly, rhx, rhy, base in ctx["rest_of"].get(name, ()):
        w = (rhx if cx <= rhx else cx) - (rlx if cx >= rlx else cx)
        h = (rhy if cy <= rhy else cy) - (rly if cy >= rly else cy)
        total += (w + h) - base
    return total


def build_wl_ctx(def_path, comps, sizes_um, units, chosen_names, args):
    """Pre-compute, for every chosen cell, the per-net bounding box of the net's
    other cells. Nets with no other movable member cannot change and are
    dropped."""
    cell2nets, net2cells, _ = parse_nets(def_path, chosen_names)
    ctr = {}
    for c in comps:
        if "x" not in c:
            continue
        wh = sizes_um.get(c["master"])
        if wh is None:
            ctr[c["name"]] = (float(c["x"]), float(c["y"]))
        else:
            ctr[c["name"]] = (c["x"] + wh[0] * units / 2.0,
                              c["y"] + wh[1] * units / 2.0)
    rest_of = {}
    for name in chosen_names:
        boxes = []
        for nid in cell2nets.get(name, ()):
            xs = []
            ys = []
            for other in net2cells[nid]:
                if other == name:
                    continue
                p = ctr.get(other)
                if p is not None:
                    xs.append(p[0]); ys.append(p[1])
            if not xs:
                continue
            rlx, rhx = min(xs), max(xs)
            rly, rhy = min(ys), max(ys)
            boxes.append((rlx, rly, rhx, rhy, (rhx - rlx) + (rhy - rly)))
        if boxes:
            rest_of[name] = boxes
    return {"rest_of": rest_of, "wl_weight": args.wl_weight,
            "disp_w": args.disp_weight}


def assign_positions(chosen, sizes_um, units, bands, row_y, row_orient, row_x0,
                   site_w, row_h, core_xh, args, obstacles=None, wl_ctx=None):
    """Greedy assignment: walk the candidates in scarcity order and give each the
    nearest feasible position, first come first served.  The ordering matters --
    the most pin-access-critical cells get first pick and are never crowded out."""
    margin = args.stripe_margin_sites * site_w
    sep_x = args.min_sep_sites * site_w
    shift_s = args.max_shift_sites
    shift_r = args.max_shift_rows
    occupied = {}          # row index -> sorted list of (x0, x1)
    obstacles = obstacles or {}
    band_lo = [b[0] - margin for b in bands]

    def hits_obstacle(r, x0, x1):
        """the DEF's own FIXED cells: overlap is illegal, but they are not
        pin-access-critical, so they need no separation and only block their row"""
        for px0, px1 in obstacles.get(r, ()):
            if x0 < px1 and x1 > px0:
                return True
        return False

    def hits_stripe(x0, x1):
        if args.ignore_stripes:
            return False
        i = bisect.bisect_right(band_lo, x1) - 1
        for j in (i, i - 1):
            if 0 <= j < len(bands):
                if x0 < bands[j][1] + margin and x1 > bands[j][0] - margin:
                    return True
        return False

    def hits_peer(r, x0, x1):
        for rr in range(r - args.min_sep_rows, r + args.min_sep_rows + 1):
            for px0, px1 in occupied.get(rr, ()):
                if x0 - sep_x < px1 and x1 + sep_x > px0:
                    return True
        return False

    stats = {"kept": 0, "moved": 0, "skipped": 0, "on_stripe": 0}
    disp = []
    results = []
    pending = list(chosen)
    npass = max(1, getattr(args, "passes", 1))
    for _pass in range(npass):
        if not pending:
            break
        if _pass:
            # Retry what the previous pass could not place, with a wider search
            # window and a looser peer separation. Abandoning a cell leaves it
            # exactly where the global placer put it -- the worst outcome for a
            # pin-access-critical cell -- so a placement that merely misses the
            # separation target still beats none.
            shift_s *= 2
            shift_r += 1
            sep_x = max(sep_x // 2, site_w)
            print("[prepass] pass %d: retry %d skipped, shift=%d/%d sep=%.0f"
                  % (_pass + 1, len(pending), shift_s, shift_r, sep_x / site_w))
        retry = []
        for sc, c in pending:
            w_um, _ = sizes_um[c["master"]]
            w = w_um * units
            r0 = min(max(int(round((c["y"] - row_y[0]) / row_h)), 0), len(row_y) - 1)
            s0 = int(round((c["x"] - row_x0) / site_w))
            was_on_stripe = hits_stripe(row_x0 + s0 * site_w, row_x0 + s0 * site_w + w)
            if was_on_stripe:
                stats["on_stripe"] += 1

            best = None
            for dr in range(0, shift_r + 1):
                for rr in ({r0 + dr, r0 - dr} if dr else {r0}):
                    if not 0 <= rr < len(row_y):
                        continue
                    for ds in range(0, shift_s + 1):
                        for ss in ({s0 + ds, s0 - ds} if ds else {s0}):
                            x0 = row_x0 + ss * site_w
                            if x0 < row_x0 or x0 + w > core_xh:
                                continue
                            if (hits_stripe(x0, x0 + w) or hits_peer(rr, x0, x0 + w)
                                    or hits_obstacle(rr, x0, x0 + w)):
                                continue
                            # Cost of this candidate site. Displacement alone is
                            # direction-blind: moving 10 sites left and 10 sites
                            # right score the same, though the nets pull one way.
                            # With wl_weight > 0 we add the real HPWL delta of the
                            # nets on this cell, making the pin-access move
                            # wirelength-aware.
                            cost = (ds * site_w) ** 2 + (dr * row_h) ** 2
                            if wl_ctx is not None:
                                cost = cost * wl_ctx["disp_w"] + wl_ctx["wl_weight"] * hpwl_delta(
                                    c["name"], x0 + w / 2.0, row_y[rr] + row_h / 2.0, wl_ctx)
                            if best is None or cost < best[0]:
                                best = (cost, rr, x0)
                    if wl_ctx is None and best is not None and best[0] <= (dr * row_h) ** 2:
                        break     # cannot beat this row's best with a farther row
                if wl_ctx is None and best is not None and best[0] <= (dr * row_h) ** 2:
                    break
            if best is None:
                retry.append((sc, c))
                continue

            _, rr, x0 = best
            y = row_y[rr]
            orient = orient_in_row(c["orient"], row_orient[rr])
            bisect.insort(occupied.setdefault(rr, []), (x0, x0 + w))
            d = ((x0 - c["x"]) ** 2 + (y - c["y"]) ** 2) ** 0.5
            disp.append(d)
            if x0 == c["x"] and y == c["y"]:
                stats["kept"] += 1
            else:
                stats["moved"] += 1
            results.append({
                "name": c["name"], "master": c["master"], "score": sc,
                "gp_x": c["x"], "gp_y": c["y"], "x": int(round(x0)), "y": int(round(y)),
                "orient": orient, "row": rr, "was_on_stripe": was_on_stripe,
                "disp_dbu": d,
            })
            c["new"] = (int(round(x0)), int(round(y)), orient)
        pending = retry
    stats["skipped"] = len(pending)

    disp.sort()
    def pct(p):
        return disp[min(int(p * len(disp)), len(disp) - 1)] if disp else 0.0
    print("[prepass] frozen=%d (kept=%d moved=%d) skipped=%d | of chosen, %d were on a stripe"
          % (len(results), stats["kept"], stats["moved"], stats["skipped"], stats["on_stripe"]))
    print("[prepass] displacement (um): p50=%.3f p90=%.3f p99=%.3f max=%.3f"
          % (pct(.5) / units, pct(.9) / units, pct(.99) / units, (disp[-1] / units) if disp else 0))

    return results, stats, disp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gp_def", required=True, help="input (global) placement DEF")
    ap.add_argument("--pdn_def", required=True, help="DEF carrying SPECIALNETS (PDN)")
    ap.add_argument("--lefs", required=True, help="comma-separated cell LEFs")
    ap.add_argument("--reports", required=True, help="comma-separated pin-scarcity reports")
    ap.add_argument("--out_def", required=True)
    ap.add_argument("--out_json", default="", help="dump chosen instances here")
    ap.add_argument("--skip_check", action="store_true",
                    help="skip the post-write legality self-check (not recommended: "
                         "an illegal DEF makes Innovus checkPlace fail and leaves "
                         "refinePlace behaviour undefined)")
    ap.add_argument("--wl_weight", type=float, default=0.0,
                    help="weight on the HPWL delta of a candidate site. 0 keeps the "
                         "original displacement-only cost (direction blind).")
    ap.add_argument("--disp_weight", type=float, default=1.0,
                    help="weight on the squared-displacement term when --wl_weight > 0")
    ap.add_argument("--score_sigma", type=float, default=1.0,
                    help="select masters with score >= mean + sigma*std")
    ap.add_argument("--area_budget", type=float, default=0.05,
                    help="max frozen area / total movable cell area")
    ap.add_argument("--stripe_margin_sites", type=float, default=1.0)
    ap.add_argument("--min_sep_sites", type=float, default=4.0)
    ap.add_argument("--min_sep_rows", type=int, default=1)
    ap.add_argument("--passes", type=int, default=1,
                    help="retry passes over cells that found no legal site; each pass doubles "
                         "the search window and halves the peer separation. 1 = single pass.")
    ap.add_argument("--max_shift_sites", type=int, default=24)
    ap.add_argument("--max_shift_rows", type=int, default=3)
    ap.add_argument("--ignore_stripes", action="store_true",
                    help="control run: do not avoid M3 stripes (snap/spread only)")
    ap.add_argument("--random_control", action="store_true",
                    help="control run: pick a random same-area set of NON-critical "
                         "cells instead of the pin-access-critical ones")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --random_control")
    args = ap.parse_args()

    scores = parse_scarcity_reports(args.reports.split(","))
    sizes_um = parse_lef_widths(args.lefs.split(","))
    units, die, rows = parse_def_header(args.gp_def)
    bands = parse_m3_stripes(args.pdn_def)
    lines, cstart, cend, comps = parse_components(args.gp_def)

    if not rows:
        sys.exit("no ROW statements found")
    site_w = rows[0]["step"]
    row_x0 = min(r["x0"] for r in rows)
    row_nx = max(r["nx"] for r in rows)
    rows_sorted = sorted(rows, key=lambda r: r["y"])
    row_y = [r["y"] for r in rows_sorted]
    row_orient = [r["orient"] for r in rows_sorted]
    row_h = (row_y[1] - row_y[0]) if len(row_y) > 1 else 0.0
    core_xh = row_x0 + row_nx * site_w

    vals = sorted(scores.values())
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    thresh = mean + args.score_sigma * std

    print("[prepass] units=%g site_w=%g row_h=%g #rows=%d core_x=[%g,%g]"
          % (units, site_w, row_h, len(row_y), row_x0, core_xh))
    print("[prepass] M3 vertical stripes: %d bands, width=%g, pitch~%g (DBU)"
          % (len(bands), bands[0][1] - bands[0][0] if bands else 0,
             (bands[1][0] - bands[0][0]) if len(bands) > 1 else 0))
    print("[prepass] scarcity masters=%d mean=%.6f std=%.6f -> thresh=%.6f"
          % (len(vals), mean, std, thresh))

    # ── candidate selection ───────────────────────────────────────────────────
    movable = [c for c in comps if c.get("status") in ("PLACED", "UNPLACED")]
    total_area = 0.0
    for c in movable:
        w, h = sizes_um.get(c["master"], (0.0, 0.0))
        total_area += w * h
    cands = []
    for c in movable:
        sc = scores.get(c["master"])
        if sc is not None and sc >= thresh and c["master"] in sizes_um:
            cands.append((sc, c))
    cands.sort(key=lambda t: -t[0])
    if args.random_control:
        import random
        pool = [c for c in movable
                if c["master"] in sizes_um and scores.get(c["master"], 0.0) < thresh]
        random.Random(args.seed).shuffle(pool)
        cands = [(scores.get(c["master"], 0.0), c) for c in pool]
        print("[prepass] RANDOM CONTROL: drawing from %d non-critical cells" % len(pool))

    budget = args.area_budget * total_area
    chosen, used_area = [], 0.0
    for sc, c in cands:
        w, h = sizes_um[c["master"]]
        if used_area + w * h > budget:
            continue
        used_area += w * h
        chosen.append((sc, c))
    print("[prepass] movable=%d candidates=%d (masters over thresh) chosen=%d "
          "area=%.1f/%.1f um2 (%.2f%% of movable area, budget %.2f%%)"
          % (len(movable), len(cands), len(chosen), used_area, total_area,
             100.0 * used_area / total_area, 100.0 * args.area_budget))

    # The DEF may already contain FIXED cells -- a gp.def produced from a DEF that
    # already went through a pre-pass does (xplace_work/seed_paA.def has 1903).
    # They are not candidates -- `movable` excludes them -- but nothing may be
    # placed on top of them either, so they go in as per-row obstacles.
    # NOTE: keyed by a single row index, so an obstacle taller than one row would
    #   only block the row its origin sits in. Every FIXED cell in the current
    #   inputs is single-height; revisit this when macros can arrive here.
    obstacles, unknown = {}, 0
    for c in comps:
        if c.get("status") not in ("FIXED", "COVER"):
            continue
        if c["master"] not in sizes_um:
            unknown += 1
            continue
        r = min(max(int(round((c["y"] - row_y[0]) / row_h)), 0), len(row_y) - 1)
        obstacles.setdefault(r, []).append(
            (c["x"], c["x"] + sizes_um[c["master"]][0] * units))
    for r in obstacles:
        obstacles[r].sort()
    if obstacles or unknown:
        print("[prepass] pre-placed FIXED cells: %d over %d rows (%d with an "
              "unknown master, ignored)"
              % (sum(len(v) for v in obstacles.values()), len(obstacles), unknown))

    # ── wirelength context (optional) ─────────────────────────────────────────
    wl_ctx = None
    if args.wl_weight > 0:
        t0 = time.time()
        wl_ctx = build_wl_ctx(args.gp_def, comps, sizes_um, units,
                              {c["name"] for _, c in chosen}, args)
        print("[prepass] wirelength-aware: %d/%d chosen cells have movable nets "
              "(%.1fs)" % (len(wl_ctx["rest_of"]), len(chosen), time.time() - t0))

    # ── placement of the chosen subset ────────────────────────────────────────
    results, stats, disp = assign_positions(chosen, sizes_um, units, bands, row_y,
                                            row_orient, row_x0, site_w, row_h,
                                            core_xh, args, obstacles, wl_ctx)

    # ── write DEF ─────────────────────────────────────────────────────────────
    out = lines[:cstart + 1]
    for c in comps:
        if "new" not in c:
            out.extend(c["lines"])
            continue
        x, y, orient = c["new"]
        text = "".join(c["lines"])
        text = COMP_POS.sub("+ FIXED ( %d %d ) %s" % (x, y, orient), text, count=1)
        out.append(text)
    out.extend(lines[cend:])
    with open(args.out_def, "w") as fh:
        fh.writelines(out)
    print("[prepass] wrote %s" % args.out_def)

    if not args.skip_check:
        og, ob, ost, ov = _self_check(args.out_def, bands, rows, sizes_um, units,
                                      args, results)
        print("[prepass] self-check: off-grid=%d orient-mismatch=%d on-stripe=%d "
              "same-row-overlap=%d" % (og, ob, ost, ov))
        if og or ob or ost or ov:
            print("[prepass] ERROR: %s is NOT legal -- do not route it." % args.out_def)
            sys.exit(2)

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump({"params": vars(args), "stats": stats,
                       "num_frozen": len(results), "cells": results}, fh, indent=1)
        print("[prepass] wrote %s" % args.out_json)


if __name__ == "__main__":
    main()

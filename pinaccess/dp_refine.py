#!/usr/bin/env python3
"""PDN-aware detailed placement refinement on the GPU: orientation and position of every cell by a
row-wise dynamic program.

    python3 pinaccess/dp_refine.py <model.pt|none> <cells.lef> <pdn.def> <placement.def> <out.flip>
        [--rule track --rule_w 0] [--sites 4] [--band 3] [--crit_report R --crit_z 1.0]
        [--w_pa 20] [--w_band 60] [--w_pair 20 --pair_d 1] [--w_wl 1] [--w_disp 0.05] [--device cuda|cpu]
        [--cost ours|ispd17 --minl 1]

Same interface and output as pinaccess/flip.py ("instance orientation", "instance SHIFT dx
orientation"), for pnr/pa_flip.tcl (PA_FLIP=1, PA_SHIFT=1; xplace/run_xplace_flow.sh with
PA_FLIP_SCRIPT=pinaccess/dp_refine.py). Instead of choosing each cell's mirror and shift on its
own, it optimizes all of them together, row by row, keeping the cells legal and in their order.

For every single-row movable cell i (placed, not a clock-tree cell) at x_i in a row segment, every
shift s in {-S..S} sites and both x-mirrors o of its row orientation get a cost

    c_i(s, o) = w_pa R_i(x_i + s, o)          blocked pins next to the PDN (src/pin_access.py rule)
              + w_band B_i(x_i + s)           1 for a critical cell overlapping the band of
                                              --band sites around a strap of --band_layer
              + w_wl dW_i(s, o) / site        HPWL change of its nets, the other cells kept
              + w_disp |s|,

and every two neighbours j-1, j of a chain, at gap delta = g_j + s_j - s_{j-1} sites, a cost

    p_j(delta, o_{j-1}, o_j) = w_pair [delta + e_j(o_{j-1}, o_j) <= pair_d],

e_j the smallest distance (sites, at abutment) between a pin of j-1 and a pin of j whose only access
track is the same track k: e_j = min_k a_k(j-1, o_{j-1}) + b_k(j, o_j), a_k the distance from the
rightmost such pin of a cell to its right edge, b_k from its left edge to the leftmost one. Two
single-access pins on one M2 track a site apart, across the cell boundary, fail 0.7% of the time
after routing, at two sites or more about never (NS3K stripe, 6 runs, 70k neighbour pairs).

and the shifts of the cells of a chain (consecutive movable cells between fixed objects or row
ends) solve

    min sum_j min_o c_j(s_j, o)   s.t.  x_{j-1} + w_{j-1} + s_{j-1} <= x_j + s_j  (in sites),
                                        the chain inside its segment,

exactly by dynamic programming over the states (s, o):
D_j(s, o) = c_j(s, o) + min_{s' <= s + g_j, o'} [D_{j-1}(s', o') + p_j(g_j + s - s', o', o)], g_j the gap
(sites) between the cells, a (2K x 2K) min-plus product per step. All chains run at once on the GPU,
one step per position in the chain. Fixed cells, multi-row cells and clock-tree cells (CTS_*) are
obstacles; the rows come from the DEF (cut around macros), so macros need no size.
"""
import argparse
import re
import sys
import time

import torch

INF0 = 1e9

sys.path.insert(0, '.')
from src.pin_access import PinAccessModel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('model', help='ignored with --rule (e.g. "none")')
ap.add_argument('lef'); ap.add_argument('pdn_def'); ap.add_argument('place_def'); ap.add_argument('out')
ap.add_argument('--rule', default='track', choices=['block', 'track', 'track23'])
ap.add_argument('--rule_w', type=int, default=0)
ap.add_argument('--sites', type=int, default=4, help='largest shift (sites)')
ap.add_argument('--band', type=float, default=3.0, help='half width (sites) of the band around a strap centreline')
ap.add_argument('--band_layer', default='M3')
ap.add_argument('--crit_report', default='data/pin_scarcity/NS3K_5T_track_pin_scarcity_report.txt')
ap.add_argument('--crit_z', type=float, default=1.0)
ap.add_argument('--w_pa', type=float, default=20.0, help='cost of a blocked pin (sites of wirelength)')
ap.add_argument('--w_band', type=float, default=60.0, help='cost of a critical cell in a strap band')
ap.add_argument('--w_wl', type=float, default=1.0)
ap.add_argument('--w_disp', type=float, default=0.05, help='cost per site of shift')
ap.add_argument('--w_pair', type=float, default=20.0, help='cost of a neighbour pin conflict')
ap.add_argument('--pair_d', type=float, default=1.0,
                help='facing single-access pins of two neighbours on one track this close (sites) conflict')
ap.add_argument('--tracks', type=int, default=4, help='access tracks inside a row, k = 1 .. tracks')
ap.add_argument('--track_pitch', type=float, default=0.096, help='access-track pitch (um)')
ap.add_argument('--via_half', type=float, default=0.032, help='half height of the via landing pad (um)')
ap.add_argument('--max_net_deg', type=int, default=100, help='nets with more pins are left out of the HPWL term')
ap.add_argument('--site_name', default='', help='row site of the single-row cells (default: the most frequent)')
ap.add_argument('--cost', default='ours', choices=['ours', 'ispd17'],
                help='ispd17: the pin access penalty of Ding et al., ISPD 2017 (TCPAP) as the cost; R, B and the neighbour term unused')
ap.add_argument('--minl', type=float, default=1.0, help='--cost ispd17: shortest M2 segment (sites) of the different-row penalty')
ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='cpu: the same program on the CPU (runtime comparison)')
# flip.py options, accepted and ignored
ap.add_argument('--min_gain', type=float); ap.add_argument('--shift_thr', type=float)
ap.add_argument('--shift_sites', type=int); ap.add_argument('--shift_cost', type=float)
a = ap.parse_args()
t0 = time.time()
times = []


def phase(name):
    if a.device == 'cuda':
        torch.cuda.synchronize()
    times.append((name, time.time() - t0 - sum(v for _, v in times)))


pa = PinAccessModel(None, a.pdn_def, a.lef, device=a.device)
pa.rule, pa.rule_w = a.rule, a.rule_w
dev = pa.device
t = open(a.place_def).read()

# ------------------------------------------------------------------ rows (segments of the single-row site)
rows = [(m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), int(m.group(5)), int(m.group(6)))
        for m in re.finditer(r'^ROW \S+ (\S+) (-?\d+) (-?\d+) (\S+) DO (\d+) BY 1 STEP (\d+) \d+', t, re.M)]
# the single-row site: the one whose rows hold the most single-height cells (by count of rows alone it is
# ambiguous: jpeg has as many i3d_nsfet3nm_2H_VDD rows as i3d_nsfet3nm rows), ties to the shortest name
if not a.site_name:
    cy = {}
    sec = re.search(r'\nCOMPONENTS \d+ ;(.*?)END COMPONENTS', t, re.S).group(1)
    for m in re.finditer(r'- \S+ (\S+)[^;]*?\+ (?:PLACED|FIXED) \( -?\d+ (-?\d+) \)', sec):
        e = pa.cells.get(m.group(1))
        if e is not None and e[1] <= 1.5 * pa.rowh:
            cy[int(m.group(2))] = cy.get(int(m.group(2)), 0) + 1
    hold = {}
    for r in rows:
        hold.setdefault(r[0], set()).add(r[2])
site_name = a.site_name or max(sorted(hold), key=lambda s: (sum(cy.get(y, 0) for y in hold[s]), -len(s)))
rows = [r for r in rows if r[0] == site_name]
site = rows[0][5]
seg_by_y = {}
for _, x0, y, _, n, step in rows:
    seg_by_y.setdefault(y, []).append((x0, x0 + n * step))
rowh = pa.rowh

# ------------------------------------------------------------------ components
comp = re.search(r'\nCOMPONENTS \d+ ;(.*?)END COMPONENTS', t, re.S).group(1)
code = {'N': 0, 'FS': 1, 'FN': 2, 'S': 3}
inv = {v: k for k, v in code.items()}
names, types, xs, ys, ors = [], [], [], [], []
obst = {}  # row y -> [(x1, x2)]
cpos = {}  # component -> (x, y, orient, cell) for the HPWL term
for m in re.finditer(r'- (\S+) (\S+)(.*?)\+ (PLACED|FIXED|COVER) \( (-?\d+) (-?\d+) \) (\S+)', comp, re.S):
    name, cell, _, st, x, y, o = m.groups()
    x, y = int(x), int(y)
    cpos[name] = (x, y, o, cell)
    e = pa.cells.get(cell)
    if e is None:
        continue                                   # a macro: its rows are cut already
    W, H = e[0], e[1]
    movable = (st == 'PLACED' and H <= 1.5 * rowh and not name.startswith('CTS_') and o in code
               and y in seg_by_y)
    if movable:
        names.append(name); types.append(cell); xs.append(x); ys.append(y); ors.append(code[o])
    else:
        k = 0
        while y + k * rowh < y + H - 1e-6:
            obst.setdefault(int(round(y + k * rowh)), []).append((x, x + W))
            k += 1
n = len(names)
W = [pa.cells[c][0] for c in types]

# ------------------------------------------------------------------ chains of movable cells
by_y = {}
for i in range(n):
    by_y.setdefault(ys[i], []).append(i)
chains = []  # (lo, hi, [cells])
for y, segs in seg_by_y.items():
    cells = sorted(by_y.get(y, []), key=lambda i: xs[i])
    obs = sorted(obst.get(y, []))
    for lo, hi in segs:
        items = [(xs[i], 0, i) for i in cells if lo <= xs[i] < hi] + \
                [(x1, 1, (x1, x2)) for x1, x2 in obs if x2 > lo and x1 < hi]
        items.sort()
        cur_lo, cur = lo, []
        for _, kind, v in items:
            if kind == 1:
                if cur:
                    chains.append((cur_lo, min(hi, v[0]), cur))
                cur, cur_lo = [], max(cur_lo, v[1])
            else:
                cur.append(v)
        if cur:
            chains.append((cur_lo, hi, cur))
nc = len(chains)
maxl = max(len(c[2]) for c in chains)
S = a.sites
K = 2 * S + 1
shifts = torch.arange(-S, S + 1, device=dev)

phase('parse')

# ------------------------------------------------------------------ costs c_i(s, o)
xl = torch.tensor(xs, dtype=torch.float64, device=dev)
yb = torch.tensor(ys, dtype=torch.float64, device=dev)
Wt = torch.tensor(W, dtype=torch.float64, device=dev)
cur = torch.tensor(ors, device=dev)
base = cur % 2                                   # the row's orientation: N / FS
opts = (base, base + 2)                          # as in the row, x-mirrored
pa.index_cells(types)
P = pa.pins
R = torch.zeros((n, K, 2), dtype=torch.float64, device=dev)
with torch.no_grad():
    for k, s in enumerate(range(-S, S + 1)):
        for oi, opt in enumerate(opts):
            p, _ = pa._score_o(xl + s * site, yb, opt[P['cid']])
            R[:, k, oi] = torch.zeros(n, dtype=torch.float64, device=dev).index_add(0, P['cid'], p.to(torch.float64))

# strap bands for the critical types
crit = set()
for line in open(a.crit_report):
    m = re.match(r'\s*\d+\s+(\S+)\s+score=\S+\s+z=(\S+)', line)
    if m and float(m.group(2)) >= a.crit_z:
        crit.add(m.group(1))
pt = open(a.pdn_def).read()
sp = pt[pt.index('\nSPECIALNETS'):pt.index('END SPECIALNETS')]
L = re.escape(a.band_layer)
cx = [int(m.group(2)) for m in re.finditer(r'(?:ROUTED|NEW) %s (\d+) \+ SHAPE STRIPE \( (-?\d+) -?\d+ \) \( \* -?\d+ \)' % L, sp)]
cx += [(int(m.group(1)) + int(m.group(2))) // 2 for m in re.finditer(r'RECT %s \( (-?\d+) -?\d+ \) \( (-?\d+) -?\d+ \)' % L, sp)]
B = torch.zeros((n, K), dtype=torch.float64, device=dev)
iscrit = torch.tensor([c in crit for c in types], device=dev)
if cx and iscrit.any():
    cen = torch.tensor(sorted(set(cx)), dtype=torch.float64, device=dev)
    half = a.band * site
    x1 = xl.unsqueeze(1) + shifts.unsqueeze(0) * site              # (n, K)
    x2 = x1 + Wt.unsqueeze(1)
    j = torch.searchsorted(cen, (x1 + x2) / 2)
    hit = torch.zeros_like(x1, dtype=torch.bool)
    for d in (-1, 0):
        jj = (j + d).clamp(0, cen.numel() - 1)
        c = cen[jj]
        hit |= (x2 > c - half) & (x1 < c + half)
    B = (hit & iscrit.unsqueeze(1)).to(torch.float64)

# HPWL change of the cell's nets (x only: the shifts and mirrors are horizontal)
dW = torch.zeros((n, K, 2), dtype=torch.float64, device=dev)
nets = re.search(r'\nNETS \d+ ;(.*?)END NETS', t, re.S)
idx = {nm: i for i, nm in enumerate(names)}
nwl = 0
if nets:
    io = {}
    pins_sec = re.search(r'\nPINS \d+ ;(.*?)END PINS', t, re.S)
    if pins_sec:
        for m in re.finditer(r'- (\S+) \+ NET \S+(.*?)\+ (?:PLACED|FIXED) \( (-?\d+) (-?\d+) \)', pins_sec.group(1), re.S):
            io[m.group(1)] = int(m.group(3))
    poff = {}
    for c, (Wc, Hc, _, pins) in pa.cells.items():
        poff[c] = {pn: sum((r[0] + r[2]) / 2 for r in rs) / len(rs) for pn, rs in pins.items()}
    pnet, pcomp, px, pmv = [], [], [], []     # per pin: net, movable index or -1, x now, offset for N
    netid = 0
    for m in re.finditer(r'^- (\S+)(.*?);', nets.group(1), re.S | re.M):
        conns = re.findall(r'\( (\S+) (\S+) \)', m.group(2))
        if len(conns) < 2 or len(conns) > a.max_net_deg:
            continue
        ok = []
        for cn, pn in conns:
            if cn == 'PIN':
                if pn in io:
                    ok.append((-1, io[pn], 0.0))
                continue
            if cn not in cpos:
                continue
            x, y, o, cell = cpos[cn]
            off = poff.get(cell, {}).get(pn)
            if off is None:
                continue
            Wc = pa.cells[cell][0]
            xo = x + (Wc - off if o in ('FN', 'S') else off)
            ok.append((idx.get(cn, -1), xo, off))
        if len(ok) < 2 or all(i < 0 for i, _, _ in ok):
            continue
        for i, xo, off in ok:
            pnet.append(netid); pcomp.append(i); px.append(xo); pmv.append(off)
        netid += 1
    nwl = netid
    if nwl:
        pnet = torch.tensor(pnet, device=dev); pcomp = torch.tensor(pcomp, device=dev)
        px = torch.tensor(px, dtype=torch.float64, device=dev); poffN = torch.tensor(pmv, dtype=torch.float64, device=dev)
        inf = torch.tensor(float('inf'), dtype=torch.float64, device=dev)
        # pairs (net, movable cell) and the other pins of each net: top-2 min / max over pins of other cells
        key = pnet * (n + 1) + (pcomp + 1)
        ukey, pair = torch.unique(key, return_inverse=True)
        npair = ukey.numel()
        pr_net = ukey // (n + 1); pr_cell = ukey % (n + 1) - 1
        pmin = torch.full((npair,), float('inf'), dtype=torch.float64, device=dev).scatter_reduce(0, pair, px, 'amin')
        pmax = torch.full((npair,), -float('inf'), dtype=torch.float64, device=dev).scatter_reduce(0, pair, px, 'amax')
        def top2(v, sign):
            # per net the best and second best pair value (sign 1: min, -1: max), and the best pair
            vv = sign * v
            b1 = torch.full((nwl,), float('inf'), dtype=torch.float64, device=dev).scatter_reduce(0, pr_net, vv, 'amin')
            isb = vv == b1[pr_net]
            first = torch.full((nwl,), npair, device=dev).scatter_reduce(
                0, pr_net, torch.where(isb, torch.arange(npair, device=dev), torch.full_like(pr_net, npair)), 'amin')
            v2 = torch.where(torch.arange(npair, device=dev) == first[pr_net], inf, vv)
            b2 = torch.full((nwl,), float('inf'), dtype=torch.float64, device=dev).scatter_reduce(0, pr_net, v2, 'amin')
            other = torch.where(torch.arange(npair, device=dev) == first[pr_net], b2[pr_net], b1[pr_net])
            return sign * other, sign * b1
        omin, nmin = top2(pmin, 1)
        omax, nmax = top2(pmax, -1)
        span0 = nmax - nmin
        mv = pr_cell >= 0
        pc = pr_cell[mv]; pnn = pr_net[mv]
        # the cell's pins on the net: offsets for its row orientation and the mirror
        pin_mv = pcomp >= 0
        pk = pair[pin_mv]
        offN = poffN[pin_mv]
        ci = pcomp[pin_mv]
        offs = (offN, Wt[ci] - offN)
        rel = []
        for o in offs:
            lo_ = torch.full((npair,), float('inf'), dtype=torch.float64, device=dev).scatter_reduce(0, pk, o, 'amin')
            hi_ = torch.full((npair,), -float('inf'), dtype=torch.float64, device=dev).scatter_reduce(0, pk, o, 'amax')
            rel.append((lo_[mv], hi_[mv]))
        om, oM, s0 = omin[mv], omax[mv], span0[pnn]
        for oi in range(2):
            lo_, hi_ = rel[oi]
            for k, s in enumerate(range(-S, S + 1)):
                xc = xl[pc] + s * site
                nl = torch.minimum(om, xc + lo_)
                nh = torch.maximum(oM, xc + hi_)
                d = (nh - nl) - s0
                d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
                dW[:, k, oi] += torch.zeros(n, dtype=torch.float64, device=dev).index_add(0, pc, d)

disp = shifts.abs().to(torch.float64).unsqueeze(0).unsqueeze(2).expand(n, K, 2)
C = a.w_pa * R + a.w_band * B.unsqueeze(2) + a.w_wl * dW / site + a.w_disp * disp
# keep the current orientation on ties
curo = (cur >= 2).long()
C[torch.arange(n, device=dev), :, 1 - curo] += 1e-6

# ------------------------------------------------------------------ neighbour pin profiles
# per cell and x-mirror (0: as in the row, 1: mirrored), per access track k: b_k (left edge to the leftmost
# pin whose only access track is k) and a_k (its rightmost such pin to the right edge), in sites
dbu = int(re.search(r'UNITS DISTANCE MICRONS (\d+)', t).group(1))
tp, vh, NT = a.track_pitch * dbu, a.via_half * dbu, a.tracks
prof = {}
for c in set(types):
    Wc, _, _, pins = pa.cells[c]
    bl = [[INF0] * NT for _ in range(2)]; arr = [[INF0] * NT for _ in range(2)]
    for rs in pins.values():
        acc = {}
        for x1, y1, x2, y2 in rs:
            for k in range(1, NT + 1):
                if y1 + vh - 1e-3 <= k * tp <= y2 - vh + 1e-3:
                    acc.setdefault(k, []).append((x1 + x2) / 2)
        if len(acc) != 1:
            continue
        (k, px_), = acc.items()
        lo_, hi_ = min(px_), max(px_)
        for oi, (l_, h_) in enumerate(((lo_, hi_), (Wc - hi_, Wc - lo_))):
            bl[oi][k - 1] = min(bl[oi][k - 1], l_ / site)
            arr[oi][k - 1] = min(arr[oi][k - 1], (Wc - h_) / site)
    prof[c] = (bl, arr)
BL = torch.tensor([prof[c][0] for c in types], dtype=torch.float64, device=dev)   # (n, 2, NT)
AR = torch.tensor([prof[c][1] for c in types], dtype=torch.float64, device=dev)

# ------------------------------------------------------------------ --cost ispd17: the pin access penalty of Ding et al.
# (ISPD'17) in place of R, B and the neighbour term: per pin-to-pin connection AA' (the nets cut into connections between
# their pins in x order), its pin access region PAR (the pin's height, from the pin towards A', width w), the PAP
#   blockage (eq. 1): sum over A's tapping points (access tracks) of f_w(dist to the PG M2 shape on that track, or a
#                     neighbouring one, in the connection's direction) / TP_A,
#   connections (eq. 2-4): CTP_{AA'.BB'} / (TP_A TP_B) f(.) for two pins sharing access tracks whose PARs intersect
#                     (same direction: the PAR behind reaches the other pin; facing: the two widths meet), in one cell
#                     (unary, per mirror) or across two neighbours (pairwise, in the DP like p_j),
# f_w(dist) = [dist <= w] for a same-row connection, 1 / alpha/dist + beta (f(minl) = 1, f(w) = 0) for a different-row one.
# The connection directions and the widths in eq. 2-4 are those of the input placement, as in the paper.
Pt = None
if a.cost == 'ispd17':
    NKp = pa.NK
    pname_j = {}                                   # cell type -> {pin name: index in pa.ptab}
    io_xy = {}
    pins_sec = re.search(r'\nPINS \d+ ;(.*?)END PINS', t, re.S)
    if pins_sec:
        for m in re.finditer(r'- (\S+) \+ NET \S+(.*?)\+ (?:PLACED|FIXED) \( (-?\d+) (-?\d+) \)', pins_sec.group(1), re.S):
            io_xy[m.group(1)] = (int(m.group(3)), int(m.group(4)))
    ent = []                                       # (cell, pin index, direction, partner x, same row)
    for m in re.finditer(r'^- (\S+)(.*?);', nets.group(1), re.S | re.M) if nets else ():
        conns = re.findall(r'\( (\S+) (\S+) \)', m.group(2))
        if len(conns) < 2 or len(conns) > a.max_net_deg:
            continue
        pts = []
        for cn, pn in conns:
            if cn == 'PIN':
                if pn in io_xy:
                    pts.append((io_xy[pn][0], io_xy[pn][1], -1, -1))
                continue
            if cn not in cpos:
                continue
            x, y, o, cell = cpos[cn]
            e = pa.ptab.get(cell)
            if e is None or o not in e:
                continue
            if cell not in pname_j:
                pname_j[cell] = {e_[4]: jj for jj, e_ in enumerate(e['N'])}
            j = pname_j[cell].get(pn)
            if j is None:
                continue
            pts.append((x + e[o][j][0], y, idx.get(cn, -1), j))
        pts.sort()
        for (xu, yu, iu, ju), (xv, yv, iv, jv) in zip(pts, pts[1:]):
            if iu >= 0:
                ent.append((iu, ju, 1, xv, yu == yv))
            if iv >= 0:
                ent.append((iv, jv, -1, xu, yu == yv))
    # per entry: tapping points (access levels 1..4 of the row), offsets for the two mirrors, the pin's row
    rows_e = []
    for i, j, d, px_, same in ent:
        e = pa.ptab[types[i]]
        ob = inv[ors[i] % 2]; om = inv[ors[i] % 2 + 2]
        ft = e[ob][j][3]
        mask = sum(1 << (k - 1) for k in range(1, 5) if ft[k] > 0.5)
        if mask:
            rows_e.append((i, j, d, px_, float(same), mask, e[ob][j][0], e[om][j][0], e[ob][j][2]))
    ne = len(rows_e)
    pop = torch.tensor([bin(v).count('1') for v in range(16)], dtype=torch.float64, device=dev)
    dd_ = dict(dtype=torch.float64, device=dev)
    if ne:
        cols = list(zip(*rows_e))
        e_i = torch.tensor(cols[0], device=dev); e_j = torch.tensor(cols[1], device=dev)
        e_d = torch.tensor(cols[2], **dd_); e_px = torch.tensor(cols[3], **dd_); e_same = torch.tensor(cols[4], **dd_)
        e_mask = torch.tensor(cols[5], device=dev); e_off = torch.stack([torch.tensor(cols[6], **dd_), torch.tensor(cols[7], **dd_)], 1)
        e_roff = torch.tensor(cols[8], device=dev)
    else:
        e_i = torch.zeros(0, dtype=torch.long, device=dev)
    minl = a.minl * site

    def fpen(dist, w, same):
        hit = (dist <= w) | (dist <= 0)
        alpha = 1.0 / (1.0 / minl - 1.0 / w.clamp(min=minl + 1e-3))
        dr = torch.where((dist <= minl) | (w <= minl), torch.ones_like(dist), alpha / dist.clamp(min=1e-6) - alpha / w.clamp(min=minl + 1e-3))
        return torch.where(hit, torch.where(same > 0.5, torch.ones_like(dist), dr.clamp(0, 1)), torch.zeros_like(dist))

    gxm, cumm, insm, keym = pa.cov['M2']
    INFd = torch.tensor(float('inf'), **dd_)

    def ndist(key, x, d):
        """distance from x to the nearest PG M2 coverage at (row, level) key in direction d (0 inside)"""
        if gxm.numel() == 0:
            return torch.full_like(x, float('inf'))
        q = key.to(torch.float64) * pa.BIG + x
        ix = (torch.searchsorted(gxm, q, right=True) - 1).clamp(min=0)
        same = keym[ix] == key
        nx = (ix + 1).clamp(max=gxm.numel() - 1)
        dr = torch.where(same & (keym[nx] == key) & (insm[nx] > 0.5) & (nx > ix), gxm[nx] - q, INFd)
        dl = torch.where(same & (insm[ix] < 0.5) & (cumm[ix] > 0), q - gxm[ix], INFd)
        return torch.where(same & (insm[ix] > 0.5), torch.zeros_like(x), torch.where(d > 0, dr, dl))

    # unary blockage PAP per (cell, shift, mirror)
    U = torch.zeros((n, K, 2), **dd_)
    if ne:
        prow = torch.round((yb[e_i] - pa.y0) / pa.rowh).long() + e_roff
        tp = pop[e_mask]
        for oi in range(2):
            for k, s in enumerate(range(-S, S + 1)):
                xa = xl[e_i] + s * site + e_off[:, oi]
                w = (e_d * (e_px - xa)).clamp(min=0)
                acc = torch.zeros(ne, **dd_)
                for lv in range(1, 5):
                    has = ((e_mask >> (lv - 1)) & 1) > 0
                    dmin = torch.full((ne,), float('inf'), **dd_)
                    for l2 in (lv - 1, lv, lv + 1):
                        dmin = torch.minimum(dmin, ndist(prow * NKp + l2, xa, e_d))
                    acc += torch.where(has, fpen(dmin, w, e_same), torch.zeros_like(acc))
                U[:, k, oi] += torch.zeros(n, **dd_).index_add(0, e_i, acc / tp)
        # padded per-cell entry table for the connection terms (widths at the input placement)
        order = torch.argsort(e_i, stable=True)
        cnt = torch.bincount(e_i, minlength=n)
        start = torch.cumsum(cnt, 0) - cnt
        Emax = int(cnt.max())
        slot = torch.arange(ne, device=dev) - start[e_i[order]]
        T_ = lambda v, fill: torch.full((n, Emax), fill, dtype=v.dtype, device=dev).index_put_((e_i[order], slot), v[order])
        xa0 = xl[e_i] + torch.where(curo[e_i] == 1, e_off[:, 1], e_off[:, 0])
        Ev = T_(torch.ones(ne, dtype=torch.bool, device=dev), False)
        Ed, Ew, Es = T_(e_d, 0.0), T_((e_d * (e_px - xa0)).clamp(min=0), 0.0), T_(e_same, 0.0)
        Em, Ej = T_(e_mask, 0), T_(e_j, -1)
        Eo0, Eo1 = T_(e_off[:, 0], 0.0), T_(e_off[:, 1], 0.0)
        Etp = torch.where(Ev, pop[Em], torch.ones_like(Ew))

        def ppen(dA, dB, wA, wB, sA, sB, dist):
            """eq. 2-4 for A left of B at distance dist: PAP on A's plus on B's connection"""
            br, bl, fc = (dA > 0) & (dB > 0), (dA < 0) & (dB < 0), (dA > 0) & (dB < 0)
            ws = (wA + wB).clamp(min=1e-6)
            z = torch.zeros_like(dist)
            pA = torch.where(br, fpen(dist, wA, sA), torch.where(bl, fpen(dist, wB, sA), torch.where(fc, fpen(wA / ws * dist, wA, sA), z)))
            pB = torch.where(br, fpen(dist, wA, sB), torch.where(bl, fpen(dist, wB, sB), torch.where(fc, fpen(wB / ws * dist, wB, sB), z)))
            return pA + pB

        # in one cell: per mirror, every two entries of different pins
        cin = max(1, int(4e6 // (Emax * Emax)))
        for c0 in range(0, n, cin):
            c1 = min(n, c0 + cin)
            v = Ev[c0:c1].unsqueeze(2) & Ev[c0:c1].unsqueeze(1) & (Ej[c0:c1].unsqueeze(2) != Ej[c0:c1].unsqueeze(1))
            v &= torch.triu(torch.ones(Emax, Emax, dtype=torch.bool, device=dev), 1)
            coef = pop[Em[c0:c1].unsqueeze(2) & Em[c0:c1].unsqueeze(1)] / (Etp[c0:c1].unsqueeze(2) * Etp[c0:c1].unsqueeze(1))
            coef = torch.where(v, coef, torch.zeros_like(coef))
            g = lambda X: (X[c0:c1].unsqueeze(2), X[c0:c1].unsqueeze(1))
            (d1, d2), (w1, w2), (s1, s2) = g(Ed), g(Ew), g(Es)
            for oi, Eo in enumerate((Eo0, Eo1)):
                x1, x2 = g(Eo)
                lt = x1 <= x2
                dist = (x2 - x1).abs()
                pen = torch.where(lt, ppen(d1, d2, w1, w2, s1, s2, dist), ppen(d2, d1, w2, w1, s2, s1, dist))
                U[c0:c1, :, oi] += (coef * pen).sum((1, 2)).unsqueeze(1)
        # across neighbours: Pt[i, t + 2S, o', o] for cell i, its left neighbour at gap g + t sites (t = s_i - s_left)
        left = torch.full((n,), -1, dtype=torch.long, device=dev); gapc = torch.zeros(n, **dd_)
        for lo, hi, cells in chains:
            for j in range(1, len(cells)):
                left[cells[j]] = cells[j - 1]
                gapc[cells[j]] = round((xs[cells[j]] - xs[cells[j - 1]] - W[cells[j - 1]]) / site)
        Pt = torch.zeros((n, 4 * S + 1, 2, 2), **dd_)
        hasl = torch.nonzero(left >= 0).squeeze(1)
        cp = max(1, int(2e6 // (Emax * Emax)))
        for c0 in range(0, hasl.numel(), cp):
            ib = hasl[c0:c0 + cp]; ia = left[ib]
            A_ = lambda X: X[ia].unsqueeze(2); B_ = lambda X: X[ib].unsqueeze(1)
            v = A_(Ev) & B_(Ev)
            coef = pop[A_(Em) & B_(Em)] / (A_(Etp) * B_(Etp))
            coef = torch.where(v, coef, torch.zeros_like(coef)).unsqueeze(-1).unsqueeze(-1)       # (m, E, E, 1, 1)
            offA = torch.stack([Eo0[ia], Eo1[ia]], -1).unsqueeze(2).unsqueeze(-1)                 # (m, E, 1, 2', 1)
            offB = torch.stack([Eo0[ib], Eo1[ib]], -1).unsqueeze(1).unsqueeze(-2)                 # (m, 1, E, 1, 2)
            base_d = Wt[ia].view(-1, 1, 1, 1, 1) - offA + offB
            ex = lambda X: X.unsqueeze(-1).unsqueeze(-1)
            dA, dB, wA, wB, sA, sB = ex(A_(Ed)), ex(B_(Ed)), ex(A_(Ew)), ex(B_(Ew)), ex(A_(Es)), ex(B_(Es))
            for tt in range(-2 * S, 2 * S + 1):
                dist = ((gapc[ib] + tt) * site).view(-1, 1, 1, 1, 1) + base_d
                Pt[ib, tt + 2 * S] = (coef * ppen(dA, dB, wA, wB, sA, sB, dist.clamp(min=0))).sum((1, 2))
    C = a.w_pa * U + a.w_wl * dW / site + a.w_disp * disp
    C[torch.arange(n, device=dev), :, 1 - curo] += 1e-6

phase('cost')

# ------------------------------------------------------------------ the dynamic program over all chains
INF = float('inf')
cidx = torch.full((nc, maxl), -1, dtype=torch.long, device=dev)
gap = torch.zeros((nc, maxl), dtype=torch.float64, device=dev)
lens = torch.tensor([len(c[2]) for c in chains], device=dev)
lo_gap = torch.zeros(nc, dtype=torch.float64, device=dev); hi_gap = torch.zeros(nc, dtype=torch.float64, device=dev)
cl, cj, cv, gv = [], [], [], []
for q, (lo, hi, cells) in enumerate(chains):
    for j, i in enumerate(cells):
        cl.append(q); cj.append(j); cv.append(i)
        gv.append((xs[i] - (xs[cells[j - 1]] + W[cells[j - 1]])) / site if j else 0.0)
    lo_gap[q] = (xs[cells[0]] - lo) / site
    hi_gap[q] = (hi - xs[cells[-1]] - W[cells[-1]]) / site
cl = torch.tensor(cl, device=dev); cj = torch.tensor(cj, device=dev)
cidx[cl, cj] = torch.tensor(cv, device=dev)
gap[cl, cj] = torch.tensor(gv, dtype=torch.float64, device=dev)
sv = shifts.to(torch.float64)
K2 = 2 * K
D = torch.full((nc, K, 2), INF, dtype=torch.float64, device=dev)
ptr = torch.zeros((maxl, nc, K, 2), dtype=torch.int16, device=dev)
ar = torch.arange(K, device=dev)
ident = torch.arange(K2, device=dev).view(1, K, 2).expand(nc, K, 2).to(torch.int16)
dss = ar.view(1, K) - ar.view(K, 1)                  # (s', s): s - s'
chunk = max(1, int(4e7 // (K2 * K2)))
for j in range(maxl):
    real = cidx[:, j] >= 0
    i_ = cidx[:, j].clamp(min=0)
    cj_cost = torch.where(real.view(nc, 1, 1), C[i_], torch.zeros((nc, K, 2), dtype=torch.float64, device=dev))
    if j == 0:
        Dn = torch.where((sv.unsqueeze(0) >= -lo_gap.unsqueeze(1)).unsqueeze(2), cj_cost, torch.full_like(cj_cost, INF))
        p = ident
    else:
        ip = cidx[:, j - 1].clamp(min=0)
        Dn = torch.empty_like(D); p = torch.empty_like(ident)
        for q0 in range(0, nc, chunk):
            q1 = min(nc, q0 + chunk)
            delta = gap[q0:q1, j].round().view(-1, 1, 1) + dss.unsqueeze(0)                    # (m, K', K)
            e = (AR[ip[q0:q1]].unsqueeze(2) + BL[i_[q0:q1]].unsqueeze(1)).amin(-1)             # (m, 2', 2)
            if Pt is None:
                conf = (delta.view(-1, K, 1, K, 1) + e.view(-1, 1, 2, 1, 2)) <= a.pair_d + 1e-6     # (m, K', 2', K, 2)
                T = D[q0:q1].view(-1, K, 2, 1, 1) + a.w_pair * conf
            else:                                                                              # ispd17: Pt[t + 2S]
                pen = Pt[i_[q0:q1]][:, dss + 2 * S]                                             # (m, K', K, 2', 2)
                T = D[q0:q1].view(-1, K, 2, 1, 1) + a.w_pa * pen.permute(0, 1, 3, 2, 4)
            T = torch.where((delta >= 0).view(-1, K, 1, K, 1), T, torch.full_like(T, INF))
            Mv, Mi = T.reshape(q1 - q0, K2, K, 2).min(dim=1)
            Dn[q0:q1] = cj_cost[q0:q1] + Mv
            p[q0:q1] = Mi.to(torch.int16)
        Dn = torch.where(real.view(nc, 1, 1), Dn, D)                                     # padding: carry
        p = torch.where(real.view(nc, 1, 1), p, ident)
    last = lens - 1 == j
    Dn = torch.where((last.unsqueeze(1) & (sv.unsqueeze(0) > hi_gap.unsqueeze(1))).unsqueeze(2), torch.full_like(Dn, INF), Dn)
    D, ptr[j] = Dn, p
state = D.view(nc, K2).argmin(dim=1)
feasible = torch.isfinite(D.view(nc, K2).min(dim=1).values)
sel = torch.zeros((nc, maxl), dtype=torch.long, device=dev)
for j in range(maxl - 1, -1, -1):
    sel[:, j] = state
    state = ptr[j].view(nc, K2).gather(1, state.unsqueeze(1)).squeeze(1).long()
real = cidx >= 0
ci = cidx[real]
s_of = torch.full((n,), S, dtype=torch.long, device=dev); o_of = curo.clone()
sk = sel[real]; okc = feasible.unsqueeze(1).expand(nc, maxl)[real]                # infeasible chain: unchanged
s_of[ci] = torch.where(okc, sk // 2, torch.full_like(sk, S))
o_of[ci] = torch.where(okc, sk % 2, curo[ci])
newo = torch.where(o_of == 1, base + 2, base)
dx = s_of - S

# neighbour pin conflicts before / after
pj = cj >= 1
qa, ja = cl[pj], cj[pj]
i_b, i_a = cidx[qa, ja], cidx[qa, ja - 1]
g0 = gap[qa, ja].round()
def nconf(sa, oa, sb, ob):
    e = (AR[i_a, oa] + BL[i_b, ob]).amin(-1)
    return int(((g0 + (sb - sa) + e) <= a.pair_d + 1e-6).sum())
pair_before = nconf(S, curo[i_a], S, curo[i_b])
pair_after = nconf(s_of[i_a], o_of[i_a], s_of[i_b], o_of[i_b])
if Pt is not None:
    def tcpap(s_, o_):
        v = float(U[torch.arange(n, device=dev), s_, o_].sum())
        if hasl.numel():
            la = left[hasl]
            v += float(Pt[hasl, s_[hasl] - s_[la] + 2 * S, o_[la], o_[hasl]].sum())
        return v
    print('pa dp ispd17: %d connection ends, TCPAP %.1f -> %.1f' % (ne, tcpap(torch.full((n,), S, dtype=torch.long, device=dev), curo),
                                                                 tcpap(s_of, o_of)))

phase('dp')

# ------------------------------------------------------------------ output and summary
dxl, newol, curl = dx.tolist(), newo.tolist(), cur.tolist()
moved = flipped = 0
with open(a.out, 'w') as f:
    for i in range(n):
        if dxl[i] != 0:
            f.write('%s SHIFT %d %s\n' % (names[i], dxl[i], inv[newol[i]])); moved += 1
        elif newol[i] != curl[i]:
            f.write('%s %s\n' % (names[i], inv[newol[i]])); flipped += 1
phase('output')
ar_n = torch.arange(n, device=dev)
r_before = float(R[ar_n, S, curo].sum()); r_after = float(R[ar_n, s_of, o_of].sum())
b_before = float(B[:, S].sum()); b_after = float(B[ar_n, s_of].sum())
w_after = float(dW[ar_n, s_of, o_of].sum()) / site
print('pa dp: %d cells in %d chains (max %d), %d flipped, %d shifted (mean |dx| %.2f sites); blocked pins %.0f -> %.0f, '
      'critical cells in a band %.0f -> %.0f, neighbour pin conflicts %d -> %d, HPWL %+.0f sites (%d nets), %.1f s' % (
          n, nc, maxl, flipped, moved, float(dx.abs()[dx != 0].float().mean()) if moved else 0.0,
          r_before, r_after, b_before, b_after, pair_before, pair_after, w_after, nwl, time.time() - t0))
print('pa dp time (%s): %s' % (a.device, ', '.join('%s %.2f s' % kv for kv in times)))
print('pa flip: %d cells, %d flipped; rule %s/%d blocked pins %.1f -> %.1f' % (n, flipped + moved, a.rule, a.rule_w, r_before, r_after))

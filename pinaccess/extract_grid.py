#!/usr/bin/env python3
"""Per-pin dataset with per-layer occupancy grids (PDN-agnostic), from one routed run.

    extract_grid.py <run_dir> <cells_lef> <out_prefix> [--half 6]

Writes <out_prefix>.csv.gz (ids, pin-access features, label) and <out_prefix>.npy
(float16 grids, one row per CSV row). No PDN type is given: the PDN is described
only by what it occupies near the pin.

Grid window: columns at x = pin centre + j * 176 dbu (half a site = one M3 pitch),
j = -half..half; levels k = 0..5 at y = row bottom + k * 192 (the M2/M4 tracks of
the pin's row; k = 0 / 5 are the bottom / top rail tracks). Each cell holds the
coverage (0..1, overlap length / bin width) of a 176-dbu bin centred on the
column, at that level, by:
  M2, M3, M4   VDD/VSS shapes on that layer (stripe wires and the pads of the rail
               via stacks: VIA12/VIA23 -> M2, VIA23/VIA34 -> M3, VIA34/VIA45 -> M4)
  PIN          access points of the other cells' signal pins: each pin's x extent
               on the tracks a via can land on it
Grid order in the .npy row: M2, M3, M4, PIN, each levels x columns (row-major).
A pin is kept when any PG shape lies in its M2/M3/M4 window.
"""
import argparse, bisect, collections, gzip, os, re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('lef'); ap.add_argument('out')
ap.add_argument('--half', type=int, default=6)
a = ap.parse_args()
H = a.half; NC = 2 * H + 1; NK = 6
BIN = 176.0; SITE = 352.0

# ---------------------------------------------------------------- cell LEF
lef = open(a.lef).read()
cells = {}
for m in re.finditer(r'^MACRO (\S+)(.*?)^END \1', lef, re.S | re.M):
    name, body = m.groups()
    W, Hc = map(float, re.search(r'SIZE (\S+) BY (\S+)', body).groups())
    pins = {}
    for p in re.finditer(r'PIN (\S+)(.*?)END \1\b', body, re.S):
        if re.search(r'USE (POWER|GROUND)', p.group(2)):
            continue
        rects = [tuple(map(float, r.split())) for r in re.findall(r'RECT ([-\d. ]+) ;', p.group(2))]
        for poly in re.findall(r'POLYGON ([-\d. ]+) ;', p.group(2)):
            v = list(map(float, poly.split()))
            rects.append((min(v[0::2]), min(v[1::2]), max(v[0::2]), max(v[1::2])))
        if rects:
            pins[p.group(1)] = rects
    cells[name] = (W, Hc, pins)

# ---------------------------------------------------------------- DEF
d = [f for f in os.listdir(a.run) if f.endswith('.def') and not f.endswith('.floorplan.def')][0]
t = open(os.path.join(a.run, d)).read()
dbu = int(re.search(r'UNITS DISTANCE MICRONS (\d+)', t).group(1))
m = re.search(r'TRACKS Y (-?\d+) DO \d+ STEP (\d+) LAYER M2 ;', t)
TY0, TP = int(m.group(1)), int(m.group(2))
via_half = 0.032 * dbu
unesc = lambda s: s.replace('\\', '')

sp = re.search(r'\nSPECIALNETS \d+ ;(.*?)END SPECIALNETS', t, re.S).group(1)
rails = set()
wires = collections.defaultdict(list)     # layer -> [(x1, x2, y1, y2)] vertical PG wires
vias = []                                 # (x, y, via name)
for w in re.finditer(r'(?:ROUTED|NEW) (M\d+) (\d+) \+ SHAPE (\w+) \( (-?\d+) (-?\d+) \)(?: \( (\S+) (\S+) \))?\s*(\S+)?', sp):
    lay, wid, shape, x1, y1, x2, y2, via = w.groups()
    wid, x1, y1 = int(wid), int(x1), int(y1)
    if shape == 'FOLLOWPIN' and lay == 'M0':
        rails.add(y1)
    elif wid > 0 and x2 == '*':                       # vertical wire
        yy = int(y2)
        wires[lay].append((x1 - wid // 2, x1 + wid // 2, min(y1, yy), max(y1, yy)))
    elif wid > 0 and y2 == '*':                       # horizontal wire (non-rail)
        xx = int(x2)
        if shape != 'FOLLOWPIN':
            wires[lay].append((min(x1, xx), max(x1, xx), y1 - wid // 2, y1 + wid // 2))
    elif wid == 0 and via and re.match(r'VIA\d\d', via):
        vias.append((x1, y1, via))
rails = sorted(rails)
rail_hh = 48                                      # half height of an M0 rail (0.012 um 1x)
# pad half width: exact crossover = half width of the vertical PG wire at that x
strap_hw = {}
for lay in wires:
    for x1, x2, y1, y2 in wires[lay]:
        if y2 - y1 > x2 - x1:
            strap_hw[(x1 + x2) // 2] = (x2 - x1) / 2
shapes = collections.defaultdict(list)            # layer -> [(x1, x2, y1, y2)]
for lay in ('M2', 'M3', 'M4'):
    shapes[lay] += wires.get(lay, [])
VIA_LAYERS = {'VIA12': ('M2',), 'VIA23': ('M2', 'M3'), 'VIA34': ('M3', 'M4'), 'VIA45': ('M4',)}
for x, y, v in vias:
    key = v[:5]
    if key not in VIA_LAYERS:
        continue
    hw = strap_hw.get(x, 192)
    for lay in VIA_LAYERS[key]:
        shapes[lay].append((x - hw, x + hw, y - rail_hh, y + rail_hh))
# index: layer -> sorted by x1, with the max width for the bisect window
index = {}
for lay, lst in shapes.items():
    lst.sort()
    index[lay] = ([s[0] for s in lst], lst, max((s[1] - s[0] for s in lst), default=0))

def row_of(y):
    i = bisect.bisect_right(rails, y) - 1
    if i < 0 or i + 1 >= len(rails):
        return None
    return rails[i], rails[i + 1]

def cover(x1, x2, xc):
    """coverage of the NC bins around xc by [x1, x2]"""
    out = []
    for j in range(-H, H + 1):
        b1 = xc + j * BIN - BIN / 2; b2 = b1 + BIN
        o = min(x2, b2) - max(x1, b1)
        out.append(o / BIN if o > 0 else 0.0)
    return out

def pg_grid(lay, xc, rb):
    g = np.zeros((NK, NC), dtype=np.float32)
    if lay not in index:
        return g
    xs, lst, wmax = index[lay]
    lo = xc - H * BIN - BIN / 2; hi = xc + H * BIN + BIN / 2
    i = bisect.bisect_left(xs, lo - wmax)
    while i < len(lst) and lst[i][0] <= hi:
        x1, x2, y1, y2 = lst[i]; i += 1
        if x2 < lo:
            continue
        for k in range(NK):
            y = rb + k * TP
            if y1 <= y <= y2:
                g[k] = np.maximum(g[k], cover(x1, x2, xc))
    return g

# ---------------------------------------------------------------- placed pins
comp = re.search(r'\nCOMPONENTS \d+ ;(.*?)END COMPONENTS', t, re.S).group(1)
inst = {}
for c in re.finditer(r'- (\S+) (\S+) .*?\+ (?:PLACED|FIXED) \( (-?\d+) (-?\d+) \) (\S+)', comp, re.S):
    if c.group(2) in cells:
        inst[unesc(c.group(1))] = (c.group(2), int(c.group(3)), int(c.group(4)), c.group(5))

def place_rect(cell, ox, oy, o, r):
    W, Hc, _ = cells[cell]
    x1, y1, x2, y2 = r
    if o in ('S', 'FN'):
        x1, x2 = W - x2, W - x1
    if o in ('S', 'FS'):
        y1, y2 = Hc - y2, Hc - y1
    return (ox + x1 * dbu, oy + y1 * dbu, ox + x2 * dbu, oy + y2 * dbu)

pinrec = {}
row_pins = collections.defaultdict(list)          # row bottom -> [(x1, x2, ks, inst)]
for iname, (cell, ox, oy, o) in inst.items():
    for pname, rects in cells[cell][2].items():
        prs = [place_rect(cell, ox, oy, o, r) for r in rects]
        bx1 = min(r[0] for r in prs); bx2 = max(r[2] for r in prs)
        by1 = min(r[1] for r in prs); by2 = max(r[3] for r in prs)
        acc = set()
        for r in prs:
            k0 = int((r[1] + via_half - TY0) // TP)
            for k in range(k0, k0 + 12):
                ty = TY0 + k * TP
                if ty > r[3] - via_half:
                    break
                if ty >= r[1] + via_half:
                    acc.add(ty)
        rw = row_of((by1 + by2) / 2)
        if rw is None:
            continue
        rb, rt = rw
        ks = set()
        for ty in acc:
            r2 = row_of(ty)
            k = int(round((ty - rb) / TP))
            if r2 is not None and r2[0] == rb and 0 <= k < NK:   # (rails can be missing: a "row" can span two)
                ks.add(k)
        pinrec[(iname, pname)] = dict(inst=iname, cell=cell, pin=pname, x1=bx1, x2=bx2,
                                      xc=(bx1 + bx2) / 2, rb=rb, rt=rt, n_acc=len(acc), ks=ks)
        row_pins[rb].append((bx1, bx2, frozenset(ks), iname))
for rb in row_pins:
    row_pins[rb].sort()

# ---------------------------------------------------------------- labels (as extract.py)
nets = re.search(r'\nNETS \d+ ;(.*?)END NETS', t, re.S).group(1)
net_pins = {}
for n in re.finditer(r'^- (\S+)(.*?)(?=^- |\Z)', nets, re.S | re.M):
    body = n.group(2).split('+ ROUTED')[0].split('+ USE')[0]
    net_pins[unesc(n.group(1))] = [(unesc(i), p) for i, p in re.findall(r'\( (\S+) (\S+) \)', body)
                                   if i != 'PIN' and not re.fullmatch(r'-?\d+|\*', i)]
drc = open(os.path.join(a.run, 'rpt', 'drc.rpt')).read()
pos = set(); nmark = 0
for mk in re.finditer(r'^\w+: \( [^)]* \) (.*?)\s+\( \S+ \)\nBounds : \( (\S+), (\S+) \) \( (\S+), (\S+) \)', drc, re.M):
    nmark += 1
    mx = (float(mk.group(2)) + float(mk.group(4))) / 2 * dbu
    my = (float(mk.group(3)) + float(mk.group(5))) / 2 * dbu
    for net in re.findall(r'Regular Wire of Net (\S+)', mk.group(1)):
        best = None
        for ip in net_pins.get(unesc(net), []):
            r = pinrec.get(ip)
            if r is None or not (r['rb'] <= my <= r['rt']):
                continue
            dx = max(abs(mx - r['xc']) - (r['x2'] - r['x1']) / 2, 0)
            if dx <= SITE and (best is None or dx < best[0]):
                best = (dx, ip)
        if best:
            pos.add(best[1])

# ---------------------------------------------------------------- rows
runid = '/'.join(a.run.rstrip('/').split('/')[-3:])
cols = ['run', 'inst', 'cell', 'pin', 'n_acc', 'acc_k1', 'acc_k2', 'acc_k3', 'acc_k4', 'pin_w', 'y']
grids = []; kept = kpos = 0
with gzip.open(a.out + '.csv.gz', 'wt') as fo:
    fo.write(','.join(cols) + '\n')
    for key, r in pinrec.items():
        xc, rb = r['xc'], r['rb']
        g2 = pg_grid('M2', xc, rb); g3 = pg_grid('M3', xc, rb); g4 = pg_grid('M4', xc, rb)
        if g2.sum() + g3.sum() + g4.sum() == 0:
            continue
        gp = np.zeros((NK, NC), dtype=np.float32)
        rp = row_pins[rb]
        lo = xc - H * BIN - BIN / 2; hi = xc + H * BIN + BIN / 2
        i = bisect.bisect_left(rp, (lo - 4 * SITE,))
        while i < len(rp) and rp[i][0] <= hi:
            x1, x2, ks, iname = rp[i]; i += 1
            if iname == r['inst'] or x2 < lo:
                continue
            cv = cover(x1, x2, xc)
            for k in ks:
                gp[k] += cv
        grids.append(np.concatenate([g2.ravel(), g3.ravel(), g4.ravel(), gp.ravel()]).astype(np.float16))
        ks = r['ks']; y = 1 if key in pos else 0
        kept += 1; kpos += y
        fo.write(','.join(map(str, [runid, r['inst'], r['cell'], r['pin'], r['n_acc'],
                                    int(1 in ks), int(2 in ks), int(3 in ks), int(4 in ks),
                                    round((r['x2'] - r['x1']) / SITE, 3), y])) + '\n')
np.save(a.out + '.npy', np.stack(grids) if grids else np.zeros((0, 4 * NK * NC), np.float16))
print(f'{runid}: kept {kept} pins near PG ({kpos} of {len(pos)} positive pins), markers {nmark}')

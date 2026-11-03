#!/usr/bin/env python3
"""Per-pin pin-access dataset from one routed Innovus run (DEF + verify_drc report).

    extract.py <run_dir> <cells_lef> <out.csv.gz> [--pdn stripe|pillar] [--window 4]

One row per signal pin of a standard cell whose row has a PG pad (rail via stack
of a stripe/strap) or an M3 stripe within --window sites of the pin. Columns:

  ids      run, inst, cell, pin
  access   n_acc            M2 tracks that can land a via on the pin (pad fully on it)
           acc_k1..acc_k4   which of the 4 tracks inside the row (k = 1 next to the
                            bottom rail ... 4 next to the top rail), after orientation
           n_edge, n_mid    accessible tracks next to a rail (k 1/4) / in the middle (2/3)
           pin_w            pin width (sites)
  pdn      gap_bot, gap_top gap (sites) between the pin and the nearest PG pad on the
                            row's bottom / top rail (< 0: overlap in x; capped at window+1)
           gap_edge         gap to the pad on the rail next to an accessible edge track
                            (window+1 if the pin has no edge track)
           gap_m3           gap to the nearest M3 PG stripe crossing the row (stripe PDN)
           pillar           1 for the pillar PDN (PG on M5, only pads on M1..M4)
  context  nb_pins          signal pins of other cells within +-2 sites in the row
           nb_fragile       of those, pins with a single accessible track
  label    y                1 if a verify_drc marker naming the pin's net lies on it
                            (marker centre within 1 site in x and inside the row)

Coordinates are DEF database units (the 4x layout); one site = 352 dbu (0.044 um 1x).
"""
import argparse, bisect, collections, gzip, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('lef'); ap.add_argument('out')
ap.add_argument('--pdn', choices=['stripe', 'pillar'], required=True)
ap.add_argument('--window', type=float, default=4.0)
a = ap.parse_args()

# ---------------------------------------------------------------- cell LEF (4x um)
lef = open(a.lef).read()
cells = {}
for m in re.finditer(r'^MACRO (\S+)(.*?)^END \1', lef, re.S | re.M):
    name, body = m.groups()
    W, H = map(float, re.search(r'SIZE (\S+) BY (\S+)', body).groups())
    pins = {}
    for p in re.finditer(r'PIN (\S+)(.*?)END \1\b', body, re.S):
        if re.search(r'USE (POWER|GROUND)', p.group(2)):
            continue
        rects = []
        for r in re.findall(r'RECT ([-\d. ]+) ;', p.group(2)):
            rects.append(tuple(map(float, r.split())))
        for poly in re.findall(r'POLYGON ([-\d. ]+) ;', p.group(2)):
            v = list(map(float, poly.split()))
            xs, ys = v[0::2], v[1::2]
            rects.append((min(xs), min(ys), max(xs), max(ys)))
        if rects:
            pins[p.group(1)] = rects
    cells[name] = (W, H, pins)

# ---------------------------------------------------------------- DEF
d = [f for f in os.listdir(a.run) if f.endswith('.def') and not f.endswith('.floorplan.def')][0]
t = open(os.path.join(a.run, d)).read()
dbu = int(re.search(r'UNITS DISTANCE MICRONS (\d+)', t).group(1))
SITE = 352.0
m = re.search(r'TRACKS Y (-?\d+) DO \d+ STEP (\d+) LAYER M2 ;', t)
TY0, TP = int(m.group(1)), int(m.group(2))
via_half = 0.032 * dbu        # M1 landing pad half height (VIA12: 0.01 cut + 0.003 enclosure, 1x -> 4x)
unesc = lambda s: s.replace('\\', '')

sp = re.search(r'\nSPECIALNETS \d+ ;(.*?)END SPECIALNETS', t, re.S).group(1)
rails = set()
straps = []                   # (x1, x2) of vertical PG stripes/straps (M3 or M5)
m3 = []                       # (x1, x2, y1, y2) of M3 PG stripes
pads = collections.defaultdict(list)   # rail y -> [(xc, half width)] of the VIA12 pads on it
for w in re.finditer(r'(?:ROUTED|NEW) (M\d+) (\d+) \+ SHAPE (\w+) \( (-?\d+) (-?\d+) \)(?: \( (\S+) (\S+) \))?\s*(\S+)?', sp):
    lay, wid, shape, x1, y1, x2, y2, via = w.groups()
    wid, x1, y1 = int(wid), int(x1), int(y1)
    if shape == 'FOLLOWPIN' and lay == 'M0':
        rails.add(y1)
    elif shape == 'STRIPE' and wid > 0 and x2 == '*':
        yy = int(y2)
        straps.append((x1 - wid // 2, x1 + wid // 2, x1))
        if lay == 'M3':
            m3.append((x1 - wid // 2, x1 + wid // 2, min(y1, yy), max(y1, yy)))
    elif shape == 'STRIPE' and wid == 0 and via and via.startswith('VIA12'):
        pads[y1].append(x1)
strap_hw = {}
for s1, s2, xc in straps:
    strap_hw[xc] = (s2 - s1) / 2
rails = sorted(rails)
pad_x = {}
for y, xs in pads.items():
    xs = sorted(set(xs))
    # pad half width = the strap's half width (exact crossover)
    pad_x[y] = (xs, [strap_hw.get(x, 192) for x in xs])
m3.sort()

def row_of(y):
    i = bisect.bisect_right(rails, y) - 1
    if i < 0 or i + 1 >= len(rails):
        return None
    return rails[i], rails[i + 1]

def gap_to_pads(y, xc, hw):
    if y not in pad_x:
        return a.window + 1
    xs, hws = pad_x[y]
    i = bisect.bisect_left(xs, xc)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(xs):
            g = (abs(xc - xs[j]) - hws[j] - hw) / SITE
            best = g if best is None else min(best, g)
    return min(best, a.window + 1) if best is not None else a.window + 1

def gap_to_m3(ylo, yhi, xc, hw):
    best = a.window + 1
    for s1, s2, sy1, sy2 in m3:
        if sy2 < ylo or sy1 > yhi:
            continue
        g = (max(s1 - xc, xc - s2, 0) - hw) / SITE if not (s1 <= xc <= s2) else -((min(s2 - xc, xc - s1)) + hw) / SITE
        best = min(best, g)
    return best

# ---------------------------------------------------------------- components -> placed pins
comp = re.search(r'\nCOMPONENTS \d+ ;(.*?)END COMPONENTS', t, re.S).group(1)
inst = {}
for c in re.finditer(r'- (\S+) (\S+) .*?\+ (?:PLACED|FIXED) \( (-?\d+) (-?\d+) \) (\S+)', comp, re.S):
    if c.group(2) in cells:
        inst[unesc(c.group(1))] = (c.group(2), int(c.group(3)), int(c.group(4)), c.group(5))

def place_rect(cell, ox, oy, o, r):
    W, H, _ = cells[cell]
    x1, y1, x2, y2 = r
    if o in ('S', 'FN'):        # mirror x
        x1, x2 = W - x2, W - x1
    if o in ('S', 'FS'):        # mirror y
        y1, y2 = H - y2, H - y1
    return (ox + x1 * dbu, oy + y1 * dbu, ox + x2 * dbu, oy + y2 * dbu)

pinrec = {}                    # (inst, pin) -> dict
row_pins = collections.defaultdict(list)   # row bottom -> [(xc, fragile)]
for iname, (cell, ox, oy, o) in inst.items():
    for pname, rects in cells[cell][2].items():
        prs = [place_rect(cell, ox, oy, o, r) for r in rects]
        bx1 = min(r[0] for r in prs); bx2 = max(r[2] for r in prs)
        by1 = min(r[1] for r in prs); by2 = max(r[3] for r in prs)
        xc, hw = (bx1 + bx2) / 2, (bx2 - bx1) / 2
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
            if r2 is None:
                continue
            ks.add(int(round((ty - r2[0]) / TP)))
        rec = dict(inst=iname, cell=cell, pin=pname, xc=xc, hw=hw, rb=rb, rt=rt,
                   y1=by1, y2=by2, n_acc=len(acc), ks=ks)
        pinrec[(iname, pname)] = rec
        row_pins[rb].append((xc, len(acc) == 1, iname))
for rb in row_pins:
    row_pins[rb].sort()

# ---------------------------------------------------------------- nets and DRC markers
nets = re.search(r'\nNETS \d+ ;(.*?)END NETS', t, re.S).group(1)
net_pins = {}
for n in re.finditer(r'^- (\S+)(.*?)(?=^- |\Z)', nets, re.S | re.M):
    body = n.group(2).split('+ ROUTED')[0].split('+ USE')[0]
    conn = [(unesc(i), p) for i, p in re.findall(r'\( (\S+) (\S+) \)', body)
            if i != 'PIN' and not re.fullmatch(r'-?\d+|\*', i)]
    net_pins[unesc(n.group(1))] = conn
drc = open(os.path.join(a.run, 'rpt', 'drc.rpt')).read()
pos = set(); nmark = nmatch = 0
for mk in re.finditer(r'^\w+: \( [^)]* \) (.*?)\s+\( \S+ \)\nBounds : \( (\S+), (\S+) \) \( (\S+), (\S+) \)', drc, re.M):
    nmark += 1
    mx = (float(mk.group(2)) + float(mk.group(4))) / 2 * dbu
    my = (float(mk.group(3)) + float(mk.group(5))) / 2 * dbu
    hit = False
    for net in re.findall(r'Regular Wire of Net (\S+)', mk.group(1)):
        best = None
        for ip in net_pins.get(unesc(net), []):
            r = pinrec.get(ip)
            if r is None or not (r['rb'] <= my <= r['rt']):
                continue
            dx = max(abs(mx - r['xc']) - r['hw'], 0)
            if dx <= SITE and (best is None or dx < best[0]):
                best = (dx, ip)
        if best:
            pos.add(best[1]); hit = True
    nmatch += hit

# ---------------------------------------------------------------- rows
runid = '/'.join(a.run.rstrip('/').split('/')[-3:])
pillar = 1 if a.pdn == 'pillar' else 0
cols = ['run', 'inst', 'cell', 'pin', 'n_acc', 'acc_k1', 'acc_k2', 'acc_k3', 'acc_k4', 'n_edge', 'n_mid',
        'pin_w', 'gap_bot', 'gap_top', 'gap_edge', 'gap_m3', 'pillar', 'nb_pins', 'nb_fragile', 'y']
kept = kpos = 0
with gzip.open(a.out, 'wt') as fo:
    fo.write(','.join(cols) + '\n')
    for key, r in pinrec.items():
        gb = gap_to_pads(r['rb'], r['xc'], r['hw'])
        gt = gap_to_pads(r['rt'], r['xc'], r['hw'])
        gm = gap_to_m3(r['rb'], r['rt'], r['xc'], r['hw']) if m3 else a.window + 1
        if min(gb, gt, gm) > a.window:
            continue
        ks = r['ks']
        ge = min(([gb] if 1 in ks else []) + ([gt] if 4 in ks else []) or [a.window + 1])
        rp = row_pins[r['rb']]
        i1 = bisect.bisect_left(rp, (r['xc'] - 2 * SITE,))
        i2 = bisect.bisect_right(rp, (r['xc'] + 2 * SITE, True, '￿'))
        nb = [q for q in rp[i1:i2] if q[2] != r['inst']]
        y = 1 if key in pos else 0
        kept += 1; kpos += y
        fo.write(','.join(map(str, [
            runid, r['inst'], r['cell'], r['pin'], r['n_acc'],
            int(1 in ks), int(2 in ks), int(3 in ks), int(4 in ks),
            int(1 in ks) + int(4 in ks), int(2 in ks) + int(3 in ks),
            round(2 * r['hw'] / SITE, 3), round(gb, 3), round(gt, 3), round(ge, 3), round(gm, 3), pillar,
            len(nb), sum(1 for q in nb if q[1]), y])) + '\n')
print(f'{runid}: {len(pinrec)} signal pins, kept {kept} near PG; markers {nmark}, matched to a pin {nmatch} '
      f'({100 * nmatch / max(nmark, 1):.0f}%); positive pins {len(pos)}, of them kept {kpos}')

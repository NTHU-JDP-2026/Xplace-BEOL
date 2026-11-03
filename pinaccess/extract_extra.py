#!/usr/bin/env python3
"""Extra per-pin columns for the pins of an extract_grid.py dataset: label by
violation type and net-direction features.

    extract_extra.py <run_dir> <cells_lef> <grid_prefix>

Reads <grid_prefix>.csv.gz (the pins kept there, same order) and writes
<grid_prefix>.extra.csv.gz with, per pin:
  y_pg, y_ss   markers on the pin: signal vs PG (Special Wire) / signal vs signal
  deg          number of pins on the pin's net (instance pins + IO pins)
  n_left, n_right, n_up, n_down
               other pins of the net to the left/right (beyond +-1 site) and in
               rows above/below (beyond the pin's own row)
  bb_w, bb_h   net bounding box width (sites) and height (rows), pins only
  same_row     share of the other pins in the pin's own row
Marker to pin matching as in extract_grid.py (marker centre in the pin's row,
within one site of the pin in x).
"""
import bisect, gzip, os, re, sys, collections

run, lef_f, pre = sys.argv[1:4]
SITE = 352.0
lef = open(lef_f).read()
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
d = [f for f in os.listdir(run) if f.endswith('.def') and not f.endswith('.floorplan.def')][0]
t = open(os.path.join(run, d)).read()
dbu = int(re.search(r'UNITS DISTANCE MICRONS (\d+)', t).group(1))
unesc = lambda s: s.replace('\\', '')
sp = re.search(r'\nSPECIALNETS \d+ ;(.*?)END SPECIALNETS', t, re.S).group(1)
rails = sorted({int(y) for y in re.findall(r'NEW M0 \d+ \+ SHAPE FOLLOWPIN \( -?\d+ (-?\d+) \)', sp)} |
               {int(y) for y in re.findall(r'ROUTED M0 \d+ \+ SHAPE FOLLOWPIN \( -?\d+ (-?\d+) \)', sp)})
def row_of(y):
    i = bisect.bisect_right(rails, y) - 1
    return (rails[i], rails[i + 1]) if 0 <= i < len(rails) - 1 else None
comp = re.search(r'\nCOMPONENTS \d+ ;(.*?)END COMPONENTS', t, re.S).group(1)
inst = {}
for c in re.finditer(r'- (\S+) (\S+) .*?\+ (?:PLACED|FIXED) \( (-?\d+) (-?\d+) \) (\S+)', comp, re.S):
    inst[unesc(c.group(1))] = (c.group(2), int(c.group(3)), int(c.group(4)), c.group(5))
def pin_box(iname, pname):
    if iname not in inst or inst[iname][0] not in cells:
        return None
    cell, ox, oy, o = inst[iname]
    W, Hc, pins = cells[cell]
    if pname not in pins:
        return None
    xs, ys = [], []
    for x1, y1, x2, y2 in pins[pname]:
        if o in ('S', 'FN'):
            x1, x2 = W - x2, W - x1
        if o in ('S', 'FS'):
            y1, y2 = Hc - y2, Hc - y1
        xs += [ox + x1 * dbu, ox + x2 * dbu]; ys += [oy + y1 * dbu, oy + y2 * dbu]
    return min(xs), min(ys), max(xs), max(ys)
iopin = {}
pins_sec = re.search(r'\nPINS \d+ ;(.*?)END PINS', t, re.S)
if pins_sec:
    for p in re.finditer(r'- (\S+) \+ NET (\S+).*?(?:PLACED|FIXED) \( (-?\d+) (-?\d+) \)', pins_sec.group(1), re.S):
        iopin[unesc(p.group(1))] = (int(p.group(3)), int(p.group(4)))
nets = re.search(r'\nNETS \d+ ;(.*?)END NETS', t, re.S).group(1)
net_of = {}; net_pts = {}
for n in re.finditer(r'^- (\S+)(.*?)(?=^- |\Z)', nets, re.S | re.M):
    body = n.group(2).split('+ ROUTED')[0].split('+ USE')[0]
    nn = unesc(n.group(1)); pts = []
    for i, p in re.findall(r'\( (\S+) (\S+) \)', body):
        if re.fullmatch(r'-?\d+|\*', i):
            continue
        if i == 'PIN':
            if unesc(p) in iopin:
                pts.append(('PIN', unesc(p), iopin[unesc(p)]))
            continue
        b = pin_box(unesc(i), p)
        if b:
            pts.append((unesc(i), p, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)))
            net_of[(unesc(i), p)] = nn
    net_pts[nn] = pts
# markers by type
drc = open(os.path.join(run, 'rpt', 'drc.rpt')).read()
lab = collections.defaultdict(lambda: [0, 0])
for mk in re.finditer(r'^\w+: \( [^)]* \) (.*?)\s+\( \S+ \)\nBounds : \( (\S+), (\S+) \) \( (\S+), (\S+) \)', drc, re.M):
    mx = (float(mk.group(2)) + float(mk.group(4))) / 2 * dbu
    my = (float(mk.group(3)) + float(mk.group(5))) / 2 * dbu
    typ = 0 if 'Special Wire' in mk.group(1) else 1
    for net in re.findall(r'Regular Wire of Net (\S+)', mk.group(1)):
        best = None
        for iname, pname, (cx, cy) in net_pts.get(unesc(net), []):
            if iname == 'PIN':
                continue
            b = pin_box(iname, pname); r = row_of((b[1] + b[3]) / 2)
            if r is None or not (r[0] <= my <= r[1]):
                continue
            dx = max(abs(mx - (b[0] + b[2]) / 2) - (b[2] - b[0]) / 2, 0)
            if dx <= SITE and (best is None or dx < best[0]):
                best = (dx, (iname, pname))
        if best:
            lab[best[1]][typ] = 1
out = gzip.open(pre + '.extra.csv.gz', 'wt')
out.write('y_pg,y_ss,deg,n_left,n_right,n_up,n_down,bb_w,bb_h,same_row\n')
rows = gzip.open(pre + '.csv.gz', 'rt').read().splitlines()[1:]
for line in rows:
    v = line.split(',')
    key = (v[1], v[3])
    nn = net_of.get(key)
    pts = net_pts.get(nn, []) if nn else []
    b = pin_box(*key)
    if b is None or not pts:
        out.write('%d,%d,0,0,0,0,0,0,0,0\n' % tuple(lab[key])); continue
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    r = row_of(cy) or (cy - 240, cy + 240)
    rh = r[1] - r[0]
    others = [q for q in pts if (q[0], q[1]) != key]
    nl = sum(1 for q in others if q[2][0] < cx - SITE); nr = sum(1 for q in others if q[2][0] > cx + SITE)
    nu = sum(1 for q in others if q[2][1] > r[1]); nd = sum(1 for q in others if q[2][1] < r[0])
    xs = [q[2][0] for q in pts]; ys = [q[2][1] for q in pts]
    same = sum(1 for q in others if r[0] <= q[2][1] <= r[1]) / max(len(others), 1)
    out.write('%d,%d,%d,%d,%d,%d,%d,%.2f,%.2f,%.3f\n' % (lab[key][0], lab[key][1], len(pts), nl, nr, nu, nd,
              (max(xs) - min(xs)) / SITE, (max(ys) - min(ys)) / rh, same))
out.close()
print('%s: %d pins, y_pg %d, y_ss %d' % (pre, len(rows), sum(v[0] for v in lab.values()), sum(v[1] for v in lab.values())))

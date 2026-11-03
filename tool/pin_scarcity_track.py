"""Track-aware pin scarcity of a standard-cell library, and the pin-access critical cell types it implies.

    python3 tool/pin_scarcity_track.py <cells.lef> [--site 0.176] [--track_pitch 0.096] [--tracks 4] [--via_half 0.032]
                                       [--pin_layer M1] > report.txt
    python3 tool/pin_scarcity_track.py --select 1.0 <report.txt>       # the critical types: z >= 1.0, comma separated

The score of the main branch (the *_pin_scarcity_report.txt headers):

    S(x) = sum_i (1/AP_i) N(x; x_i, sigma),   score_flip_max = int_0^W max(S(x), S_flip(x)) dx / num_sites

with S_flip the x-mirror (the orientation is not known before placement), here computed per routing track k of the
pin-access layer above the pins (NS3K: horizontal M2, 4 tracks inside the row) and maxed over k:

    S_k(x) = sum_{i : k in AP_i} (1/|AP_i|) N(x; x_i, sigma),   score = int_0^W max_k max(S_k, S_k^flip) dx / num_sites

AP_i = the tracks that land a via on pin i (the via pad, +-via_half, fully on a pin rect; a pin with none counts the
track nearest its centre). sigma = 2/3 site, as the ASAP7 reports (0.036 / 0.054). Single-access pins side by side on
one track score high (NS3K OAI22_X1: A1 A2 B1 B2 all on track 3); the same pins on alternating tracks do not (NAND4_X1:
tracks 2 / 3). The track-blind score (the main-branch one) ranks them alike, 0.904 / 0.904, while next to an M3 stripe
the first fails 41% of the time and the second 0.9% (NS3K stripe PDN, UTIL 0.8, routed, signal-only violations).

Critical types: z = (score - mean) / std over the library >= --select (the main-branch default candidate threshold,
mu + 1.0 sigma). On NS3K that is AOI222 / OAI221 / AOI221 / OAI222 / OAI22 / AOI22 (z ~ +2), the next type at +0.85.
"""
import argparse
import math
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument('path', help='cells LEF, or with --select a report written by this tool')
ap.add_argument('--select', type=float, default=None, help='print the types with z >= this, comma separated')
ap.add_argument('--site', type=float, default=0.176, help='site width (LEF units; NS3K 4x: 0.176)')
ap.add_argument('--track_pitch', type=float, default=0.096, help='pitch of the access-layer tracks (NS3K M2: 0.096)')
ap.add_argument('--tracks', type=int, default=4, help='access tracks inside a row, k = 1 .. tracks above the bottom rail')
ap.add_argument('--via_half', type=float, default=0.032, help='half height of the via landing pad on the pin')
ap.add_argument('--pin_layer', default='M1')
a = ap.parse_args()

if a.select is not None:
    sel = []
    for line in open(a.path):
        m = re.match(r'\s*\d+\s+(\S+)\s+score=\S+\s+z=(\S+)', line)
        if m and float(m.group(2)) >= a.select:
            sel.append(m.group(1))
    print(','.join(sel))
    sys.exit(0)

sig = a.site * 2 / 3
lef = open(a.path).read()
rows = []
for m in re.finditer(r'^MACRO (\S+)(.*?)^END \1', lef, re.S | re.M):
    name, body = m.groups()
    if not re.search(r'CLASS CORE', body):
        continue
    W = float(re.search(r'SIZE (\S+) BY', body).group(1))
    body = re.split(r'\n\s*OBS\b', body)[0]
    pins = []
    for p in re.finditer(r'\n\s*PIN (\S+)(.*?)\n\s*END \1\b', body, re.S):
        if re.search(r'USE (POWER|GROUND)', p.group(2)):
            continue
        rects = []
        for blk in re.split(r'LAYER ', p.group(2))[1:]:
            if blk.split()[0] == a.pin_layer:
                rects += [tuple(map(float, r.split())) for r in re.findall(r'RECT ([-\d. ]+) ;', blk)]
        if not rects:
            continue
        acc = {}
        for x1, y1, x2, y2 in rects:
            for k in range(1, a.tracks + 1):
                if y1 + a.via_half - 1e-6 <= k * a.track_pitch <= y2 - a.via_half + 1e-6:
                    acc.setdefault(k, []).append((x1 + x2) / 2)
        if not acc:
            x1, y1, x2, y2 = rects[0]
            acc = {min(a.tracks, max(1, round((y1 + y2) / 2 / a.track_pitch))): [(x1 + x2) / 2]}
        pins.append(acc)
    if not pins:
        continue
    n = 400
    dx = W / n
    norm = 1.0 / (sig * math.sqrt(2 * math.pi))

    def S(x, flip, k):
        s = 0.0
        for acc in pins:
            for c in acc.get(k, ()):
                s += norm * math.exp(-0.5 * ((x - ((W - c) if flip else c)) / sig) ** 2) / len(acc) / len(acc[k])
        return s

    xs = [(i + 0.5) * dx for i in range(n)]
    score = sum(max(max(S(x, 0, k), S(x, 1, k)) for k in range(1, a.tracks + 1)) for x in xs) * dx / (W / a.site)
    single = [sum(1 for acc in pins if list(acc) == [k]) for k in range(1, a.tracks + 1)]
    rows.append((score, name, len(pins), W / a.site, single))
rows.sort(key=lambda r: (-r[0], r[1]))
v = [r[0] for r in rows]
mu = sum(v) / len(v)
sd = (sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5
print('# track-aware pin scarcity of %s (tool/pin_scarcity_track.py)' % a.path.split('/')[-1])
print('# site %g, %d access tracks, pitch %g, via half %g, pins on %s, sigma %.4f; score mean %.5f std %.5f' % (
    a.site, a.tracks, a.track_pitch, a.via_half, a.pin_layer, sig, mu, sd))
print('# rank cell score z pins sites single-access-pins-per-track(k=1..%d)' % a.tracks)
for i, (s, name, npin, ns, single) in enumerate(rows, 1):
    print('%3d  %-12s score=%.5f  z=%+.2f  pins=%d  sites=%g  single=%s' % (
        i, name, s, (s - mu) / sd, npin, ns, '/'.join(map(str, single))))

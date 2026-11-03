"""How closely two placements pack the flops and the sinks of chosen nets.

    python3 tool/wl_analysis/sink_spread.py <a.def> <b.def> [net ...] [--flop_prefix DFF] [--scale 4]

For the flops (cells named <flop_prefix>*): the mean / median distance (L1, um at 1x) to the
nearest other flop, the 2x2 um bins they occupy and the spread of their positions; for every
given net: the mean distance from a sink to the nearest other sink of the net. The Steiner tree
of a net of thousands of sinks is about #sinks x that distance (NS3K jpeg rst: 0.23 um after
Innovus, 0.34 um after Xplace, RSMT +38%, with the flops spread alike).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from defio import read_def  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('defs', nargs=2)
ap.add_argument('nets', nargs='*')
ap.add_argument('--flop_prefix', default='DFF')
ap.add_argument('--scale', type=float, default=4.0, help='LEF scale of the DEFs (NS3K Innovus: 4x)')
a = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def nearest(p):
    # distance from every point to the nearest other one, L1, in chunks
    t = torch.tensor(np.asarray(p), dtype=torch.float32, device=dev)
    out = []
    for i in range(0, len(t), 2048):
        d = torch.cdist(t[i:i + 2048], t, p=1)
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float('inf')
        out.append(d.min(1).values.cpu())
    return torch.cat(out).numpy()


D = [read_def(f, routing=False) for f in a.defs]
for lab, (dbu, comps, _, nets) in zip('ab', D):
    s = 1.0 / dbu / a.scale
    f = np.array([(x * s, y * s) for c, x, y, _ in comps.values() if c.startswith(a.flop_prefix)])
    d = nearest(f)
    bins = len({(int(x // 2), int(y // 2)) for x, y in f})
    print('%s: %d flops  nearest-flop distance mean %.3f median %.3f um  occupied 2x2um bins %d  x-std %.1f y-std %.1f' % (
        lab, len(f), d.mean(), np.median(d), bins, f[:, 0].std(), f[:, 1].std()))
for n in a.nets:
    for lab, (dbu, comps, _, nets) in zip('ab', D):
        s = 1.0 / dbu / a.scale
        p = [(comps[i][1] * s, comps[i][2] * s) for i, _ in nets[n][0] if i != 'PIN' and i in comps]
        print('  %-10s %s: %d sinks  nearest-sink distance mean %.3f um' % (n, lab, len(p), nearest(p).mean()))

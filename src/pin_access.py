"""Per-pin signal-to-PG pin-access risk of standard cells next to the PDN.

PinAccessModel builds, for every signal pin of a standard cell,
  - the pin's access: how many M2 tracks can land a via on it and which of the
    4 tracks inside the row they are (after orientation), and its width;
  - what the PDN occupies around it: for M2, M3 and M4, the coverage of 13
    bins (half a site wide, centred on the pin, -3..+3 sites) at the 6 track
    levels of the pin's row (k = 0 / 5 are the rails).
The PDN is read from the DEF (VDD/VSS special nets: rails, stripe/strap wires,
the pads of the rail via stacks), so stripes, pillars or a mix need no change.
score() turns them into a risk per pin: rule-based (.rule, _rule_score; e.g.
"track": every M2 access track of the pin covered by PG on or next to it), or
with a learned model (pinaccess/train_grid.py, --target pg --no_pin_grid).
pinaccess/flip.py uses it to choose the x-mirror and shifts of the cells before
routing.

The coverage of a bin by the PDN shapes on one layer at one track level is a
difference of a cumulative covered length C(x) along that level. Rows and
orientations are discrete: a cell is taken in the row nearest to its bottom
edge, in the orientation Innovus gives it there (its bottom PG pin on the row's
bottom rail: N, else FS).

Coordinates are DEF database units of the design (the 4x layout for NS3K).
"""
import math
import re

import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.f(x).squeeze(1)


def _parse_cells(lef_text, dbu):
    """cell -> (W, H, bottom PG net, {pin: [rects]}) in dbu, cell coordinates."""
    cells = {}
    for m in re.finditer(r'^MACRO (\S+)(.*?)^END \1', lef_text, re.S | re.M):
        name, body = m.groups()
        size = re.search(r'SIZE (\S+) BY (\S+)', body)
        if not size:
            continue
        W, H = float(size.group(1)) * dbu, float(size.group(2)) * dbu
        pins, bottom = {}, None
        for p in re.finditer(r'PIN (\S+)(.*?)END \1\b', body, re.S):
            rects = [tuple(float(v) * dbu for v in r.split()) for r in re.findall(r'RECT ([-\d. ]+) ;', p.group(2))]
            for poly in re.findall(r'POLYGON ([-\d. ]+) ;', p.group(2)):
                v = [float(x) * dbu for x in poly.split()]
                rects.append((min(v[0::2]), min(v[1::2]), max(v[0::2]), max(v[1::2])))
            if re.search(r'USE (POWER|GROUND)', p.group(2)):
                if any(abs(r[1] + r[3]) < 1e-3 * dbu for r in rects):
                    bottom = 'VDD' if 'POWER' in p.group(2) else 'VSS'
                continue
            if rects:
                pins[p.group(1)] = rects
        cells[name] = (W, H, bottom, pins)
    return cells


def _parse_pdn(def_text):
    """rails {y: net}, and per layer a list of (x1, x2, y1, y2) VDD/VSS shapes."""
    sp = re.search(r'\nSPECIALNETS \d+ ;(.*?)END SPECIALNETS', def_text, re.S)
    rails, shapes, vias, strap_hw = {}, {}, [], {}
    if not sp:
        return rails, shapes
    for net in re.finditer(r'^- (\S+)(.*?);\s*$', sp.group(1), re.S | re.M):
        nname = net.group(1)
        for w in re.finditer(r'(?:ROUTED|NEW) (M\d+) (\d+) \+ SHAPE (\w+) \( (-?\d+) (-?\d+) \)(?: \( (\S+) (\S+) \))?\s*(\S+)?',
                             net.group(2)):
            lay, wid, shape, x1, y1, x2, y2, via = w.groups()
            wid, x1, y1 = int(wid), int(x1), int(y1)
            if shape == 'FOLLOWPIN' and lay == 'M0':
                rails[y1] = nname
            elif wid > 0 and x2 == '*':
                yy = int(y2)
                shapes.setdefault(lay, []).append((x1 - wid / 2, x1 + wid / 2, min(y1, yy), max(y1, yy)))
                strap_hw[x1] = wid / 2
            elif wid > 0 and y2 == '*' and shape != 'FOLLOWPIN':
                xx = int(x2)
                shapes.setdefault(lay, []).append((min(x1, xx), max(x1, xx), y1 - wid / 2, y1 + wid / 2))
            elif wid == 0 and via and re.match(r'VIA\d\d', via):
                vias.append((x1, y1, via[:5]))
    layers_of = {'VIA12': ('M2',), 'VIA23': ('M2', 'M3'), 'VIA34': ('M3', 'M4'), 'VIA45': ('M4',)}
    for x, y, v in vias:
        hw = strap_hw.get(x, 192)
        for lay in layers_of.get(v, ()):
            shapes.setdefault(lay, []).append((x - hw, x + hw, y - 48, y + 48))
    return rails, shapes


class PinAccessModel:
    """Scores pins; positions in dbu. Build once per design, call score()."""

    LAYERS = ('M2', 'M3', 'M4')
    ORIENTS = ('N', 'FS', 'FN', 'S')
    NK, NC, BIN, TP, SITE = 6, 13, 176.0, 192.0, 352.0

    def __init__(self, ckpt_path, def_path, cell_lef_path, dbu=2000, device='cuda', logger=None):
        """ckpt_path None: no model, for the rule-based scores only (set .rule before scoring)."""
        self.device = device
        # rule-based scores instead of the model (pinaccess/flip.py --rule)
        self.rule, self.rule_w = None, 1
        self.net = None
        if ckpt_path is not None:
            ck = torch.load(ckpt_path, map_location='cpu')
            assert ck.get('no_pin_grid', False), 'pin-access penalty needs a --no_pin_grid model'
            self.mu = torch.tensor(ck['mu'], dtype=torch.float32, device=device)
            self.sd = torch.tensor(ck['sd'], dtype=torch.float32, device=device)
            self.log_pw = math.log(ck['pos_weight'])
            nl, nk, nc = ck['grid']
            assert (nk, nc) == (self.NK, self.NC)
            self.net = _MLP(len(ck['acols']) + nl * nk * nc).to(device)
            self.net.load_state_dict(ck['state'])
            self.net.eval()
            for p in self.net.parameters():
                p.requires_grad_(False)
        self.cells = _parse_cells(open(cell_lef_path).read(), dbu)
        def_text = open(def_path).read()
        rails, shapes = _parse_pdn(def_text)
        ys = sorted(rails)
        self.y0 = ys[0]
        self.rowh = min(b - a for a, b in zip(ys, ys[1:]))
        self.nrows = int(round((ys[-1] - ys[0]) / self.rowh))
        # bottom rail net of every row boundary (missing rails: by parity)
        nets = {}
        for y, n in rails.items():
            nets[int(round((y - self.y0) / self.rowh))] = n
        first = nets.get(0, 'VSS')
        other = 'VDD' if first == 'VSS' else 'VSS'
        self.rail_net = [nets.get(i, first if i % 2 == 0 else other) for i in range(self.nrows + 1)]
        self._build_cover(shapes)
        self._build_pins()
        if logger:
            logger.info('PinAccess: %d rails, %d rows, PDN shapes %s, %d cell types with signal pins'
                        % (len(rails), self.nrows, {k: len(v) for k, v in shapes.items()}, len(self.ptab)))

    # ------------------------------------------------------------ PDN coverage
    def _build_cover(self, shapes):
        """Per layer: sorted breakpoints of the covered x-intervals at every
        (row, level) key, with the cumulative covered length, for C(x)."""
        self.cov = {}
        nkeys = self.nrows * self.NK
        self.BIG = 2.0 ** 22
        for lay in self.LAYERS:
            per_key = {}
            for x1, x2, y1, y2 in shapes.get(lay, []):
                r0 = max(int(math.floor((y1 - self.y0) / self.rowh)) - 1, 0)
                r1 = min(int(math.ceil((y2 - self.y0) / self.rowh)) + 1, self.nrows - 1)
                for r in range(r0, r1 + 1):
                    rb = self.y0 + r * self.rowh
                    for k in range(self.NK):
                        y = rb + k * self.TP
                        if y1 <= y <= y2:
                            per_key.setdefault(r * self.NK + k, []).append((x1, x2))
            gx, cum, inside, keyof = [], [], [], []
            for key in sorted(per_key):
                iv = sorted(per_key[key])
                merged = []
                for a, b in iv:
                    if merged and a <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], b)
                    else:
                        merged.append([a, b])
                c = 0.0
                gx.append(key * self.BIG - 1.0); cum.append(0.0); inside.append(0.0); keyof.append(key)
                for a, b in merged:
                    gx.append(key * self.BIG + a); cum.append(c); inside.append(1.0); keyof.append(key)
                    c += b - a
                    gx.append(key * self.BIG + b); cum.append(c); inside.append(0.0); keyof.append(key)
            dd = dict(dtype=torch.float64, device=self.device)
            self.cov[lay] = (torch.tensor(gx, **dd), torch.tensor(cum, **dd),
                             torch.tensor(inside, **dd), torch.tensor(keyof, dtype=torch.long, device=self.device))
        self.nkeys = nkeys

    def _C(self, lay, key, x):
        """cumulative covered length up to x at (row, level) key; x: float64 tensor."""
        gx, cum, inside, keyof = self.cov[lay]
        if gx.numel() == 0:
            return torch.zeros_like(x)
        q = key.to(torch.float64) * self.BIG + x
        idx = torch.searchsorted(gx, q.detach(), right=True) - 1
        idx = idx.clamp(min=0)
        ok = (keyof[idx] == key) & (idx >= 0)
        base = gx[idx] - key.to(torch.float64) * self.BIG
        c = cum[idx] + inside[idx] * (x - base)
        return torch.where(ok, c, torch.zeros_like(c))

    # ------------------------------------------------------------ pins
    def _build_pins(self):
        """Per cell type, both orientations (N, FS): every signal pin's centre
        offset, half width, row offset, and access features."""
        self.ptab = {}
        for name, (W, H, bottom, pins) in self.cells.items():
            if not pins:
                continue
            ent = {'W': W, 'H': H, 'bottom': bottom, 'N': [], 'FS': [], 'FN': [], 'S': []}
            for pname, rects in pins.items():
                for o in ('N', 'FS', 'FN', 'S'):
                    rs = rects
                    if o in ('FS', 'S'):            # mirror y
                        rs = [(x1, H - y2, x2, H - y1) for x1, y1, x2, y2 in rs]
                    if o in ('FN', 'S'):            # mirror x
                        rs = [(W - x2, y1, W - x1, y2) for x1, y1, x2, y2 in rs]
                    bx1 = min(r[0] for r in rs); bx2 = max(r[2] for r in rs)
                    by1 = min(r[1] for r in rs); by2 = max(r[3] for r in rs)
                    yc = (by1 + by2) / 2
                    roff = int(math.floor(yc / self.rowh))
                    acc = set()
                    for x1, y1, x2, y2 in rs:
                        for k in range(int(H / self.TP) + 1):
                            ty = k * self.TP
                            if y1 + 64 <= ty <= y2 - 64:
                                acc.add(ty)
                    # tracks inside the pin's own row, as in extract_grid.py
                    ks = set()
                    for ty in acc:
                        kk = int(round((ty - roff * self.rowh) / self.TP))
                        if roff * self.rowh <= ty < (roff + 1) * self.rowh and 0 <= kk < self.NK:
                            ks.add(kk)
                    feat = [float(len(acc)), float(1 in ks), float(2 in ks), float(3 in ks), float(4 in ks),
                            (bx2 - bx1) / self.SITE]
                    ent[o].append(((bx1 + bx2) / 2, (bx2 - bx1) / 2, roff, feat, pname))
            self.ptab[name] = ent

    def index_cells(self, cell_types):
        """Pin table for a list of cell type names (one per cell): pin -> cell
        index, offsets and features for both orientations."""
        cid, hw, bot, hgt = [], [], [], []
        off = {o: [] for o in self.ORIENTS}; roff = {o: [] for o in self.ORIENTS}; ft = {o: [] for o in self.ORIENTS}
        self.pin_names = []
        for i, ct in enumerate(cell_types):
            e = self.ptab.get(ct)
            if e is None:
                continue
            for j in range(len(e['N'])):
                cid.append(i); hw.append(e['N'][j][1]); self.pin_names.append((i, e['N'][j][4]))
                for o in self.ORIENTS:
                    off[o].append(e[o][j][0]); roff[o].append(e[o][j][2]); ft[o].append(e[o][j][3])
                bot.append(1 if e['bottom'] == 'VDD' else 0); hgt.append(e['H'])
        d = self.device
        t = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt, device=d)
        # per orientation (N, FS, FN, S) stacked on dim 0
        self.pins = dict(cid=t(cid, torch.long), hw=t(hw, torch.float64), bot=t(bot, torch.long), H=t(hgt, torch.float64),
                         off=torch.stack([t(off[o], torch.float64) for o in self.ORIENTS]),
                         roff=torch.stack([t(roff[o], torch.long) for o in self.ORIENTS]),
                         feat=torch.stack([t(ft[o]) for o in self.ORIENTS]))
        self.rail_vdd = t([1 if n == 'VDD' else 0 for n in self.rail_net], torch.long)
        return len(cid)

    def _rule_score(self, g, feat=None):
        """Hand-written baselines on the model's own inputs (pinaccess/flip.py --rule).

        block: a pin is blocked (1) when any PG shape (M2, M3 or M4, any track
          level, rails included) covers a bin within +-rule_w half-sites of its
          centre; else 0. On the pillar PDN the risky pins are the ones with a via
          pad right over or under them (rule_w 0).
        track: track-aware. A pin is blocked when EVERY M2 track it can be
          accessed on (acc_k1..4, the interior tracks a via can land on) is
          taken: a PG shape on M2 on that track or on a neighbouring one (spacing:
          the pillar pads sit on the rail tracks 0 / 5, right next to tracks 1 / 4)
          within +-rule_w half-sites of the pin centre; a pin with a free access
          track is free (0).
        track23: as track, an access track also counting as taken when M3 above
          it is covered there (a stripe over the pin: no way up)."""
        c0, w = self.NC // 2, self.rule_w
        if self.rule == 'block':
            return (g[:, :, c0 - w:c0 + w + 1].amax(dim=(1, 2)) > 0).float()
        NK = self.NK
        m2 = g[:, 0:NK, c0 - w:c0 + w + 1].amax(dim=2) > 0               # (m, NK) per track level
        cov = m2.clone()
        cov[:, 1:] |= m2[:, :-1]                                         # neighbour below
        cov[:, :-1] |= m2[:, 1:]                                         # neighbour above
        if self.rule == 'track23':
            cov = m2 | (g[:, NK:2 * NK, c0 - w:c0 + w + 1].amax(dim=2) > 0)
        acc = feat[:, 1:5] > 0.5                                         # acc_k1..4: levels 1..4
        n_acc = acc.sum(1)
        blocked = (acc & cov[:, 1:5]).sum(1)
        return ((n_acc > 0) & (blocked == n_acc)).float()

    def score(self, xl, yb, orient=None, window_only=True):
        """xl, yb: lower-left corner (dbu) of the indexed cells, float64 tensors
        (xl may require grad). orient: per-cell orientation code (0 N, 1 FS,
        2 FN, 3 S), or None: the row's orientation (N / FS by rail parity), and
        the risk averaged over the two x-mirror options (Innovus picks the flip).
        Returns (per-pin probability, selected pin mask)."""
        if orient is not None:
            return self._score_o(xl, yb, orient[self.pins['cid']])
        P = self.pins
        r = torch.round((yb.detach()[P['cid']] - self.y0) / self.rowh).long().clamp(0, self.nrows - 1)
        isN = self.rail_vdd[r] == P['bot']
        o1 = torch.where(isN, 0, 1)                    # N / FS
        p1, s1 = self._score_o(xl, yb, o1, window_only)
        p2, s2 = self._score_o(xl, yb, o1 + 2, window_only)   # FN / S
        return (p1 + p2) / 2, s1 | s2

    def cell_risk_minflip(self, xl, yb, ncells, tau=0.05):
        """Per-cell risk with the x-mirror chosen per cell: soft minimum
        (temperature tau) of the summed pin risk of the two mirror options of
        the row's orientation. Returns (cell risk, cell risk of each option,
        selected pins)."""
        P = self.pins
        r = torch.round((yb.detach()[P['cid']] - self.y0) / self.rowh).long().clamp(0, self.nrows - 1)
        o1 = torch.where(self.rail_vdd[r] == P['bot'], 0, 1)
        pa_, s1 = self._score_o(xl, yb, o1)
        pb_, s2 = self._score_o(xl, yb, o1 + 2)
        ca = torch.zeros(ncells, dtype=pa_.dtype, device=self.device).index_add(0, P['cid'], pa_)
        cb = torch.zeros(ncells, dtype=pb_.dtype, device=self.device).index_add(0, P['cid'], pb_)
        soft = -tau * torch.logsumexp(torch.stack([-ca / tau, -cb / tau]), 0)
        soft = torch.clamp(soft, min=0)
        return soft, ca, cb, s1 | s2

    def _score_o(self, xl, yb, o, window_only=True):
        """o: per-pin orientation code."""
        P = self.pins
        n = o.shape[0]
        ar = torch.arange(n, device=self.device)
        x = xl[P['cid']]
        r = torch.round((yb.detach()[P['cid']] - self.y0) / self.rowh).long().clamp(0, self.nrows - 1)
        off = P['off'][o, ar]
        roff = P['roff'][o, ar]
        feat = P['feat'][o, ar]
        pr = (r + roff).clamp(0, self.nrows - 1)
        xc = x + off
        js = torch.arange(-(self.NC // 2), self.NC // 2 + 1, device=self.device, dtype=torch.float64)
        b1 = xc.unsqueeze(1) + js.unsqueeze(0) * self.BIN - self.BIN / 2          # (n, NC)
        b2 = b1 + self.BIN
        # window test (no grad): any PDN shape in the 13-bin window at any level
        with torch.no_grad():
            any_cov = torch.zeros(xc.shape[0], dtype=torch.bool, device=self.device)
            for lay in self.LAYERS:
                for k in range(self.NK):
                    key = pr * self.NK + k
                    c = self._C(lay, key, b2[:, -1].detach()) - self._C(lay, key, b1[:, 0].detach())
                    any_cov |= c > 0
        sel = torch.nonzero(any_cov).squeeze(1) if window_only else torch.arange(xc.shape[0], device=self.device)
        if sel.numel() == 0:
            return torch.zeros(xc.shape[0], dtype=torch.float32, device=self.device), any_cov
        prs, bs1, bs2 = pr[sel], b1[sel], b2[sel]
        grids = []
        for lay in self.LAYERS:
            for k in range(self.NK):
                key = (prs * self.NK + k).unsqueeze(1).expand_as(bs1)
                c = (self._C(lay, key, bs2) - self._C(lay, key, bs1)) / self.BIN
                grids.append(c.clamp(0, 1))
        g = torch.stack(grids, 1).float()                                      # (m, 3*NK, NC)
        if self.rule:
            p = self._rule_score(g, feat[sel])
        elif self.net is None:
            raise RuntimeError('PinAccessModel without a model: set .rule (pinaccess/flip.py --rule)')
        else:
            g = torch.cat([g, torch.zeros(g.shape[0], self.NK, self.NC, device=self.device)], 1)  # PIN grid = 0
            xa = (feat[sel] - self.mu) / self.sd
            logit = self.net(torch.cat([xa, g.flatten(1)], 1))
            p = torch.sigmoid(logit - self.log_pw)
        out = torch.zeros(xc.shape[0], dtype=p.dtype, device=self.device)
        out = out.index_put((sel,), p)
        return out, any_cov

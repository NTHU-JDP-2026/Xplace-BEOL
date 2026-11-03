"""
Generate pdn_structure.drawio: a schematic of the i3dPDK_NS3K M3-stripe PDN.

  (a) top view of the core: M8/M9 ring, M0 follow-pin rails, M3 VDD/VSS
      stripe pairs, M0->M3 via stacks, Xplace's d_M3 low-density band
  (b) zoom on one stripe pair over two rows, to scale (4x LEF, dbu 2000/um)
  (c) layer stack (cross-section) of the stripe style and the pillar style

Geometry in (b) is from aes .../ns3k_innf_fast/xplace/aes_cipher_top.preroute.def
and lef_x4/i3dPDK_NS3K_5T_tech_unidir.lef:
  site 352 x 960, M2/M4 pitch 192, M3 pitch 176, M0 rail 96 wide,
  M3 stripe 384 wide, VSS centre 880 right of VDD centre, pair every 45-46
  sites, M0->M3 stack VIA01/12/23_gen_2 (2 cuts, M1/M2 pad 384 x 128).
"""
import sys
from xml.sax.saxutils import escape

cells = []
_id = [10]


def nid():
    _id[0] += 1
    return f"c{_id[0]}"


def box(x, y, w, h, style, value=""):
    cells.append(
        f'<mxCell id="{nid()}" value="{escape(value, {chr(34): "&quot;"})}" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>')


def text(x, y, w, h, value, size=11, align="left", style=""):
    box(x, y, w, h, f"text;html=1;whiteSpace=wrap;align={align};verticalAlign=middle;"
        f"fontSize={size};fillColor=none;strokeColor=none;{style}", value)


def line(x1, y1, x2, y2, style):
    cells.append(
        f'<mxCell id="{nid()}" value="" style="{style}" edge="1" parent="1">'
        f'<mxGeometry relative="1" as="geometry">'
        f'<mxPoint x="{x1:.1f}" y="{y1:.1f}" as="sourcePoint"/>'
        f'<mxPoint x="{x2:.1f}" y="{y2:.1f}" as="targetPoint"/>'
        f'</mxGeometry></mxCell>')


VDD = "fillColor=#f8cecc;strokeColor=#b85450;"
VSS = "fillColor=#dae8fc;strokeColor=#6c8ebf;"
VDD_S = "fillColor=#b85450;strokeColor=#b85450;"   # solid, thin shapes
VSS_S = "fillColor=#6c8ebf;strokeColor=#6c8ebf;"
VIA = "fillColor=#333333;strokeColor=#000000;"
BAND = "fillColor=#fff2cc;strokeColor=#d6b656;dashed=1;opacity=70;"
CELL = "fillColor=#f5f5f5;strokeColor=#999999;"
PIN = "fillColor=#d5e8d4;strokeColor=#82b366;"
R = "rounded=0;html=1;"
ARROW = "endArrow=classic;endSize=5;html=1;strokeColor=#555555;"
DIM = "endArrow=open;startArrow=open;endSize=5;startSize=5;html=1;strokeColor=#333333;"
TRACK = "endArrow=none;html=1;dashed=1;strokeColor=#bbbbbb;"


def net_of(k):
    """Rail k counted from the core bottom: VDD when k is even (flow.tcl)."""
    return "VDD" if k % 2 == 0 else "VSS"


# ---------------------------------------------------------------- title
text(40, 14, 900, 30, "<b>PDN structure: i3dPDK_NS3K, M3-stripe style (PG_STYLE=stripe)</b>", 18)

# ---------------------------------------------------------------- (a) top view
text(40, 52, 560, 22, "<b>(a) Top view of the core</b>", 13)
CX, CY, CW, CH, NR = 100, 150, 400, 280, 10
RH = CH / NR
RING = {"VDD": 14, "VSS": 26}   # ring offset from the core edge (VDD inner)
RW = 6

for net, off in RING.items():
    st = R + (VDD if net == "VDD" else VSS)
    x0, y0, x1, y1 = CX - off - RW, CY - off - RW, CX + CW + off, CY + CH + off
    box(x0, y0, x1 + RW - x0, RW, st)                 # top    (M8, H)
    box(x0, y1, x1 + RW - x0, RW, st)                 # bottom (M8, H)
    box(x0, y0, RW, y1 + RW - y0, st)                 # left   (M9, V)
    box(x1, y0, RW, y1 + RW - y0, st)                 # right  (M9, V)

PAIR_P, VSS_DX, SW = 135, 11, 5
pairs = [CX + 55 + i * PAIR_P for i in range(3)]
for x0 in pairs:   # Xplace low-density band, behind everything else
    box(x0 - 10, CY, VSS_DX + SW + 20, CH, R + BAND)
box(CX, CY, CW, CH, R + "fillColor=none;strokeColor=#666666;dashed=1;")

for k in range(NR + 1):  # M0 follow-pin rails, out to their own ring
    y = CY + CH - k * RH
    net = net_of(k)
    off = RING[net] + RW / 2
    box(CX - off, y - 1, CW + 2 * off, 2, R + (VDD_S if net == "VDD" else VSS_S))
    for xr in (CX - off, CX + CW + off):  # M0->M9 stack at the ring
        box(xr - 3, y - 3, 6, 6, R + VIA)

for x0 in pairs:   # M3 stripe pairs, each ending on a rail of its own net
    for net, xs, ks in (("VDD", x0, (2, 8)), ("VSS", x0 + VSS_DX, (1, 9))):
        ya, yb = CY + CH - ks[1] * RH, CY + CH - ks[0] * RH
        box(xs, ya, SW, yb - ya, R + (VDD if net == "VDD" else VSS))
        for k in range(ks[0], ks[1] + 1, 2):        # M0->M3 stack per own rail
            box(xs + SW / 2 - 3, CY + CH - k * RH - 3, 6, 6, R + VIA)

# pair pitch dimension
line(pairs[0], 112, pairs[1], 112, DIM)
text(pairs[0], 92, PAIR_P, 18, "45–46 sites (8 µm @4×)", 10, "center")
line(pairs[0], 114, pairs[0], CY, "endArrow=none;dashed=1;strokeColor=#999999;html=1;")
line(pairs[1], 114, pairs[1], CY, "endArrow=none;dashed=1;strokeColor=#999999;html=1;")

# callouts (right of the ring)
LX = 545
callouts = [
    (165, "<b>M8 / M9 core ring</b><br>VDD inner, VSS outer", (CX + CW + 26 + RW / 2, 175)),
    (225, "<b>M0 follow-pin rails</b><br>one per row boundary,<br>VDD / VSS alternate", (CX + CW - 20, CY + CH - 5 * RH)),
    (295, "<b>M3 stripe pair</b> (V)<br>VDD + VSS", (pairs[2] + VSS_DX + SW, CY + 60)),
    (350, "<b>M0→M3 via stack</b><br>stripe × rail of same net", (pairs[2] + VSS_DX + SW / 2 + 3, CY + CH - 3 * RH)),
    (410, "<b>M0→M9 stack</b><br>rail × ring", (CX + CW + 26 + RW / 2 + 3, CY + CH - RH)),
]
for y, label, (tx, ty) in callouts:
    text(LX, y - 20, 150, 40, label, 10)
    line(LX - 2, y, tx, ty, ARROW)
text(CX, CY + CH + 40, CW, 34,
     "<span style='background:#fff2cc'>&nbsp;yellow band&nbsp;</span> = M3 bins where Xplace lowers the "
     "target density to d<sub>M3</sub> = 0.3 (target 0.85 elsewhere)", 10)

# ---------------------------------------------------------------- (b) zoom, to scale
ZX, ZY, K = 760, 120, 0.14          # px per dbu
NS, SITE, ROW, TP = 10, 352, 960, 192
ZW, ZH = NS * SITE * K, 2 * ROW * K
text(720, 52, 560, 22, "<b>(b) Zoom: one stripe pair over two rows</b> (to scale, 4× LEF, 2000 dbu/µm)", 13)


def zx(d):
    return ZX + d * K


def zy(d):   # DEF y-up -> page y-down
    return ZY + ZH - d * K


VDD_C, VSS_C, STW = 3.5 * SITE, 3.5 * SITE + 880, 384
box(zx(VDD_C - STW / 2 - SITE), ZY, (880 + STW + 2 * SITE) * K, ZH, R + BAND)

# cells and their M1 pins (full-height bars on the site centres)
placed = [  # (row, first site, width in sites)
    (0, 0, 3), (0, 7, 3), (1, 0, 2), (1, 2, 2), (1, 8, 2),
]
for row, s0, w in placed:
    y0 = zy((row + 1) * ROW) + 48 * K
    box(zx(s0 * SITE) + 1, y0, w * SITE * K - 2, (ROW - 96) * K, R + CELL)
    for s in range(s0, s0 + w):
        xc = (s + 0.5) * SITE
        box(zx(xc - 32), zy((row + 1) * ROW - 120), 64 * K, (ROW - 240) * K, R + PIN)

for k in range(2 * ROW // TP + 1):   # M2 tracks (H), pitch 192
    y = zy(k * TP)
    if k % 5:
        line(ZX - 6, y, ZX + ZW + 6, y, TRACK)
text(ZX + ZW + 8, zy(4 * TP) - 7, 90, 14, "M2 tracks (H)", 9, style="fontColor=#888888;")

for k in range(3):   # M0 rails: VSS / VDD / VSS
    net = "VSS" if k % 2 == 0 else "VDD"
    box(ZX - 6, zy(k * ROW) - 48 * K, ZW + 12, 96 * K, R + (VDD_S if net == "VDD" else VSS_S))
    text(ZX - 62, zy(k * ROW) - 8, 54, 16, f"{net} M0", 9, "right")

for net, xc in (("VDD", VDD_C), ("VSS", VSS_C)):   # M3 stripes (V), run on past both rows
    box(zx(xc - STW / 2), ZY - 18, STW * K, ZH + 36, R + (VDD if net == "VDD" else VSS) + "opacity=85;")
    text(zx(xc - STW / 2) - 10, ZY - 36, STW * K + 20, 16, f"{net} M3", 9, "center")
stacks = [(VDD_C, ROW), (VSS_C, 0), (VSS_C, 2 * ROW)]
for xc, yc in stacks:   # M0->M3 stacks: M1/M2 pad 384 x 128, two 80 x 80 cuts
    box(zx(xc - 192), zy(yc + 64), 384 * K, 128 * K, R + "fillColor=#ffffff;strokeColor=#000000;strokeWidth=1.5;")
    for cx in (xc - 120, xc + 40):
        box(zx(cx), zy(yc + 40), 80 * K, 80 * K, R + VIA)

# pins whose M2 access is crowded by the stripe / stack pads
for s, row in ((2, 1), (7, 0)):
    xc = (s + 0.5) * SITE
    box(zx(xc - 32) - 2, zy((row + 1) * ROW - 120) - 2, 64 * K + 4, (ROW - 240) * K + 4,
        R + "fillColor=none;strokeColor=#e51400;strokeWidth=2;")

# dimensions
yd = ZY + ZH + 26
line(zx(VDD_C - STW / 2), yd, zx(VDD_C + STW / 2), yd, DIM)
text(zx(VDD_C - STW / 2) - 40, yd + 2, STW * K + 80, 14, "384 dbu", 9, "center")
yd2 = ZY + ZH + 52
line(zx(VDD_C), yd2, zx(VSS_C), yd2, DIM)
text(zx(VDD_C), yd2 + 2, 880 * K, 14, "880 dbu (2.5 sites)", 9, "center")
line(ZX, ZY + ZH + 80, ZX + SITE * K, ZY + ZH + 80, DIM)
text(ZX - 20, ZY + ZH + 82, SITE * K + 40, 14, "site 352", 9, "center")
line(ZX + ZW + 22, zy(ROW), ZX + ZW + 22, zy(2 * ROW), DIM)
text(ZX + ZW + 26, zy(1.5 * ROW) - 8, 90, 16, "row 960 (5 M2 tracks)", 9)

text(ZX + 100, ZY + ZH + 76, 360, 44,
     "<span style='color:#e51400'><b>red</b></span>: pins next to the pair whose M2 access tracks run "
     "into a stripe or stack pad → signal-to-PG risk (flip / spread)", 9)

# ---------------------------------------------------------------- (c) layer stack
text(40, 520, 600, 22, "<b>(c) Layer stack (cross-section, schematic)</b>", 13)
layers = [  # name, direction, role (stripe style)
    ("M9", "V", "ring (left/right)"),
    ("M8", "H", "ring (top/bottom)"),
    ("M5–M7", "", "signal"),
    ("M4", "H", "signal, unidir."),
    ("M3", "V", "<b>PG stripe pairs</b> + signal"),
    ("M2", "H", "signal, unidir. (pin access)"),
    ("M1", "V", "cell pins (full-height bars)"),
    ("M0", "H", "<b>VDD/VSS follow-pin rails</b>"),
]
LY0, LH, LX0, LXW = 560, 26, 100, 210
for i, (name, d, role) in enumerate(layers):
    y = LY0 + i * LH
    box(LX0, y, LXW, LH - 4, R + "fillColor=#fafafa;strokeColor=#cccccc;")
    text(40, y, 56, LH - 4, f"<b>{name}</b> {d}", 10, "right")
    text(LX0 + LXW + 10, y, 200, LH - 4, role, 10)


def ly(name):
    i = [l[0] for l in layers].index(name)
    return LY0 + i * LH


def stack(xc, top, bottom, st, label):
    yt, yb = ly(top) + 4, ly(bottom) + LH - 8
    box(xc - 4, yt, 8, yb - yt, R + VIA)
    for name in [l[0] for l in layers]:
        if ly(top) <= ly(name) <= ly(bottom):
            box(xc - 14, ly(name) + 4, 28, LH - 12, R + st)
    text(xc - 60, LY0 + len(layers) * LH, 120, 30, label, 9, "center")


# stripe style: M3 stripe fed by an M0->M3 stack; rail fed by M0->M9 at the ring
stack(LX0 + 60, "M3", "M0", VDD, "stripe style:<br>M0→M3 stack")
stack(LX0 + 160, "M9", "M0", VSS, "rail → ring:<br>M0→M9 (outside core)")

# pillar style, to the right
PX = LX0 + LXW + 230
for i, (name, _, _) in enumerate(layers):
    box(PX, LY0 + i * LH, 210, LH - 4, R + "fillColor=#fafafa;strokeColor=#cccccc;")
text(PX, 522, 260, 36, "<b>PG_STYLE=pillar</b>: same pairs as<br>M5 straps, M0→M5 via pillars", 10)


def pillar(xc, st):
    yt, yb = ly("M5–M7") + 4, ly("M0") + LH - 8
    box(xc - 4, yt, 8, yb - yt, R + VIA)
    for name in ("M5–M7", "M4", "M3", "M2", "M1", "M0"):
        box(xc - 14, ly(name) + 4, 28, LH - 12, R + st)


pillar(PX + 60, VDD)
text(PX + 216, ly("M5–M7"), 120, LH - 4, "← M5 VDD/VSS straps", 9)
pillar(PX + 150, VSS)
text(PX - 20, LY0 + len(layers) * LH, 250, 30, "M3 carries only the pillar pads,<br>no stripe", 9, "center")

# ---------------------------------------------------------------- legend
LGX, LGY = 1010, 540
text(LGX, LGY - 26, 200, 20, "<b>Legend</b>", 12)
legend = [
    (R + VDD, "VDD (light: wide shape, dark: rail)"),
    (R + VSS, "VSS"),
    (R + VIA, "via cut / via stack"),
    (R + "fillColor=#ffffff;strokeColor=#000000;strokeWidth=1.5;", "M1/M2 pad of a rail via stack"),
    (R + CELL, "standard cell"),
    (R + PIN, "M1 signal pin"),
    (R + "fillColor=none;strokeColor=#e51400;strokeWidth=2;", "pin at signal-to-PG risk"),
    (R + BAND, "low-density band (d_M3 = 0.3)"),
]
for i, (st, label) in enumerate(legend):
    box(LGX, LGY + i * 24, 24, 14, st)
    text(LGX + 32, LGY + i * 24 - 3, 220, 20, label, 10)

out = sys.argv[1] if len(sys.argv) > 1 else "pdn_structure.drawio"
with open(out, "w") as f:
    f.write('<mxfile host="app.diagrams.net" type="device">\n'
            '  <diagram id="pdn" name="PDN Structure">\n'
            '    <mxGraphModel dx="1300" dy="820" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
            'arrows="1" fold="1" page="1" pageScale="1" pageWidth="1300" pageHeight="820" math="0" shadow="0">\n'
            '      <root>\n        <mxCell id="0"/>\n        <mxCell id="1" parent="0"/>\n')
    for c in cells:
        f.write("        " + c + "\n")
    f.write("      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n")
print(f"wrote {out} ({len(cells)} shapes)")

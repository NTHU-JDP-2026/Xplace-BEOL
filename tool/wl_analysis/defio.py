"""DEF/LEF reading shared by the wire-length analysis scripts of this directory."""
import re

_PT = re.compile(r'\(\s*(\S+)\s+(\S+)(?:\s+\S+)?\s*\)')


def read_cell_pins(lef):
    """{macro: (width, height, {pin: (x, y)})} in LEF microns; a pin's position is
    the mean of its RECT centers."""
    text = open(lef).read()
    cells = {}
    for m in re.finditer(r'\nMACRO (\S+)(.*?)\nEND \1', text, re.S):
        body = m.group(2)
        size = re.search(r'SIZE\s+(\S+)\s+BY\s+(\S+)', body)
        if not size:
            continue
        pins = {}
        for p in re.finditer(r'\n\s*PIN (\S+)(.*?)\n\s*END \1', body, re.S):
            r = [tuple(map(float, x)) for x in re.findall(r'RECT\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', p.group(2))]
            if r:
                pins[p.group(1)] = (sum((a[0] + a[2]) / 2 for a in r) / len(r),
                                    sum((a[1] + a[3]) / 2 for a in r) / len(r))
        cells[m.group(1)] = (float(size.group(1)), float(size.group(2)), pins)
    return cells


def _section(text, name):
    m = re.search(r'\n%s \d+ ;(.*?)\nEND %s' % (name, name), text, re.S)
    return m.group(1) if m else ''


def read_def(path, routing=True):
    """dbu, components {name: (cell, x, y, orient)}, IO pins {name: (x, y)} and
    nets {name: (sorted connections ((inst | PIN, pin), ...), routed length in
    DEF units (vias not counted), USE CLOCK)}."""
    text = open(path).read()
    dbu = float(re.search(r'UNITS DISTANCE MICRONS (\d+)', text).group(1))
    comps = {m.group(1): (m.group(2), int(m.group(3)), int(m.group(4)), m.group(5)) for m in re.finditer(
        r'\n\s*- (\S+) (\S+)\s+(?:\+ \S+ )*?\+ (?:PLACED|FIXED) \( (-?\d+) (-?\d+) \) (\S+)', _section(text, 'COMPONENTS'))}
    iopins = {}
    for blk in _section(text, 'PINS').split(';'):
        n = re.search(r'- (\S+)', blk)
        p = re.search(r'\+ (?:PLACED|FIXED) \( (-?\d+) (-?\d+) \)', blk)
        if n and p:
            iopins[n.group(1)] = (int(p.group(1)), int(p.group(2)))
    nets = {}
    for blk in re.split(r'\n\s*- ', _section(text, 'NETS'))[1:]:
        name = blk.split()[0]
        conns = tuple(sorted(re.findall(r'\(\s*(\S+)\s+(\S+)\s*\)', blk.split('+')[0])))
        wl = 0
        if routing:
            for seg in re.findall(r'(?:ROUTED|NEW)\s+M\d+\s+([^\n]*)', blk):
                px = py = None
                for x, y in _PT.findall(seg.split(' RECT')[0]):
                    if px is None:
                        px, py = int(x), int(y)
                        continue
                    nx = px if x == '*' else int(x)
                    ny = py if y == '*' else int(y)
                    wl += abs(nx - px) + abs(ny - py)
                    px, py = nx, ny
        nets[name] = (conns, wl, '+ USE CLOCK' in blk)
    return dbu, comps, iopins, nets


def pin_xy(cells, dbu, comp, pin):
    """DEF position of a cell pin (cell origin if the cell or pin is unknown)."""
    cell, x, y, orient = comp
    if cell not in cells:
        return x, y
    w, h, pins = cells[cell]
    px, py = pins.get(pin, (w / 2, h / 2))
    w, h, px, py = w * dbu, h * dbu, px * dbu, py * dbu
    if orient == 'N':
        return x + px, y + py
    if orient == 'FS':
        return x + px, y + h - py
    if orient == 'FN':
        return x + w - px, y + py
    if orient == 'S':
        return x + w - px, y + h - py
    raise ValueError('orientation %s' % orient)

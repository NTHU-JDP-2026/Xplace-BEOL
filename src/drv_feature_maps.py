"""
Build DRV prediction feature maps from XPlace's internal placement state.

All operations run on the same GPU device as the placer.
Never reads DEF files — uses PlaceData + GPDatabase directly.

Five output channels (all grid_size × grid_size float32):
  0  pin_density  – Gaussian-smeared pin-count density
  1  ap           – AP-weighted Gaussian density  (needs ap_lookup; falls back to pin_density)
  2  cell_type    – routing-complexity-weighted cell mixture
  3  aoi          – AOI*/OAI* cell density
  4  rudy         – RUDY routing-demand estimate

Normalisation: each channel divided by its per-sample max, matching
PlacementMapDataset in fcn_trainer/map_models.py so inference uses the
same input distribution as training.
"""

import os
import sys

import torch

# Reuse helpers from fcn_trainer (same repo, relative path)
_FCN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fcn_trainer')
if _FCN_DIR not in sys.path:
    sys.path.insert(0, _FCN_DIR)
from map_models import _separable_gaussian_map, _cell_category, _is_aoi_oai


def build_drv_feature_maps(
    mov_node_pos,
    data,
    gpdb,
    ap_lookup=None,
    grid_size=128,
    smooth_factor=1.5,
    min_sigma_bins=1.0,
):
    """
    Return a (5, grid_size, grid_size) float32 feature tensor on device.

    Args:
        mov_node_pos  : (N_mov, 2) current movable-node positions from the Nesterov
                        optimizer — grabbed inside the GP iteration loop.
        data          : PlaceData object from XPlace.
        gpdb          : GPDatabase C++ object; provides node_id2celltype_name().
        ap_lookup     : dict {cell_type_name: float} ap_median values pre-loaded
                        from the fcn_trainer AP JSON files, or None to fall back
                        to pin-count weighting (same as pin_density channel).
        grid_size     : output resolution — must match the training resolution (128).
        smooth_factor : Gaussian σ = (cell_size / die_size / 2) × smooth_factor.
        min_sigma_bins: minimum σ expressed in bins (avoids delta-like spikes).
    """
    device = mov_node_pos.device
    G = grid_size

    # ── Current node positions (movable updated; fixed from data) ─────────────
    mov_lhs, mov_rhs = data.movable_index
    N = data.num_nodes          # real nodes only — fillers are added later
    node_pos = data.node_pos[:N].clone()
    node_pos[mov_lhs:mov_rhs] = mov_node_pos[mov_lhs:mov_rhs].detach()

    # ── Normalise coordinates to [0, 1] inside the die ────────────────────────
    die_ll  = data.die_ll               # (2,)
    die_ur  = data.die_ur               # (2,)
    die_wh  = die_ur - die_ll           # (2,)

    real_size = data.node_size[:N]      # (N, 2) [width, height]
    cx_norm = (node_pos[:, 0] - die_ll[0]) / die_wh[0]   # (N,)
    cy_norm = (node_pos[:, 1] - die_ll[1]) / die_wh[1]

    # Gaussian sigma proportional to cell footprint
    min_sig = min_sigma_bins / G
    sx = (real_size[:, 0] / (2.0 * die_wh[0]) * smooth_factor).clamp(min=min_sig)
    sy = (real_size[:, 1] / (2.0 * die_wh[1]) * smooth_factor).clamp(min=min_sig)

    bins = (torch.arange(G, dtype=torch.float32, device=device) + 0.5) / G

    # Exclude zero-area nodes (IO ports, dummy nodes with no physical footprint)
    valid = (real_size[:, 0] > 1e-6) & (real_size[:, 1] > 1e-6)

    # Pin weights normalised over all valid cells
    pin_w = data.node_to_num_pins[:N, 0].float()   # (N,)
    pin_w = pin_w / (pin_w.sum() + 1e-9)

    def _gmap(weights):
        """Gaussian density map for an arbitrary per-node weight vector."""
        w = weights[valid]
        w = w / (w.sum() + 1e-9)
        m = _separable_gaussian_map(
            cx_norm[valid], cy_norm[valid], sx[valid], sy[valid], w, bins)
        mx = m.max()
        return (m / mx) if mx > 0 else m

    # ── 1. pin_density ─────────────────────────────────────────────────────────
    pin_density = _gmap(pin_w)

    # ── 2. ap ──────────────────────────────────────────────────────────────────
    # gpdb gives "COMB/INV_X1"-style names; we need only the part after "/"
    celltype_full = gpdb.node_id2celltype_name()        # list[str], len >= N
    macro_names   = [s.split('/')[-1] for s in celltype_full[:N]]

    if ap_lookup:
        default_ap = float(sum(ap_lookup.values()) / len(ap_lookup)) if ap_lookup else 1.0
        ap_vals = torch.tensor(
            [ap_lookup.get(m, default_ap) for m in macro_names],
            dtype=torch.float32, device=device,
        )
        ap_vals = ap_vals / (ap_vals.sum() + 1e-9)
        ap_map = _gmap(ap_vals)
    else:
        ap_map = pin_density   # identical fallback

    # ── 3. cell_type ───────────────────────────────────────────────────────────
    cats = torch.tensor([_cell_category(m) for m in macro_names],
                        dtype=torch.long, device=device)   # (N,)
    # Weights match CellTypeMapModel._WEIGHTS in map_models.py
    _CAT_W = [1.0, 2.0, 3.0, 5.0]   # simple, sequential, complex-comb, std-comb
    M = torch.zeros(G, G, device=device)
    for g, w_cat in enumerate(_CAT_W):
        mask = valid & (cats == g)
        if not mask.any():
            continue
        w = pin_w[mask] / (pin_w[mask].sum() + 1e-9)
        D_g = _separable_gaussian_map(
            cx_norm[mask], cy_norm[mask], sx[mask], sy[mask], w, bins)
        M = M + w_cat * D_g
    mx = M.max()
    cell_type = (M / mx) if mx > 0 else M

    # ── 4. aoi ─────────────────────────────────────────────────────────────────
    is_aoi = torch.tensor([_is_aoi_oai(m) for m in macro_names],
                          dtype=torch.bool, device=device)
    aoi_mask = valid & is_aoi
    if aoi_mask.any():
        w = pin_w[aoi_mask] / (pin_w[aoi_mask].sum() + 1e-9)
        aoi_raw = _separable_gaussian_map(
            cx_norm[aoi_mask], cy_norm[aoi_mask], sx[aoi_mask], sy[aoi_mask], w, bins)
        mx = aoi_raw.max()
        aoi = (aoi_raw / mx) if mx > 0 else aoi_raw
    else:
        aoi = torch.zeros(G, G, device=device)

    # ── 5. RUDY ────────────────────────────────────────────────────────────────
    rudy = _build_rudy(node_pos, data, G, device)

    # ── Stack + per-channel max-normalise ─────────────────────────────────────
    feat = torch.stack([pin_density, ap_map, cell_type, aoi, rudy], dim=0)  # (5,G,G)
    mx = feat.flatten(1).max(dim=1).values.clamp(min=1e-9).view(5, 1, 1)
    feat = feat / mx

    return feat


def _build_rudy(node_pos, data, G, device):
    """
    RUDY routing-demand map via 2-D difference array + cumsum.
    Complexity: O(N_pins) scatter_reduce + O(G^2) cumsum.

    pin_rel_cpos stores offsets relative to the node centre ('cpos').
    Requires PyTorch >= 1.12 for scatter_reduce_ with reduce='amin'/'amax'.
    """
    nid   = data.pin_id2node_id                 # (P,)
    pin_x = node_pos[nid, 0] + data.pin_rel_cpos[:, 0]
    pin_y = node_pos[nid, 1] + data.pin_rel_cpos[:, 1]

    die_ll = data.die_ll
    die_ur = data.die_ur
    die_w  = (die_ur[0] - die_ll[0])
    die_h  = (die_ur[1] - die_ll[1])

    pin_xn = ((pin_x - die_ll[0]) / die_w).clamp(0.0, 1.0)
    pin_yn = ((pin_y - die_ll[1]) / die_h).clamp(0.0, 1.0)

    pin_net = data.pin_id2net_id                 # (P,)
    num_nets = data.num_nets

    INF = float('inf')
    net_xn0 = pin_xn.new_full((num_nets,),  INF)
    net_xn1 = pin_xn.new_full((num_nets,), -INF)
    net_yn0 = pin_yn.new_full((num_nets,),  INF)
    net_yn1 = pin_yn.new_full((num_nets,), -INF)

    net_xn0.scatter_reduce_(0, pin_net, pin_xn, reduce='amin', include_self=True)
    net_xn1.scatter_reduce_(0, pin_net, pin_xn, reduce='amax', include_self=True)
    net_yn0.scatter_reduce_(0, pin_net, pin_yn, reduce='amin', include_self=True)
    net_yn1.scatter_reduce_(0, pin_net, pin_yn, reduce='amax', include_self=True)

    # Keep only valid nets (degree in [2, ignore_net_degree])
    mask = data.net_mask
    net_xn0, net_xn1 = net_xn0[mask], net_xn1[mask]
    net_yn0, net_yn1 = net_yn0[mask], net_yn1[mask]

    # RUDY demand = (W + H) / (W × H)
    min_bin = 1.0 / G
    W = (net_xn1 - net_xn0).clamp(min=min_bin)
    H = (net_yn1 - net_yn0).clamp(min=min_bin)
    demand = (W + H) / (W * H)

    # Integer bin indices
    col0 = (net_xn0 * G).long().clamp(0, G - 1)
    col1 = (net_xn1 * G).long().clamp(0, G - 1)
    row0 = (net_yn0 * G).long().clamp(0, G - 1)
    row1 = (net_yn1 * G).long().clamp(0, G - 1)

    # 2-D difference array: add d to rect [row0,row1] × [col0,col1]
    #   diff[r0,c0]+=d  diff[r0,c1+1]-=d  diff[r1+1,c0]-=d  diff[r1+1,c1+1]+=d
    # Then cumsum(0).cumsum(1) gives the accumulated RUDY map.
    S    = G + 1
    diff = torch.zeros(S * S, device=device)
    diff.scatter_add_(0,  row0       * S +  col0,         demand)
    diff.scatter_add_(0,  row0       * S + (col1 + 1),   -demand)
    diff.scatter_add_(0, (row1 + 1)  * S +  col0,        -demand)
    diff.scatter_add_(0, (row1 + 1)  * S + (col1 + 1),    demand)

    rudy = diff.view(S, S)[:G, :G].cumsum(0).cumsum(1)
    mx   = rudy.max()
    return (rudy / mx) if mx > 0 else rudy


def load_ap_lookup(json_paths):
    """
    Load AP JSON files and return {cell_type_name: ap_median}.
    json_paths: list of file paths (the same files used by map_models.py).
    Returns empty dict if no paths given.
    """
    import json
    ap_lookup = {}
    for path in json_paths:
        if not path or not os.path.isfile(path):
            continue
        with open(path) as f:
            for entry in json.load(f):
                ap_lookup[entry['cell']] = float(entry['ap_median'])
    return ap_lookup

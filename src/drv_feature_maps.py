"""
Build DRV prediction feature maps from XPlace's internal placement state.

All operations run on the same GPU device as the placer.
Never reads DEF files — uses PlaceData + GPDatabase directly.

Six output channels (all grid_size × grid_size float32), matching the
fcn_trainer `dynamic_v19_aes2048_dropaoi` checkpoint's --features order
(pin_density,ap,cell_type,rudy,net_count,m3_inv_dist):
  0  pin_density  – Gaussian-smeared pin-count density
  1  ap           – AP-weighted Gaussian density  (needs ap_lookup; falls back to pin_density)
  2  cell_type    – routing-complexity-weighted cell mixture
  3  rudy         – RUDY routing-demand estimate
  4  net_count    – unweighted net-count congestion density
  5  m3_inv_dist  – PDN proximity: 1 / (1 + distance to nearest M3 power stripe edge, in bins)

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
from map_models import _separable_gaussian_map, _cell_category


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
    Return a (6, grid_size, grid_size) float32 feature tensor on device, channel
    order [pin_density, ap, cell_type, rudy, net_count, m3_inv_dist].

    Args:
        mov_node_pos  : (N_mov, 2) current movable-node positions from the Nesterov
                        optimizer — grabbed inside the GP iteration loop.
        data          : PlaceData object from XPlace.
        gpdb          : GPDatabase C++ object; provides node_id2celltype_name()
                        and snet_info_tensor() (for the M3 stripe channel).
        ap_lookup     : dict {cell_type_name: float} ap_median values pre-loaded
                        from the fcn_trainer AP JSON files, or None to fall back
                        to pin-count weighting (same as pin_density channel).
        grid_size     : output resolution — fully independent of XPlace's own
                        density bins (data.num_bin_x/y); must match the
                        checkpoint's training resolution family. The dynamic_v*
                        checkpoints are fully-convolutional and were trained
                        across multiple bucket resolutions (256/512/1024/2048),
                        so any multiple of 16 works.
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

    # ── 4. RUDY + net_count (share the same net bounding-box pass) ────────────
    net_bbox = _net_bboxes(node_pos, data, G)
    rudy      = _accumulate_bbox_map(net_bbox, G, device, weighted=True)
    net_count = _accumulate_bbox_map(net_bbox, G, device, weighted=False)

    # ── 5. m3_inv_dist ─────────────────────────────────────────────────────────
    m3_inv_dist = _build_m3_stripe_map(data, gpdb, G, device)

    # ── Stack + per-channel max-normalise ─────────────────────────────────────
    feat = torch.stack(
        [pin_density, ap_map, cell_type, rudy, net_count, m3_inv_dist], dim=0)  # (6,G,G)
    mx = feat.flatten(1).max(dim=1).values.clamp(min=1e-9).view(6, 1, 1)
    feat = feat / mx

    return feat


def _net_bboxes(node_pos, data, G):
    """
    Per-net normalised bounding boxes (xn0, xn1, yn0, yn1), masked to valid
    signal nets. Shared by RUDY and net_count, which only differ in what
    "demand" value they scatter over each net's bbox.

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

    min_bin = 1.0 / G
    W = (net_xn1 - net_xn0).clamp(min=min_bin)
    H = (net_yn1 - net_yn0).clamp(min=min_bin)
    return net_xn0, net_xn1, net_yn0, net_yn1, W, H


def _accumulate_bbox_map(net_bbox, G, device, weighted):
    """
    Accumulate a per-net scalar uniformly over its bbox via 2-D difference
    array + cumsum. Complexity: O(N_nets) scatter_add + O(G^2) cumsum.

    weighted=True  -> RUDY demand      = (W + H) / (W * H)
    weighted=False -> net_count demand = 1.0  (unweighted congestion density)
    """
    net_xn0, net_xn1, net_yn0, net_yn1, W, H = net_bbox
    demand = (W + H) / (W * H) if weighted else torch.ones_like(W)

    # Integer bin indices
    col0 = (net_xn0 * G).long().clamp(0, G - 1)
    col1 = (net_xn1 * G).long().clamp(0, G - 1)
    row0 = (net_yn0 * G).long().clamp(0, G - 1)
    row1 = (net_yn1 * G).long().clamp(0, G - 1)

    # 2-D difference array: add d to rect [row0,row1] × [col0,col1]
    #   diff[r0,c0]+=d  diff[r0,c1+1]-=d  diff[r1+1,c0]-=d  diff[r1+1,c1+1]+=d
    # Then cumsum(0).cumsum(1) gives the accumulated map.
    S    = G + 1
    diff = torch.zeros(S * S, device=device)
    diff.scatter_add_(0,  row0       * S +  col0,         demand)
    diff.scatter_add_(0,  row0       * S + (col1 + 1),   -demand)
    diff.scatter_add_(0, (row1 + 1)  * S +  col0,        -demand)
    diff.scatter_add_(0, (row1 + 1)  * S + (col1 + 1),    demand)

    # cumsum has no deterministic CUDA implementation; the (G, G) map is small
    # enough that doing the prefix sum on CPU is cheap and keeps results
    # reproducible across runs.
    mat = diff.view(S, S)[:G, :G]
    out = mat.cpu().cumsum(0).cumsum(1).to(device)
    mx  = out.max()
    return (out / mx) if mx > 0 else out


def _build_m3_stripe_map(data, gpdb, G, device):
    """
    PDN proximity map: 1 / (1 + distance_to_nearest_M3_stripe_edge_in_bins).

    Varies only along x (stripes run the full die height): a set of vertical
    bands, highest at/inside a stripe footprint and decaying with horizontal
    distance to the nearest stripe edge. Mirrors M3StripeMapModel in
    fcn_trainer/map_models.py, but reads stripe geometry directly from the
    GPDatabase (gpdb.snet_info_tensor()) instead of parsing a DEF file — same
    source already used by _apply_m3_snet_density_reduction() in initializer.py.
    """
    snet_lpos_all, snet_size_all, snet_layer_all = gpdb.snet_info_tensor()
    snet_lpos_all = snet_lpos_all.to(device)
    snet_size_all = snet_size_all.to(device)
    snet_layer_all = snet_layer_all.to(device)

    m3_mask = snet_layer_all == 2  # M3 is layer index 2
    if m3_mask.sum() == 0:
        return torch.zeros(G, G, device=device)

    # Same shift/scale as node_pos, so stripe geometry lands in the same
    # normalised frame as die_ll/die_ur (see initializer.py's identical
    # transform for the M3 density-blockage pass).
    m3_lpos = (snet_lpos_all[m3_mask] - data.die_shift) / data.die_scale
    m3_size = snet_size_all[m3_mask] / data.die_scale

    die_ll = data.die_ll
    die_w  = data.die_ur[0] - die_ll[0]

    centers_bin = (m3_lpos[:, 0] + m3_size[:, 0] / 2 - die_ll[0]) / die_w * G
    half_w_bin  = (m3_size[:, 0] / 2) / die_w * G

    xs = torch.arange(G, dtype=torch.float32, device=device) + 0.5   # (G,)
    dist = (xs.unsqueeze(1) - centers_bin.unsqueeze(0)).abs() - half_w_bin.unsqueeze(0)  # (G, N_stripe)
    dist = dist.clamp(min=0).min(dim=1).values          # (G,) nearest stripe edge, per column
    inv_dist = 1.0 / (1.0 + dist)
    return inv_dist.unsqueeze(0).expand(G, G).contiguous()


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

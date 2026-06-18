"""
Differentiable DRV feature maps for embedding into XPlace's Nesterov objective.

Gradient flows: DRV_loss → feat (5, G, G) → mov_node_pos

Channels 0-3 (Gaussian-based): exact autograd via _separable_gaussian_map.
Channel 4 (RUDY): differentiable via LSE soft min/max + bilinear scatter + cumsum.

Drop-in replacement for build_drv_feature_maps() — same signature, same output
shape, but every op keeps mov_node_pos in the computation graph.
"""

import os
import sys

import torch

_FCN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fcn_trainer')
if _FCN_DIR not in sys.path:
    sys.path.insert(0, _FCN_DIR)
from map_models import _separable_gaussian_map, _cell_category, _is_aoi_oai


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_drv_feature_maps_diff(
    mov_node_pos,
    data,
    gpdb,
    ap_lookup=None,
    grid_size=128,
    smooth_factor=1.5,
    min_sigma_bins=1.0,
    rudy_lse_beta=30.0,
):
    """
    Differentiable version of build_drv_feature_maps().

    mov_node_pos must be part of the computation graph (requires_grad or
    derived from v_k in the Nesterov loop). Gradients flow back to it.

    Extra arg:
        rudy_lse_beta: LSE temperature for soft min/max in RUDY bounding boxes.
                       Higher → sharper approximation of true min/max, smaller
                       gradient signal for interior pins.
    """
    device = mov_node_pos.device
    G = grid_size

    # ── Build node_pos without in-place assignment (preserves autograd graph) ─
    mov_lhs, mov_rhs = data.movable_index
    N = data.num_nodes

    pieces = []
    if mov_lhs > 0:
        pieces.append(data.node_pos[:mov_lhs].to(device))
    pieces.append(mov_node_pos[mov_lhs:mov_rhs])
    if mov_rhs < N:
        pieces.append(data.node_pos[mov_rhs:N].to(device))
    node_pos = torch.cat(pieces, dim=0)  # (N, 2), grad flows through movable slice

    # ── Normalise to [0, 1] ───────────────────────────────────────────────────
    die_ll = data.die_ll
    die_ur = data.die_ur
    die_wh = die_ur - die_ll

    real_size = data.node_size[:N]
    cx_norm = (node_pos[:, 0] - die_ll[0]) / die_wh[0]   # (N,), has grad
    cy_norm = (node_pos[:, 1] - die_ll[1]) / die_wh[1]

    min_sig = min_sigma_bins / G
    # sigma depends only on cell size (constant), no need to differentiate
    sx = (real_size[:, 0] / (2.0 * die_wh[0]) * smooth_factor).clamp(min=min_sig).detach()
    sy = (real_size[:, 1] / (2.0 * die_wh[1]) * smooth_factor).clamp(min=min_sig).detach()

    bins = (torch.arange(G, dtype=torch.float32, device=device) + 0.5) / G
    valid = (real_size[:, 0] > 1e-6) & (real_size[:, 1] > 1e-6)

    # pin_w is a constant weight (doesn't depend on position)
    pin_w = data.node_to_num_pins[:N, 0].float().detach()
    pin_w = pin_w / (pin_w.sum() + 1e-9)

    def _gmap(weights):
        w = weights[valid].detach()           # weights are always constant
        w = w / (w.sum() + 1e-9)
        m = _separable_gaussian_map(
            cx_norm[valid], cy_norm[valid],   # these carry grad
            sx[valid], sy[valid], w, bins)
        mx = m.max().detach()                 # normalise with stop-grad on scale
        return (m / mx.clamp(min=1e-9)) if mx > 0 else m

    # ── Channel 0: pin_density ────────────────────────────────────────────────
    pin_density = _gmap(pin_w)

    # ── Channel 1: ap ─────────────────────────────────────────────────────────
    celltype_full = gpdb.node_id2celltype_name()
    macro_names = [s.split('/')[-1] for s in celltype_full[:N]]

    if ap_lookup:
        default_ap = float(sum(ap_lookup.values()) / len(ap_lookup))
        ap_vals = torch.tensor(
            [ap_lookup.get(m, default_ap) for m in macro_names],
            dtype=torch.float32, device=device,
        ).detach()
        ap_vals = ap_vals / (ap_vals.sum() + 1e-9)
        ap_map = _gmap(ap_vals)
    else:
        ap_map = pin_density

    # ── Channel 2: cell_type ──────────────────────────────────────────────────
    cats = torch.tensor(
        [_cell_category(m) for m in macro_names], dtype=torch.long, device=device)
    _CAT_W = [1.0, 2.0, 3.0, 5.0]
    M = torch.zeros(G, G, device=device)
    for g, w_cat in enumerate(_CAT_W):
        mask = valid & (cats == g)
        if not mask.any():
            continue
        w = pin_w[mask].detach()
        w = w / (w.sum() + 1e-9)
        D_g = _separable_gaussian_map(
            cx_norm[mask], cy_norm[mask], sx[mask], sy[mask], w, bins)
        M = M + w_cat * D_g
    mx = M.max().detach()
    cell_type = (M / mx.clamp(min=1e-9)) if mx > 0 else M

    # ── Channel 3: aoi ────────────────────────────────────────────────────────
    is_aoi = torch.tensor(
        [_is_aoi_oai(m) for m in macro_names], dtype=torch.bool, device=device)
    aoi_mask = valid & is_aoi
    if aoi_mask.any():
        w = pin_w[aoi_mask].detach()
        w = w / (w.sum() + 1e-9)
        aoi_raw = _separable_gaussian_map(
            cx_norm[aoi_mask], cy_norm[aoi_mask],
            sx[aoi_mask], sy[aoi_mask], w, bins)
        mx = aoi_raw.max().detach()
        aoi = (aoi_raw / mx.clamp(min=1e-9)) if mx > 0 else aoi_raw
    else:
        aoi = torch.zeros(G, G, device=device)

    # ── Channel 4: RUDY (differentiable) ─────────────────────────────────────
    rudy = _build_rudy_diff(node_pos, data, G, device, lse_beta=rudy_lse_beta)

    # ── Stack + per-channel max-normalise ─────────────────────────────────────
    feat = torch.stack([pin_density, ap_map, cell_type, aoi, rudy], dim=0)  # (5,G,G)
    mx = feat.flatten(1).max(dim=1).values.clamp(min=1e-9).detach().view(5, 1, 1)
    return feat / mx


# ──────────────────────────────────────────────────────────────────────────────
# Differentiable RUDY
# ──────────────────────────────────────────────────────────────────────────────

def _lse_scatter_max(values, index, num_targets, beta, device):
    """
    Per-group soft max via log-sum-exp: (1/β) log Σ_i exp(β x_i).

    Gradient flows to all group members (not just the argmax), with weight
    proportional to exp(β x_i) / Σ exp(β x_j).  Higher β → sharper winner.

    Uses a detached per-group pivot for numerical stability (straight-through
    on the pivot selection — pivot has no effect on the gradient direction,
    only on numerical conditioning).
    """
    # Stable pivot: hard max per group (detached — only used for numerics)
    pivot = torch.full((num_targets,), -1e9, dtype=values.dtype, device=device)
    pivot.scatter_reduce_(0, index, values, reduce='amax', include_self=True)
    pivot = pivot.detach()

    shifted = torch.exp(beta * (values - pivot[index]))
    sum_exp = torch.zeros(num_targets, dtype=values.dtype, device=device)
    sum_exp.scatter_add_(0, index, shifted)
    return pivot + torch.log(sum_exp.clamp(min=1e-9)) / beta


def _build_rudy_diff(node_pos, data, G, device, lse_beta=30.0):
    """
    Differentiable RUDY routing-demand map.

    Gradient path:
      cumsum ← diff array ← bilinear scatter
              ← demand=(W+H)/(W*H), bbox corners
              ← LSE soft min/max
              ← pin positions
              ← node_pos (which includes movable nodes with grad)
    """
    nid = data.pin_id2node_id
    # pin positions — grad flows here from node_pos
    pin_x = node_pos[nid, 0] + data.pin_rel_cpos[:, 0]
    pin_y = node_pos[nid, 1] + data.pin_rel_cpos[:, 1]

    die_ll = data.die_ll
    die_ur = data.die_ur
    pin_xn = ((pin_x - die_ll[0]) / (die_ur[0] - die_ll[0])).clamp(0.0, 1.0)
    pin_yn = ((pin_y - die_ll[1]) / (die_ur[1] - die_ll[1])).clamp(0.0, 1.0)

    pin_net = data.pin_id2net_id
    num_nets = data.num_nets

    # Soft bounding box via LSE — all pins in a net contribute to gradient
    net_xn1 = _lse_scatter_max( pin_xn, pin_net, num_nets, lse_beta, device)
    net_xn0 = -_lse_scatter_max(-pin_xn, pin_net, num_nets, lse_beta, device)
    net_yn1 = _lse_scatter_max( pin_yn, pin_net, num_nets, lse_beta, device)
    net_yn0 = -_lse_scatter_max(-pin_yn, pin_net, num_nets, lse_beta, device)

    mask = data.net_mask
    net_xn0, net_xn1 = net_xn0[mask], net_xn1[mask]
    net_yn0, net_yn1 = net_yn0[mask], net_yn1[mask]

    min_bin = 1.0 / G
    W = (net_xn1 - net_xn0).clamp(min=min_bin)
    H = (net_yn1 - net_yn0).clamp(min=min_bin)
    demand = (W + H) / (W * H)          # RUDY demand, grad flows here

    # Difference-array corners in continuous bin coordinates.
    # Corresponds to the 4 scatter positions in the original integer RUDY:
    #   (+d) at (row0, col0)       (-d) at (row0,   col1+1)
    #   (-d) at (row1+1, col0)     (+d) at (row1+1, col1+1)
    r0_f = net_yn0 * G
    r1_f = net_yn1 * G + 1.0           # "+1" is the "past end" sentinel offset
    c0_f = net_xn0 * G
    c1_f = net_xn1 * G + 1.0

    # Use a slightly larger buffer (G+2)² so bilinear at position G+1 doesn't OOB
    S = G + 2
    diff = (
        _bilinear_scatter(r0_f, c0_f,  demand, S, device) +
        _bilinear_scatter(r0_f, c1_f, -demand, S, device) +
        _bilinear_scatter(r1_f, c0_f, -demand, S, device) +
        _bilinear_scatter(r1_f, c1_f,  demand, S, device)
    )

    # cumsum is natively differentiable in PyTorch
    rudy = diff.view(S, S)[:G, :G].cumsum(0).cumsum(1)
    mx = rudy.max().detach()
    return (rudy / mx.clamp(min=1e-9)) if mx > 0 else rudy


def _bilinear_scatter(row_f, col_f, values, S, device):
    """
    Scatter `values` at fractional positions (row_f, col_f) onto a flat (S²,)
    tensor using bilinear weights. Returns a new tensor (out-of-place).

    Gradient flows through `values` and the fractional parts of row_f / col_f.
    The discrete bin index (integer part) does NOT carry gradient — this is the
    standard bilinear interpolation approximation used in spatial transformers.
    """
    r0 = row_f.long().clamp(0, S - 2)
    c0 = col_f.long().clamp(0, S - 2)
    r1 = r0 + 1          # already clamped to S-1 via r0 <= S-2
    c1 = c0 + 1

    # Fractional offsets — these carry the gradient w.r.t. row_f / col_f
    ar = (row_f - r0.float()).clamp(0.0, 1.0)
    ac = (col_f - c0.float()).clamp(0.0, 1.0)

    def idx(r, c):
        return (r * S + c).clamp(0, S * S - 1)

    # Each scatter_add_ operates on a fresh zero tensor → safe for autograd
    out = torch.zeros(S * S, dtype=values.dtype, device=device)
    out.scatter_add_(0, idx(r0, c0), values * (1 - ar) * (1 - ac))
    out.scatter_add_(0, idx(r0, c1), values * (1 - ar) * ac)
    out.scatter_add_(0, idx(r1, c0), values * ar * (1 - ac))
    out.scatter_add_(0, idx(r1, c1), values * ar * ac)
    return out

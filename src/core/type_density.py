import torch
import torch.nn.functional as F
from cpp_to_py import density_map_cuda


class TypeDensity:
    """Strap repulsion for the pin-access critical cell types (--type_density_cells).

    Only a few cell types fail next to a power strap: on the NS3K M3-stripe PDN
    (UTIL 0.8, routed), 99.5-100% of the signal violations lie within 3 sites of
    a stripe, and per near-stripe instance OAI221 / AOI221 fail 59-70%, OAI22 /
    AOI22 31-45%, most other types under 3%. These are the types whose
    track-aware pin scarcity stands out (tool/pin_scarcity_track.py, z >= 1).

    The listed cells get a penalty of their own, next to the usual density term
    of all cells: their overlap with a potential phi, the mask of the bins that
    hold a strap of --type_density_layers blurred over one bin each side in x,

        E = sum_i sum_b overlap(cell i, bin b) phi(b),

    so a listed cell is pushed out of the strap bins (in x) while it overlaps
    them and feels nothing elsewhere; the other cells may fill those bins. The
    gradient is scaled by density_weight x --type_density_weight, like the
    density term. (A density system of their own, with a virtual blockage over
    the strap bins, also spread the listed cells evenly outside the straps: on
    jpeg, whose 10.8k XNOR2 sit in clusters, that cost +26..31% HPWL after
    legalization.)
    """

    def __init__(self, data, args, density_map_layer, strap_map, logger, mov_node_size):
        mask = (strap_map > 0).float()
        names = set(c.strip() for c in args.type_density_cells.replace(',', ' ').split())
        mov_lhs, mov_rhs = data.movable_index
        types = data.node_id2celltype_name
        idx = [i - mov_lhs for i in range(mov_lhs, mov_rhs) if types[i].split('/')[-1] in names]
        dev = strap_map.device
        self.idx = torch.tensor(idx, dtype=torch.long, device=dev)
        area = torch.prod(mov_node_size[self.idx], 1) if idx else mov_node_size.new_zeros(0)
        _, order = torch.sort(area, descending=True)
        self.sorted_map = order.contiguous()
        # phi: mask blurred in x; field -dphi/dx by central differences (per bin)
        k = torch.tensor([0.25, 0.5, 0.25], device=dev).view(1, 1, 3)
        phi = F.conv1d(mask.t().unsqueeze(1), k, padding=1).squeeze(1).t()
        fx = torch.zeros_like(phi)
        fx[1:-1] = -(phi[2:] - phi[:-2]) / 2
        self.grad_mat = torch.vstack((fx.unsqueeze(0), torch.zeros_like(fx).unsqueeze(0))).contiguous()
        self.layer = density_map_layer
        self.weight = args.type_density_weight
        found = sorted(set(types[i + mov_lhs].split('/')[-1] for i in idx))
        logger.info("Type-aware PDN density (local): %d cells of %s, %d of %d bins blocked, weight %.2f" % (
            len(idx), ','.join(found), int(mask.sum()), mask.numel(), self.weight))

    def grad(self, mov_node_pos, mov_node_size):
        layer = self.layer
        n = self.idx.numel()
        g_all = torch.zeros_like(mov_node_pos)
        if n == 0:
            return g_all
        node_weight = layer.get_cache_var(mov_node_pos, mov_node_size, None)
        pos, size = mov_node_pos[self.idx].contiguous(), mov_node_size[self.idx].contiguous()
        info = size.new_empty((n, 5))
        info = density_map_cuda.pre_normalize(
            pos, size, node_weight[self.idx].contiguous(), layer.expand_ratio[self.idx].contiguous(),
            layer.unit_len, info, layer.num_bin_x, layer.num_bin_y, n)
        g = info.new_zeros((n, 2))
        g = density_map_cuda.backward(info, self.grad_mat, self.sorted_map, g, -1.0,
                                      layer.num_bin_x, layer.num_bin_y, n, layer.deterministic)
        g_all[self.idx] = g
        return g_all

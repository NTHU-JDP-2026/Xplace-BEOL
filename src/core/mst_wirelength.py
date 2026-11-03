import numba as nb
import numpy as np
import torch


@nb.njit(parallel=True, cache=True)
def _mst_edges(x, y, start, end, edge_start):
    """L1 minimum spanning tree (Prim, O(n^2)) of each net's pins; net k's pins are
    x/y[start[k]:end[k]], its n - 1 edges go to rows edge_start[k]... of the result
    as (index of pin a, index of pin b), indices into x/y."""
    edges = np.empty((edge_start[-1], 2), dtype=np.int64)
    for k in nb.prange(len(start)):
        s, n = start[k], end[k] - start[k]
        dist = np.empty(n, dtype=np.float32)
        parent = np.zeros(n, dtype=np.int64)
        used = np.zeros(n, dtype=np.bool_)
        used[0] = True
        for j in range(n):
            dist[j] = abs(x[s + j] - x[s]) + abs(y[s + j] - y[s])
        e = edge_start[k]
        for _ in range(n - 1):
            best, bj = np.inf, -1
            for j in range(n):
                if not used[j] and dist[j] < best:
                    best, bj = dist[j], j
            used[bj] = True
            edges[e, 0] = s + parent[bj]
            edges[e, 1] = s + bj
            e += 1
            for j in range(n):
                if not used[j]:
                    d = abs(x[s + j] - x[s + bj]) + abs(y[s + j] - y[s + bj])
                    if d < dist[j]:
                        dist[j] = d
                        parent[j] = bj
    return edges


class MSTNetModel:
    """MST decomposition of the high-degree nets (--mst_degree).

    The routed length of a net of thousands of sinks (reset, enables) is its
    Steiner tree, ~ #sinks x the distance between neighboring sinks, while its
    HPWL is ~ the die half perimeter whatever the sinks do (NS3K jpeg, same
    netlist, CTS only: the sinks of rst were 0.23 um from the nearest other rst
    sink after Innovus, 0.34 um after Xplace, tree +38%). WA-HPWL only acts on the
    outermost pins, and a pull toward the net center (a star model, tried
    and dropped: the big nets' trees -0.7%, the other nets +2..14%) does not
    bring neighboring sinks together. Here every net above the degree is
    replaced by the edges of the L1 minimum spanning tree of its pins (~ RSMT /
    0.9), recomputed every --mst_update iterations, each edge a 2-pin net:

        WL_edge = mst_weight * sum_c d_c * tanh(d_c / (2 g_c)) * s_c

    the WA wirelength of two pins (d = pin distance, g = wa_coeff, s =
    hpwl_scale), so the smoothing matches the WA-HPWL of the other nets.

    Only from wa_coeff <= --mst_start_gamma on (the nets keep the plain WA-HPWL
    model before): from the start of GP the tree edges, springs between the
    sinks while all cells are still near the center, packed the flops of the
    big control nets together and pushed their logic out (jpeg: tree length of
    the nets of over 100 pins -37%, but the HPWL of the other nets +90%).
    """

    def __init__(self, data, args):
        degree = data.net_to_num_pins
        mst = degree > args.mst_degree
        self.wa_net_mask = torch.logical_and(data.net_mask, ~mst)
        net_ids = torch.nonzero(mst).squeeze(1)
        self.num_nets = net_ids.numel()
        self.update = max(1, args.mst_update)
        self.weight = args.mst_weight
        self.start_gamma = args.mst_start_gamma
        self.edges = None
        self.last_iter = None
        if self.num_nets == 0:
            return
        end = data.hyperedge_list_end.long()
        start = torch.cat([end.new_zeros(1), end[:-1]])
        s, e = start[net_ids], end[net_ids]
        n = e - s
        seg = torch.repeat_interleave(torch.arange(self.num_nets, device=n.device), n)
        offs = torch.arange(int(n.sum()), device=n.device) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
        self.pins = data.hyperedge_list.long()[s[seg] + offs]
        cs = torch.cumsum(n, 0)
        self.start = (cs - n).cpu().numpy().astype(np.int64)
        self.end = cs.cpu().numpy().astype(np.int64)
        self.edge_start = np.concatenate([[0], np.cumsum(self.end - self.start - 1)]).astype(np.int64)

    def _pin_pos(self, conn_node_pos, data):
        nodes = data.pin_id2node_id.long()[self.pins]
        return nodes, conn_node_pos[nodes] + data.pin_rel_cpos[self.pins]

    def active(self, gamma):
        return self.num_nets > 0 and gamma <= self.start_gamma

    def loss_grad(self, conn_node_pos, data, gamma, iteration):
        grad = torch.zeros_like(conn_node_pos)
        if self.num_nets == 0:
            return conn_node_pos.new_zeros(()), grad
        nodes, pos = self._pin_pos(conn_node_pos, data)
        if self.edges is None or iteration is None or self.last_iter is None or iteration - self.last_iter >= self.update or iteration < self.last_iter:
            p = (pos * data.hpwl_scale.view(1, 2)).detach().cpu().numpy().astype(np.float32)
            e = _mst_edges(p[:, 0].copy(), p[:, 1].copy(), self.start, self.end, self.edge_start)
            self.edges = torch.from_numpy(e).to(pos.device)
            self.last_iter = iteration
        a, b = self.edges[:, 0], self.edges[:, 1]
        scale = data.hpwl_scale.view(1, 2)
        d = (pos[a] - pos[b]) * scale
        g = gamma * scale
        u = d / (2 * g)
        t = torch.tanh(u)
        loss = self.weight * torch.sum(d * t)
        # d/dd [d tanh(d / 2g)] = tanh(u) + u (1 - tanh(u)^2)
        gd = self.weight * (t + u * (1 - t * t)) * scale
        grad.index_add_(0, nodes[a], gd)
        grad.index_add_(0, nodes[b], -gd)
        return loss, grad


def get_mst_model(data, args):
    if getattr(args, "mst_degree", 0) <= 0:
        return None
    model = getattr(data, "mst_model", None)
    if model is None:
        model = MSTNetModel(data, args)
        data.mst_model = model
    return model

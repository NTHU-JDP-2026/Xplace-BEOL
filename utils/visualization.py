from typing import List, Tuple
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

import logging

matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.INFO)


def draw_fig_with_cairo(
    mov_node_pos,
    mov_node_size,
    fix_node_pos,
    fix_node_size,
    filler_node_pos,
    filler_node_size,
    data,
    info,
    args,
    base_size=2048,
):
    import cairocffi as cairo

    iteration, hpwl, design_name = info
    filename = "%s_iter%s.png" % (design_name, iteration)
    res_root = os.path.join(args.result_dir, args.exp_id)
    png_path = os.path.join(res_root, args.eval_dir, filename)
    if not os.path.exists(os.path.dirname(png_path)):
        os.makedirs(os.path.dirname(png_path))

    lx, ly, hx, hy = data.ori_die_lx, data.ori_die_ly, data.ori_die_hx, data.ori_die_hy
    WIDTH = base_size
    HEIGHT = int(WIDTH * (hx - lx) / (hy - ly))
    num_bin_x = data.num_bin_x
    num_bin_y = data.num_bin_y
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surface)
    # Scale Image
    ratio0, ratio1 = WIDTH / (hx - lx), HEIGHT / (hy - ly)
    ctx.translate(-lx * ratio0, HEIGHT + ly * ratio1)
    ctx.scale(ratio0, -ratio1)
    # White Background
    ctx.rectangle(lx, ly, hx - lx, hy - ly)
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.fill()
    # Bins / Grids
    ctx.set_line_width(0.0005)
    ctx.set_source_rgb(0.3, 0.3, 0.3)
    for i in range(1, num_bin_x):
        ctx.move_to(i * (hx - lx) / num_bin_x + lx, ly)
        ctx.line_to(i * (hx - lx) / num_bin_x + lx, hy)
        ctx.stroke()
    for i in range(1, num_bin_y):
        ctx.move_to(lx, i * (hy - ly) / num_bin_y + ly)
        ctx.line_to(hx, i * (hy - ly) / num_bin_y + ly)
        ctx.stroke()
    # Movable Nodes
    if mov_node_pos is not None and mov_node_size is not None:
        mov_node_pos = mov_node_pos.cpu()
        mov_node_size = mov_node_size.cpu()
        for i in range(mov_node_pos.shape[0]):
            pos_x = round(mov_node_pos[i][0].item() * (hx - lx) + lx)
            pos_y = round(mov_node_pos[i][1].item() * (hy - ly) + ly)
            size_x = round(mov_node_size[i][0].item() * (hx - lx))
            size_y = round(mov_node_size[i][1].item() * (hy - ly))
            ctx.rectangle(pos_x - size_x / 2, pos_y - size_y / 2, size_x, size_y)
            ctx.set_source_rgba(0.475, 0.706, 0.718, 0.8)
            ctx.fill()
    # Fixed Nodes
    if fix_node_pos is not None and fix_node_size is not None:
        fix_node_pos = fix_node_pos.cpu()
        fix_node_size = fix_node_size.cpu()
        for i in range(fix_node_pos.shape[0]):
            pos_x = round(fix_node_pos[i][0].item() * (hx - lx) + lx)
            pos_y = round(fix_node_pos[i][1].item() * (hy - ly) + ly)
            size_x = round(fix_node_size[i][0].item() * (hx - lx))
            size_y = round(fix_node_size[i][1].item() * (hy - ly))
            ctx.rectangle(pos_x - size_x / 2, pos_y - size_y / 2, size_x, size_y)
            ctx.set_source_rgba(0.878, 0.365, 0.365, 0.8)
            ctx.fill()
    # Filler Nodes
    if filler_node_pos is not None and filler_node_size is not None:
        filler_node_pos = filler_node_pos.cpu()
        filler_node_size = filler_node_size.cpu()
        for i in range(filler_node_pos.shape[0]):
            pos_x = round(filler_node_pos[i][0].item() * (hx - lx) + lx)
            pos_y = round(filler_node_pos[i][1].item() * (hy - ly) + ly)
            size_x = round(filler_node_size[i][0].item() * (hx - lx))
            size_y = round(filler_node_size[i][1].item() * (hy - ly))
            ctx.rectangle(pos_x - size_x / 2, pos_y - size_y / 2, size_x, size_y)
            ctx.set_source_rgba(0.082, 0.176, 0.208, 0.33)
            ctx.fill()
    surface.write_to_png(png_path)


def draw_fig_with_cairo_cpp(node_pos, node_size, data, info, args, base_size=2048):
    from cpp_to_py import draw_placement

    die_info = tuple(data.__ori_die_info__.tolist())
    scaleX, scaleY = data.die_scale[0].cpu(), data.die_scale[1].cpu()
    shiftX, shiftY = data.die_shift[0].cpu(), data.die_shift[1].cpu()
    lx, hx, ly, hy = die_info

    node_pos_x: List[float] = (node_pos.cpu()[:, 0] * scaleX + shiftX).tolist()
    node_pos_y: List[float] = (node_pos.cpu()[:, 1] * scaleY + shiftY).tolist()
    node_size_x: List[float] = (node_size.cpu()[:, 0] * scaleX).tolist()
    node_size_y: List[float] = (node_size.cpu()[:, 1] * scaleY).tolist()
    node_name: List[str] = ["%d" % i for i in range(node_pos.shape[0])]

    iteration, hpwl, design_name = info
    filename = "%s_iter%s.png" % (design_name, iteration)
    res_root = os.path.join(args.result_dir, args.exp_id)
    png_path: str = os.path.join(res_root, args.eval_dir, filename)
    if not os.path.exists(os.path.dirname(png_path)):
        os.makedirs(os.path.dirname(png_path))

    site_info = (data.site_width, data.site_height)
    bin_size_info = (
        round(1 / data.num_bin_x * (hx - lx)),
        round(1 / data.num_bin_y * (hy - ly)),
    )
    node_type_indices = data.node_type_indices
    ele_type_to_rgba_vec: List[Tuple[str, float, float, float, float]] = [
        ("Bin", 0.1, 0.1, 0.1, 1.0),
        ("Mov", 0.475, 0.706, 0.718, 0.8),
        ("Filler", 0.8, 0.8, 0.8, 0.8),
        ("Buffer", 0.65, 0.08, 0.9, 0.8),
        ("FF", 0.65, 0.9, 0.08, 0.7),
    ]
    
    node_special_type: List[int] = (data.node_special_type.cpu()).tolist()
    width = base_size
    height = round(width * (hy - ly) / (hx - lx))
    draw_contents: List[str] = ["Nodes", "NodesText"]

    status = draw_placement.draw(
        node_pos_x,
        node_pos_y,
        node_size_x,
        node_size_y,
        node_name,
        die_info,
        site_info,
        bin_size_info,
        node_type_indices,
        ele_type_to_rgba_vec,
        node_special_type,
        png_path,
        width,
        height,
        draw_contents,
    )


def visualize_electronic_variables(density_map, potential_map, force_map, info, args):
    import cv2
    iteration, design_name = info
    M, N = density_map.shape

    def get_png_path(filename):
        res_root = os.path.join(args.result_dir, args.exp_id)
        png_path = os.path.join(res_root, args.eval_dir, filename)
        if not os.path.exists(os.path.dirname(png_path)):
            os.makedirs(os.path.dirname(png_path))
        return png_path

    # 1) Visualize density_map
    filename = "%s_iter%s_density.png" % (design_name, iteration)
    png_path = get_png_path(filename)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(density_map.cpu().numpy(), cmap="YlGnBu")
    fig.colorbar(im, ax=ax)
    ax.title.set_text("Density Map")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    # 2) Visualize potential_map
    filename = "%s_iter%s_potential.png" % (design_name, iteration)
    png_path = get_png_path(filename)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(potential_map.cpu().numpy(), cmap="YlGnBu")
    fig.colorbar(im, ax=ax)
    ax.title.set_text("Potential Map")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    # 3) Visualize force_map
    filename = "%s_iter%s_force.png" % (design_name, iteration)
    png_path = get_png_path(filename)
    # 3.1) Init background image
    GRID_SIZE = 100
    img = np.ones((M * GRID_SIZE, N * GRID_SIZE, 3)) * 255

    # 3.2) Draw grid line
    for i in range(0, M * GRID_SIZE - 1, GRID_SIZE):
        cv2.line(img, (i, 0), (i, N * GRID_SIZE), (0, 0, 0), 1, 1)
    for j in range(0, N * GRID_SIZE - 1, GRID_SIZE):
        cv2.line(img, (0, j), (M * GRID_SIZE, j), (0, 0, 0), 1, 1)

    # 3.3) Normalize force
    max_force = torch.sum(torch.pow(force_map, 2), axis=0).sqrt().max().item()
    force_map = (force_map / max_force).cpu().numpy()

    # 3.4) Draw force arrows
    for i in range(0, M, 1):
        centre_x = i * GRID_SIZE + GRID_SIZE / 2
        for j in range(0, N, 1):
            centre_y = j * GRID_SIZE + GRID_SIZE / 2
            cv2.arrowedLine(
                img,
                (
                    int(centre_x - force_map[0][i][j] * GRID_SIZE / 2),
                    int(centre_y - force_map[1][i][j] * GRID_SIZE / 2),
                ),
                (
                    int(centre_x + force_map[0][i][j] * GRID_SIZE / 2),
                    int(centre_y + force_map[1][i][j] * GRID_SIZE / 2),
                ),
                color=(230, 216, 173),
                thickness=10,
                tipLength=0.3,
            )

    cv2.imwrite(png_path, img)


def draw_placement_with_pdn(node_pos, node_size, gpdb, data, info, args, base_size=2048):
    """Draw post-GP placement with PDN power stripes and placement rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection

    iteration, hpwl, design_name = info
    filename = "%s_iter%s_pdn.png" % (design_name, iteration)
    res_root = os.path.join(args.result_dir, args.exp_id)
    png_path = os.path.join(res_root, args.eval_dir, filename)
    if not os.path.exists(os.path.dirname(png_path)):
        os.makedirs(os.path.dirname(png_path))

    # Die bounds: __ori_die_info__ is [lx, hx, ly, hy]
    lx, hx, ly, hy = data.__ori_die_info__
    die_w, die_h = hx - lx, hy - ly

    # Convert normalized node coords back to DB units
    scaleX = data.die_scale[0].cpu().item()
    scaleY = data.die_scale[1].cpu().item()
    shiftX = data.die_shift[0].cpu().item()
    shiftY = data.die_shift[1].cpu().item()
    node_pos_cpu = node_pos.detach().cpu()
    node_size_cpu = node_size.detach().cpu()
    pos_x = (node_pos_cpu[:, 0] * scaleX + shiftX).numpy()
    pos_y = (node_pos_cpu[:, 1] * scaleY + shiftY).numpy()
    sz_x = (node_size_cpu[:, 0] * scaleX).numpy()
    sz_y = (node_size_cpu[:, 1] * scaleY).numpy()

    fig_w = base_size / 100.0
    fig_h = fig_w * die_h / die_w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.set_xlim(lx, hx)
    ax.set_ylim(ly, hy)
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    # --- Placement rows (alternating gray bands) ---
    try:
        coreLX, coreHX, coreLY, coreHY = gpdb.coreInfo()
        site_h = gpdb.siteHeight()
        if site_h > 0:
            num_rows = int(round((coreHY - coreLY) / site_h))
            row_w = coreHX - coreLX
            even_patches, odd_patches = [], []
            for i in range(num_rows):
                ry = coreLY + i * site_h
                rect = mpatches.Rectangle((coreLX, ry), row_w, site_h)
                (even_patches if i % 2 == 0 else odd_patches).append(rect)
            for patches, fc in ((even_patches, "#f2f2f2"), (odd_patches, "#e6e6e6")):
                if patches:
                    ax.add_collection(PatchCollection(patches, facecolor=fc, edgecolor="none", zorder=1))
    except Exception:
        pass

    # --- PDN power stripes, colored by metal layer ---
    # Layer color table: (facecolor, alpha)
    _layer_style = [
        ("#E55C5C", 0.55),  # M1  red
        ("#F5A623", 0.55),  # M2  orange
        ("#4A90D9", 0.60),  # M3  blue
        ("#7ED321", 0.60),  # M4  green
        ("#9B59B6", 0.60),  # M5  purple
        ("#F0E442", 0.60),  # M6  yellow
        ("#1ABC9C", 0.60),  # M7  teal
        ("#E91E8C", 0.60),  # M8  pink
    ]
    try:
        snet_tensors = gpdb.snet_info_tensor()
        if len(snet_tensors) == 3 and snet_tensors[0].numel() > 0:
            snet_lpos_t, snet_sz_t, snet_layer_t = snet_tensors
            snet_lpos_np = snet_lpos_t.numpy()
            snet_sz_np = snet_sz_t.numpy()
            snet_layer_np = snet_layer_t.numpy()
            for layer_idx in np.unique(snet_layer_np):
                mask = snet_layer_np == layer_idx
                patches = [
                    mpatches.Rectangle(
                        (snet_lpos_np[i, 0], snet_lpos_np[i, 1]),
                        snet_sz_np[i, 0], snet_sz_np[i, 1],
                    )
                    for i in np.where(mask)[0]
                ]
                fc, alpha = _layer_style[int(layer_idx) % len(_layer_style)]
                ax.add_collection(
                    PatchCollection(patches, facecolor=fc, alpha=alpha, edgecolor="none", zorder=3)
                )
    except Exception:
        pass

    # --- Cells ---
    n = node_pos_cpu.shape[0]
    mov_lhs, mov_rhs = data.movable_index
    fix_lhs, fix_rhs = data.fixed_index

    def _cell_patches(start, end):
        patches = []
        for i in range(start, min(end, n)):
            patches.append(
                mpatches.Rectangle(
                    (pos_x[i] - sz_x[i] / 2, pos_y[i] - sz_y[i] / 2), sz_x[i], sz_y[i]
                )
            )
        return patches

    mov_patches = _cell_patches(mov_lhs, mov_rhs)
    if mov_patches:
        ax.add_collection(
            PatchCollection(mov_patches, facecolor="#79B4BA", alpha=0.75, edgecolor="none", zorder=4)
        )

    fix_patches = _cell_patches(fix_lhs, fix_rhs)
    if fix_patches:
        ax.add_collection(
            PatchCollection(fix_patches, facecolor="#E05D5D", alpha=0.85, edgecolor="none", zorder=4)
        )

    # Die outline
    ax.add_patch(
        mpatches.Rectangle((lx, ly), die_w, die_h, fill=False, edgecolor="black", linewidth=1.5, zorder=5)
    )

    ax.set_title("%s  iter%s  HPWL=%.2E" % (design_name, iteration, hpwl), fontsize=10)
    ax.set_xlabel("X (DB units)")
    ax.set_ylabel("Y (DB units)")
    legend_handles = [
        mpatches.Patch(facecolor="#f2f2f2", edgecolor="gray", label="Rows"),
        mpatches.Patch(facecolor="#79B4BA", label="Movable cells"),
        mpatches.Patch(facecolor="#E05D5D", label="Fixed cells"),
    ]
    for li, (fc, _) in enumerate(_layer_style[:4]):
        legend_handles.append(mpatches.Patch(facecolor=fc, label="PDN M%d" % (li + 1)))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    logging.getLogger(__name__).info("PDN placement figure saved to %s" % png_path)


def draw_grad_abs_mean(
    wl_grads, density_grads, iterations, info, args,
):
    iteration, design_name = info
    filename = "%s_iter%s_grad_magnitude_mean.png" % (design_name, iteration)
    res_root = os.path.join(args.result_dir, args.exp_id)
    png_path = os.path.join(res_root, args.eval_dir, filename)
    if not os.path.exists(os.path.dirname(png_path)):
        os.makedirs(os.path.dirname(png_path))

    colors = ["tab:blue", "tab:red"]
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("iterations")
    ax1.set_ylabel("Wirelength Gradient Magnitude", color=colors[0])
    ax1.plot(iterations, wl_grads, color=colors[0])
    ax1.tick_params(axis="y", labelcolor=colors[0])

    ax2 = ax1.twinx()

    ax2.set_ylabel("Density Gradient Magnitude", color=colors[1])
    ax2.plot(iterations, density_grads, color=colors[1])
    ax2.tick_params(axis="y", labelcolor=colors[1])

    plt.title("Gradient Magnitude Mean")
    fig.tight_layout()
    plt.savefig(png_path)
    plt.close()

"""
DRV map predictor: wraps a trained fcn_trainer U-Net checkpoint for
use inside the XPlace global-placement loop.

Usage (inside run_placement_nesterov.py):
    predictor = DRVPredictor(args.drv_checkpoint, device, args.result_dir)
    # ... inside GP loop ...
    feat = build_drv_feature_maps(mov_node_pos, data, gpdb, predictor.ap_lookup)
    drv_pred = predictor.predict(feat)
"""

import os
import sys

import torch

_FCN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fcn_trainer')
if _FCN_DIR not in sys.path:
    sys.path.insert(0, _FCN_DIR)
from train import UNet
from map_models import pick_grid_size


def resolve_drv_grid_size(data, gpdb, target_um_per_bin, candidates=(256, 512, 1024, 2048)):
    """
    Pick a DRV feature-map grid_size from this design's physical die size,
    mirroring fcn_trainer's process_batch_dynamic() training-data bucketing
    (same pick_grid_size() call, same candidates). Die size is read from the
    *original* (pre shift/scale) die extents and the DEF's DBU-per-micron
    factor, so it matches what map_models.py's DEF-only _die_size_um() would
    compute for the same design.
    """
    dbu_per_micron = gpdb.microns()
    die_w_um = (data.ori_die_hx - data.ori_die_lx) / dbu_per_micron
    die_h_um = (data.ori_die_hy - data.ori_die_ly) / dbu_per_micron
    return pick_grid_size(die_w_um, die_h_um, target_um_per_bin, candidates)


class DRVPredictor:
    """
    Loads a trained U-Net checkpoint and runs inference on (C, H, W) feature maps
    (C is inferred from the checkpoint itself — see build_drv_feature_maps()).

    Predictions are saved as:
      {out_dir}/drv_iter{N:05d}_ovfl{overflow:.3f}.pt   – (1, H, W) raw tensor
      {out_dir}/drv_iter{N:05d}_ovfl{overflow:.3f}.png  – heatmap (if matplotlib OK)
    """

    def __init__(self, checkpoint_path, device, out_dir, design_name='design'):
        self.device      = device
        self.out_dir     = out_dir
        self.design_name = design_name
        self.ap_lookup   = {}       # populated externally via load_ap_lookup()

        os.makedirs(out_dir, exist_ok=True)

        self.model, self.epoch, self.best_loss = self._load(checkpoint_path)
        print(f"[DRVPredictor] Loaded checkpoint '{checkpoint_path}' "
              f"(epoch={self.epoch}, best_loss={self.best_loss:.5f})")

        # Try to import matplotlib for visualisation — not required
        self._has_mpl = False
        try:
            import matplotlib
            matplotlib.use('Agg')
            self._has_mpl = True
        except ImportError:
            print("[DRVPredictor] matplotlib not available; .png output disabled")

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        sd   = ckpt['model']
        # Infer architecture from state-dict weight shapes
        in_ch   = sd['enc1.block.0.weight'].shape[1]
        base_ch = sd['enc1.block.0.weight'].shape[0]
        model   = UNet(in_ch=in_ch, base_ch=base_ch).to(self.device)
        model.load_state_dict(sd)
        model.eval()
        for p in model.parameters():   # frozen — we never update weights here
            p.requires_grad_(False)
        return model, ckpt.get('epoch', '?'), ckpt.get('loss', float('nan'))

    # ── Prediction ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, feature_maps):
        """
        Args:
            feature_maps : (C, H, W) float32 tensor on device (C must match checkpoint)
        Returns:
            (1, H, W) float32 DRV density prediction on device
        """
        return self.model(feature_maps.unsqueeze(0)).squeeze(0)

    # ── Visualisation ──────────────────────────────────────────────────────────

    def save_drv_png(self, drv_pred, iteration, overflow, l2_loss, path):
        """Save a standalone DRV prediction heatmap (no feature map panels)."""
        import matplotlib.pyplot as plt

        pred_np = drv_pred.cpu().numpy()[0]    # (H, W)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig.patch.set_facecolor('#12121e')
        im = ax.imshow(pred_np, origin='lower', cmap='hot', vmin=0, vmax=1,
                       interpolation='nearest')
        ax.set_title(
            f'DRV  iter={iteration}  ovfl={overflow:.4f}  L2={l2_loss:.3f}',
            color='white', fontsize=10)
        ax.set_facecolor('#12121e')
        ax.tick_params(colors='#888888', labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors='#aaaaaa', labelsize=7)
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()

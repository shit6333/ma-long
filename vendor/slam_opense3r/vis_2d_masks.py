"""
slam_opense3r/vis_2d_masks.py
==============================
Optional debug visualization: per-frame 2D instance masks.

During SLAM inference, for each frame this takes the query head's top-K
objectness slots, upsamples each predicted mask to the original image
resolution, overlays it as a single (constant) RED alpha mask on the RGB
frame, and saves all K overlays as ONE grid PNG (one file per frame).

Notes
-----
* All masks use the SAME red colour — this is a "what instances did the
  head propose on this frame" debug view, NOT instance-identity colouring.
* Each grid tile shares the same base RGB frame; tiles differ only by which
  instance slot is overlaid (`raw is same frame, diff inst`).
* Only meaningful for the query instance head (needs `mask_logits` +
  `objectness` in the forward output).  Silently no-ops otherwise.

Enabled per-pipeline via the `save_2d_masks` attribute (default off), set
from run.py's `--save_2d_inst_masks` flag.
"""
from __future__ import annotations

import os
import math

import numpy as np
import torch


# Constant overlay colour for every mask (RGB).
_RED = np.array([255.0, 0.0, 0.0], dtype=np.float32)


def _denorm_image(img_chw: torch.Tensor) -> np.ndarray:
    """[3,H,W] float in [-1,1] → uint8 HxWx3 RGB."""
    img = img_chw.detach().float().cpu()
    img = ((img + 1.0) / 2.0).clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
    return (img * 255.0 + 0.5).astype(np.uint8)


@torch.no_grad()
def dump_frame_instance_masks(
    images:      torch.Tensor,
    sem_res:     dict,
    frame_start: int,
    out_dir:     str,
    demo_name:   str = 'scene',
    top_k:       int = 15,
    mask_thresh: float = 0.5,
    alpha:       float = 0.5,
):
    """Save one grid PNG per frame of the top-`top_k` objectness masks.

    Args
    ----
    images:      [1,T,3,H,W] or [T,3,H,W] RGB in [-1,1] for THIS chunk.
    sem_res:     forward output dict; must carry
                   'mask_logits' [1,T,N,Hf,Wf]  and  'objectness' [1,T,N].
    frame_start: global frame index of this chunk's first frame (for naming).
    out_dir:     directory to write `<demo>_frameXXXX_inst2d.png` into.
    top_k:       number of highest-objectness slots to render per frame.
    mask_thresh: sigmoid(mask_logit) threshold for the binary overlay.
    alpha:       blend weight of the red overlay on masked pixels.
    """
    if sem_res is None:
        return
    if 'mask_logits' not in sem_res or 'objectness' not in sem_res:
        return
    if sem_res['mask_logits'] is None or sem_res['objectness'] is None:
        return

    # Matplotlib is only needed when this debug path is active.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    if images.dim() == 5:
        images = images[0]                                       # [T,3,H,W]
    mask_logits = sem_res['mask_logits'][0].detach().float().cpu()   # [T,N,Hf,Wf]
    objectness  = sem_res['objectness'][0].detach().float().cpu()    # [T,N]
    T, N, _, _ = mask_logits.shape
    H, W = images.shape[-2], images.shape[-1]
    k = min(int(top_k), N)

    for t in range(T):
        gidx = frame_start + t
        base = _denorm_image(images[t]).astype(np.float32)      # HxWx3

        obj   = objectness[t].sigmoid()                         # [N]
        slots = torch.topk(obj, k).indices.tolist()

        # Upsample the selected slots' masks to full image resolution once.
        ml_up = torch.nn.functional.interpolate(
            mask_logits[t, slots].unsqueeze(0),                 # [1,k,Hf,Wf]
            size=(H, W), mode='bilinear', align_corners=False,
        )[0]                                                    # [k,H,W]
        probs = ml_up.sigmoid().numpy()

        # Grid: original frame as tile 0, then one tile per instance slot.
        n_tiles = k + 1
        cols = int(math.ceil(math.sqrt(n_tiles)))
        rows = int(math.ceil(n_tiles / cols))
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * 3.0, rows * 3.0))
        axes = np.atleast_1d(axes).ravel()

        for ax in axes:
            ax.axis('off')

        axes[0].imshow(base.astype(np.uint8))
        axes[0].set_title('raw', fontsize=8)

        for j, slot in enumerate(slots):
            ax = axes[j + 1]
            m = probs[j] > mask_thresh                          # HxW bool
            over = base.copy()
            over[m] = (1.0 - alpha) * over[m] + alpha * _RED
            ax.imshow(over.astype(np.uint8))
            ax.set_title(f"slot {slot}  obj={obj[slot]:.2f}", fontsize=8)

        fig.suptitle(f"{demo_name}  frame {gidx}  (top-{k} objectness)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_path = os.path.join(out_dir,
                                f"{demo_name}_frame{gidx:04d}_inst2d.png")
        fig.savefig(out_path, dpi=100)
        plt.close(fig)

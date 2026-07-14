"""
Method 2: Per-instance mask-crop forward through semantic backbone.

For each instance:
  1. Compute its bbox (with optional margin).
  2. Crop the image (mode='bbox') or crop and zero-out outside (mode='masked').
  3. Resize to backbone's input resolution.
  4. Forward through backbone's standard image encoder → global embedding.
  5. Broadcast the global embedding back into the mask area to form dense GT.

All instances of one frame are batched into a single backbone forward to
amortize cost.
"""

import torch
import torch.nn.functional as F


def collect_instance_crops(
    image: torch.Tensor,            # [3, H, W] in [0, 1]
    instance_mask: torch.Tensor,    # [H, W]  long
    crop_size: int = 224,
    margin: float = 0.1,
    crop_mode: str = 'bbox',        # 'bbox' or 'masked'
) -> tuple:
    """
    Returns:
        crops:     [num_inst, 3, crop_size, crop_size]  in [0, 1]
        ids:       [num_inst]   instance IDs in same order as crops
        mask_oh:   [num_inst, H, W]  one-hot float for broadcast
    """
    H, W = instance_mask.shape
    device = image.device
    dtype  = image.dtype

    ids_all = instance_mask.unique()
    ids_all = ids_all[ids_all > 0]

    crops      = []
    valid_ids  = []
    mask_oh_l  = []

    for iid in ids_all.tolist():
        mask = (instance_mask == iid)
        ys, xs = torch.where(mask)
        if ys.numel() == 0:
            continue
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1

        bh, bw = y2 - y1, x2 - x1
        my, mx = int(bh * margin), int(bw * margin)
        y1 = max(0, y1 - my); y2 = min(H, y2 + my)
        x1 = max(0, x1 - mx); x2 = min(W, x2 + mx)
        if y2 <= y1 or x2 <= x1:
            continue

        if crop_mode == 'bbox':
            crop = image[:, y1:y2, x1:x2]
        elif crop_mode == 'masked':
            mb = mask[y1:y2, x1:x2].to(dtype)
            crop = image[:, y1:y2, x1:x2] * mb.unsqueeze(0)
        else:
            raise ValueError(f"Unknown crop_mode: {crop_mode}")

        crop = F.interpolate(
            crop.unsqueeze(0), size=(crop_size, crop_size),
            mode='bilinear', align_corners=False,
        ).squeeze(0)
        crops.append(crop)
        valid_ids.append(iid)
        mask_oh_l.append(mask.to(dtype))

    if not crops:
        return (
            torch.zeros(0, 3, crop_size, crop_size, device=device, dtype=dtype),
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, H, W, device=device, dtype=dtype),
        )

    return (
        torch.stack(crops),
        torch.tensor(valid_ids, dtype=torch.long, device=device),
        torch.stack(mask_oh_l),
    )


@torch.no_grad()
def mask_crops_supervision_per_frame(
    image: torch.Tensor,            # [3, H, W]  in [0, 1]
    instance_mask: torch.Tensor,    # [H, W]
    backbone,                        # PECoreBackbone-like, has .encode_image_global() and .output_dim
    crop_size: int = 224,
    margin: float = 0.1,
    crop_mode: str = 'bbox',
) -> torch.Tensor:
    """
    Build dense GT for one frame via per-instance crop forward + broadcast.

    Returns:
        dense_gt: [output_dim, H, W]
    """
    H, W = instance_mask.shape
    device = image.device
    dtype  = image.dtype

    crops, ids, mask_oh = collect_instance_crops(
        image, instance_mask, crop_size=crop_size,
        margin=margin, crop_mode=crop_mode,
    )

    if crops.shape[0] == 0:
        return torch.zeros(backbone.output_dim, H, W, device=device, dtype=dtype)

    embs = backbone.encode_image_global(crops)         # [num_inst, output_dim]

    # Broadcast: dense[c,y,x] = sum_i embs[i,c] * mask_oh[i,y,x]
    dense = torch.einsum('id,ihw->dhw', embs.to(dtype), mask_oh)
    return dense

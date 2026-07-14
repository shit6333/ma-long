"""
opense3r/model_mapanything.py
==============================
OpenSe3r variant where the frozen geometry backbone is Meta's MapAnything
(https://github.com/facebookresearch/map-anything) instead of DA3.

Why a subclass instead of a parallel class?
-------------------------------------------
The trainable semantic / instance branch (`DepthEncoder`, `CausalBranch`,
`FeatureDPTHead`, etc.) is bit-identical across the DA3 and MapAnything
backends — only the geometry source changes.  We therefore subclass
`OpenSe3rDA3`, pass `_skip_geo_backbone=True` so the parent doesn't load
DA3, build MapAnything ourselves, and override `_da3_forward` to call
MapAnything's `infer` instead.  Every other code path (`forward`,
`forward_geometry`, `forward_chunk`, `forward_semantic`, the shared
`_run_semantic_branch`) comes for free with the right behaviour because
the parent's `forward` / `forward_geometry` now thread optional
`depth_z` / `intrinsics` from `frames` into `_da3_forward`.

Multi-modal mode (depth input)
------------------------------
MapAnything's killer feature is that it can fuse a partial depth signal
into its 3D regression for noticeably better geometry.  When the SLAM
pipeline has access to a depth source (e.g. ScanNet GT depth via
ScannetSLAMGTDataset), it can put a `depth_z` tensor into the `frames`
dict; the parent forward will forward it here, and we add it to each
view's MapAnything input.  When `depth_z` is None, MapAnything runs in
its image-only mode (still produces metric geometry).

Coord convention
----------------
MapAnything's `camera_poses` is OpenCV cam2world in the world frame.  We
return it directly as `c2w` — same convention as DA3 and our downstream
SLAM (`OpenSe3rMemory.poses`).

Reference view
--------------
Without `camera_poses` inputs, MapAnything is trained to output poses
"in the frame of the reference view 0" (see `model.py:690-691` in the
upstream repo).  So predicted `camera_poses[0] ≈ identity`, matching
DA3's `ref_view_strategy="first"` convention.

Coord pipeline (important — same as DA3)
----------------------------------------
The geometry this class returns from `forward_geometry` is in the
CHUNK-LOCAL frame (anchored to view 0 of the chunk batch).  The SLAM
pipeline aligns it to the SLAM-global frame inside
`memory.update_geo` via `coordinate_alignment`, which then writes
SLAM-global `memory.poses` and `memory._pending_pts_new`.  These global
values are what get passed to `forward_semantic` (the OpenSe3r CausalBranch
input).  Because `OpenSe3rMapAnything` is a subclass of `OpenSe3rDA3`,
the entire `_geo_forward → update_geo → forward_semantic` pipeline plumbing
is bit-identical to DA3 — no extra coord/scale wiring needed.

Default `apply_mask` / `mask_edges` policy
-----------------------------------------
MapAnything's `infer` defaults `apply_mask=True` + `mask_edges=True`,
which ZEROES the masked pixels in dense outputs (`pts3d`, `pts3d_cam`,
`depth_z`).  That would change the depth distribution → bias
`box_scale = depth.median()` inside `forward_semantic` AND remove
samples that `coordinate_alignment` relies on.  To keep behaviour
parity with DA3 (no masking) we DISABLE both by default; expose
`ma_apply_mask` / `ma_mask_edges` for users who want them back.

Confidence range
----------------
MapAnything's UniCeption confidence adaptor is in the DUSt3R/MASt3R
family — empirically `conf ≥ 1` (the same `expp1 = exp(x) + 1`
activation DA3 uses).  Downstream `OpenSe3rMemory._conf_sig(c) = (c-1)/c`
assumes that range.  We log the chunk-0 conf range once so the user
can sanity-check on the first run; if conf turns out to live in [0,1]
on a future ckpt, the log will make the mismatch obvious.
"""

from __future__ import annotations
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Make the bundled map-anything package importable without
# `pip install -e thirdparty/map-anything`.
_MA_THIRDPARTY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'thirdparty', 'map-anything',
)
if os.path.isdir(_MA_THIRDPARTY) and _MA_THIRDPARTY not in sys.path:
    sys.path.insert(0, _MA_THIRDPARTY)

from .model_da3 import OpenSe3rDA3


# ---------------------------------------------------------------------------
# Default MapAnything model id on the HF hub.  The "apache" variant is the
# Apache-2.0-licensed release; use the full "facebook/map-anything" via the
# CLI (`--ma_model_id`) when the stricter licence isn't a constraint.
# ---------------------------------------------------------------------------
_DEFAULT_MA_MODEL_ID = "facebook/map-anything"


class OpenSe3rMapAnything(OpenSe3rDA3):
    """OpenSe3r with MapAnything as the frozen geometry backbone."""

    def __init__(
        self,
        ma_model_id: str = _DEFAULT_MA_MODEL_ID,
        ma_cache_dir: Optional[str] = None,
        # MapAnything `infer` knobs.
        # `apply_mask` / `mask_edges` default OFF here (vs MapAnything's own
        # True/True) so dense outputs don't get zeroed — matches DA3 and
        # protects `box_scale` + `coordinate_alignment` downstream.  Flip
        # back to True if you specifically want MapAnything to suppress
        # sky / edge-artifact pixels at the SLAM input boundary.
        ma_memory_efficient_inference: bool = True,
        ma_apply_mask:              bool  = False,
        ma_mask_edges:              bool  = False,
        ma_apply_confidence_mask:   bool  = False,
        ma_confidence_percentile:   float = 10.0,
        ma_use_multiview_confidence: bool = False,
        ma_ignore_calibration_inputs: bool = False,
        ma_ignore_depth_inputs:       bool = False,
        ma_ignore_pose_inputs:        bool = True,
        ma_ignore_depth_scale_inputs: bool = False,
        ma_ignore_pose_scale_inputs:  bool = False,
        ma_data_norm_type:            str  = 'dinov2',
        **da3_kwargs,
    ):
        # Skip DA3 build in the parent.  We accept (and ignore) `da3_ckpt` so
        # that callers using the same kwarg surface as OpenSe3rDA3 keep
        # working.  The parent reuses its `_da3_mean` / `_da3_std` buffers,
        # which happen to be ImageNet stats — exactly what the MapAnything
        # DINOv2 encoder also wants.
        super().__init__(_skip_geo_backbone=True, **da3_kwargs)

        # ------------------------------------------------------------------
        # Frozen MapAnything backbone.  `cache_dir` (if given) redirects the
        # HF hub download/cache off the default `~/.cache/huggingface/hub`
        # so multiple users can share a project-local checkpoint store.
        # ------------------------------------------------------------------
        from mapanything.models import MapAnything
        from_pretrained_kwargs = {}
        if ma_cache_dir is not None:
            os.makedirs(ma_cache_dir, exist_ok=True)
            from_pretrained_kwargs['cache_dir'] = ma_cache_dir
        self.mapanything = MapAnything.from_pretrained(
            ma_model_id, **from_pretrained_kwargs,
        )
        cache_note = f" (cache_dir={ma_cache_dir})" if ma_cache_dir else ""
        print(f"[OpenSe3rMapAnything] MapAnything loaded from {ma_model_id}{cache_note}")
        for p in self.mapanything.parameters():
            p.requires_grad_(False)
        self.mapanything.eval()

        # Cache `infer` defaults so we don't pass them every forward.
        # We disable our outer bf16 autocast around the whole `infer` call
        # (because MapAnything's post-processing calls torch.linalg.solve
        # which has no bf16 cuSOLVER path), so we let MapAnything's OWN
        # `use_amp=True` re-enter mixed precision for the forward (just not
        # the linalg-heavy post-processing).  README-recommended setting.
        self._ma_infer_kwargs = dict(
            memory_efficient_inference  = ma_memory_efficient_inference,
            use_amp                     = True,
            amp_dtype                   = 'bf16',
            apply_mask                  = ma_apply_mask,
            mask_edges                  = ma_mask_edges,
            apply_confidence_mask       = ma_apply_confidence_mask,
            confidence_percentile       = ma_confidence_percentile,
            use_multiview_confidence    = ma_use_multiview_confidence,
            ignore_calibration_inputs   = ma_ignore_calibration_inputs,
            ignore_depth_inputs         = ma_ignore_depth_inputs,
            ignore_pose_inputs          = ma_ignore_pose_inputs,
            ignore_depth_scale_inputs   = ma_ignore_depth_scale_inputs,
            ignore_pose_scale_inputs    = ma_ignore_pose_scale_inputs,
        )
        self._ma_data_norm_type = ma_data_norm_type
        # Fires once on the first `_da3_forward` call so the user can
        # sanity-check (a) view 0's c2w is ≈ identity (the SLAM
        # coordinate_alignment relies on this) and (b) conf is in the
        # expected [1, ∞) range that `OpenSe3rMemory._conf_sig` assumes.
        self._ma_first_call_diag_done = False

    # ------------------------------------------------------------------
    # Override the parent's geo predict method.  The parent's
    # `forward` / `forward_geometry` route to this via `_da3_forward(...)`
    # — the name is a back-compat lie; in this subclass it runs MapAnything.
    # ------------------------------------------------------------------

    def _da3_forward(
        self,
        images_01: torch.Tensor,                       # [B, T, 3, H, W] in [0, 1]
        depth_z:    Optional[torch.Tensor] = None,     # [B, T, H, W]    metric (optional)
        intrinsics: Optional[torch.Tensor] = None,     # [B, T, 3, 3]    (optional)
        **_ignored,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Run frozen MapAnything to obtain depth / depth_conf / c2w / K.

        Returns the same 5-tuple as the parent's DA3 version so the rest of
        `forward` / `forward_geometry` is bit-identical.

        Args:
            images_01:  RGB in [0, 1].
            depth_z:    optional per-frame Z-depth (metric, metres).  When
                        provided, MapAnything fuses it for refined geo (and
                        also requires intrinsics — we forward that too).
            intrinsics: optional per-frame pinhole K.  When `depth_z` is
                        provided this MUST be set (MapAnything's contract);
                        when only intrinsics are provided it just uses them
                        as calibration hint.
        """
        B, T, _, H, W = images_01.shape
        device = images_01.device

        # MapAnything's DINOv2 encoder normalises like DA3's (ImageNet).
        # The parent's `_da3_mean` / `_da3_std` buffers are ImageNet stats
        # so we just reuse them — no need to register a duplicate.
        images_norm = (images_01 - self._da3_mean) / self._da3_std        # [B,T,3,H,W]

        # Build one view dict per frame.  Each view's batch dim = B.
        views: List[Dict] = []
        for t in range(T):
            view: Dict = {
                'img':             images_norm[:, t],                     # [B, 3, H, W]
                'data_norm_type':  [self._ma_data_norm_type] * B,
                'true_shape':      [(H, W)] * B,
                'idx':             [t] * B,
                'instance':        [str(t)] * B,
            }
            if intrinsics is not None:
                view['intrinsics'] = intrinsics[:, t].to(device)          # [B, 3, 3]
            if depth_z is not None:
                if intrinsics is None:
                    raise ValueError(
                        "OpenSe3rMapAnything: when `depth_z` is provided "
                        "intrinsics must also be provided (MapAnything's "
                        "multi-modal contract — depth without K is rejected)."
                    )
                # NOTE: model.py docstring claims (B, H, W, 1) but the
                # actual `preprocess_input_views_for_inference` consumes
                # `depth_z[..., None] * ray_directions` where ray_directions
                # is (B, H, W, 3) — so depth_z must be (B, H, W) (3D); 4D
                # blows up broadcasting.  `preprocess_inputs` (the public
                # helper) also produces 3D, confirming the real contract.
                view['depth_z']         = depth_z[:, t].to(device)                  # [B, H, W]
                view['is_metric_scale'] = torch.ones(B, dtype=torch.bool, device=device)
            views.append(view)

        # MapAnything's `infer` runs its own validation + AMP + preprocessing.
        # CRITICAL: the SLAM pipeline wraps `_geo_forward` in a bf16 autocast
        # context, but MapAnything's post-processing calls `torch.linalg.solve`
        # in `recover_pinhole_intrinsics_from_ray_directions`, and cuSOLVER's
        # `lu_factor_cusolver` does not support BFloat16 → NotImplementedError.
        # Disable the outer autocast for the whole `infer` call and let
        # MapAnything manage its own internal mixed precision via use_amp.
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            preds = self.mapanything.infer(views, **self._ma_infer_kwargs)

        # `preds[t]['depth_z']`        [B, H, W, 1]
        # `preds[t]['conf']`           [B, H, W]
        # `preds[t]['camera_poses']`   [B, 4, 4]   c2w OpenCV
        # `preds[t]['intrinsics']`     [B, 3, 3]
        depth      = torch.stack([p['depth_z'].squeeze(-1) for p in preds], dim=1)   # [B, T, H, W]
        depth_conf = torch.stack([p['conf']                 for p in preds], dim=1)   # [B, T, H, W]
        c2w        = torch.stack([p['camera_poses']         for p in preds], dim=1)   # [B, T, 4, 4]
        K_out      = torch.stack([p['intrinsics']           for p in preds], dim=1)   # [B, T, 3, 3]

        # One-time sanity diagnostic.  Confirms (a) view 0's c2w is ≈
        # identity (DA3-equivalent convention assumed by the SLAM
        # coordinate_alignment) and (b) conf range fits the expp1-like
        # [1, ∞) that `OpenSe3rMemory._conf_sig` expects.  Pure logging.
        if not self._ma_first_call_diag_done:
            self._ma_first_call_diag_done = True
            with torch.no_grad():
                # Use the first sample in the batch for the report.
                c2w0   = c2w[0, 0].float().cpu().numpy()
                t0_err = float(((c2w0[:3, 3]) ** 2).sum() ** 0.5)
                R0_err = float(((c2w0[:3, :3] - torch.eye(3).numpy()) ** 2).sum() ** 0.5)
                cmin = float(depth_conf.min().item())
                cmed = float(depth_conf.median().item())
                cmax = float(depth_conf.max().item())
            print(f"[OpenSe3rMapAnything diag] view-0 c2w drift from "
                  f"identity:  ||t||={t0_err:.4f}  ||R-I||_F={R0_err:.4f}")
            print(f"[OpenSe3rMapAnything diag] conf range  min={cmin:.4f}  "
                  f"median={cmed:.4f}  max={cmax:.4f}  "
                  f"(memory._conf_sig assumes c≥1 → if min<1 we'll "
                  f"silently produce negative conf_sig values)")

        # MapAnything outputs metric scale (its raison d'être).  Mark in meta
        # so the SLAM coord_alignment seeds `scale=1.0` like for DA3-nested.
        meta = {
            'is_metric':    True,
            'scale_factor': 1.0,
        }
        return depth, depth_conf, c2w, K_out, meta

"""OpenSe3r query-instance semantic model behind a per-chunk streaming API.

The model is built FROM THE CHECKPOINT (``build_model_from_ckpt``): its stored
training args give the architecture, and ``_infer_arch_from_weights`` overrides
them wherever the tensors disagree (the descfix P/Q checkpoint records
``query_descriptor_mlp_layers=0`` but carries 3 layers — building from the args
would drop the descriptor MLP that the S1 loop verification lives on; likewise
``pq_head`` presence). Loading is ``strict=False`` because the frozen backbones
are loaded inside ``__init__``, so the TRAINABLE key sets are checked explicitly
and any mismatch raises (a silent mis-load cost multi-day debugging before —
docs/opense3r_model_handoff.md §5).

FastSAM query seeding happens INSIDE the model (``fastsam_query=True`` builds
``_fastsam_seg`` + ``region_pool``), so no external segmenter is needed here.

Chunk contract (CPU float32 torch tensors, from ``bridge``):
    images                 [1,T,3,H,W]  in [-1,1]
    depth                  [1,T,H,W]    metric z
    intrinsics             [1,T,3,3]    K at (H,W)
    c2w                    [1,T,4,4]    cam->world, sequence-frame-0 anchored (gauge)
    world_points_anchored  [1,T,H,W,3]  same anchored frame; invalid pixels zeroed
    valid                  [1,T,H,W]    bool geometry validity

Output: the model's prediction dict — ``mask_logits`` [1,T,N,H,W], ``objectness``
[1,T,N], ``descriptor`` [1,T,N,256], ``semantic_feat_expanded`` (None for A1-type
checkpoints that dropped the dense head), ``pq_p``/``pq_q`` [1,T,N] (P/Q head
checkpoints only), ``box_scale``, ``kv_cache`` (consumed here), ...
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from ma_slam_semantic.config import SemConfig, ensure_opense3r
from ma_slam_semantic.kv_policies import make_kv_policy


def build_model_from_ckpt(ckpt_path: str, backbone_paths: dict, device: str = "cuda",
                          verbose: bool = True) -> Tuple[torch.nn.Module, object, dict]:
    """Checkpoint-described construction (semantic_mapping.model helpers) with the
    frozen-backbone paths re-pointed at ``backbone_paths`` (the checkpoint stores
    OpenSe3r-repo-relative paths; vendor/ renames them). Returns
    (model on device, ckpt args, load-report dict)."""
    from semantic_mapping.model import (read_ckpt_args, _arch_kwargs, _resolve,
                                        _infer_arch_from_weights, TRAINABLE_PREFIXES,
                                        OPTIONAL_DROP_PREFIXES)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    a = read_ckpt_args(ckpt_path)
    kw = _arch_kwargs(a)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    state = state.get("model", state)
    kw = _infer_arch_from_weights(state, kw, verbose)
    if kw.get("fastsam_query"):
        kw["fastsam_ckpt"] = backbone_paths["fastsam_ckpt"]
    backend = getattr(a, "backend", "vggt")
    if backend == "da3":
        from opense3r.model_da3 import OpenSe3rDA3
        # gt-geometry route: DA3 never runs -> not built (geometry enters via
        # forward_semantic — exactly the contract ma_slam / the bridge feeds).
        model = OpenSe3rDA3(
            da3_ckpt=_resolve(a.da3_ckpt),
            vggt_ckpt=backbone_paths["vggt_ckpt"],
            lbotv_ckpt=_resolve(getattr(a, "lbotv_ckpt", None)),
            dinov3_ckpt=_resolve(getattr(a, "dinov3_ckpt", None)),
            lseg_ckpt=_resolve(a.lseg_ckpt),
            _skip_geo_backbone=True,
            **kw)
    elif backend == "vggt":
        from opense3r.model import OpenSe3r
        model = OpenSe3r(vggt_ckpt=backbone_paths["vggt_ckpt"],
                         lseg_ckpt=_resolve(a.lseg_ckpt), **kw)
    else:
        raise ValueError(f"unsupported backend {backend!r} (vggt | da3)")

    msg = model.load_state_dict(state, strict=False)
    miss = [k for k in msg.missing_keys
            if k.startswith(TRAINABLE_PREFIXES) and not k.startswith(OPTIONAL_DROP_PREFIXES)]
    unexp = [k for k in msg.unexpected_keys
             if k.startswith(TRAINABLE_PREFIXES) and not k.startswith(OPTIONAL_DROP_PREFIXES)]
    n_pq_ckpt = sum(k.startswith("pq_head.") for k in state)
    info = dict(backend=backend, model_class=type(model).__name__,
                n_missing_trainable=len(miss), n_unexpected_trainable=len(unexp),
                n_missing_frozen=len(msg.missing_keys) - len(miss),
                n_pq_tensors_in_ckpt=n_pq_ckpt,
                pq_head_built=getattr(model, "pq_head", None) is not None,
                desc_mlp_layers=kw.get("query_descriptor_mlp_layers"),
                seed_sem=kw.get("seed_sem"), no_dense_sem=kw.get("no_dense_sem"),
                use_sem_conf_head=kw.get("use_sem_conf_head"))
    if verbose:
        print(f"[sem_model] {os.path.basename(ckpt_path)}: backend={backend} "
              f"class={info['model_class']} | trainable missing={len(miss)} "
              f"unexpected={len(unexp)} | frozen missing={info['n_missing_frozen']} (expected)")
    if miss or unexp:
        raise RuntimeError(
            f"checkpoint/architecture mismatch: {len(miss)} missing and {len(unexp)} "
            f"unexpected TRAINABLE keys.\n  missing (first 5): {miss[:5]}\n"
            f"  unexpected (first 5): {unexp[:5]}")
    if n_pq_ckpt and not info["pq_head_built"]:
        # pit #2 of the handoff doc: pq_head. used to be outside the trainable
        # whitelist and a dropped P/Q head passed silently.
        raise RuntimeError(f"checkpoint carries {n_pq_ckpt} pq_head tensors but the "
                           f"model was built without one")
    model.to(device).eval()
    if getattr(a, "query_merge", False) and hasattr(model, "set_merge_active"):
        model.set_merge_active(True)
    return model, a, info


class SemanticModel:
    def __init__(self, cfg: SemConfig, device: Optional[str] = None):
        ensure_opense3r(cfg.opense3r_root)
        self.cfg = cfg
        self.device = device or cfg.device
        self.model, self.ckpt_args, self.load_info = build_model_from_ckpt(
            cfg.ckpt_path, cfg.backbone_paths(), self.device)
        self.has_pq = bool(self.load_info["pq_head_built"])
        li = self.load_info
        variant = ("A1-type (textregion seeding, dense sem head off)"
                   if li["seed_sem"] == "textregion" or li["no_dense_sem"]
                   else "semconf-type (dense sem head)")
        print(f"[sem_model] variant: {variant} | P/Q head: {'yes' if self.has_pq else 'no'}"
              f" | descriptor MLP layers: {li['desc_mlp_layers']} | ckpt: {cfg.ckpt_path}")
        if cfg.ranker != "obj" and not self.has_pq:
            print(f"[sem_model] NOTE ranker={cfg.ranker!r} needs the P/Q head; this "
                  f"checkpoint has none -> export will rank by objectness")
        self.kv = self.model.init_kv_cache()
        self.policy = make_kv_policy(cfg, model=self.model)
        self._box_scale: Optional[torch.Tensor] = None  # [1] sequence-global scale

    @property
    def semantic_backbone(self):
        """The frozen VLM teacher (TIPS) — proposals pool textregion features from
        it when the checkpoint has no dense semantic head."""
        return self.model.semantic_backbone

    # ------------------------------------------------------------------ scale
    def set_box_scale(self, scale) -> None:
        """Lock the sequence-global box_scale (golden gt mode: gt_seq_median)."""
        self._box_scale = torch.as_tensor(scale, dtype=torch.float32).reshape(1)
        print(f"[sem_model] box_scale locked = {float(self._box_scale):.4f}")

    def _maybe_capture_box_scale(self, out: dict) -> None:
        if self._box_scale is not None or self.cfg.box_scale_mode != "first_chunk":
            return
        bs = out.get("box_scale")
        if bs is None:
            print("[sem_model] WARN: box_scale_mode=first_chunk but model returned no "
                  "'box_scale' — staying in per-chunk mode.")
            return
        self._box_scale = bs.detach().float().cpu().reshape(1)
        print(f"[sem_model] box_scale captured from first chunk = {float(self._box_scale):.4f}")

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward_chunk(self, chunk: dict) -> dict:
        from opense3r.model import fill_invalid_world_points
        dev = self.device
        valid = chunk["valid"][0]                                        # [T,H,W] bool
        pts = fill_invalid_world_points(
            chunk["world_points_anchored"][0].float(), valid).unsqueeze(0).to(dev)
        bs = None if self._box_scale is None else self._box_scale.to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model.forward_semantic(
                {"images": chunk["images"].to(dev)},
                depth=chunk["depth"].float().to(dev),
                intrinsics=chunk["intrinsics"].float().to(dev),
                c2w=chunk["c2w"].float().to(dev),
                world_points=pts,
                world_points_conf=valid.float().unsqueeze(0).unsqueeze(-1).to(dev),
                box_scale_override=bs,
                kv_cache_list=self.kv,
                mode=self.cfg.attn_mode,
                return_poses=False,
            )
        self._maybe_capture_box_scale(out)
        self.kv = self.policy.after_chunk(out["kv_cache"])
        return out

    def reset_kv(self) -> None:
        self.kv = self.model.init_kv_cache()
        print("[sem_model] KV cache reset")

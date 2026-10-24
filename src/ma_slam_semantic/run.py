"""CLI runner for ma_slam_semantic.

Golden-test mode (S2/S3 correctness harness — GT geometry, bypasses ma_slam; must
reproduce the reference OpenSe3r 005 GT run):

    PYTHONPATH=src python -m ma_slam_semantic.run --geometry gt \
        --data /mnt/HDD4/ricky/data/instance_eval_data/scenenn/oni/005 \
        --demo_type scenenn_gt --out outputs/ma_sem/005_gt --save_2d_masks

ma_slam-geometry mode (S4) plugs SemanticHooks into ``ma_slam.run`` instead.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ma_slam_semantic.config import SemConfig, ensure_opense3r, DEFAULT_CKPT
from ma_slam_semantic.bridge import gt_chunks, gt_seq_box_scale
from ma_slam_semantic.sem_model import SemanticModel
from ma_slam_semantic.proposals import make_proposals
from ma_slam_semantic.instance_map import InstanceTable, StaticGeoResolver


def _load_gt_batch(data_path: str, demo_type: str, res, max_frames: int = 0) -> dict:
    ensure_opense3r()
    if demo_type == "scenenn_gt":
        from slam_opense3r.datasets.scenenn import SceneNNSLAMGTDataset
        ds = SceneNNSLAMGTDataset(data_path, resolution=list(res))
    elif demo_type == "scannet_gt":
        from slam_opense3r.datasets.scannet import ScannetSLAMGTDataset
        ds = ScannetSLAMGTDataset(data_path, resolution=list(res))
    else:
        raise ValueError(f"unknown demo_type {demo_type!r}")
    batch = ds[0]
    if max_frames:
        batch = {k: (v[:max_frames] if torch.is_tensor(v) and v.dim() >= 1 else v)
                 for k, v in batch.items()}
    return batch


def _assoc_kwargs(a):
    """Association-cue overrides: --assoc preset first, then any explicit --w_* / thresholds."""
    kw = {}
    if a.assoc == "new":
        kw.update(w_desc=0.70, w_vox=0.0, w_geo=0.30, match_threshold=0.65)
    elif a.assoc == "old":
        kw.update(w_desc=0.30, w_vox=0.30, w_geo=0.25, match_threshold=0.32)
    for k in ("w_desc", "w_vox", "w_geo", "w_sem", "match_threshold", "merge_mode", "merge_desc", "merge_desc_high"):
        v = getattr(a, k, None)
        if v is not None:
            kw[k] = v
    return kw


def run_ma_slam(a, cfg: SemConfig) -> None:
    """ma_slam-geometry mode: full online SLAM + semantic hooks."""
    import copy
    import glob

    import numpy as np

    from model.inputs import load_intrinsics
    from ma_slam.solver import MaSlam, DEFAULT_CONFIG
    from ma_slam_semantic.hooks import SemanticHooks

    def _frames(d):
        fs = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG", "*.tiff", "*.TIFF"):
            fs += glob.glob(os.path.join(d, ext))
        return sorted(fs)

    name = a.name or os.path.basename(os.path.normpath(a.data))
    image_paths = _frames(os.path.join(a.data, "rgb"))
    if a.max_frames:
        image_paths = image_paths[: a.max_frames]
    need_depth = "depth" in a.mode
    depth_paths = None
    intrinsics = None
    depth_scale = 1000.0                     # uint16-mm convention (ScanNet PNG)
    if need_depth:
        depth_paths = _frames(os.path.join(a.data, "depth"))
        if not depth_paths:
            raise FileNotFoundError(f"no depth frames under {a.data}/depth")
        n = min(len(image_paths), len(depth_paths))
        if len(image_paths) != len(depth_paths):
            print(f"[ma_sem] color/depth count mismatch "
                  f"({len(image_paths)}/{len(depth_paths)}); truncating to {n}")
        image_paths, depth_paths = image_paths[:n], depth_paths[:n]
        # float depth files (e.g. .TIFF) are already metres -> scale 1.0
        from PIL import Image
        d0 = np.asarray(Image.open(depth_paths[0]))
        if np.issubdtype(d0.dtype, np.floating):
            depth_scale = 1.0
            print("[ma_sem] float depth detected -> depth_scale=1.0 (metres)")
    if ("intr" in a.mode) or need_depth:
        intrinsics = load_intrinsics(os.path.join(a.data, "intrinsic.txt"))

    # LK keyframe gate (mirrors ma_slam.run, default ON): drop low-parallax frames so
    # submap_size counts keyframes. GT for ATE eval is subsampled to the same keyframes
    # (per-index correspondence); hooks' gt_positions stays the FULL array — Submap
    # frame_ids are parsed from filenames, so original ordinals survive the subset.
    gt_positions = None; gt_poses = None
    if a.gt:
        _g = np.loadtxt(a.gt, comments="#").reshape(-1, 8)
        gt_positions = _g[:, 1:4]
        from scipy.spatial.transform import Rotation as _Rot
        gt_poses = np.tile(np.eye(4), (len(_g), 1, 1)); gt_poses[:, :3, :3] = _Rot.from_quat(_g[:, 4:8]).as_matrix(); gt_poses[:, :3, 3] = _g[:, 1:4]
    if a.keyframe_disparity > 0:
        from ma_slam.keyframe_flow import select_keyframes_by_disparity
        kf = select_keyframes_by_disparity(image_paths, a.keyframe_disparity)
        print(f"[ma_sem] keyframe gate: kept {len(kf)}/{len(image_paths)} frames "
              f"(disparity > {a.keyframe_disparity:g}px)")
        image_paths = [image_paths[i] for i in kf]
        if depth_paths is not None:
            depth_paths = [depth_paths[i] for i in kf]
        if a.gt:
            gt_lines = [ln for ln in open(a.gt) if ln.strip() and not ln.startswith("#")]
            os.makedirs(a.out, exist_ok=True)
            kf_gt = os.path.join(a.out, "gt_kf.txt")
            with open(kf_gt, "w") as f:
                f.write("# timestamp tx ty tz qx qy qz qw\n")
                f.write("".join(gt_lines[i] for i in kf))
            a.gt = kf_gt

    slam_cfg = copy.deepcopy(DEFAULT_CONFIG)
    slam_cfg["submap_size"] = a.submap_size
    slam_cfg["Loop"]["enable"] = not a.no_loop
    slam_cfg["Loop"]["half_window"] = a.loop_half_window
    slam_cfg["Loop"]["rot_max_deg"] = a.rot_max_deg
    if a.coloc_ratio is not None:
        slam_cfg["Loop"]["coloc_ratio"] = a.coloc_ratio      # e.g. 1e9 = no geometric co-location gate (instance gate decides alone)
    slam_cfg["Graph"]["manifold"] = a.manifold
    slam_cfg["Loop"]["Instance"].update(
        enable=(a.sem_lc != "off"), shadow=(a.sem_lc not in ("enforce", "veto")),
        veto=(a.sem_lc == "veto"),
        tier_b_floor=a.sem_lc_floor, max_tier_b=a.sem_lc_max,
        coloc_ratio_b=(a.sem_lc_coloc if a.sem_lc_coloc > 0 else None),
        unknown_fail=(a.sem_lc_unknown == "fail"),
        s2_hard_reject=tuple(x for x in a.s2_hard_reject.split(",") if x),
        joint=(a.sem_lc_mode == "joint"))

    backend = None
    if a.backend == "da3":
        if "depth" in a.mode:
            raise SystemExit("backend da3 supports rgb / rgb+intr only (no depth input)")
        from model.da3_infer import Da3ChunkModel
        backend = Da3ChunkModel(device="cuda")

    from ma_slam_semantic.bidesc_lc import BiDescConfig
    lc_cfg = BiDescConfig(mode=a.sem_lc_mode, rigid_tau=a.rigid_tau, rigid_eps_rel=a.rigid_eps,
                          rigid_min_inliers=a.rigid_min_inliers, rigid_lowobj=a.rigid_lowobj, rigid_geom=(a.rigid_geom == "on"),
                          s2_t_noise_aware=(a.s2_t_noise == "on"), s2_lvc_first=(a.s2_lvc_first == "on"),
                          n_min=a.bd_nmin, tau_pair=a.bd_pair, tau_dir=a.bd_dir,
                          tau_bi=a.bd_bi, n_uniq=a.bd_nuniq, k_bag=a.bd_kbag,
                          desc_src=a.bd_desc_src, s2_enable=(a.bd_s2 == "on"),
                          dump_dir=(os.path.join(a.out, "semlc_dump")
                                    if a.sem_lc_dump else ""))
    hooks = SemanticHooks(cfg, a.out, name=name, gt_positions=gt_positions, gt_poses=gt_poses, align_mode=a.align, gt_mesh=a.gt_mesh,
                          lc_cfg=lc_cfg)
    if a.sim_threshold is not None:
        slam_cfg["Loop"]["sim_threshold"] = a.sim_threshold
    if a.max_per_submap is not None:
        slam_cfg["Loop"]["max_per_submap"] = a.max_per_submap
    slam = MaSlam(config=slam_cfg, model=backend, hooks=hooks)
    res = slam.run(image_paths, a.out, mode=a.mode, depth_paths=depth_paths,
                   intrinsics=intrinsics, depth_scale=depth_scale,
                   depth_max=(a.depth_max if a.depth_max > 0 else None))

    if a.gt:
        from eval import ate_rmse
        s = ate_rmse(res["poses_txt"], a.gt, "sim3")
        se = ate_rmse(res["poses_txt"], a.gt, "se3")
        print(f"[ma_sem] Sim3-ATE={s['ate_rmse']:.4f} | SE3-ATE={se['ate_rmse']:.4f} | "
              f"loops={res['n_loops']}")

    if getattr(a, "deploy", False):
        from ma_slam_semantic.deploy import run_deploy
        run_deploy(a.out, name)


def main():
    ap = argparse.ArgumentParser(description="ma_slam_semantic runner")
    ap.add_argument("--geometry", default="ma", choices=["gt", "ma"],
                    help="'gt' = golden-test geometry (OpenSe3r GT dataset); "
                         "'ma' = full ma_slam online SLAM + semantic hooks")
    ap.add_argument("--data", required=True)
    ap.add_argument("--demo_type", default="scenenn_gt", choices=["scenenn_gt", "scannet_gt"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None, help="export prefix (default: basename of --data)")
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--save_2d_masks", action="store_true",
                    help="dump per-frame top-K instance-mask grids (S2 eyeball gate)")
    ap.add_argument("--mask_top_k", type=int, default=15)
    # ma_slam-geometry mode args
    ap.add_argument("--mode", default="rgb",
                    help="[ma] ma_slam input mode (rgb / rgb+intr / rgb+depth / rgb+depth+intr)")
    ap.add_argument("--backend", default="da3", choices=["ma", "da3"],
                    help="[ma] reconstruction backbone: MapAnything | DA3 (rgb-only, metric)")
    ap.add_argument("--submap_size", type=int, default=32, help="[ma] frames per submap")
    ap.add_argument("--no_loop", action="store_true", help="[ma] disable loop closure")
    ap.add_argument("--sim_threshold", type=float, default=None, help="[ma] SALAD cos threshold for tier-A loop candidates (default: solver config 0.40)")
    ap.add_argument("--align", default="pos", choices=["pos", "pose", "sim3", "sim3map"], help="[ma] export alignment to GT: 'pos' = SE3 Umeyama on camera centres (scale fixed 1); 'pose' = rotation from camera orientations + centroid shift (uses GT orientations; robust for planar trajectories); 'sim3' = Umeyama similarity (scale + R + t) — absorbs the backbone's metric-scale error, scale is logged")
    ap.add_argument("--assoc", default=None, choices=["new", "old"],
                    help="[ma] association cue preset: 'new' (2026-09-04 main method) w_desc .70/w_vox 0/w_geo .30 @ thr .65; "
                         "'old' w_desc .30/w_vox .30/w_geo .25 @ thr .32. Individual --w_* / --match_threshold override it.")
    ap.add_argument("--w_desc", type=float, default=None); ap.add_argument("--w_vox", type=float, default=None)
    ap.add_argument("--w_geo", type=float, default=None); ap.add_argument("--w_sem", type=float, default=None)
    ap.add_argument("--match_threshold", type=float, default=None)
    ap.add_argument("--merge_mode", default=None, choices=["desc", "fused"],
                    help="[ma] export final-merge rule: 'desc' = pure descriptor cosine >= --merge_desc (reference, cue-invariant); "
                         "'fused' = legacy fused score >= --merge_desc_high (pre-2026-09-04 runs)")
    ap.add_argument("--merge_desc", type=float, default=None); ap.add_argument("--merge_desc_high", type=float, default=None)
    ap.add_argument("--gt_mesh", default=None, help="[ma] GT surface ply for --align sim3map (SE3 from trajectory + one global scale fitted to the surface)")
    ap.add_argument("--resolution", default="auto", help="[ma] semantic model resolution WxH; 'auto' = 518x336, or 336x518 when the rgb frames are portrait")
    ap.add_argument("--coloc_ratio", type=float, default=None, help="[ma] override Loop.coloc_ratio (default: solver config 0.70); 1e9 disables the coloc gate")
    ap.add_argument("--max_per_submap", type=int, default=None, help="[ma] max tier-A candidates verified per submap (default: solver config 5)")
    ap.add_argument("--rot_max_deg", type=float, default=0.0,
                    help="[ma] reject loop candidates above this graph-estimated relative rotation (0 = off)")
    ap.add_argument("--loop_half_window", type=int, default=0,
                    help="[ma] coloc verification window half-size (frames each side; "
                         "0 = just the 2 candidate frames)")
    ap.add_argument("--keyframe_disparity", type=float, default=50.0,
                    help="[ma] LK disparity keyframe gate in px (mirrors ma_slam.run; "
                         "0 = off, feed every frame)")
    # instance-LC (semantic-assisted tier-B loop candidates; [ma] geometry only)
    ap.add_argument("--sem_lc", default="veto", choices=["off", "shadow", "enforce", "veto"],
                    help="[ma] instance-bag gating of SALAD tier-B candidates "
                         "(shadow = log SEMLC verdicts only)")
    ap.add_argument("--sem_lc_floor", type=float, default=0.45,
                    help="[ma] tier-B SALAD floor")
    ap.add_argument("--sem_lc_max", type=int, default=2,
                    help="[ma] max tier-B candidates per submap")
    ap.add_argument("--bd_nmin", type=int, default=2,
                    help="[ma] bidesc: instances per side below this -> UNKNOWN")
    ap.add_argument("--bd_pair", type=float, default=0.85,
                    help="[ma] bidesc: MNN unique-match cosine (tau_pair)")
    ap.add_argument("--bd_dir", type=float, default=0.85,
                    help="[ma] bidesc: per-direction MeanTopK threshold (tau_dir)")
    ap.add_argument("--bd_bi", type=float, default=0.85,
                    help="[ma] bidesc: harmonic-mean threshold (tau_bi)")
    ap.add_argument("--bd_nuniq", type=int, default=2,
                    help="[ma] bidesc: min unique MNN matches for PASS")
    ap.add_argument("--bd_kbag", type=int, default=16,
                    help="[ma] bidesc: per-frame instance-id cap (by objectness)")
    ap.add_argument("--s2_hard_reject", default="LARGE_VIEW_CHANGE,T_FAR,M_LOW",
                    help="comma list of Stage-2 statuses that hard-reject without coloc; "
                         "statuses left out fall through to coloc with the raised (veto) bar")
    ap.add_argument("--s2_lvc_first", default="off", choices=["on", "off"],
                    help="Stage-2: theta_R > tau_theta hard-rejects before the t/Ed/ER gates")
    ap.add_argument("--s2_t_noise", default="off", choices=["on", "off"],
                    help="noise-aware T_FAR gate: widen by d_ref*sin(E_R)")
    ap.add_argument("--proposal_conf", default="gate", choices=["gate", "prefer"],
                    help="gate = proposals own only geometry-valid pixels (shipped); prefer = ownership on "
                         "all finite pixels, 3D footprint from valid pixels when any (see SemConfig)")
    ap.add_argument("--lc_proposal_conf", default="prefer", choices=["gate", "prefer"],
                    help="proposal mode for the current frame's LOOP-VERIFICATION evidence only "
                         "(transient bag, never enters the InstanceTable); map keeps --proposal_conf")
    ap.add_argument("--lc_history", default="bag", choices=["map", "bag"],
                    help="rigid mode history side: map instances of the history frame, or that frame's own transient bag")
    ap.add_argument("--sem_lc_mode", default="rigid", choices=["bidesc", "rigid", "joint"],
                    help="rigid = the shipped IDLV verifier (MNN pairs >= rigid_tau; with --rigid_geom off "
                         "PASS iff >= rigid_min_inliers pairs); bidesc = old S1 Bi-Desc (+S2 layout) rules")
    ap.add_argument("--rigid_tau", type=float, default=0.85)
    ap.add_argument("--rigid_eps", type=float, default=0.15, help="inlier radius as a fraction of median matched depth")
    ap.add_argument("--rigid_min_inliers", type=int, default=4)
    ap.add_argument("--rigid_geom", default="off", choices=["on", "off"],
                    help="off = shipped method: accept on matched-pair count alone; on = + triplet-Kabsch layout check (ablation)")
    ap.add_argument("--rigid_lowobj", default="reject", choices=["reject", "coloc"],
                    help="frames with < rigid_min_inliers instances: reject, or hand to coloc")
    ap.add_argument("--sem_lc_unknown", default="fail", choices=["coloc", "fail"],
                    help="S1 UNKNOWN handling: 'fail' = hard-reject (default; uniform "
                         "with M_LOW — no instance evidence, no edge), 'coloc' = fall "
                         "through to the normal coloc bar")
    ap.add_argument("--snapshot_pts", type=int, default=50000,
                    help="points kept per submap in --snapshot records (raise for close-up videos)")
    ap.add_argument("--snapshot", action="store_true",
                    help="write per-submap reconstruction snapshots to <out>/snapshots "
                         "(render with scripts/render_snapshots.py)")
    ap.add_argument("--bd_s2", default="off", choices=["on", "off"],
                    help="[bidesc mode] Stage-2 layout-graph verdict (on = S1+S2 ablation)")
    ap.add_argument("--sem_lc_dump", action="store_true",
                    help="dump per-candidate scene-graph JSONs to <out>/semlc_dump "
                         "(render with scripts/render_semlc_graph.py)")
    ap.add_argument("--bd_desc_src", default="proposal", choices=["map", "proposal"],
                     help="bidesc current-side descriptors: map running-mean (id-based) or the frame's raw proposal descs (association-independent)")
    ap.add_argument("--sem_lc_coloc", type=float, default=0.50,
                    help="[ma] stricter coloc gate for tier-B edges (weak SALAD "
                         "prior needs stronger geometry); <=0 uses coloc_ratio")
    ap.add_argument("--depth_max", type=float, default=0.0, help="[ma] clamp far depth (m)")
    ap.add_argument("--gt", default=None, help="[ma] GT trajectory for ATE report")
    ap.add_argument("--conf_gate_coef", type=float, default=1.0,
                    help="[ma] scale on the per-submap mean-conf validity gate "
                         "(<1 keeps more points -> more coverage, noisier voxels)")
    ap.add_argument("--manifold", default="se3", choices=["se3", "sim3"],
                    help="[ma] pose-graph manifold: se3 (metric backbone, default) or sim3 "
                         "(per-submap scale — absorbs DA3 scale drift on textureless scenes)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="semantic checkpoint: a CKPTS name (semconf | semconf_pq | a1 | a1_pq) "
                         "or a .pth path; architecture is read from the checkpoint")
    ap.add_argument("--ranker", default="p_top25q", choices=["obj", "p_top25q", "pq", "max"],
                    help="export cluster confidence (p_top25q needs the P/Q head; "
                         "falls back to objectness otherwise)")
    ap.add_argument("--sem_agg", default="auto", choices=["auto", "dense", "textregion"],
                    help="per-proposal semantic source: dense-head mask-mean or "
                         "frozen-TIPS textregion pool (auto = by checkpoint)")
    ap.add_argument("--kv_policy", default="h2o", choices=["none", "window", "h2o"],
                    help="CausalBranch KV retention: 'none' (flat, unbounded), "
                         "'window' (flat trim), 'h2o' (STAC temporal compression: "
                         "pinned frame-0 + recent window + heavy-hitter tokens)")
    ap.add_argument("--kv_window_size", type=int, default=None,
                    help="[window] sliding-window frames")
    ap.add_argument("--kv_keyframe_stride", type=int, default=None,
                    help="[window] keep every S-th past frame")
    ap.add_argument("--h2o_recent", type=int, default=8,
                    help="[h2o] recent-window budget (frames)")
    ap.add_argument("--h2o_hh", type=int, default=8,
                    help="[h2o] heavy-hitter budget (frame-equivalents)")
    ap.add_argument("--h2o_decay", type=float, default=0.9,
                    help="[h2o] attention-score decay γ")
    ap.add_argument("--h2o_subsample", type=float, default=0.25,
                    help="[h2o] query subsample ratio for colsum scores")
    ap.add_argument("--deploy", action="store_true",
                    help="[ma] after SLAM+export, also write the robot artifact set "
                         "(trajectory json, multi-level instance tables, inst/rgb/sem "
                         "per-point PLYs) into <out>/deploy")
    a = ap.parse_args()

    kv_kwargs = dict(kv_policy=a.kv_policy,
                     kv_window_size=a.kv_window_size,
                     kv_keyframe_stride=a.kv_keyframe_stride,
                     h2o_recent_frames=a.h2o_recent, h2o_hh_frames=a.h2o_hh,
                     h2o_decay=a.h2o_decay, h2o_subsample=a.h2o_subsample)

    model_kwargs = dict(ckpt=a.ckpt, ranker=a.ranker, sem_agg=a.sem_agg,
                        snapshot_dir=(os.path.join(a.out, "snapshots") if a.snapshot else ""),
                        snapshot_pts=a.snapshot_pts)
    if a.geometry == "ma":
        res_kw = {}
        if a.resolution == "auto":                     # portrait input (e.g. 3RScan upright) -> 336x518, as the GT-geometry runs
            import glob as _g, cv2 as _cv2
            _fs = sorted(_g.glob(os.path.join(a.data, "rgb", "*")))
            if _fs:
                _im = _cv2.imread(_fs[0])
                if _im is not None and _im.shape[0] > _im.shape[1]:
                    res_kw["resolution"] = (336, 518)
                    print("[ma_sem] portrait input detected -> semantic resolution 336x518")
        elif a.resolution:
            w_, h_ = [int(x) for x in a.resolution.lower().split("x")]; res_kw["resolution"] = (w_, h_)
        res_kw.update(_assoc_kwargs(a))
        cfg = SemConfig(box_scale_mode="first_chunk", conf_gate_coef=a.conf_gate_coef,
                        **kv_kwargs, **model_kwargs, proposal_conf_mode=a.proposal_conf,
                        lc_proposal_conf_mode=a.lc_proposal_conf, lc_history=a.lc_history, **res_kw)
        run_ma_slam(a, cfg)
        return

    cfg = SemConfig(chunk_size=a.chunk_size, box_scale_mode="gt_seq_median", proposal_conf_mode=a.proposal_conf,
                    save_2d_masks=a.save_2d_masks, mask_2d_top_k=a.mask_top_k,
                    **kv_kwargs, **model_kwargs)
    name = a.name or os.path.basename(os.path.normpath(a.data))
    os.makedirs(a.out, exist_ok=True)

    batch = _load_gt_batch(a.data, a.demo_type, cfg.resolution, a.max_frames)
    T = batch["images"].shape[0]
    print(f"[ma_sem] {T} frames at {tuple(batch['images'].shape[-2:])} from {a.data}")

    model = SemanticModel(cfg)
    model.set_box_scale(gt_seq_box_scale(batch))
    resolver = StaticGeoResolver()
    table = InstanceTable(cfg, resolver)

    t0 = time.time()
    n_done = 0
    for chunk in gt_chunks(batch, cfg.chunk_size):
        out = model.forward_chunk(chunk)
        n = chunk["images"].shape[1]
        if a.save_2d_masks:
            from slam_opense3r.vis_2d_masks import dump_frame_instance_masks
            dump_frame_instance_masks(
                images=chunk["images"].to(model.device), sem_res=out,
                frame_start=chunk["frame_indices"][0],
                out_dir=os.path.join(a.out, "2d_inst_masks"),
                demo_name=name, top_k=cfg.mask_2d_top_k)
        props, pix_map, pts_hd = make_proposals(out, chunk, cfg,
                                                sem_backbone=model.semantic_backbone)
        for t, key in enumerate(chunk["frame_keys"]):
            resolver.register(key, pts_hd[t])
        table.match_chunk(props, chunk["frame_keys"], pix_map)
        n_done += n
        print(f"[ma_sem] frames {n_done:4d}/{T}  proposals={len(props):3d}  "
              f"instances={len(table)}  {n_done / (time.time() - t0):.1f} fps")
        if hasattr(model.policy, "stats"):
            info = model.policy.stats()
            if info is not None:
                print(f"[ma_sem]   h2o-kv: tokens/layer={info['kvcache_size'][0]}"
                      f"  used={info['kvcache_used_mb']:.0f}MB"
                      f"  prune={model.policy.prune_ms:.1f}ms")

    print(f"[ma_sem] streaming done ({time.time() - t0:.1f}s): {len(table)} instances")
    from ma_slam_semantic.export import export_all
    res = export_all(table, num_frames=T, out_dir=a.out, name=name, cfg=cfg)
    print(f"[ma_sem] exported {res['n_instances']} instances "
          f"({res['n_dropped']} noise-filtered) -> {a.out}")


if __name__ == "__main__":
    main()

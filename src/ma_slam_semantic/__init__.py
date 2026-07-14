"""ma_slam_semantic — query-instance semantic mapping on top of ma_slam.

Port of OpenSe3r's `pipeline_query_gt.py` path (DINOv2+TIPSv2-L14 → 12-block causal
transformer with KV cache → Mask2Former query head, FastSAM query seeding) driven by
ma_slam geometry through the S1 hooks. See docs/semantic_branch_plan.md.

Phase-1 imports OpenSe3r code via ``SemConfig.opense3r_root`` (sys.path); nothing from
OpenSe3r is vendored here.
"""

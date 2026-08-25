"""PARTIAL vendor of OpenSe3r's ``semantic_mapping`` package (2026-08-25).

Only the checkpoint-driven model builder helpers (``model.py``: ``read_ckpt_args``,
``_arch_kwargs``, ``_infer_arch_from_weights``, ``TRAINABLE_PREFIXES``) and the
cluster ranking functions (``stages/ranking.py``) are vendored — ma_slam_semantic
has its own pipeline/config/data. The live package's ``__init__`` imports its whole
pipeline, so it is deliberately NOT mirrored here.
"""

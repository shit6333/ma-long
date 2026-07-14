"""
GPU / CPU memory diagnostics for slam_opense3r pipelines.

Every pipeline (``pipeline.py``, ``pipeline_query.py``, ``*_voxel_kv.py``,
``*_gt.py``) appends a one-line ``[MEM]`` breakdown after its per-step log so
you can see where memory is going:

    [MEM]  GPU cur=2103MB res=3072MB  geo_pk=4218MB  sem_pk=3805MB
           kv=412MB(12L/2912tok)  sem_store=58MB(cpu)  ins_store=2MB(cpu)
           inst_table=4MB(cpu, 37 inst)

Important caveats:

* ``VoxelFeatureMap`` / ``PerPointFeatureStore`` / ``InstanceTable`` keep all
  their tensors on **CPU**.  They do not consume VRAM — reported separately
  under the ``(cpu)`` suffix.
* ``kv=...`` covers only the flat per-layer KV cache (``CausalBranch.global_blocks``).
  For the ``ThreeTierKVManager`` variants (voxel_kv pipelines) use
  :func:`voxel_kv_manager_breakdown` instead — that splits the cache into
  anchor / window / voxel-pivot byte counts.
* Peak figures (``geo_pk``, ``sem_pk``) are captured via
  ``torch.cuda.reset_peak_memory_stats`` before each forward and
  ``max_memory_allocated`` after.  They represent peak DURING that call, not
  current.

This module is deliberately defensive — it never raises on attribute errors,
just returns zero — so wiring it in cannot break a working pipeline.
"""
from __future__ import annotations
from typing import Iterable, Optional, Tuple

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mb(num_bytes: float) -> float:
    return num_bytes / 1024.0 ** 2


def _tensor_bytes(t) -> int:
    if not torch.is_tensor(t):
        return 0
    return t.numel() * t.element_size()


# ──────────────────────────────────────────────────────────────────────────────
# GPU snapshot + peak tracking
# ──────────────────────────────────────────────────────────────────────────────

def gpu_alloc_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return _mb(torch.cuda.memory_allocated())


def gpu_reserved_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return _mb(torch.cuda.memory_reserved())


def reset_gpu_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return _mb(torch.cuda.max_memory_allocated())


class GpuPeakTracker:
    """``with GpuPeakTracker(slam._mem_peaks, 'geo'): ...``

    Resets the CUDA peak counter on enter and writes the post-block peak (MB)
    into ``target[name]`` on exit.
    """

    def __init__(self, target: dict, name: str):
        self.target = target
        self.name = name

    def __enter__(self):
        reset_gpu_peak()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.target[self.name] = gpu_peak_mb()
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Flat KV cache (list-of-[K,V] format used by CausalBranch.global_blocks)
# ──────────────────────────────────────────────────────────────────────────────

def kv_cache_stats(kv_cache_list) -> Tuple[int, int, int, int]:
    """Returns ``(n_populated_layers, gpu_bytes, cpu_bytes, past_tokens)``.

    ``past_tokens`` is the K-cache token count (last 2nd dim of the K tensor),
    which is the same across all populated layers in a CausalBranch cache.
    """
    if kv_cache_list is None:
        return (0, 0, 0, 0)
    gpu_b = 0
    cpu_b = 0
    n_pop = 0
    tokens = 0
    for layer in kv_cache_list:
        if layer is None:
            continue
        try:
            k_cache, v_cache = layer[0], layer[1]
        except (IndexError, TypeError):
            continue
        if k_cache is None:
            continue
        n_pop += 1
        for t in (k_cache, v_cache):
            b = _tensor_bytes(t)
            if torch.is_tensor(t) and t.is_cuda:
                gpu_b += b
            else:
                cpu_b += b
        if k_cache.dim() >= 2:
            tokens = max(tokens, k_cache.shape[-2])
    return (n_pop, gpu_b, cpu_b, tokens)


# ──────────────────────────────────────────────────────────────────────────────
# Feature stores  (VoxelFeatureMap / PerPointFeatureStore — all CPU)
# ──────────────────────────────────────────────────────────────────────────────

def store_bytes(store) -> int:
    """Bytes used by a feature store's backing tensors."""
    if store is None:
        return 0
    total = 0
    # VoxelFeatureMap: _feat_sum + _conf_sum (+ optional coord arrays)
    for attr in ('_feat_sum', '_conf_sum', '_coords', '_coord_to_idx'):
        total += _tensor_bytes(getattr(store, attr, None))
    # PerPointFeatureStore: list-of-tensors
    for attr in ('_pts_list', '_feat_list'):
        lst = getattr(store, attr, None)
        if isinstance(lst, (list, tuple)):
            for t in lst:
                total += _tensor_bytes(t)
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Instance table  (CPU)
# ──────────────────────────────────────────────────────────────────────────────

_INST_REC_TENSOR_ATTRS = (
    'descriptor', 'sem_query', 'center', 'bbox_min', 'bbox_max',
    'pred_bbox_min', 'pred_bbox_max',
)


def instance_table_bytes(table) -> Tuple[int, int]:
    """Returns ``(approx_bytes, num_records)``.

    Walks the small per-record tensors plus the table's own bbox cache.  The
    per-record ``voxels`` set / ``voxel_votes`` dict are Python objects (not
    tensors) — we approximate them as ``num_voxels * 56 B`` (rough size of
    a 3-int tuple + dict entry overhead).
    """
    if table is None:
        return (0, 0)
    records = getattr(table, 'records', None) or getattr(table, '_records', None)
    if records is None:
        return (0, 0)
    items = records.values() if hasattr(records, 'values') else records
    total = 0
    n = 0
    for rec in items:
        n += 1
        for attr in _INST_REC_TENSOR_ATTRS:
            total += _tensor_bytes(getattr(rec, attr, None))
        voxels = getattr(rec, 'voxels', None)
        if voxels is not None:
            total += len(voxels) * 56
    # Bbox cache
    for attr in ('_cache_bmin', '_cache_bmax'):
        arr = getattr(table, attr, None)
        if arr is not None and hasattr(arr, 'nbytes'):
            total += int(arr.nbytes)
    return (total, n)


# ──────────────────────────────────────────────────────────────────────────────
# ThreeTierKVManager (voxel_kv pipelines)
# ──────────────────────────────────────────────────────────────────────────────

def _frame_kv_bytes(frame) -> Tuple[int, int]:
    """Return (gpu_b, cpu_b) for a single ``FrameKV``."""
    gpu_b, cpu_b = 0, 0
    for lst_name in ('K_per_layer', 'V_per_layer'):
        lst = getattr(frame, lst_name, None) or []
        for t in lst:
            b = _tensor_bytes(t)
            if torch.is_tensor(t) and t.is_cuda:
                gpu_b += b
            else:
                cpu_b += b
    return gpu_b, cpu_b


def voxel_kv_manager_breakdown(mgr) -> dict:
    """Anchor / window / voxel store byte breakdown for a ThreeTierKVManager.

    Returns a dict with keys ``anchor_gpu/cpu``, ``window_gpu/cpu``,
    ``voxel_gpu/cpu``, ``total_gpu/cpu``.  All values are bytes.
    """
    out = {k: 0 for k in (
        'anchor_gpu', 'anchor_cpu',
        'window_gpu', 'window_cpu',
        'voxel_gpu',  'voxel_cpu',
        'total_gpu',  'total_cpu',
    )}
    if mgr is None:
        return out

    # ── Anchor ──
    anc = getattr(mgr, 'anchor', None)
    if anc is not None:
        if getattr(anc, 'first_frame_kv', None) is not None:
            g, c = _frame_kv_bytes(anc.first_frame_kv)
            out['anchor_gpu'] += g
            out['anchor_cpu'] += c
        for fr in getattr(anc, 'pool', {}).values():
            g, c = _frame_kv_bytes(fr)
            out['anchor_gpu'] += g
            out['anchor_cpu'] += c

    # ── Window ──
    win = getattr(mgr, 'window', None)
    if win is not None:
        for chunk in getattr(win, 'chunks', ()):
            for fr in chunk:
                g, c = _frame_kv_bytes(fr)
                out['window_gpu'] += g
                out['window_cpu'] += c

    # ── Voxel store ──
    vox = getattr(mgr, 'voxel', None)
    if vox is not None:
        for pivot_list in getattr(vox, 'store', {}).values():
            for pv in pivot_list:
                for attr in ('K_per_layer', 'V_per_layer'):
                    t = getattr(pv, attr, None)
                    if torch.is_tensor(t):
                        b = _tensor_bytes(t)
                        if t.is_cuda:
                            out['voxel_gpu'] += b
                        else:
                            out['voxel_cpu'] += b

    out['total_gpu'] = out['anchor_gpu'] + out['window_gpu'] + out['voxel_gpu']
    out['total_cpu'] = out['anchor_cpu'] + out['window_cpu'] + out['voxel_cpu']
    return out


# ──────────────────────────────────────────────────────────────────────────────
# One-line formatter
# ──────────────────────────────────────────────────────────────────────────────

def format_mem_log(memory,
                   *,
                   peaks: Optional[dict] = None,
                   kv_cache_list = None,
                   voxel_kv_manager = None,
                   extra: str = "") -> str:
    """Format a one-line ``[MEM]`` summary for the given SLAM step.

    Args:
        memory: ``OpenSe3rMemory`` / ``OpenSe3rQueryMemory`` (any object with
            ``sem_store`` / ``ins_store`` / optional ``instance_table``).
        peaks: dict of ``{name: peak_mb}`` — usually
            ``{'geo': ..., 'sem': ...}`` populated by ``GpuPeakTracker``.
        kv_cache_list: flat ``[[K, V], ...]`` cache (non-voxel pipelines).
        voxel_kv_manager: ``ThreeTierKVManager`` (voxel_kv pipelines); if
            given, replaces ``kv=...`` with an anchor/window/voxel split.
        extra: optional trailing string (e.g. extra diagnostics).
    """
    parts = [f"GPU cur={gpu_alloc_mb():.0f}MB res={gpu_reserved_mb():.0f}MB"]

    if peaks:
        if 'geo' in peaks:
            parts.append(f"geo_pk={peaks['geo']:.0f}MB")
        if 'sem' in peaks:
            parts.append(f"sem_pk={peaks['sem']:.0f}MB")
        # any other peaks the pipeline tracks
        for k, v in peaks.items():
            if k in ('geo', 'sem'):
                continue
            parts.append(f"{k}_pk={v:.0f}MB")

    # KV cache — voxel manager takes precedence (its own three-tier breakdown)
    if voxel_kv_manager is not None:
        br = voxel_kv_manager_breakdown(voxel_kv_manager)
        parts.append(
            f"kv=anc{_mb(br['anchor_gpu']):.0f}/win{_mb(br['window_gpu']):.0f}"
            f"/vox{_mb(br['voxel_gpu']):.0f}MB(gpu)"
        )
        if br['total_cpu'] > 0:
            parts.append(f"kv_cpu={_mb(br['total_cpu']):.0f}MB")
    else:
        n_pop, kv_gpu_b, kv_cpu_b, kv_tok = kv_cache_stats(kv_cache_list)
        if n_pop > 0 or kv_gpu_b > 0:
            parts.append(
                f"kv={_mb(kv_gpu_b):.0f}MB(gpu,{n_pop}L/{kv_tok}tok)"
            )
            if kv_cpu_b > 0:
                parts.append(f"kv_cpu={_mb(kv_cpu_b):.0f}MB")

    sem_b = store_bytes(getattr(memory, 'sem_store', None))
    parts.append(f"sem_store={_mb(sem_b):.0f}MB(cpu)")

    ins_b = store_bytes(getattr(memory, 'ins_store', None))
    if ins_b > 0:
        parts.append(f"ins_store={_mb(ins_b):.0f}MB(cpu)")

    inst_b, n_inst = instance_table_bytes(getattr(memory, 'instance_table', None))
    if inst_b > 0 or n_inst > 0:
        parts.append(f"inst_table={_mb(inst_b):.0f}MB(cpu,{n_inst} inst)")

    if extra:
        parts.append(extra)

    return "  [MEM] " + "  ".join(parts)

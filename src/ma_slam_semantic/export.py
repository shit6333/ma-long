"""Export — writes the reference-format artifacts via OpenSe3r's OWN writers
(``amb3r/vis_query_instance_slam.py``), duck-typed over our provenance table, so the
old repo's eval / query scripts run unmodified on the output:

    <name>_query_instance_table.json      instance metadata
    <name>_query_instance_voxels.ply      per-voxel instance colors
    <name>_query_instance_boxes.ply       AABB wireframes
    <name>_instance_labels.npz            centers/cluster_id/... (eval_instance_npz_*)
    <name>_instance_sem_queries.npz       inst_ids + sem_queries [K,1024] + voxel_size

Export sequence: final re-lift of every instance from provenance (through the FINAL
graph) -> final merge pass (heals pre-LC duplicate spawns) -> noise filter -> write.
"""

from __future__ import annotations

import os
from typing import Dict

from ma_slam_semantic.config import SemConfig
from ma_slam_semantic.instance_map import InstanceTable


class _MemShim:
    """Just enough of ``OpenSe3rQueryMemory`` for the export writers."""

    def __init__(self, table: InstanceTable, num_frames: int):
        self.instance_table = table
        self.num_frames = num_frames


def export_all(table: InstanceTable, num_frames: int, out_dir: str, name: str,
               cfg: SemConfig, do_final_merge: bool = True) -> Dict:
    from amb3r.vis_query_instance_slam import (
        export_query_instance_voxels_ply,
        export_query_instance_boxes_ply,
        dump_instance_table_json,
        dump_instance_labels_npz,
        dump_instance_sem_queries_npz,
        color_map_for_table,
        filter_noise_instances,
    )
    os.makedirs(out_dir, exist_ok=True)

    n_relift = table.rekey_all(force=True)          # final heal through the final graph
    n_merged = table.final_merge() if do_final_merge else 0
    print(f"[export] re-lifted {n_relift} instances, final-merged away {n_merged} "
          f"-> {len(table)} instances")

    mem = _MemShim(table, num_frames)
    dropped = filter_noise_instances(
        table, total_frames=num_frames,
        min_count=cfg.filter_min_count,
        end_grace=cfg.filter_end_grace,
        max_size_m3=cfg.filter_max_size_m3,
        oversized_max_count=cfg.filter_oversized_max_count,
        tiny_voxels=cfg.filter_tiny_voxels,
        tiny_max_count=cfg.filter_tiny_max_count,
        min_avg_objectness=cfg.filter_min_avg_objectness,
        max_extent_m=cfg.filter_max_extent_m,
        elongated_max_count=cfg.filter_elongated_max_count,
    )
    color_map = color_map_for_table(table, mode="desc")
    print(f"[export] noise filter dropped {len(dropped)}/{len(table)}")

    paths = {}
    paths["voxels_ply"] = os.path.join(out_dir, f"{name}_query_instance_voxels.ply")
    export_query_instance_voxels_ply(mem, paths["voxels_ply"], transformation=None,
                                     color_map=color_map, color_mode="desc",
                                     dropped_ids=dropped, verbose=True)
    paths["boxes_ply"] = os.path.join(out_dir, f"{name}_query_instance_boxes.ply")
    export_query_instance_boxes_ply(mem, paths["boxes_ply"], transformation=None,
                                    color_map=color_map, color_mode="desc",
                                    dropped_ids=dropped, verbose=True)
    paths["table_json"] = os.path.join(out_dir, f"{name}_query_instance_table.json")
    dump_instance_table_json(mem, paths["table_json"], transformation=None,
                             dropped_ids=dropped, verbose=True)
    paths["labels_npz"] = os.path.join(out_dir, f"{name}_instance_labels.npz")
    dump_instance_labels_npz(mem, paths["labels_npz"], transformation=None,
                             dropped_ids=dropped, verbose=True, objectness_agg="mean")
    paths["sem_queries_npz"] = os.path.join(out_dir, f"{name}_instance_sem_queries.npz")
    dump_instance_sem_queries_npz(mem, paths["sem_queries_npz"],
                                  dropped_ids=dropped, verbose=True)
    return dict(n_instances=len(table), n_dropped=len(dropped),
                n_relift=n_relift, n_merged=n_merged, paths=paths)

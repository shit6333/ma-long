"""Text-query labelling of exported instances — thin wrapper over OpenSe3r's
``query_instance_semantic.py`` (TIPSv2 SigLIP text head; one label per instance,
broadcast to its voxels). Runs on the export artifacts, so it works identically for
golden-gt and ma_slam-geometry outputs.

    from ma_slam_semantic.query_text import label_instances
    label_instances(out_dir, name, benchmark=20)

or from the shell (identical to the reference repo usage):

    cd /mnt/HDD4/ricky/OpenSe3r && python query_instance_semantic.py \
        --instance_labels_npz <out>/<name>_instance_labels.npz \
        --instance_sem_queries_npz <out>/<name>_instance_sem_queries.npz \
        --benchmark 20 --backbone tips-v2-l14 --output_dir <out> --demo_name <name>
"""

from __future__ import annotations

import os
import subprocess
import sys

from ma_slam_semantic.config import OPENSE3R_ROOT


def label_instances(out_dir: str, name: str, benchmark: int = 20,
                    backbone: str = "tips-v2-l14", geo_ply: str = None,
                    python: str = sys.executable) -> str:
    """Label exported instances via text queries; returns the output dir."""
    out_dir = os.path.abspath(out_dir)
    cmd = [
        python, os.path.join(OPENSE3R_ROOT, "query_instance_semantic.py"),
        "--instance_labels_npz", os.path.join(out_dir, f"{name}_instance_labels.npz"),
        "--instance_sem_queries_npz", os.path.join(out_dir, f"{name}_instance_sem_queries.npz"),
        "--benchmark", str(benchmark),
        "--backbone", backbone,
        "--output_dir", out_dir,
        "--demo_name", name,
    ]
    if geo_ply:
        cmd += ["--geo_ply", os.path.abspath(geo_ply)]
    print(f"[query_text] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=OPENSE3R_ROOT, check=True)
    return out_dir

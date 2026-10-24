"""Run VGGT-SLAM 1.0 (Sim3 path) INSIDE the amb3r_bw env — the env of our semantic stack.

Their SL(4) mode needs gtsam-develop (py3.11 only); the Sim3 path only needs Pose3, the
facebookresearch `vggt` we already vendor, and SALAD. Viewer-only deps are stubbed, and
`pytorch_lightning` (SALAD's VPRModel base class) gets a 3-line shim. Usage = their main.py:

    PYTHONPATH=src python scripts/vggt1_lc/run_vggt1_in_amb3r.py --image_folder ... --use_sim3 ...
"""
import os
import runpy
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = os.path.join(ROOT, "thirdparty", "vggt-slam-1", "VGGT-SLAM")
for p in (os.path.join(ROOT, "vendor", "thirdparty"),          # facebookresearch vggt (vendored)
          os.path.join(V1, "third_party", "salad"), V1):
    if p not in sys.path:
        sys.path.insert(0, p)
from unittest import mock
for m in ("viser", "viser.transforms", "gradio", "trimesh",       # viewer-only in 1.0;
          "pytorch_metric_learning", "pytorch_metric_learning.losses",   # SALAD builds a train
          "pytorch_metric_learning.miners", "pytorch_metric_learning.distances",  # loss/miner in
          "pytorch_metric_learning.reducers", "prettytable"):        # __init__ (unused at inference)
    sys.modules.setdefault(m, mock.MagicMock(name=m))              # Solver() builds a
                                                                  # ViserServer even w/o --vis_map
class _LightningShim(torch.nn.Module):                            # SALAD VPRModel base, inference only
    def save_hyperparameters(self, *a, **k): pass
    def log(self, *a, **k): pass
pl = types.ModuleType("pytorch_lightning")
pl.LightningModule = _LightningShim
sys.modules.setdefault("pytorch_lightning", pl)
os.chdir(V1)                                                      # their relative paths (hub ckpt etc.)
sys.argv[0] = os.path.join(V1, "main.py")
runpy.run_path(sys.argv[0], run_name="__main__")

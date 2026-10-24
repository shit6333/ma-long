"""Run VGGT-SLAM 1.0's main.py (any env) with torch.hub's GitHub validation disabled.

Their SALAD loads `serizba/salad` and `facebookresearch/dinov2` via torch.hub; even with a warm
cache torch.hub validates the repo against the GitHub API on every start and blocks forever when
GitHub is slow / rate-limited (observed: hours-long hangs with an open socket to github.com).
Usage: python scripts/vggt1_lc/run_v1_nohub.py <path/to/VGGT-SLAM/main.py> [main.py args...]
"""
import os, runpy, sys
import torch.hub as hub
for name in ("_validate_not_a_forked_repo",):
    if hasattr(hub, name):
        setattr(hub, name, lambda *a, **k: None)
main = os.path.abspath(sys.argv[1])
os.chdir(os.path.dirname(main))
sys.argv = [main] + sys.argv[2:]
runpy.run_path(main, run_name="__main__")

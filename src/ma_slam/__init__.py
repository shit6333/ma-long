"""ma_slam — VGGT-SLAM-style submap SLAM on a MapAnything backbone.

Faithful to VGGT-SLAM's architecture (submaps + global factor graph + incremental
re-optimization + SALAD loop closure), but optimizing on **metric SE3** (upstream gtsam
``Pose3``) instead of SL(4), since MapAnything is calibrated/metric. Supports the four
MapAnything input modes (rgb / rgb+intr / rgb+depth / rgb+depth+intr).
"""

from ma_slam.solver import MaSlam, DEFAULT_CONFIG

__all__ = ["MaSlam", "DEFAULT_CONFIG"]

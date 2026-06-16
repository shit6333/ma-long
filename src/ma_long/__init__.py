"""ma_long — MapAnything-based long-sequence SLAM / reconstruction.

A standalone long pipeline (chunk -> overlap SE3 alignment -> loop closure ->
global pose-graph optimization -> merge) built on top of Meta's MapAnything,
with multi-modal input support (rgb / rgb+depth / rgb+intr / rgb+depth+intr).

The geometry / optimization utilities under `align` and `fastloop`
are vendored (and lightly adapted) from VGGT-Long / DA3-Streaming so that the
package is self-contained and portable (no runtime dependency on the sibling
thirdparty repos). The only external model dependency is `mapanything`.
"""

__version__ = "0.0.1"

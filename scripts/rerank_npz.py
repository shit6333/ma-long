"""Re-score an exported ``*_instance_labels.npz`` with a different cluster ranker,
WITHOUT re-running SLAM — same clusters, only ``cluster_score`` changes.

This is the control used for the P/Q ranking claim (docs/opense3r_model_handoff.md
§2A / §6): evaluate the same map ranked by objectness vs p_top25q.

    PYTHONPATH=src python scripts/rerank_npz.py --run outputs/.../<scene_dir> \
        --ranker obj --suffix _objrank
    -> <scene>_instance_labels<suffix>.npz next to the original

Member scores come from ``<name>_query_instance_table.json`` (member_objectness /
member_pq_p / member_pq_q per instance, written by ma_slam_semantic.export).
"""
import argparse
import glob
import json
import os

import numpy as np

from ma_slam_semantic.config import ensure_opense3r


class _Rec:
    def __init__(self, d):
        self.member_objectness = list(d["member_objectness"])
        self.member_pq_p = list(d.get("member_pq_p", []))
        self.member_pq_q = list(d.get("member_pq_q", []))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="scene output dir")
    ap.add_argument("--ranker", default="obj", choices=["obj", "p", "q", "pq", "p_top25q"])
    ap.add_argument("--suffix", default=None, help="default _<ranker>rank")
    a = ap.parse_args()
    ensure_opense3r()
    from semantic_mapping.stages.ranking import score_record

    npz = [f for f in glob.glob(os.path.join(a.run, "*_instance_labels*.npz"))
           if "rank" not in os.path.basename(f)]
    tab = glob.glob(os.path.join(a.run, "*_query_instance_table.json"))
    if len(npz) != 1 or len(tab) != 1:
        raise SystemExit(f"expected one labels npz + one table json in {a.run}: {npz} {tab}")
    d = dict(np.load(npz[0]))
    T = json.load(open(tab[0]))
    inst = T["instances"] if isinstance(T, dict) else T
    recs = {int(r["inst_id"]): _Rec(r) for r in inst}
    ids = d["score_ids"].astype(int)
    new = np.array([score_record(recs[i], a.ranker) if i in recs else 0.0 for i in ids],
                   dtype=np.float32)
    n_pq = sum(1 for i in ids if i in recs and recs[i].member_pq_p)
    print(f"[rerank] {len(ids)} instances, {n_pq} with P/Q, ranker={a.ranker}: "
          f"score mean {d['cluster_score'].mean():.3f} -> {new.mean():.3f}")
    d["cluster_score"] = new
    suf = a.suffix or f"_{a.ranker}rank"
    out = npz[0].replace("_instance_labels.npz", f"_instance_labels{suf}.npz")
    np.savez_compressed(out, **d)
    print("[rerank] ->", out)


if __name__ == "__main__":
    main()

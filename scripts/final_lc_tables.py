"""Aggregate every instance-LC experiment into the final tables (TUM fr1 50/25 px, Clio default/relaxed),
with GT co-visibility labels of accepted loops (true / weak / FALSE) for our runs.

    PYTHONPATH=src:scripts/vggt1_lc python scripts/final_lc_tables.py [--no_gt]
"""
import argparse, csv, glob, os, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vggt1_lc"))
from loop_precision import load_gt, covis  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQS = ["360", "desk", "desk2", "floor", "plant", "room", "rpy", "teddy", "xyz"]
TUM_ROWS = [("SALAD only (no coloc, no inst)", "SALADONLY"), ("base (no inst-LC)", None), ("+S1", "a1pq_w32_kf{kf}:s1"), ("A: S1+S2 old rules", "a1pq_w32_kf{kf}:s1s2"),
            ("B: rigid", "a1pq_w32_kf{kf}_rigid:s1s2"), ("B-S1: pairs only", "a1pq_w32_kf{kf}_bs1:s1s2"),
            ("B lowobj->coloc", "a1pq_w32_kf{kf}_rigidcoloc:s1s2"),
            ("S1 only, NO coloc", "a1pq_w32_kf{kf}_nocoloc_bs1:s1s2"), ("S1+S2, NO coloc", "a1pq_w32_kf{kf}_nocoloc_rigid:s1s2"),
            ("joint", "a1pq_w32_kf{kf}_joint:s1s2"),
            ("joint-S1: pairs only", "a1pq_w32_kf{kf}_jointS1:s1s2")]
ACC = re.compile(r"LOOPCAND ACCEPT coloc=([\d.]+) sim=([\d.]+).*?frame_(\d+)\.(?:png|jpg) mimg=.*?frame_(\d+)\.(?:png|jpg)")
_gt = {}


def label_loops(run_dir, data_dir, depth_scale):
    """(n_true, n_weak, n_false) over unique accepted loops in run_dir/loops.txt."""
    f = os.path.join(run_dir, "loops.txt")
    if not os.path.exists(f) or not os.path.exists(os.path.join(data_dir, "gt_pose.txt")):
        return None
    if data_dir not in _gt:
        kf = "depth_intrinsic.txt" if os.path.exists(os.path.join(data_dir, "depth_intrinsic.txt")) else "intrinsic.txt"
        _gt[data_dir] = (load_gt(os.path.join(data_dir, "gt_pose.txt")), np.loadtxt(os.path.join(data_dir, kf))[:3, :3])
    gt, K = _gt[data_dir]
    seen, c = set(), [0, 0, 0]
    for l in open(f):
        m = ACC.search(l)
        if not m:
            continue
        q, r = int(m.group(3)), int(m.group(4))
        if (q, r) in seen:
            continue
        seen.add((q, r))
        cv = covis(gt, q, r, K, os.path.join(data_dir, "depth"), depth_scale)
        c[0 if cv >= 0.2 else (1 if cv >= 0.05 else 2)] += 1
    return tuple(c)


def fmt_loops(c):
    if c is None:
        return "?"
    t, w, f = c
    return f"{t}" + (f"+{w}w" if w else "") + (f"+**{f}F**" if f else "")


def tum(kf, use_gt):
    print(f"\n### TUM fr1, {kf} px keyframer, w32 (Sim3-ATE; loops = accepted true[+weak][+FALSE])\n")
    print("| method | " + " | ".join(SEQS) + " | avg | loops |")
    print("|---|" + "---|" * (len(SEQS) + 2))
    # base from ma_slam runs
    for name, key in TUM_ROWS:
        vals, loops = [], []
        for s in SEQS:
            if key == "SALADONLY":
                if kf != 50:
                    rd = None; log = None; vals.append(None); loops.append(None); continue
                rd = os.path.join(ROOT, "outputs", "ma_slam", "tum_kf50_saladonly", "da3_rgb", f"fr1_{s}")
                log = os.path.join(rd, "run.log")
            elif key is None:
                rd = os.path.join(ROOT, "outputs", "ma_slam", "tum_kf50", "da3_rgb", f"fr1_{s}") if kf == 50 else \
                     os.path.join(ROOT, "outputs", "sem_lc_shadow", "tum_w32", f"fr1_{s}_base")   # the RESULTS.md 25 px base (0.0557)
                log = os.path.join(rd, "run.log")
            else:
                tag, cfg = key.format(kf=kf).split(":")
                if kf == 25 and tag == "a1pq_w32_kf25":
                    tag = "newmodel_w32"          # the 25 px S1 / S1+S2 (old rules) runs
                rd = os.path.join(ROOT, "outputs", "sem_lc_shadow", tag, cfg, f"fr1_{s}")
                log = os.path.join(rd, "run.log")
            v = None
            if log and os.path.exists(log):
                for l in open(log):
                    m = re.search(r"Sim3-ATE=([\d.]+)", l)
                    if m:
                        v = float(m.group(1))
            vals.append(v)
            loops.append(label_loops(rd, os.path.join(ROOT, "data", "tum", f"fr1_{s}_dist"), 5000.0) if (use_gt and rd) else None)
        if all(v is None for v in vals):
            continue
        avg = np.mean([v for v in vals if v is not None]) if all(v is not None for v in vals) else None
        print(f"| {name} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in vals) + f" | {'—' if avg is None else f'{avg:.4f}'} | " + "/".join(fmt_loops(c) for c in loops) + " |")


TUM_RELAX_ROWS = [("SALAD only (no coloc, no inst)", ("ma_slam/tum_kf50_relax_saladonly/da3_rgb", "")), ("base (no inst-LC)", ("ma_slam/tum_kf50_relax/da3_rgb", "")), ("A: S1+S2 old rules", ("sem_lc_shadow/a1pq_w32_kf50_relax", "s1s2")),
                  ("B: rigid", ("sem_lc_shadow/a1pq_w32_kf50_relax_rigid", "s1s2")), ("B-S1: pairs only", ("sem_lc_shadow/a1pq_w32_kf50_relax_bs1", "s1s2")),
                  ("S1 only, NO coloc", ("sem_lc_shadow/a1pq_w32_kf50_relax_nocoloc_bs1", "s1s2")), ("S1+S2, NO coloc", ("sem_lc_shadow/a1pq_w32_kf50_relax_nocoloc_rigid", "s1s2"))]


def tum_relax(use_gt):
    print("\n### TUM fr1, 50 px, RELAXED retrieval (SALAD cos 0.40, 5/submap) (Sim3-ATE; loops = true[+weak][+FALSE])\n")
    print("| method | " + " | ".join(SEQS) + " | avg | loops |")
    print("|---|" + "---|" * (len(SEQS) + 2))
    for name, (root, cfg) in TUM_RELAX_ROWS:
        vals, loops = [], []
        for s in SEQS:
            rd = os.path.join(ROOT, "outputs", root, cfg, f"fr1_{s}") if cfg else os.path.join(ROOT, "outputs", root, f"fr1_{s}")
            log = os.path.join(rd, "run.log"); v = None
            if os.path.exists(log):
                for l in open(log):
                    m = re.search(r"Sim3-ATE=([\d.]+)", l)
                    if m:
                        v = float(m.group(1))
            vals.append(v)
            loops.append(label_loops(rd, os.path.join(ROOT, "data", "tum", f"fr1_{s}_dist"), 5000.0) if use_gt else None)
        if all(v is None for v in vals):
            continue
        avg = np.mean(vals) if all(v is not None for v in vals) else None
        print(f"| {name} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in vals) + f" | {'—' if avg is None else f'{avg:.4f}'} | " + "/".join(fmt_loops(c) for c in loops) + " |")


def clio(variant, use_gt):
    out = os.path.join(ROOT, "outputs", "clio_kf50" + ("_relax" if variant == "relax" else ""))
    rows = list(csv.DictReader(open(os.path.join(out, "summary.csv"))))
    order = ["v1_sl4", "v1_sim3", "pA", "pB", "pC", "base", "s1", "s1s2", "rigid", "bs1", "rigidcoloc", "joint", "jointS1"]
    names = {"v1_sl4": "1.0 SL(4)", "v1_sim3": "1.0 Sim3", "pA": "1.0 Sim3 plug-in L2 0.80", "pB": "1.0 relaxed, no gate",
             "pC": "1.0 relaxed + our gate", "base": "ours base", "s1": "+S1", "s1s2": "A: S1+S2", "rigid": "B: rigid",
             "bs1": "B-S1: pairs only", "rigidcoloc": "B lowobj->coloc", "joint": "joint", "jointS1": "joint-S1: pairs only"}
    scenes = ["apartment", "office", "cubicle"]
    print(f"\n### Clio, 50 px, {'RELAXED retrieval (cos 0.40, 5/submap; 1.0 L2 1.10)' if variant=='relax' else 'default retrieval'}\n")
    print("| method | " + " | ".join(scenes) + " |")
    print("|---|---|---|---|")
    for m in order:
        cells = []
        for sc in scenes:
            r = [x for x in rows if x["scene"] == sc and x["method"] == m and x["sim3_ate"]]
            if not r:
                cells.append("—"); continue
            r = r[-1]
            lab = label_loops(os.path.join(out, f"{sc}_{m}"), os.path.join(ROOT, "data", "clio", sc), 1000.0) if (use_gt and m in ("base", "s1", "s1s2", "rigid", "bs1", "rigidcoloc", "joint", "jointS1")) else None
            cells.append(f"{float(r['sim3_ate']):.3f} ({fmt_loops(lab) if lab else r['n_loops']})")
        if all(c == "—" for c in cells):
            continue
        print(f"| {names[m]} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--no_gt", action="store_true"); a = ap.parse_args()
    for kf in (50, 25):
        tum(kf, not a.no_gt)
    tum_relax(not a.no_gt)
    for v in ("default", "relax"):
        clio(v, not a.no_gt)

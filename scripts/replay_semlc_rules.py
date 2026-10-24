"""Offline replay of the semantic-LC gate rules over RECORDED candidates, GT-labelled.

For every tier-A candidate in a semantic run's loops.txt (BIDESC / BILAYOUT lines carry
the S1 status and the S2 metrics), join the coloc value the SAME candidate got in the
matching base run (no semantic gate -> LOOPCAND lines carry coloc), label truth with GT
co-visibility, and re-decide under several rule variants. Reports, per variant, how many
true / false loops end up ADDED. No GPU; ATE needs a real run, this is the loop-level view.

    python scripts/replay_semlc_rules.py --sem outputs/sem_lc_shadow/a1pq_w32_kf50/s1s2 \
        --base outputs/ma_slam/tum_kf50/da3_rgb --tag 50px
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vggt1_lc"))
from loop_precision import covis, load_gt  # noqa: E402

BID = re.compile(r"BIDESC A (\w+) .*?frame_(\d+)\.png mimg=.*frame_(\d+)\.png")
LAY = re.compile(r"BILAYOUT A (\w+) Ed=([\d.]+) thR=(\d+) ER=(\d+) M=(\d+).*?tnorm=([\d.]+) s2=([\d.]+) dref=([\d.]+).*?frame_(\d+)\.png mimg=.*frame_(\d+)\.png")
CAND = re.compile(r"LOOPCAND (ACCEPT|REJECT) coloc=([\d.]+) sim=([\d.]+).*?frame_(\d+)\.png mimg=.*frame_(\d+)\.png")

COLOC, COLOC_VETO = 0.70, 0.45


def s2_status(m, variant):
    """Re-derive the S2 status from logged metrics under a rule variant."""
    if m is None:
        return None
    st, ed, thr, er, M, tn, dr = m["st"], m["ed"], m["thr"], m["er"], m["M"], m["tn"], m["dr"]
    if M < 3:
        return "M_LOW"
    if M < 4:
        return "PASS" if ed <= 0.3 else "FAIL"
    if variant.get("lvc_first") and thr > 50:
        return "LARGE_VIEW_CHANGE"      # layout says the views are rotated >50deg: not co-visible
    gate = 1.0 + (dr * np.sin(np.radians(min(er, 90))) if variant.get("noise") else 0.0)
    if tn > gate:
        return "T_FAR"
    if ed <= 0.3 and er <= 25:
        return "PASS" if thr <= 50 else "LARGE_VIEW_CHANGE"
    return "FAIL"


def decide(c, variant):
    s1, m, cl = c["s1"], c["lay"], c["coloc"]
    if s1 == "UNKNOWN":
        if variant.get("unknown") == "fail":
            return "REJ:UNKFAIL"
        return "ADD" if (cl is not None and cl <= COLOC) else "REJ:coloc"
    st = s2_status(m, variant)
    hard = variant.get("hard", ("LARGE_VIEW_CHANGE", "T_FAR", "M_LOW"))
    if variant.get("fail_hard") and (st == "FAIL"):
        return "REJ:S2:FAIL"
    if st in hard:
        return f"REJ:S2:{st}"
    fail = s1 == "FAIL" or (st is not None and st != "PASS")
    bar = COLOC_VETO if fail else COLOC
    if cl is None:
        return "REJ:nocoloc"
    return "ADD" if cl <= bar else f"REJ:coloc{'V' if fail else ''}"


VARIANTS = {
    "current (unk fail, T_FAR/LVC/M_LOW hard)": dict(unknown="fail"),
    "V1 unknown->coloc": dict(unknown="coloc"),
    "V2 V1 + T_FAR/LVC abstain (raised bar)": dict(unknown="coloc", hard=("M_LOW",)),
    "V3 V1 + noise-aware T_FAR": dict(unknown="coloc", noise=True),
    "V4 V3 + LVC abstain": dict(unknown="coloc", noise=True, hard=("T_FAR", "M_LOW")),
    "V5 V3 + M_LOW abstain (raised bar)": dict(unknown="coloc", noise=True, hard=("LARGE_VIEW_CHANGE", "T_FAR")),
    "V6 V3 + LVC checked first (thR>50 hard)": dict(unknown="coloc", noise=True, lvc_first=True),
    "V7 V3 + S2 FAIL hard": dict(unknown="coloc", noise=True, fail_hard=True),
    "V8 V1 + LVC first (no noise-aware)": dict(unknown="coloc", lvc_first=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem", required=True, help="dir with fr1_<seq>/loops.txt (semantic run)")
    ap.add_argument("--base", required=True, help="dir with fr1_<seq>*/loops.txt (base run, coloc values)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cands = []
    for f in sorted(glob.glob(os.path.join(a.sem, "fr1_*", "loops.txt"))):
        seq = re.search(r"fr1_(\w+?)(?:_dist)?/loops", f).group(1)
        bf = glob.glob(os.path.join(a.base, f"fr1_{seq}*", "loops.txt")) + \
            glob.glob(os.path.join(a.base, f"fr1_{seq}_dist*", "loops.txt"))
        coloc = {}
        for b in bf:
            for ln in open(b):
                m = CAND.search(ln)
                if m:
                    coloc[(int(m.group(4)), int(m.group(5)))] = float(m.group(2))
        K = np.loadtxt(f"data/tum/fr1_{seq}_dist/intrinsic.txt")[:3, :3]
        gt = load_gt(f"data/tum/fr1_{seq}_dist/gt_pose.txt")
        rec = {}
        for ln in open(f):
            m = BID.search(ln)
            if m:
                key = (int(m.group(2)), int(m.group(3)))
                rec.setdefault(key, dict(seq=seq, q=key[0], r=key[1], s1=m.group(1), lay=None))
            m = LAY.search(ln)
            if m:
                key = (int(m.group(9)), int(m.group(10)))
                rec.setdefault(key, dict(seq=seq, q=key[0], r=key[1], s1="?", lay=None))
                rec[key]["lay"] = dict(st=m.group(1), ed=float(m.group(2)), thr=float(m.group(3)),
                                       er=float(m.group(4)), M=int(m.group(5)), tn=float(m.group(6)),
                                       dr=float(m.group(8)))
            m = CAND.search(ln)
            if m:
                coloc.setdefault((int(m.group(4)), int(m.group(5))), float(m.group(2)))
        for key, c in rec.items():
            c["coloc"] = coloc.get(key)
            cv = covis(gt, c["q"], c["r"], K, f"data/tum/fr1_{seq}_dist/depth", 5000.0)
            c["truth"] = "true" if cv >= 0.2 else ("weak" if cv >= 0.05 else "FALSE")
            c["covis"] = cv
            cands.append(c)
    print(f"[{a.tag}] {len(cands)} tier-A candidates "
          f"({sum(c['truth']=='true' for c in cands)} true / {sum(c['truth']=='FALSE' for c in cands)} false / "
          f"{sum(c['coloc'] is None for c in cands)} without coloc value)")
    for name, var in VARIANTS.items():
        add_t = add_f = rej_t = 0
        flips = []
        for c in cands:
            d = decide(c, var)
            if d == "ADD":
                if c["truth"] == "FALSE":
                    add_f += 1; flips.append(f"ADD-FALSE {c['seq']} f{c['q']:05d}<->f{c['r']:05d}")
                elif c["truth"] == "true":
                    add_t += 1
            elif c["truth"] == "true":
                rej_t += 1; flips.append(f"REJ-TRUE {c['seq']} f{c['q']:05d}<->f{c['r']:05d} ({d}, coloc={c['coloc']})")
        print(f"  {name:42s} added true={add_t:2d} false={add_f:2d} | rejected true={rej_t:2d}")
        for fl in flips:
            print(f"      {fl}")


if __name__ == "__main__":
    main()

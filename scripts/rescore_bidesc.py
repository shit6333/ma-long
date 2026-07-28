"""Offline threshold sweep over BIDESC shadow logs — no SLAM rerun needed.

BIDESC lines log raw scores (a_qr/a_rq/a_bi, nq/nr) and BIDESCPAIRS logs every
mutual-NN pair's raw sim, so tau_pair / tau_dir / tau_bi / n_uniq / n_min can all
be re-applied post-hoc; the coloc columns (sync tier-A + dry tier-B) provide the
ground truth.

Usage:
  PYTHONPATH=src python scripts/rescore_bidesc.py <run_dir> [<run_dir> ...] \
      [--pair 0.6,0.7] [--dir 0.5,0.55,0.6] [--bi 0.5,0.55,0.6] \
      [--nuniq 2,3] [--nmin 2,3]
Prints, per parameter combo, the aggregate confusion over all given runs:
  FAIL x ACCEPT (missed real)   PASS x REJECT (aliasing let through)
  FAIL x REJECT (compute saved) PASS x ACCEPT (correct pass-through)
"""
import argparse
import itertools
import re
import sys

BIDESC = re.compile(
    r"BIDESC (A|B) (\w+) a_qr=([\d.]+) a_rq=([\d.]+) a_bi=([\d.]+) "
    r"nuniq=(\d+) nq/nr=(\d+)/(\d+).*q=(\S+) m=(\S+) qimg")
PAIRS = re.compile(r"BIDESCPAIRS q=(\S+) m=(\S+) (.*)")
SEG = re.compile(r"q\d+~r\d+:s([\d.]+):(IN|out)")
LOOP = re.compile(r"LOOPCAND (ACCEPT|REJECT)(-DRY)? coloc=([\d.]+) sim=[\d.]+ .*q=(\S+) m=(\S+) qimg")


def load(run_dir):
    rows = {}
    for ln in open(f"{run_dir}/loops.txt"):
        b = BIDESC.match(ln)
        if b:
            rows.setdefault((b.group(9), b.group(10)), {}).update(
                tier=b.group(1), a_qr=float(b.group(3)), a_rq=float(b.group(4)),
                a_bi=float(b.group(5)), nq=int(b.group(7)), nr=int(b.group(8)))
            continue
        p = PAIRS.match(ln)
        if p:
            sims = [float(m.group(1)) for m in SEG.finditer(p.group(3))]
            rows.setdefault((p.group(1), p.group(2)), {})["pair_sims"] = sims
            continue
        m = LOOP.match(ln)
        if m:
            rows.setdefault((m.group(4), m.group(5)), {})["coloc"] = m.group(1)
    return [r for r in rows.values() if "coloc" in r and "a_bi" in r]


def verdict(r, pair, dirt, bi, nuniq, nmin):
    if r["nq"] < nmin or r["nr"] < nmin:
        return "UNKNOWN"
    nu = sum(1 for s in r.get("pair_sims", []) if s > pair)
    if (nu >= nuniq and r["a_qr"] > dirt and r["a_rq"] > dirt and r["a_bi"] > bi):
        return "PASS"
    return "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--pair", default="0.6")
    ap.add_argument("--dir", dest="dirt", default="0.55")
    ap.add_argument("--bi", default="0.55")
    ap.add_argument("--nuniq", default="2")
    ap.add_argument("--nmin", default="2")
    a = ap.parse_args()
    rows = [r for run in a.runs for r in load(run)]
    print(f"{len(rows)} candidates with ground truth from {len(a.runs)} run(s)\n")
    grid = itertools.product(*[[float(x) for x in getattr(a, k).split(",")]
                               for k in ("pair", "dirt", "bi")],
                             *[[int(x) for x in getattr(a, k).split(",")]
                               for k in ("nuniq", "nmin")])
    hdr = "pair dir  bi   nu nm | FAILxACC(miss) PASSxREJ FAILxREJ(saved) PASSxACC UNK"
    print(hdr)
    for pair, dirt, bi, nuniq, nmin in grid:
        c = {"FAILxACCEPT": 0, "PASSxREJECT": 0, "FAILxREJECT": 0,
             "PASSxACCEPT": 0, "UNK": 0}
        for r in rows:
            v = verdict(r, pair, dirt, bi, nuniq, nmin)
            if v == "UNKNOWN":
                c["UNK"] += 1
            else:
                c[f"{v}x{r['coloc']}"] += 1
        print(f"{pair:.2f} {dirt:.2f} {bi:.2f} {nuniq:2d} {nmin:2d} | "
              f"{c['FAILxACCEPT']:14d} {c['PASSxREJECT']:8d} "
              f"{c['FAILxREJECT']:15d} {c['PASSxACCEPT']:8d} {c['UNK']:3d}")


if __name__ == "__main__":
    main()

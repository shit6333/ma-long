"""Summarise outputs/profile/*.json (from ma_slam_semantic.profile_pipeline) into per-component tables.

Top-level grouping of the hierarchical sections:
  geo prediction  = submap/infer + submap/place
  loop closure    = submap/lc/**  (retrieval, inst_verify, coloc incl. its DA3 joint infer, optimize) + submap/optimize
  Rememb3r        = submap/rememb3r/**  (forward, proposals, associate, other)
  export          = export (once per run; excluded from streaming fps)
"""
import glob, json, os, sys


def group(sec):
    g = {"geo prediction": 0.0, "Rememb3r": 0.0, "loop closure": 0.0, "other": 0.0}
    pk = {k: 0.0 for k in g}
    top = {k: v for k, v in sec.items() if k.count("/") == 1 and k.startswith("submap/")}
    for k, v in top.items():
        leaf = k.split("/")[1]
        name = {"infer": "geo prediction", "place": "geo prediction", "rememb3r": "Rememb3r",
                "lc": "loop closure", "optimize": "loop closure"}.get(leaf, "other")
        g[name] += v["seconds"]; pk[name] = max(pk[name], v["peak_vram_gb"])
    total = sum(v["seconds"] for k, v in sec.items() if k == "submap")
    g["other"] += max(0.0, total - sum(g.values()))
    return g, pk, total


def main(root):
    files = [f for f in sorted(glob.glob(os.path.join(root, "*.json"))) if not os.path.basename(f).startswith("_")]
    print("# Pipeline profile (per-component time and peak VRAM)\n")
    for f in files:
        r = json.load(open(f)); sec = r["sections"]
        log = f[:-5] + ".log"; nkf = None
        if os.path.exists(log):
            import re
            m = re.findall(r"keyframe gate: kept (\d+)/", open(log, errors="ignore").read())
            nkf = int(m[0]) if m else None
        g, pk, total = group(sec)
        nkf = nkf or 1
        print(f"## {os.path.basename(f)[:-5]}  ({nkf} keyframes, streaming {total:.1f} s = {nkf/total:.2f} fps)\n")
        print("| component | time (s) | ms / keyframe | share | fps if alone | peak VRAM in section (GB) |")
        print("|---|---|---|---|---|---|")
        for k in ("geo prediction", "Rememb3r", "loop closure", "other"):
            ms = 1000 * g[k] / nkf
            print(f"| {k} | {g[k]:.1f} | {ms:.0f} | {100*g[k]/max(total,1e-9):.0f} % | {(1000/ms if ms > 0 else float('inf')):.2f} | {pk[k]:.2f} |")
        print(f"| **total (streaming)** | **{total:.1f}** | **{1000*total/nkf:.0f}** | 100 % | **{nkf/total:.2f}** | **{max(v['peak_vram_gb'] for v in sec.values()):.2f}** |")
        w = r.get("weights_gb", {})
        print(f"\nweights: " + ", ".join(f"{k} {v:.2f} GB" for k, v in w.items()) + f" | export (once) {sec.get('export', {}).get('seconds', 0):.1f} s\n")
        print("<details><summary>all sections</summary>\n\n| section | calls | s | peak GB |\n|---|---|---|---|")
        for k in sorted(sec):
            print(f"| {k} | {sec[k]['calls']} | {sec[k]['seconds']:.2f} | {sec[k]['peak_vram_gb']:.2f} |")
        print("\n</details>\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/profile")

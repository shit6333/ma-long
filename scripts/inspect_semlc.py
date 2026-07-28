"""Visual review page for instance-LC shadow runs.

Parses ``loops.txt`` (LOOPCAND + SEMLC lines) of one or more runs and writes a
self-contained HTML gallery: each loop candidate as a row with the query/match
image pair (embedded thumbnails), SALAD sim, SEMLC verdict (inliers/matched/
score) and the coloc result (sync tier-A, or -DRY shadow_coloc tier-B).

Run:  PYTHONPATH=src python scripts/inspect_semlc.py outputs/sem_lc_shadow/scene0231_00 [...]
Writes <run_dir>/semlc_review.html per run dir.
"""
import base64
import io
import os
import re
import sys

from PIL import Image

LOOPCAND = re.compile(
    r"LOOPCAND (ACCEPT|REJECT)(-DRY)? coloc=([\d.]+) sim=([\d.]+) hw=\d+ "
    r"q=(\d+)\+(\d+) m=(\d+)\+(\d+) qimg=(\S+) mimg=(\S+)")
SEMLC = re.compile(
    r"SEMLC (A|B) (PASS|FAIL|ABSTAIN) inliers=(\d+) matched=(\d+) bags=(\d+)/(\d+) "
    r"score=([\d.]+)(?: \(([^)]*)\))? sim=([\d.]+) "
    r"q=(\d+)\+(\d+) m=(\d+)\+(\d+) qimg=(\S+) mimg=(\S+)")
SEMLCPAIRS = re.compile(r"SEMLCPAIRS q=(\d+)\+(\d+) m=(\d+)\+(\d+) (.*)")
BIDESC = re.compile(
    r"BIDESC (A|B) (PASS|FAIL|UNKNOWN) a_qr=([\d.]+) a_rq=([\d.]+) a_bi=([\d.]+) "
    r"nuniq=(\d+) nq/nr=(\d+)/(\d+)(?: \(([^)]*)\))? sim=([\d.]+) "
    r"q=(\d+)\+(\d+) m=(\d+)\+(\d+) qimg=(\S+) mimg=(\S+)")
BIDESCPAIRS = re.compile(r"BIDESCPAIRS q=(\d+)\+(\d+) m=(\d+)\+(\d+) (.*)")


def thumb(path, width=300):
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGB")
    im.thumbnail((width, width * 3))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def parse(loops_txt):
    """-> {(q,m): row} with keys 'coloc'/'sem' merged per candidate pair."""
    rows = {}
    for line in open(loops_txt):
        line = line.strip()
        m = LOOPCAND.match(line)
        if m:
            key = (f"{m.group(5)}+{m.group(6)}", f"{m.group(7)}+{m.group(8)}")
            r = rows.setdefault(key, dict(q=key[0], m=key[1],
                                          qimg=m.group(9), mimg=m.group(10)))
            r["coloc_verdict"] = m.group(1) + (m.group(2) or "")
            r["coloc"] = float(m.group(3))
            r["sim"] = float(m.group(4))
            continue
        s = SEMLC.match(line)
        if s:
            key = (f"{s.group(10)}+{s.group(11)}", f"{s.group(12)}+{s.group(13)}")
            r = rows.setdefault(key, dict(q=key[0], m=key[1],
                                          qimg=s.group(14), mimg=s.group(15)))
            r.update(tier=s.group(1), sem=s.group(2), inliers=int(s.group(3)),
                     matched=int(s.group(4)), bags=f"{s.group(5)}/{s.group(6)}",
                     score=float(s.group(7)), reason=s.group(8) or "",
                     sim=float(s.group(9)))
            continue
        b = BIDESC.match(line)
        if b:
            key = (f"{b.group(11)}+{b.group(12)}", f"{b.group(13)}+{b.group(14)}")
            r = rows.setdefault(key, dict(q=key[0], m=key[1],
                                          qimg=b.group(15), mimg=b.group(16)))
            r.update(tier=b.group(1), sem=b.group(2),
                     inliers=int(b.group(6)),                   # nuniq in bidesc
                     matched=f"a_qr {b.group(3)} · a_rq {b.group(4)} · a_bi {b.group(5)}",
                     bags=f"{b.group(7)}/{b.group(8)}",
                     score=float(b.group(5)), reason=b.group(9) or "",
                     sim=float(b.group(10)))
            continue
        for rex in (SEMLCPAIRS, BIDESCPAIRS):
            p = rex.match(line)
            if p:
                key = (f"{p.group(1)}+{p.group(2)}", f"{p.group(3)}+{p.group(4)}")
                if key in rows:
                    rows[key]["pairs"] = p.group(5)
                break
    return list(rows.values())


BADGE = {"PASS": "#1a7f37", "FAIL": "#cf222e", "ABSTAIN": "#9a6700", "UNKNOWN": "#9a6700",
         "ACCEPT": "#1a7f37", "REJECT": "#cf222e",
         "ACCEPT-DRY": "#2da44e", "REJECT-DRY": "#d1242f", None: "#57606a"}


def badge(text, color):
    return (f"<span style='background:{color};color:#fff;border-radius:4px;"
            f"padding:1px 7px;font-weight:600'>{text}</span>")


def render(run_dir, rows):
    def sort_key(r):
        tier = r.get("tier", "A")
        return (tier, -r.get("sim", 0))
    rows.sort(key=sort_key)
    h = ["<meta charset='utf-8'><title>SEMLC review — %s</title>" % os.path.basename(run_dir),
         "<style>body{font-family:sans-serif;background:#f6f8fa;margin:16px}"
         ".card{background:#fff;border:1px solid #d0d7de;border-radius:8px;"
         "padding:10px;margin:10px 0;display:flex;gap:12px;align-items:flex-start}"
         ".imgs img{border-radius:4px;display:block}"
         ".imgs{display:flex;gap:6px}.meta{font-size:14px;line-height:1.9}"
         "code{background:#eee;padding:0 4px;border-radius:3px}</style>",
         f"<h2>Instance-LC review — {os.path.basename(run_dir)} "
         f"({len(rows)} candidates)</h2>",
         "<p>tier A = SALAD ≥ 0.5 (現行路徑, SEMLC 只是 shadow 對照) · "
         "tier B = [floor, 0.5) (SEMLC gate; coloc 為 -DRY 空跑) · "
         "左 query / 右 match</p>"]
    for r in rows:
        qi, mi = thumb(r["qimg"]), thumb(r["mimg"])
        imgs = "".join(f"<img src='data:image/jpeg;base64,{d}'>"
                       for d in (qi, mi) if d) or "<i>images not found</i>"
        sem = r.get("sem")
        cv = r.get("coloc_verdict")
        meta = [f"tier <b>{r.get('tier','A?')}</b> · SALAD sim <b>{r.get('sim','?')}</b>",
                f"SEMLC {badge(sem, BADGE.get(sem))} "
                f"matched=<b>{r.get('matched','-')}</b> → "
                f"inliers=<b>{r.get('inliers','-')}</b> bags={r.get('bags','-')} "
                f"score={r.get('score','-')}"
                + (f" <i>({r['reason']})</i>" if r.get("reason") else "")
                if sem else "SEMLC —",
                f"coloc {badge(cv, BADGE.get(cv))} = {r.get('coloc','')}"
                if cv else "coloc — (not run)",
                f"q=<code>{r['q']}</code> {os.path.basename(r['qimg'])} · "
                f"m=<code>{r['m']}</code> {os.path.basename(r['mimg'])}"]
        if r.get("pairs"):
            meta.append(f"<span style='font-family:monospace;font-size:11px;"
                        f"color:#57606a'>{r['pairs']}</span>")
        h.append(f"<div class='card'><div class='imgs'>{imgs}</div>"
                 f"<div class='meta'>{'<br>'.join(meta)}</div></div>")
    out = os.path.join(run_dir, "semlc_review.html")
    with open(out, "w") as f:
        f.write("\n".join(h))
    return out


def main():
    for run_dir in sys.argv[1:]:
        loops = os.path.join(run_dir, "loops.txt")
        if not os.path.exists(loops):
            print(f"skip {run_dir}: no loops.txt")
            continue
        rows = parse(loops)
        out = render(run_dir, rows)
        n_sem = sum(1 for r in rows if r.get("sem"))
        agree = sum(1 for r in rows
                    if r.get("sem") and r.get("coloc_verdict")
                    and (r["sem"] == "PASS") == r["coloc_verdict"].startswith("ACCEPT"))
        both = sum(1 for r in rows if r.get("sem") and r.get("coloc_verdict"))
        print(f"{run_dir}: {len(rows)} candidates, {n_sem} with SEMLC verdicts, "
              f"sem/coloc agreement {agree}/{both} -> {out}")


if __name__ == "__main__":
    main()

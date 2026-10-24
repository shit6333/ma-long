"""Open-vocabulary class per exported instance (sem_query vs tips-v2-l14 text, OVO template) ->
<run>/<scene>_instance_classes.json  {inst_id: class}. Used by render_stream_video.py --hide_struct.

    PYTHONPATH=src python scripts/classify_instances.py --run outputs/videos/replica_room0 [--run ...]
"""
import argparse, glob, json, os, sys

import numpy as np
import torch

OSR = "/mnt/HDD4/ricky/OpenSe3r"
VOCAB = ["wall", "floor", "ceiling", "chair", "table", "sofa", "bed", "cabinet", "door", "window", "bookshelf",
         "picture", "counter", "desk", "curtain", "refrigerator", "toilet", "sink", "bathtub", "cushion", "pillow",
         "rug", "lamp", "plant", "blinds", "book", "box", "bag", "shelf", "monitor", "keyboard", "television",
         "towel", "clothes", "bottle", "cup", "basket", "vase", "stool", "whiteboard", "radiator", "trash can",
         "backpack", "laptop", "mirror", "washing machine", "oven", "microwave", "blanket", "toy"]
STRUCT = {"wall", "floor", "ceiling"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    a = ap.parse_args()
    sys.path.insert(0, OSR)
    from eval_semantic_npz_scannet import encode_text_via_backbone
    te = encode_text_via_backbone(tuple(VOCAB), ["This is a photo of a {}."], "tips-v2-l14", "cuda")
    te = torch.nn.functional.normalize(te.float().cuda(), dim=-1)
    for run in a.run:
        f = glob.glob(os.path.join(run, "*_instance_sem_queries.npz"))[0]
        z = np.load(f)
        q = torch.nn.functional.normalize(torch.from_numpy(z["sem_queries"]).float().cuda(), dim=-1)
        lab = (q @ te.T).argmax(-1).cpu().numpy()
        out = {int(i): VOCAB[l] for i, l in zip(z["inst_ids"], lab)}
        p = f.replace("_instance_sem_queries.npz", "_instance_classes.json")
        json.dump(out, open(p, "w"), indent=0)
        n = sum(v in STRUCT for v in out.values())
        print(f"{run}: {len(out)} instances, {n} structural (wall/floor/ceiling) -> {p}")


if __name__ == "__main__":
    main()

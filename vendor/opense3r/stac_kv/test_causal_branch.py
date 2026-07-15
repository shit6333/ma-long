"""CausalBranch-level integration test for the STAC temporal KV manager.

Run:  cd /mnt/HDD4/ricky/OpenSe3r && \
      CUDA_VISIBLE_DEVICES=0 <env-python> -m opense3r.stac_kv.test_causal_branch

  I1  Streaming forward with a HUGE-budget manager == legacy cat path (E1).
  I2  Small budget: cache stays bounded across many chunks; outputs finite.
  I3  reset/clear lifecycle: second sequence starts clean.
"""

import sys

import torch

from opense3r.causal_branch import CausalBranch
from opense3r.stac_kv import HeavyHittersKV

DEV = torch.device('cuda')

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def make_branch(seed=0):
    torch.manual_seed(seed)
    cb = CausalBranch(
        dino_token_dim=32, lseg_dim=0, causal_dim=64,
        n_blocks=2, n_heads=4, depth_cross_every=3,
        qk_norm=True, rope_freq=100,
    ).to(DEV).eval()
    return cb


def stream(cb, chunks, patch_h, patch_w, use_mgr=False, budget=None):
    """Run chunks through streaming forward; returns list of last-block outputs."""
    if use_mgr:
        cb.register_kv_mgr(
            HeavyHittersKV,
            chunk_size=max(c.shape[1] for c in chunks),
            recent_size=budget['recent'], hh_size=budget['hh'],
            pinned_idx=[0], temperature=0.9,
            subsample_ratio=1.0, attn_backend='triton',
            dtype=torch.bfloat16, device=DEV,
        )
        cb.reset_kv_mgr()
    kv = cb.init_kv_cache()
    outs = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for dino in chunks:
            res = cb(
                dino_tokens=dino, lseg_feat=None, depth_tokens=None,
                patch_h=patch_h, patch_w=patch_w,
                mode='causal', kv_cache_list=kv,
            )
            tokens_list = res[0] if isinstance(res, tuple) else res
            outs.append(tokens_list[-1].float().clone())
            if use_mgr:
                cb.prune_kv_mgr()
    return outs


def main():
    patch_h = patch_w = 4          # 16 patch tokens/frame
    T_chunk, n_chunks = 2, 4
    torch.manual_seed(42)
    chunks = [torch.randn(1, T_chunk, patch_h * patch_w, 32, device=DEV)
              for _ in range(n_chunks)]

    # I1: huge budget manager == legacy path -------------------------------
    print("I1: manager (huge budget) vs legacy cat path")
    ref = stream(make_branch(), chunks, patch_h, patch_w, use_mgr=False)
    got = stream(make_branch(), chunks, patch_h, patch_w, use_mgr=True,
                 budget={'recent': n_chunks * T_chunk + 1, 'hh': 0})
    err = max((a - b).abs().max().item() for a, b in zip(ref, got))
    check("outputs match legacy", err < 5e-2, f"max_abs={err:.2e}")

    # I2: small budget stays bounded ---------------------------------------
    print("I2: small budget bounds the cache")
    cb = make_branch()
    long_chunks = [torch.randn(1, T_chunk, patch_h * patch_w, 32, device=DEV)
                   for _ in range(12)]
    outs = stream(cb, long_chunks, patch_h, patch_w, use_mgr=True,
                  budget={'recent': 2, 'hh': 1})
    info = cb.get_kv_mgr_info()
    tpf = patch_h * patch_w
    hot_budget = (1 + 2 + 1) * tpf          # pinned + recent + hh
    sizes = info['kvcache_size']
    check("cache ≤ hot budget after prune", all(s <= hot_budget for s in sizes),
          f"sizes={sizes} budget={hot_budget}")
    check("outputs finite", all(torch.isfinite(o).all() for o in outs))
    check("lazy build picked token_per_frame",
          cb.kv_manager.token_per_f == tpf,
          f"token_per_f={cb.kv_manager.token_per_f}")

    # I3: lifecycle ----------------------------------------------------------
    print("I3: reset / clear lifecycle")
    cb.reset_kv_mgr()
    check("reset empties cache", all(o == 0 for o in cb.kv_manager.get_offset()))
    outs2 = stream(cb, long_chunks[:3], patch_h, patch_w, use_mgr=False)  # mgr still attached
    check("second sequence runs", all(torch.isfinite(o).all() for o in outs2))
    cb.clear_kv_mgr()
    check("clear detaches manager", cb.kv_manager is None
          and all(getattr(cb.global_blocks[i].attn, 'kv_manager', None) is None
                  for i in range(cb.n_blocks)))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == '__main__':
    assert torch.cuda.is_available()
    main()

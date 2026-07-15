"""Standalone smoke + equivalence tests for the vendored STAC temporal KV.

Run:  cd /mnt/HDD4/ricky/OpenSe3r && \
      CUDA_VISIBLE_DEVICES=0 <env-python> -m opense3r.stac_kv.test_smoke

Covers:
  T1  Triton colsum kernel vs SDPA reference (Blackwell smoke).
  T2  HeavyHittersKV decode path vs cat+SDPA reference, no prune (E1).
  T3  Prune behaviour: pinned ∪ recent ∪ per-head HH kept, budget respected (E2).
  T4  SparseStreamAttention swap: managed streaming path vs original cat path.
"""

import math
import sys

import torch
import torch.nn.functional as F

from opense3r.stac_kv import HeavyHittersKV, SparseStreamAttention
from opense3r.stac_kv.flash_attn_triton import fa_forward_colsum_fast_sub

DEV = torch.device('cuda')
DT = torch.bfloat16

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def sdpa_ref_with_colsum(q, k, v):
    """q,k,v: [B, T, H, D] (seq-major). Returns out [B,T,H,D], colsum [B,H,Tk]."""
    qh = q.transpose(1, 2).float()   # [B,H,Tq,D]
    kh = k.transpose(1, 2).float()
    vh = v.transpose(1, 2).float()
    attn = torch.softmax(qh @ kh.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)
    out = (attn @ vh).transpose(1, 2)          # [B,Tq,H,D]
    colsum = attn.sum(dim=2)                    # [B,H,Tk]
    return out, colsum


# ─────────────────────────────────────────────────────────────────────────
def t1_kernel():
    print("T1: triton colsum kernel vs SDPA reference")
    torch.manual_seed(0)
    B, M, N, H, D = 1, 512, 2048, 16, 64
    q = torch.randn(B, M, H, D, device=DEV, dtype=DT)
    k = torch.randn(B, N, H, D, device=DEV, dtype=DT)
    v = torch.randn(B, N, H, D, device=DEV, dtype=DT)

    out, _, colsum = fa_forward_colsum_fast_sub(q, k, v, write_o=True,
                                                subsample_ratio=1.0)
    ref_out, ref_cs = sdpa_ref_with_colsum(q, k, v)

    err_o = (out.float() - ref_out).abs().max().item()
    rel_cs = ((colsum.float() - ref_cs).abs() / ref_cs.abs().clamp(min=1e-3)).mean().item()
    check("kernel out  vs ref", err_o < 3e-2, f"max_abs={err_o:.2e}")
    check("kernel colsum vs ref", rel_cs < 3e-2, f"mean_rel={rel_cs:.2e}")

    # Subsampled colsum drops QUERY rows, so a key that only one query attends
    # to rightly vanishes under subsampling.  A true heavy hitter is attended
    # by MANY queries — plant n_hot directions, each supported by ~M/n_hot
    # noisy queries, and require the subsampled ranking to recover them.
    n_hot = 8
    dirs = torch.randn(n_hot, H, D, device=DEV, dtype=DT)
    assign = torch.randint(0, n_hot, (M,), device=DEV)
    q_hot = (dirs[assign] + 0.5 * torch.randn(M, H, D, device=DEV, dtype=DT)
             ).unsqueeze(0)                                   # [1, M, H, D]
    k_hot = k.clone()
    k_hot[0, :n_hot] = dirs
    _, _, cs_sub = fa_forward_colsum_fast_sub(q_hot, k_hot, v, write_o=False,
                                              subsample_ratio=0.25)
    top_sub = set(cs_sub[0].float().sum(0).topk(n_hot).indices.tolist())
    recovered = len(top_sub & set(range(n_hot))) / n_hot
    check("subsample recovers heavy hitters", recovered > 0.9,
          f"recovered={recovered:.3f}")


# ─────────────────────────────────────────────────────────────────────────
def make_mgr(L, H, D, tpf, chunk, recent, hh, pinned=(0,), buffer_size=None):
    return HeavyHittersKV(
        num_layers=L, num_heads=H, head_dim=D, token_per_frame=tpf,
        chunk_size=chunk, recent_size=recent, hh_size=hh,
        pinned_idx=list(pinned),
        buffer_size=buffer_size or (chunk + len(pinned) + recent + hh),
        temperature=0.9, attn_backend='triton', subsample_ratio=1.0,
        dtype=DT, device=DEV,
    )


def t2_manager_equivalence():
    print("T2: HeavyHittersKV decode vs cat+SDPA (no prune, E1)")
    torch.manual_seed(1)
    L, H, D, tpf = 2, 4, 32, 10
    n_chunks, chunk_frames = 3, 2
    # Budget large enough that prune never fires.
    mgr = make_mgr(L, H, D, tpf, chunk=chunk_frames,
                   recent=n_chunks * chunk_frames + 1, hh=0)
    mgr.reset()

    ref_k = [None] * L
    ref_v = [None] * L
    max_err = 0.0
    for c in range(n_chunks):
        for layer in range(L):
            Tn = chunk_frames * tpf
            q = torch.randn(1, H, Tn, D, device=DEV, dtype=DT)
            k = torch.randn(1, H, Tn, D, device=DEV, dtype=DT)
            v = torch.randn(1, H, Tn, D, device=DEV, dtype=DT)

            mgr.append_kv(k, v, layer_idx=layer)
            out = mgr.decode_sparse_attn(q.transpose(1, 2), layer_idx=layer)

            ref_k[layer] = k if ref_k[layer] is None else torch.cat([ref_k[layer], k], dim=2)
            ref_v[layer] = v if ref_v[layer] is None else torch.cat([ref_v[layer], v], dim=2)
            ref_out, _ = sdpa_ref_with_colsum(
                q.transpose(1, 2), ref_k[layer].transpose(1, 2), ref_v[layer].transpose(1, 2))
            max_err = max(max_err, (out.float() - ref_out).abs().max().item())
        mgr.prune_kv()  # under-budget → must be a no-op

    offs = mgr.get_offset()
    check("decode == cat+SDPA", max_err < 3e-2, f"max_abs={max_err:.2e}")
    check("no-op prune keeps all", all(o == n_chunks * chunk_frames * tpf for o in offs),
          f"offsets={offs}")


# ─────────────────────────────────────────────────────────────────────────
def t3_prune():
    print("T3: prune keeps pinned ∪ recent ∪ per-head HH (E2)")
    torch.manual_seed(2)
    L, H, D, tpf = 2, 4, 32, 4
    chunk, recent, hh = 2, 2, 1
    mgr = make_mgr(L, H, D, tpf, chunk=chunk, recent=recent, hh=hh)
    mgr.reset()

    n_frames = 10
    for c in range(n_frames // chunk):
        for layer in range(L):
            Tn = chunk * tpf
            q = torch.randn(1, H, Tn, D, device=DEV, dtype=DT)
            k = torch.randn(1, H, Tn, D, device=DEV, dtype=DT)
            v = torch.randn(1, H, Tn, D, device=DEV, dtype=DT)
            mgr.append_kv(k, v, layer_idx=layer)
            mgr.decode_sparse_attn(q.transpose(1, 2), layer_idx=layer)
        mgr.prune_kv()

    hot_budget = (1 + recent + hh) * tpf          # pinned + recent + hh
    offs = mgr.get_offset()
    check("budget respected", all(o <= hot_budget for o in offs),
          f"offsets={offs} budget={hot_budget}")

    ti = mgr.get_token_indices()                   # [L, H, buffer_T] global ids
    live = offs[0]
    ids = ti[:, :, :live]                          # kept global token ids
    frame_of = ids // tpf
    # pinned frame 0 present in every (layer, head)
    pin_ok = bool((frame_of == 0).any(dim=-1).all())
    # recent frames fully present in every (layer, head)
    rec_frames = list(range(n_frames - recent, n_frames))
    rec_ok = all(bool((frame_of == f).sum(dim=-1).eq(tpf).all()) for f in rec_frames)
    # per-head HH: kept sets may DIFFER across heads (token-level H2O)
    heads_differ = not torch.equal(ids[0, 0], ids[0, 1]) or not torch.equal(ids[0, 1], ids[0, 2])
    check("pinned frame-0 kept everywhere", pin_ok)
    check("recent frames fully kept", rec_ok, f"recent={rec_frames}")
    check("per-head HH sets can differ", heads_differ,
          "(identical across heads would suggest frame-level pruning)")


# ─────────────────────────────────────────────────────────────────────────
def t4_sparse_attention():
    print("T4: SparseStreamAttention managed path vs original cat path")
    torch.manual_seed(3)
    sys.path.append('/mnt/HDD4/ricky/OpenSe3r/thirdparty')
    sys.path.append('/mnt/HDD4/ricky/OpenSe3r/thirdparty/STream3R')
    from stream3r.models.components.layers.attention import Attention

    dim, H = 128, 4
    tpf, chunk, n_chunks = 8, 2, 3
    attn = Attention(dim=dim, num_heads=H, qk_norm=True).to(DEV).eval()

    mgr = make_mgr(1, H, dim // H, tpf, chunk=chunk,
                   recent=n_chunks * chunk + 1, hh=0)
    mgr.reset()
    sp = SparseStreamAttention.from_attention(attn, layer_idx=0)
    sp.kv_manager = mgr

    kv_ref = [None, None]
    kv_dummy = [None, None]
    max_err = 0.0
    with torch.no_grad(), torch.autocast('cuda', dtype=DT):
        for c in range(n_chunks):
            x = torch.randn(1, chunk * tpf, dim, device=DEV)
            y_ref, kv_ref = attn(x, pos=None, kv_cache=kv_ref)
            y_new, kv_dummy = sp(x, pos=None, kv_cache=kv_dummy)
            max_err = max(max_err, (y_new.float() - y_ref.float()).abs().max().item())

    check("swap-path == original", max_err < 5e-2, f"max_abs={max_err:.2e}")
    check("dummy cache untouched", kv_dummy == [None, None])
    check("params shared (zero-copy)", sp.qkv.weight.data_ptr() == attn.qkv.weight.data_ptr())


# ─────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    assert torch.cuda.is_available(), "CUDA required"
    print(f"device={torch.cuda.get_device_name(0)}  torch={torch.__version__}")
    t1_kernel()
    t2_manager_equivalence()
    t3_prune()
    t4_sparse_attention()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)

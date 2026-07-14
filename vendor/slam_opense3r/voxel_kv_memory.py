"""
3-tier KV memory for OpenSe3r streaming SLAM (experimental).

Replaces the flat "anchor[0] + sliding window" KV cache with:

    first_frame_kv  ⊕  anchor_pool  ⊕  window  ⊕  retrieved_voxel_pivots

  - first_frame_kv : 1 fixed slot, always frame 0's KV (camera-cond anchor).
  - anchor_pool   : up to `anchor_cap` keyframes, rotate or sticky mode.
  - window        : last `window_chunks` chunks' KV (sliding).
  - voxel store   : spatial pool of pivots, indexed by 3D position;
                    retrieved by new chunk's patch positions + neighbour ball.

Per-pivot eviction uses count-saturated update + diversity-maximise policy
to preserve `pivot_cap` distinct prototypes per voxel.  Pivot similarity is
computed on a configurable signal (`instance_feat` by default).

NOT used by the default `OpenSe3rSLAM` / `OpenSe3rSLAM_GT` pipelines — opt
in via `--memory_mode voxel_kv` (which uses `OpenSe3rSLAM_VoxelKV` in
`pipeline_voxel_kv.py`).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VoxelKVConfig:
    # Anchor
    anchor_cap: int           = 10        # geometry-keyframe slots (first_frame separate)
    anchor_mode: str          = 'sticky'  # 'sticky' | 'rotate'

    # Window
    window_chunks: int        = 6

    # Voxel store
    voxel_size: float         = 0.1       # metres
    voxel_pivot_cap: int      = 4
    voxel_retrieval_radius: int = 1       # 3×3×3 = 27 voxels per query
    voxel_layers: Optional[List[int]] = None  # None → all layers
    pivot_signal: str         = 'instance_feat'  # 'instance_feat' | 'K' | 'V'
    merge_threshold: float    = 0.85
    update_p: float           = 1.0        # α = 1/(1+count)^p

    # Maintenance
    update_positions_on_alignment: bool = False  # re-bucket pivots when alignment refines poses

    # Where to hold KV tensors.
    #   'cpu'    : everything on host (memory-friendly, slower copies each step)
    #   'gpu'    : everything on device (fastest, biggest VRAM)
    #   'hybrid' : hot tier (first_frame + anchor + window) on GPU, voxel pool on CPU
    storage_location: str     = 'hybrid'


# ─────────────────────────────────────────────────────────────────────────────
# Spatial hash voxel index
# ─────────────────────────────────────────────────────────────────────────────

class BinaryVoxel:
    """3D position → integer voxel ID.  Pure-Python dict, GPU-safe (CPU index)."""

    def __init__(self, voxel_size: float = 0.1):
        self.voxel_size = float(voxel_size)
        self._next_id = 0
        self._key_to_id: Dict[Tuple[int, int, int], int] = {}
        self._id_to_key: Dict[int, Tuple[int, int, int]] = {}

    def reset(self):
        self._next_id = 0
        self._key_to_id.clear()
        self._id_to_key.clear()

    def positions_to_keys(self, positions: torch.Tensor) -> np.ndarray:
        """positions: [..., 3] → keys: numpy [..., 3] int64."""
        keys = (positions.float() / self.voxel_size).floor().to(torch.int64).cpu().numpy()
        return keys

    def get_or_create_ids(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: [..., 3]. Returns long ids matching positions.shape[:-1]."""
        keys = self.positions_to_keys(positions).reshape(-1, 3)
        out = np.empty(keys.shape[0], dtype=np.int64)
        for i in range(keys.shape[0]):
            k = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
            vid = self._key_to_id.get(k)
            if vid is None:
                vid = self._next_id
                self._key_to_id[k] = vid
                self._id_to_key[vid] = k
                self._next_id += 1
            out[i] = vid
        return torch.from_numpy(out).reshape(positions.shape[:-1])

    def neighbor_ids(self, voxel_ids: List[int], radius: int = 1) -> List[int]:
        """For each input voxel_id, return all existing neighbour ids (incl self)
        within max-norm radius."""
        if radius < 0:
            radius = 0
        results = set()
        for vid in voxel_ids:
            ijk = self._id_to_key.get(int(vid))
            if ijk is None:
                continue
            i, j, k = ijk
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    for dk in range(-radius, radius + 1):
                        nb_key = (i + di, j + dj, k + dk)
                        nb_id = self._key_to_id.get(nb_key)
                        if nb_id is not None:
                            results.add(nb_id)
        return list(results)


# ─────────────────────────────────────────────────────────────────────────────
# Pivot record (per-voxel)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Pivot:
    """A single pivot in the voxel store.  K, V are stacked across all layers
    in `voxel_layers` (so retrieval can index by layer cheaply)."""
    K_per_layer:   torch.Tensor   # [L, H, D]  (L = number of stored layers)
    V_per_layer:   torch.Tensor   # [L, H, D]
    position:      torch.Tensor   # [3]
    instance_feat: torch.Tensor   # [ins_dim]
    frame_idx:     int
    count:         int = 1


# ─────────────────────────────────────────────────────────────────────────────
# Voxel KV store (cross-layer pivot identity)
# ─────────────────────────────────────────────────────────────────────────────

class VoxelKVStore:
    """Spatial KV pivots, keyed by voxel id.

    Each pivot stores K, V for ALL `voxel_layers` simultaneously.  Pivot
    identity (which pivot lives at a voxel) is shared across layers because
    eviction is driven by a single cross-layer signal (e.g. `instance_feat`).
    """

    def __init__(self, cfg: VoxelKVConfig):
        self.cfg = cfg
        self.voxel = BinaryVoxel(cfg.voxel_size)
        self.store: Dict[int, List[Pivot]] = defaultdict(list)

    # ── public ops ──────────────────────────────────────────────────────────

    def reset(self):
        self.voxel.reset()
        self.store.clear()

    def num_pivots(self) -> int:
        return sum(len(p) for p in self.store.values())

    @torch.no_grad()
    def add_or_update(
        self,
        K_per_layer: torch.Tensor,     # [L, H, D]
        V_per_layer: torch.Tensor,     # [L, H, D]
        position: torch.Tensor,        # [3]
        instance_feat: torch.Tensor,   # [ins_dim]
        frame_idx: int,
    ):
        vid = int(self.voxel.get_or_create_ids(position.unsqueeze(0))[0].item())
        pivots = self.store[vid]

        signal_new = self._signal_of(instance_feat, K_per_layer, V_per_layer)

        # 1. Merge with closest existing pivot if similarity > threshold
        if pivots:
            sigs = torch.stack([
                self._signal_of(p.instance_feat, p.K_per_layer, p.V_per_layer)
                for p in pivots
            ])
            sims = sigs @ signal_new                          # [n_pivots]
            best_sim, best_idx = sims.max(dim=0)
            if best_sim.item() > self.cfg.merge_threshold:
                self._count_saturated_update(
                    pivots[int(best_idx)],
                    K_per_layer, V_per_layer, position, instance_feat,
                )
                return

        # 2. Cap not yet hit → append as new pivot
        if len(pivots) < self.cfg.voxel_pivot_cap:
            pivots.append(Pivot(
                K_per_layer   = K_per_layer.clone(),
                V_per_layer   = V_per_layer.clone(),
                position      = position.clone(),
                instance_feat = instance_feat.clone(),
                frame_idx     = int(frame_idx),
                count         = 1,
            ))
            return

        # 3. Cap full + below merge threshold → evict for diversity
        sigs = torch.stack([
            self._signal_of(p.instance_feat, p.K_per_layer, p.V_per_layer)
            for p in pivots
        ])
        sim_mat = sigs @ sigs.t()                              # [cap, cap]
        sim_mat.fill_diagonal_(-1.0)
        flat = sim_mat.flatten()
        i_flat = int(flat.argmax().item())
        i, j = divmod(i_flat, sim_mat.size(0))
        # Evict the one with smaller count (less stable); tie-break: smaller frame_idx (older)
        if pivots[i].count < pivots[j].count:
            evict = i
        elif pivots[i].count > pivots[j].count:
            evict = j
        else:
            evict = i if pivots[i].frame_idx <= pivots[j].frame_idx else j
        pivots[evict] = Pivot(
            K_per_layer   = K_per_layer.clone(),
            V_per_layer   = V_per_layer.clone(),
            position      = position.clone(),
            instance_feat = instance_feat.clone(),
            frame_idx     = int(frame_idx),
            count         = 1,
        )

    @torch.no_grad()
    def retrieve(
        self,
        new_chunk_positions: torch.Tensor,    # [N_query, 3]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Find voxels overlapping with the query positions + their neighbours.
        Returns (K, V) each [N_retrieved, L, H, D], or (None, None) if empty.
        """
        if new_chunk_positions.numel() == 0:
            return None, None
        query_keys = self.voxel.positions_to_keys(new_chunk_positions).reshape(-1, 3)
        seen_voxels = set()
        for i in range(query_keys.shape[0]):
            k = (int(query_keys[i, 0]), int(query_keys[i, 1]), int(query_keys[i, 2]))
            vid = self.voxel._key_to_id.get(k)
            if vid is not None:
                seen_voxels.add(vid)
        if not seen_voxels:
            return None, None
        target_ids = self.voxel.neighbor_ids(list(seen_voxels),
                                              radius=self.cfg.voxel_retrieval_radius)
        all_pivots: List[Pivot] = []
        for vid in target_ids:
            all_pivots.extend(self.store.get(vid, []))
        if not all_pivots:
            return None, None
        K = torch.stack([p.K_per_layer for p in all_pivots], dim=0)
        V = torch.stack([p.V_per_layer for p in all_pivots], dim=0)
        return K, V

    @torch.no_grad()
    def update_pivot_positions(self, frame_to_delta: Dict[int, torch.Tensor]):
        """Apply per-frame SE(3) deltas to all pivots from those frames.
        `frame_to_delta`: {frame_idx → 4×4 c2w delta (= T_c2w_new @ T_c2w_old^-1)}.
        Re-buckets pivots whose source frame's pose was refined."""
        if not self.cfg.update_positions_on_alignment:
            return
        if not frame_to_delta:
            return
        new_voxel = BinaryVoxel(self.cfg.voxel_size)
        new_store: Dict[int, List[Pivot]] = defaultdict(list)
        for pivots in self.store.values():
            for p in pivots:
                T = frame_to_delta.get(p.frame_idx)
                if T is not None:
                    pos_h = torch.cat([p.position.float(), torch.ones(1)])
                    p.position = (T.cpu().float() @ pos_h)[:3]
                new_vid = int(new_voxel.get_or_create_ids(p.position.unsqueeze(0))[0].item())
                new_store[new_vid].append(p)
        self.store = new_store
        self.voxel = new_voxel

    # ── helpers ─────────────────────────────────────────────────────────────

    def _signal_of(
        self,
        instance_feat: torch.Tensor,
        K_per_layer: torch.Tensor,
        V_per_layer: torch.Tensor,
    ) -> torch.Tensor:
        sig = self.cfg.pivot_signal
        if sig == 'instance_feat':
            v = instance_feat
        elif sig == 'K':
            v = K_per_layer.reshape(-1)
        elif sig == 'V':
            v = V_per_layer.reshape(-1)
        else:
            raise ValueError(f"Unknown pivot_signal: {sig}")
        return F.normalize(v.float(), dim=-1)

    def _count_saturated_update(
        self,
        p: Pivot,
        K_new: torch.Tensor, V_new: torch.Tensor,
        pos_new: torch.Tensor, inst_new: torch.Tensor,
    ):
        alpha = 1.0 / (1.0 + p.count) ** self.cfg.update_p
        a = float(alpha)
        p.K_per_layer   = (1 - a) * p.K_per_layer   + a * K_new.to(p.K_per_layer.dtype)
        p.V_per_layer   = (1 - a) * p.V_per_layer   + a * V_new.to(p.V_per_layer.dtype)
        p.instance_feat = (1 - a) * p.instance_feat + a * inst_new.to(p.instance_feat.dtype)
        p.position      = (1 - a) * p.position      + a * pos_new.to(p.position.dtype)
        p.count += 1


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame KV record (used by anchor + window)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameKV:
    frame_idx:     int
    # Per-layer K, V — list of length n_blocks, each [num_heads, P, head_dim]
    # P = N_patch + patch_start_idx (1 pose token prepended when pose_cond=True).
    # The cache is stored with the FULL P tokens (including pose token) so that
    # `materialize_kv_cache` can concat them and forward_chunk's attention
    # sees the same layout it was trained with.
    K_per_layer:   List[torch.Tensor]
    V_per_layer:   List[torch.Tensor]
    # Spatial / semantic metadata covers ONLY the patch tokens (skip the
    # pose token because it has no 3D position).
    positions:     torch.Tensor      # [N_patch, 3]  world frame
    instance_feat: torch.Tensor      # [N_patch, ins_dim]
    instance_sig:  torch.Tensor      # [ins_dim]    aggregate signature
    count:         int = 1
    # Patch-token layout (how many leading tokens are special, e.g. pose)
    patch_start_idx: int = 0


def _make_instance_sig(instance_feat: torch.Tensor) -> torch.Tensor:
    """Mean of per-patch instance feat, then L2-norm.  Used as anchor frame
    signature for diversity-maximise eviction."""
    return F.normalize(instance_feat.mean(dim=0).float(), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Anchor pool (first_frame slot + keyframe pool)
# ─────────────────────────────────────────────────────────────────────────────

class AnchorMemory:
    """First-frame slot + keyframe pool.  In `sticky` mode, frames in the
    pool stay until cap-driven eviction (diversity-based); in `rotate` mode,
    the pool tracks the SLAM keyframe set (frames removed from
    `cur_kf_idx` are also dropped here).
    """

    def __init__(self, cfg: VoxelKVConfig):
        self.cfg = cfg
        self.first_frame_kv: Optional[FrameKV] = None
        self.pool: Dict[int, FrameKV] = {}      # frame_idx → record

    def reset(self):
        self.first_frame_kv = None
        self.pool.clear()

    def has_frame(self, frame_idx: int) -> bool:
        if self.first_frame_kv is not None and self.first_frame_kv.frame_idx == frame_idx:
            return True
        return frame_idx in self.pool

    def set_first_frame(self, kv: FrameKV):
        self.first_frame_kv = kv

    @torch.no_grad()
    def try_admit(self, candidate: FrameKV) -> Optional[FrameKV]:
        """Try to add `candidate` to the keyframe pool.  Returns the evicted
        FrameKV (or candidate itself, if it gets rejected) — caller should
        push it to the voxel store.  Returns None if accepted with no
        eviction (pool below cap)."""
        if candidate.frame_idx in self.pool:
            return None
        if (self.first_frame_kv is not None
                and self.first_frame_kv.frame_idx == candidate.frame_idx):
            return None

        if len(self.pool) < self.cfg.anchor_cap:
            self.pool[candidate.frame_idx] = candidate
            return None

        # Pool full — diversity-maximise eviction
        return self._diversity_evict(candidate)

    @torch.no_grad()
    def sync_with_keyframes(self, current_keyframes: List[int]) -> List[FrameKV]:
        """Used in `rotate` mode: drop frames no longer in `current_keyframes`.
        Returns the evicted records (caller pushes them to voxel store)."""
        if self.cfg.anchor_mode != 'rotate':
            return []
        kf_set = set(int(x) for x in current_keyframes)
        evicted = []
        for fidx in list(self.pool.keys()):
            if fidx not in kf_set:
                evicted.append(self.pool.pop(fidx))
        return evicted

    @torch.no_grad()
    def collect_all(self) -> List[FrameKV]:
        """Iteration order: first_frame, then pool by ascending frame_idx."""
        out = []
        if self.first_frame_kv is not None:
            out.append(self.first_frame_kv)
        for fidx in sorted(self.pool.keys()):
            out.append(self.pool[fidx])
        return out

    @torch.no_grad()
    def _diversity_evict(self, candidate: FrameKV) -> FrameKV:
        """Find the most-similar pair across (pool ∪ {candidate}) and evict
        the one with smaller count.  Returns the evicted record."""
        # First-frame slot is excluded from this pool (handled separately).
        pool_items = list(self.pool.items())              # [(fidx, FrameKV)]
        sigs = [r.instance_sig for _, r in pool_items] + [candidate.instance_sig]
        S = torch.stack(sigs)                              # [n+1, ins_dim]
        sim_mat = S @ S.t()
        sim_mat.fill_diagonal_(-1.0)
        flat = sim_mat.flatten()
        i_flat = int(flat.argmax().item())
        n = sim_mat.size(0)
        i, j = divmod(i_flat, n)
        # Map back to records (last index = candidate)
        items_full = pool_items + [(candidate.frame_idx, candidate)]

        def _key(rec: FrameKV):
            return (rec.count, -rec.frame_idx)            # lower count → evict first, then older

        ri = items_full[i][1]; rj = items_full[j][1]
        evict_rec = ri if _key(ri) < _key(rj) else rj
        # Update pool
        if evict_rec is candidate:
            return candidate                              # candidate rejected
        # Remove evicted from pool, install candidate
        del self.pool[evict_rec.frame_idx]
        self.pool[candidate.frame_idx] = candidate
        return evict_rec


# ─────────────────────────────────────────────────────────────────────────────
# Window memory (sliding chunks)
# ─────────────────────────────────────────────────────────────────────────────

class WindowMemory:
    """Sliding queue of the last `window_chunks` chunks' FrameKV records.
    Drops oldest chunk when overflow; returns the dropped frames."""

    def __init__(self, cfg: VoxelKVConfig):
        self.cfg = cfg
        self.chunks: deque[List[FrameKV]] = deque()  # each chunk = list of FrameKV

    def reset(self):
        self.chunks.clear()

    def push_chunk(self, frames: List[FrameKV]) -> List[FrameKV]:
        """Append a new chunk of FrameKV records.  Returns evicted FrameKV
        records (from the oldest chunk if window is full)."""
        evicted: List[FrameKV] = []
        self.chunks.append(frames)
        while len(self.chunks) > self.cfg.window_chunks:
            old = self.chunks.popleft()
            evicted.extend(old)
        return evicted

    def collect_all(self) -> List[FrameKV]:
        out = []
        for chunk in self.chunks:
            out.extend(chunk)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3-tier KV manager  (orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class ThreeTierKVManager:
    """Coordinates first_frame + anchor + window + voxel store.

    Per-chunk usage:
        manager.materialize_kv_cache(new_chunk_positions)
          → returns kv_cache_list to pass into model.forward_chunk
        manager.absorb_chunk(new_chunk_kv_per_layer,  # [B, H, T_new_total, D] per layer
                              T_new_in_cache,           # how many tokens at the end belong to this chunk
                              new_chunk_meta)           # frame indices, positions, instance feat, keyframe mask
    """

    def __init__(
        self,
        n_blocks: int,
        num_heads: int,
        head_dim: int,
        ins_dim: int,
        patch_start_idx: int,
        cfg: VoxelKVConfig,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.n_blocks  = n_blocks
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.ins_dim   = ins_dim
        self.patch_start_idx = int(patch_start_idx)
        self.device    = device
        self.dtype     = dtype
        self.cfg       = cfg

        # Storage device split — hot tier (anchor + window) vs cold (voxel pool).
        loc = cfg.storage_location
        if loc == 'cpu':
            self._hot_device, self._cold_device = 'cpu', 'cpu'
        elif loc == 'gpu':
            self._hot_device, self._cold_device = device, device
        elif loc == 'hybrid':
            self._hot_device, self._cold_device = device, 'cpu'
        else:
            raise ValueError(f"Unknown storage_location: {loc}")
        # Which layers participate in voxel store
        self.voxel_layers = (list(range(n_blocks)) if cfg.voxel_layers is None
                              else list(cfg.voxel_layers))
        self._voxel_layer_set = set(self.voxel_layers)
        # State
        self.anchor   = AnchorMemory(cfg)
        self.window   = WindowMemory(cfg)
        self.voxel    = VoxelKVStore(cfg)

    # ── public lifecycle ────────────────────────────────────────────────────

    def reset(self):
        self.anchor.reset()
        self.window.reset()
        self.voxel.reset()

    @torch.no_grad()
    def materialize_kv_cache(
        self,
        new_chunk_positions: torch.Tensor,     # [B*T_new*N_patch, 3]
        batch_size: int = 1,
    ) -> Tuple[List[List[torch.Tensor]], int]:
        """Build [first ⊕ anchor ⊕ window ⊕ retrieved] per layer for the
        upcoming model.forward_chunk call.

        Returns:
            kv_cache_list: list of length n_blocks; each entry is [K, V] with
                K, V shape [B, num_heads, T_cache, head_dim].
            T_cache: total number of tokens in cache (the same value for all
                layers; the model will append T_new tokens past this).
        """
        # 1. Collect FrameKV from anchor + window (cross-layer).
        anchor_frames = self.anchor.collect_all()
        window_frames = self.window.collect_all()

        # 2. Voxel retrieve: K, V each [N_retrieved, L_voxel, num_heads, head_dim]
        K_ret, V_ret = self.voxel.retrieve(new_chunk_positions.float())
        n_retrieved = 0 if K_ret is None else K_ret.size(0)

        # 3. Build per-layer cache. Token-dim layout: [H, T, D] then unsqueeze
        #    to [B, H, T, D] for the attention.
        kv_cache_list: List[List[torch.Tensor]] = []
        for layer_idx in range(self.n_blocks):
            parts_k: List[torch.Tensor] = []
            parts_v: List[torch.Tensor] = []
            for frame in anchor_frames + window_frames:
                parts_k.append(frame.K_per_layer[layer_idx])           # [H, P, D]
                parts_v.append(frame.V_per_layer[layer_idx])
            # Retrieved pivots — only for layers in voxel_layers
            if K_ret is not None and layer_idx in self._voxel_layer_set:
                vidx = self.voxel_layers.index(layer_idx)
                # K_ret[:, vidx] : [N_ret, H, D] → transpose → [H, N_ret, D]
                parts_k.append(K_ret[:, vidx].transpose(0, 1).contiguous())
                parts_v.append(V_ret[:, vidx].transpose(0, 1).contiguous())
            if parts_k:
                K = torch.cat(parts_k, dim=-2)                          # [H, T, D]
                V = torch.cat(parts_v, dim=-2)
                K = K.unsqueeze(0).expand(batch_size, -1, -1, -1).to(self.device, self.dtype)
                V = V.unsqueeze(0).expand(batch_size, -1, -1, -1).to(self.device, self.dtype)
                kv_cache_list.append([K, V])
            else:
                kv_cache_list.append([None, None])

        # Diagnostic counts.
        n_aw_tokens = sum(f.K_per_layer[0].shape[-2] for f in anchor_frames + window_frames)
        return kv_cache_list, n_aw_tokens, n_retrieved

    @torch.no_grad()
    def absorb_chunk(
        self,
        new_chunk_K_per_layer: List[torch.Tensor],   # length n_blocks; each [B, H, T_full, D]
        new_chunk_V_per_layer: List[torch.Tensor],
        T_cache_anchor_window: int,                   # tokens before retrieved+new
        n_retrieved_tokens: int,                       # retrieved pivots count
        new_chunk_positions: torch.Tensor,            # [B*T_new*N_patch, 3]
        new_chunk_instance_feat: torch.Tensor,        # [B*T_new*N_patch, ins_dim]
        new_chunk_frame_indices: torch.Tensor,        # [T_new]  global sequence indices
        new_chunk_is_keyframe_mask: torch.Tensor,     # [T_new]  bool (current chunk)
        current_keyframes: Optional[List[int]] = None,  # SLAM-wide keyframe set
        N_patch: int = None,
        batch: int = 0,
    ):
        """Slice the new chunk's KV out of post-forward cache and update
        anchor/window/voxel state."""
        B = new_chunk_K_per_layer[0].shape[0]
        T_new = int(new_chunk_frame_indices.numel())
        P = N_patch + self.patch_start_idx                            # tokens per frame in cache

        # Slice the new chunk's tail from the per-layer full caches.
        # The model returns K,V of shape [B, H, T_cache_in + T_new*P, D]
        # — we strip the cache-in portion and keep T_new * P tokens.
        new_K_per_layer_full: List[torch.Tensor] = []
        new_V_per_layer_full: List[torch.Tensor] = []
        T_new_tokens = T_new * P
        for layer_idx in range(self.n_blocks):
            K = new_chunk_K_per_layer[layer_idx]
            V = new_chunk_V_per_layer[layer_idx]
            new_K_per_layer_full.append(K[..., -T_new_tokens:, :].contiguous())
            new_V_per_layer_full.append(V[..., -T_new_tokens:, :].contiguous())

        # Build per-frame records — each frame holds its full P tokens of K,V
        # (so re-materialised cache replays the same layout). Patch-only
        # metadata (positions / instance_feat) is stored separately.
        records: List[FrameKV] = []
        for t in range(T_new):
            t_slice = slice(t * P, (t + 1) * P)
            K_layers = [new_K_per_layer_full[l][batch, :, t_slice, :]
                        for l in range(self.n_blocks)]
            V_layers = [new_V_per_layer_full[l][batch, :, t_slice, :]
                        for l in range(self.n_blocks)]
            pos_t  = new_chunk_positions.view(B, T_new, N_patch, 3)[batch, t]
            inst_t = new_chunk_instance_feat.view(B, T_new, N_patch, self.ins_dim)[batch, t]
            sig    = _make_instance_sig(inst_t)
            hot = self._hot_device
            rec = FrameKV(
                frame_idx       = int(new_chunk_frame_indices[t]),
                K_per_layer     = [k.contiguous().to(hot) for k in K_layers],
                V_per_layer     = [v.contiguous().to(hot) for v in V_layers],
                # Position / instance feat / signature stay on CPU regardless —
                # they're small + used in Python loops for retrieval / eviction.
                positions       = pos_t.contiguous().cpu().float(),
                instance_feat   = inst_t.contiguous().cpu().float(),
                instance_sig    = sig.cpu(),
                patch_start_idx = self.patch_start_idx,
            )
            records.append(rec)

        # Handle first-frame slot (frame 0 = global anchor) — exclusive from
        # window/anchor/voxel to avoid duplicate K,V in attention context.
        records_for_window: List[FrameKV] = []
        for rec in records:
            if rec.frame_idx == 0 and self.anchor.first_frame_kv is None:
                self.anchor.set_first_frame(rec)
                continue
            records_for_window.append(rec)

        # Push new chunk into window; collect frames evicted from window.
        evicted_from_window = self.window.push_chunk(records_for_window)

        # Keyframe set used to decide if window-evicted frames get promoted
        # to anchor.  Prefer the SLAM-wide `current_keyframes` if given (works
        # for OLDER frames being evicted now); fall back to this chunk's
        # mask (only covers frames in this chunk).
        if current_keyframes is not None:
            kf_set = set(int(x) for x in current_keyframes)
        else:
            kf_set = set(int(f) for f, m in zip(
                new_chunk_frame_indices.tolist(),
                new_chunk_is_keyframe_mask.tolist(),
            ) if bool(m))

        # For each evicted frame:
        #   - if a keyframe → try anchor admit; if anchor rejects → push to voxel
        #   - else → push to voxel directly
        for rec in evicted_from_window:
            if rec.frame_idx == 0 and self.anchor.first_frame_kv is rec:
                continue   # never push first frame anywhere else
            if rec.frame_idx in kf_set:
                evicted = self.anchor.try_admit(rec)
                if evicted is not None and evicted.frame_idx != 0:
                    self._push_frame_to_voxel(evicted)
            else:
                self._push_frame_to_voxel(rec)

        # `rotate` mode: also sync anchor with current SLAM keyframes
        # (caller calls .sync_anchor_with_keyframes after each step).

    @torch.no_grad()
    def sync_anchor_with_keyframes(self, current_keyframes: List[int]):
        """Used in rotate mode.  Drop anchor frames no longer in keyframe set
        and push them to voxel.  No-op in sticky mode."""
        evicted = self.anchor.sync_with_keyframes(current_keyframes)
        for rec in evicted:
            if rec.frame_idx != 0:
                self._push_frame_to_voxel(rec)

    @torch.no_grad()
    def update_frame_positions(self, frame_to_delta: Dict[int, torch.Tensor]):
        """Per-frame SE(3) refinement applied to every stored token from each
        listed frame.

        `frame_to_delta`: dict {frame_idx → 4×4 c2w delta = T_new @ T_old^-1}.
        The delta is applied to 3D *world points* coming from frame f (since
        the camera-to-world matrix moved, the inferred world point also moves
        by the same transform).

        - Anchor (first frame + pool) and Window FrameKV `positions` updated.
        - Voxel pivots are re-bucketed in the spatial hash (positions may
          cross voxel boundaries).
        - No-op when `update_positions_on_alignment=False`.
        """
        if not self.cfg.update_positions_on_alignment:
            return
        if not frame_to_delta:
            return
        for rec in self.anchor.collect_all():
            T = frame_to_delta.get(rec.frame_idx)
            if T is not None:
                self._transform_record_positions(rec, T.cpu().float())
        for rec in self.window.collect_all():
            T = frame_to_delta.get(rec.frame_idx)
            if T is not None:
                self._transform_record_positions(rec, T.cpu().float())
        self.voxel.update_pivot_positions(frame_to_delta)

    # ── private helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _push_frame_to_voxel(self, rec: FrameKV):
        """Push every patch token of a frame as a voxel pivot candidate.
        Pose token (if any) is skipped — it has no 3D position.

        K,V are moved to `_cold_device` (typically CPU under 'hybrid' storage)
        because the voxel pool is cold storage — only a small subset is
        retrieved each chunk."""
        N_patch = rec.positions.shape[0]
        s = rec.patch_start_idx                       # skip pose token
        cold = self._cold_device
        # Per-layer K, V slice to patches only, then stack across voxel_layers
        # → [L_voxel, H, N_patch, D]
        K_stack = torch.stack(
            [rec.K_per_layer[l][:, s:s + N_patch, :].to(cold) for l in self.voxel_layers],
            dim=0,
        )
        V_stack = torch.stack(
            [rec.V_per_layer[l][:, s:s + N_patch, :].to(cold) for l in self.voxel_layers],
            dim=0,
        )
        for n in range(N_patch):
            self.voxel.add_or_update(
                K_per_layer   = K_stack[:, :, n, :],
                V_per_layer   = V_stack[:, :, n, :],
                position      = rec.positions[n],
                instance_feat = rec.instance_feat[n],
                frame_idx     = rec.frame_idx,
            )

    @staticmethod
    @torch.no_grad()
    def _transform_record_positions(rec: FrameKV, T: torch.Tensor):
        ones = torch.ones(rec.positions.shape[0], 1, dtype=rec.positions.dtype)
        pts_h = torch.cat([rec.positions, ones], dim=-1)               # [N, 4]
        rec.positions = (T @ pts_h.t()).t()[:, :3].contiguous()

    # ── diagnostics ─────────────────────────────────────────────────────────

    def diagnostics(self) -> Dict[str, int]:
        return {
            'first_frame_pinned': int(self.anchor.first_frame_kv is not None),
            'anchor_pool_size'  : len(self.anchor.pool),
            'window_chunks'     : len(self.window.chunks),
            'window_frames'     : sum(len(c) for c in self.window.chunks),
            'voxel_count'       : len(self.voxel.store),
            'voxel_pivots'      : self.voxel.num_pivots(),
        }

# OpenSe3r 模型換代 — 給 ma-slam semantic branch 的交接文件

**日期** 2026-08-25 ·  **來源 repo** `/mnt/HDD4/ricky/OpenSe3r` ·  **對象** `src/ma_slam_semantic/`

ma-slam 目前掛的是 **naive semconf** 模型。要換成兩個新模型:

| 代號 | checkpoint | 狀態 |
|---|---|---|
| **A. semconf+PQ** | `outputs/opense3r_pq_only_on_semconf/checkpoint-last-descfix.pth` | ✅ 已完成,論文所有數字都出自它 |
| **B. A1** | `outputs/opense3r_fastsam_query_a1/checkpoint-last.pth` | 🔄 訓練中(ep33/59,約 2026-08-28 收工);之後還會**再加 PQ head** |

---

## 0. 先講結論:建議整段換掉 `FROZEN_MODEL_ARGS`

**A1 模型目前根本無法用 ma-slam 現在的路徑建起來。**

`sem_model.py` 走 `slam_opense3r.run.load_model(cfg.model_args())`,而 `slam_opense3r/run.py` **沒有** `--seed_sem` / `--no_dense_sem` 這兩個旗標(只有 `--use_pq_head`)。A1 少了這兩個就會建成錯的架構,而且因為是 `strict=False` 載入,**會靜默地載成一個壞模型**,不會報錯。

**建議做法:改用 `semantic_mapping/model.py` 的 `build_model()`,把 `FROZEN_MODEL_ARGS` 整個刪掉。**

```python
# 取代 sem_model.py 裡的 load_model(cfg.model_args())
import sys; sys.path.append(OPENSE3R_ROOT)
from semantic_mapping.model import build_model
model, ckpt_args = build_model(cfg.ckpt_path, device=cfg.device)
```

它做兩件 `FROZEN_MODEL_ARGS` 做不到的事:

1. **從 checkpoint 自己的 `args` 推架構** → 三個模型、以後每個新模型都不用再改 config
2. **`_infer_arch_from_weights()`:用權重當 ground truth 覆寫 args**(`semantic_mapping/model.py:140`)。這不是防禦性程式碼,是修真的 bug —— descfix checkpoint 的 args 寫 `query_descriptor_mlp_layers=0`,但檔案裡實際有 3 層 descriptor MLP。照 args 建就會**第二次**把 descriptor 丟掉,而 ma-slam 的 S1 loop verification 正好整個建立在那些 descriptor 上。

同一個函式也會偵測 `pq_head.*` 是否存在並修正 `use_pq_head`。

如果你堅持保留 `FROZEN_MODEL_ARGS`,下面第 3 節有逐欄位的 diff,但你要自己承擔每次換模型都得手動同步的風險。

---

## 1. ma-slam 現況(我讀到的)

| 項目 | 現值 |
|---|---|
| checkpoint | `opense3r_fastsam_query_semconf/checkpoint-last.pth`(ep44) |
| 架構來源 | `config.py:66-121` 靜態 `FROZEN_MODEL_ARGS`,`use_sem_conf_head=True`,**無** `use_pq_head` |
| code 來源 | `vendor/opense3r/`(2026-07-15/16 的複本,**已過時 40 天**) |
| 語意怎麼來 | `proposals.py`:對 dense `semantic_feat_expanded` 做 mask-mean → `sem_query` [K,1024] |
| cluster 排序 | `mean_objectness` |

---

## 2. 兩個新模型分別改了什麼

### A. semconf+PQ — 「加一個排序頭」

trunk 跟現在這顆**完全一樣**(PQ 是在凍結 trunk 上單獨訓練的),只多一個 `pq_head`(28 個張量),輸出兩個 per-query 標量:

- **P** = thingness(視角穩定)
- **Q** = mask quality(視角相依)

用途**只有一個:cluster 排序**。P/Q 拿來做 gate / filter / accumulation 全部測過是 NO-GO(recall 掉或持平)。

```python
# semantic_mapping/stages/ranking.py:62  — 直接抄這段
def rank_p_top25q(rec):
    if not (rec.member_pq_p and rec.member_pq_q):
        return rank_obj(rec)                       # 沒 PQ head → 自動退回 objectness
    return mean(rec.member_pq_p) * top_quantile_mean(rec.member_pq_q, 0.25)
```

P 取平均、Q 取**前 25% 的平均**(問「這個 record 最好的幾個視角有多好」——垃圾提案沒有任何好視角)。

**值多少**:SceneNN-12 上 **+2.96 AP / +5.01 AP50 / +5.79 AP25**(28.95/46.56/62.29 → 31.91/51.57/68.08),同一組 cluster、只換分數。ScanNet 上 +2.9 AP50。

**要做的事**:記錄每個 observation 的 `pq_p` / `pq_q`(`instance_map.py` 的 record 加兩個 list),匯出時用上面的函式算 `cluster_score`。

### B. A1 — 「拿掉 dense semantic head」

這顆改的是 **seeding 的語意半邊**:原本 FastSAM region 的語意種子是對 dense semantic feature 做 scatter-mean,A1 改成 **frozen TIPS-v2 對 FastSAM mask 做 textregion pool**(mask 限制的 attention-V pooling,沒有 per-pixel context bleed)。

因為語意已經由 frozen teacher 提供,**dense semantic head 整個拿掉**:

```python
# opense3r/model_da3.py:1359
if self._skip_dense_sem and not (self.fastsam_query and self.seed_sem == 'dense'):
    sem_feat = sem_conf = sem_feat_expanded = None      # ← A1 走這條
```

> ⚠️ **對 ma-slam 最重要的一句話**:A1 模型的 `semantic_feat` / `semantic_feat_expanded` **是 `None`**。
> `proposals.py` 現在算 `sem_query` 的 mask-mean 會直接拿到 None → 語意整條斷掉。
>
> `semantic_head.*` 的權重在 checkpoint 裡**還在**(64 個張量,為了檔名相容而保留),但**從來沒被訓練也沒被跑過**。不要拿它們來算任何東西。

**替代方案**(已驗證,照抄即可):`semantic_mapping/stages/proposals.py:224` 的 `_textregion_sem_query()`——拿 proposal 的 owner map 當 instance mask,餵 frozen TIPS 的 `forward_supervision(sem_supervision='textregion')`,得到 per-proposal 的 [K,1024],下游(running-mean 累積、merge、text query)完全不用改。

**成本**:每個 chunk 多一次 frozen TIPS forward。

**值多少**(A1 模型 vs semconf trunk,兩邊都用 A1 聚合):

| | SceneNN-12 instance | ScanNet-5 SN20 instsem | Replica-8 instsem |
|---|---|---|---|
| A1 (ep28) | **31.31 / 50.50 / 63.69** | **43.89 / 63.33** | **29.26** / 40.15 |
| semconf | 28.95 / 46.56 / 62.29 | 42.96 / 62.00 | 29.13 / **41.97** |

**mask 大贏(+3.94 AP50,11/12 場),語意持平。** 這是設計使然:語意由 frozen teacher 決定,訓練的模型只負責把 mask 做好。

> 兩邊都是 objectness 排序(A1 還沒 PQ head)。A1 加上 PQ 後外推 ≈ **34.3 / 55.5 / 69.5**。

---

## 3. `FROZEN_MODEL_ARGS` 逐欄位 diff(若你不改用 `build_model`)

只列**影響架構**的欄位。其餘訓練超參(lr / w_* / epochs / dataset …)建模時不讀,可忽略。

| key | ma-slam 現值 | A. semconf+PQ | B. A1 |
|---|---|---|---|
| `use_pq_head` | *(缺)* | **`True`** | `False` → 之後 `True` |
| `use_sem_conf_head` | `True` | `True` | **`False`** |
| `seed_sem` | *(缺)* | `'dense'` | **`'textregion'`** |
| `no_dense_sem` | *(缺)* | `False` | **`True`** |
| `pq_sem_source` | *(缺)* | `'dense'` | `'dense'` → 之後 **`'textregion'`** |
| `inst_sem_head` | *(缺)* | `False` | `False` |
| `query_descriptor_mlp_layers` | `3` | **args 寫 0,權重是 3 → 必須傳 3** | `3` |

其餘(`dino_backbone=vggt`、`semantic_backbone=tips-v2-l14`、`prior_method=native`、`sem_supervision=textregion`、`pos3d_cond=True`、`no_depth_cross_attn=True`、`depth_cond=False`、`n_blocks=12`、`sem_dim=1024`、`ins_dim=16`、query head 的 100/256/256/8/9/3、`fastsam_query=True` + conf 0.2 / iou 0.9 / max_regions 100)**三顆完全一致**,不用動。

`region_pool.seed_mlp.0.weight` 三顆都是 `(256, 1568)` —— A1 的 textregion 種子維度沒變(仍是 causal 512 + sem 1024 + geo 32),只是**算法**不同。所以 region_pool 本身不用改。

---

## 4. 必須重新 vendor 的檔案

`vendor/opense3r/` 是 7/15-16 的複本。我確認過裡面**完全沒有** A1 需要的東西:

```
vendor/opense3r/semantic_backbones/_base.py         textregion_from_shared  → 0 處
vendor/opense3r/semantic_backbones/tips_backbone.py native_from_shared      → 0 處
vendor/opense3r/fastsam_seg.py                      seed_sem                → 0 處
vendor/opense3r/region_pool.py                      sem_regions 參數        → 無
vendor/opense3r/model_da3.py                        seed_sem/no_dense_sem   → 0 處
```

要從 live repo 複製過來的:

| 檔案 | 為什麼 |
|---|---|
| `opense3r/model_da3.py` | `seed_sem` / `no_dense_sem` / `pq_sem_source` / `sem_residual` / `inst_sem_head` 建構子參數 + dense-head skip 邏輯 |
| `opense3r/semantic_backbones/_base.py` | `textregion_from_shared()` |
| `opense3r/semantic_backbones/tips_backbone.py` | `native_from_shared()`、`_v_forward` 多回傳 `'x'` |
| `opense3r/fastsam_seg.py` | `build_seed_queries(..., sem_backbone=, shared=)` |
| `opense3r/region_pool.py` | `forward(..., sem_regions=)`(給了就直接用,跳過 scatter-mean) |
| `opense3r/instance_head_query_m2f.py` | `pq_head` + `pq_sem_backbone`/`pq_shared` |
| `semantic_mapping/model.py` | `build_model` + `_infer_arch_from_weights`(建議整個改用它) |
| `semantic_mapping/stages/ranking.py` | P/Q 排序(A 模型要) |
| `semantic_mapping/stages/proposals.py` | `_textregion_sem_query()`(B 模型要) |

**只有一次 TIPS forward**:A1 在同一次 encoder pass 裡同時產生 causal branch 的 dense prior 和 seeding 用的 last-layer V(`native_from_shared` 與獨立呼叫 `forward_prior_native` 驗證過 max|diff| = 1.9e-6)。移植時不要拆成兩次 forward,會白白多一倍成本。

---

## 5. 三個會**靜默失敗**的坑

1. **`strict=False` 載入不會報錯。** 架構對不上就是安靜地載成錯的模型。載完務必印 missing/unexpected 的分類統計,**trainable 的 missing 必須是 0**(frozen backbone 的 missing 是正常的)。

2. **`pq_head.` 曾經漏在 trainable-key 白名單外**,結果 P/Q head 被整個丟掉也沒人發現(legacy `slam_opense3r` 的舊 bug)。移植後檢查 `pq_head.proj_q.0.weight` 真的有載進去。

3. **ckpt args ≠ ckpt 內容。** descfix 那顆的 `query_descriptor_mlp_layers` args 寫 0、實際有 3。務必用 `_infer_arch_from_weights`。

另外兩個非靜默但值得知道的:

- **`chunk_size` 是行為旗標不是效能旗標**(chunk_bidir / causal 都是),chunk 4 vs 6 在 SceneNN 上差 3.5 AP50。ma-slam 的參考設定是 **6**,換模型時不要順手改。
- **forward 不是 bit-deterministic**(同輸入跑 3 次 `mask_logits` max|Δ| = 10.0),所以任何 parity 驗證都要留容差。

---

## 6. 驗收數字

換完之後跑 SceneNN 應該落在:

| 模型 | 排序 | AP / AP50 / AP25 |
|---|---|---|
| semconf(現況) | objectness | 28.95 / 46.56 / 62.29 |
| **A. semconf+PQ** | p_top25q | **31.91 / 51.57 / 68.08** |
| **B. A1**(ep28) | objectness | **31.31 / 50.50 / 63.69** |
| B. A1 + PQ | p_top25q | ~34.3 / 55.5 / 69.5(外推) |

評估協定:SceneNN-12,`chunk_size 6`,`frame_stride 1`,`--benchmark_void --segment_snap`(segs 來自 `thirdparty/segmentator` 自產)。

**如果 A 模型換完 SceneNN 沒有跳到 ~51.6 AP50**,第一個要查的是 P/Q 有沒有真的載進去、有沒有真的被拿去算 `cluster_score`(退回 objectness 是**靜默**的 —— 見 `rank_p_top25q` 的 fallback)。

---

## 7. 環境

```bash
export PATH=/mnt/HDD4/ricky/envs/amb3r_bw/bin:$PATH   # amb3r_bw,不是 amb3r
```

GPU 使用規範:**跑任何吃顯存的工作前先問使用者**(兩張 PRO 6000 都在跑訓練);檢查要看 `memory.used` 不是 `utilization`;**絕不 kill 不是自己啟動的程序**。

有問題可以直接讀 `/mnt/HDD4/ricky/OpenSe3r/EXPERIMENTS.md`(一行一實驗的帳本,很多想法已經有定論)和 `paper/09_todo.md`(裡面 §「re-vendoring is not a file copy」已經點出過同一個問題)。

---

## 8. ma-slam 端實作狀態(2026-08-25,semantic branch)

**已完成接線(依 §0 建議整段換掉 `FROZEN_MODEL_ARGS`):**

| 項目 | 位置 | 說明 |
|---|---|---|
| checkpoint 自述建模 | `src/ma_slam_semantic/sem_model.py::build_model_from_ckpt` | 用 vendored `semantic_mapping/model.py` 的 `read_ckpt_args / _arch_kwargs / _infer_arch_from_weights`;frozen-backbone 路徑依 root 覆寫(vendor/weights 改名版);trainable key 任何 missing/unexpected → raise;ckpt 有 `pq_head.*` 但模型沒建出 → raise(坑 #2) |
| checkpoint 註冊表 | `config.py::CKPTS` / `--ckpt` | `semconf`(ep44,golden 參考)/ `semconf_pq`(descfix,**預設**)/ `a1`(訓練中)/ 或任意 `.pth` 路徑 |
| P/Q 記錄 + 排序 | `proposals.py`(每 proposal 記 `pq_p/pq_q`)→ `instance_map.py`(`member_pq_p/q`,merge/final_merge 串接)→ `export.py`(`objectness_agg='p_top25q'` 交給 re-vendor 後的 `amb3r/vis_query_instance_slam.py::_record_cluster_score`) | `--ranker p_top25q` 預設;無 P/Q head 自動退回 objectness,export 會印 `** NO P/Q on any record **`(§6 靜默 fallback 的解法) |
| A1 語意來源 | `proposals.py::_textregion_sem_query`(port 自 `semantic_mapping/stages/proposals.py:224`) | `--sem_agg auto`:模型有 dense head 用 mask-mean,`semantic_feat` 為 None 時走 frozen-TIPS textregion;`hooks.py` / `run.py` 把 `model.semantic_backbone` 傳進 `make_proposals` |
| 離線重排序控制組 | `scripts/rerank_npz.py` | 同一份 map 換 ranker 重寫 `cluster_score`(對應 §2A「同一組 cluster、只換分數」的驗法) |

**Re-vendor(§4 清單 + 補遺):** `opense3r/{model_da3,model,instance_head_query_m2f,fastsam_seg,region_pool,instance_post,dpt_heads}.py`、`semantic_backbones/{_base,tips_backbone,text_region_attn}.py`、`amb3r/vis_query_instance_slam.py`、**新增 `opense3r/pq_head.py`**(清單漏列但 `model_da3` 會 import)、`vendor/semantic_mapping/{model.py, stages/ranking.py}`(部分 vendor,`__init__` 為 stub)。

**⚠️ re-vendor 時發現的事(下次換代要記得):**
- **H2O KV 是 ma-slam 本地 patch**:`opense3r/stac_kv/`、`causal_branch.py` 的 `register_kv_mgr/...`(114 行)、`model.py`/`model_da3.py` `init_kv_cache` 內的 `reset_kv_mgr()` —— **live repo 現在都沒有**。這次 `model*.py` 是「live 版 + 補回 3 行」的三方合併,`causal_branch.py` 維持 vendor 版。照抄 live 會把 H2O 弄壞且不報錯。
- `instance_post.py`(不在 §4 清單)也 re-vendor 了:差異全是帶預設值的新參數(pq / select_scores / geo_cue),預設路徑行為不變;`make_proposals_cpu` 靠它的新 `pq_p/pq_q` kwargs 記錄 P/Q。
- `text_region_attn.py` 的 `textregion_regions` 是 `_base.textregion_from_shared` 的依賴,§4 也沒列。

**CPU 建置驗證(無 GPU,三顆全過):** trainable missing/unexpected 皆 0;`semconf_pq` descriptor MLP 由 args 的 0 校正為 3、`pq_head` 建起且 `pq_head.proj_q.0.weight` 與 ckpt 逐位相等;`a1` → `_skip_dense_sem=True, seed_sem=textregion`,TIPS backbone 帶 `textregion_from_shared / native_from_shared / forward_supervision`。

**GPU 驗收(GPU1,進行中):** (G) 舊 semconf 走新 loader 的 golden 回歸(005, chunk 8,對照 7 月 gate 26.17/41.55/53.65)→ (P) semconf_pq SceneNN-12(chunk 6, p_top25q + 同 cluster 的 obj 重排序控制組)→ (A) a1 單場 smoke。結果補在下方。

### 8.1 GPU 驗收結果(2026-08-25 晚,GPU1;`outputs/ma_sem/modelswap/`)

**P) semconf+PQ,SceneNN-12,chunk 6,`--ranker p_top25q`(+ 同一份 map 用 `scripts/rerank_npz.py --ranker obj` 重排序的控制組):**

| | AP | AP50 | AP25 |
|---|---|---|---|
| ma-slam `semconf_pq` **p_top25q** | **29.65** | **49.56** | **67.14** |
| ma-slam `semconf_pq` obj 重排序(同 cluster) | 27.92 | 46.35 | 62.12 |
| → P/Q 排序增益 | +1.73 | +3.21 | +5.02 |
| 對照:legacy `slam_opense3r` 管線 p_top25q(`_scenenn_instance_ap_descfixP25q`,08-05) | 29.38 | 49.05 | 66.55 |
| 對照:legacy obj ledger(EXPERIMENTS 08-05c) | 28.3 | 46.5 | 61.7 |
| 對照:`semantic_mapping` rewrite Gate 2(§6 的 31.91/51.57/68.08) | 31.91 | 51.57 | 68.08 |

解讀:ma-slam 的 P/Q 鏈路**逐場**重現了它移植來源(legacy `slam_opense3r` 管線)的數字
(12 場 p_top25q 皆在 ±3 內,平均 +0.3/+0.5/+0.6),排序增益 +1.7/+3.2/+5.0 與 legacy
ledger 的 +1.1/+2.5/+4.9 同量級。§6 的 31.91/51.57 是 `semantic_mapping` rewrite 自身的
額外增益(chunk_bidir 等),不是 ma-slam 接線的缺口 —— 驗收標準應以 legacy 管線的
29.4/49.1/66.6 為準,**通過**。每場 P/Q 覆蓋率 100%(`records with P/Q: N/N`),無靜默 fallback。

**A) A1 smoke(ep~32 中途 ckpt,obj 排序,chunk 6,scene 005):** 27.55 / 43.56 / 60.65,
對同管線同場景的 semconf trunk obj 排序 25.36 / 41.10 / 52.39 = **+2.2 / +2.5 / +8.3** ——
textregion sem_query 路徑端到端可用,方向與 §2B「mask 大贏」一致;正式數字等 08-28 收工的 ckpt(+PQ head)。
throughput ~7 fps(每幀多一次 frozen-TIPS forward)。

**G) golden 回歸(舊 `semconf` ckpt,scene 005,chunk 8,obj 排序)— 通過(零回歸):**

| 程式碼 | env | AP / AP50 / AP25 |
|---|---|---|
| 舊碼 27fcd9d(`FROZEN_MODEL_ARGS` + `load_model`)| 7 月 gate 紀錄 | 26.17/41.55/53.65、25.61/40.62/54.70 |
| 舊碼 27fcd9d(git worktree,**今天的 env**)×2 | 今天 | **24.25/38.72/51.16、24.21/38.87/51.23** |
| 新碼(checkpoint-driven builder + re-vendor)×2 | 今天 | **24.16/38.74/51.48、24.52/39.09/53.97** |

新舊碼在同一 env 下逐位重合(Δ < 0.4,同碼重跑噪聲),表 statistics 也一致(總 observations
2659 vs 2660、mean objectness 0.732 vs 0.731)。對 7 月的 −1.5 AP 是**共用 env `amb3r_bw`
自 7 月以來的漂移**(FastSAM seeding / kernel 版本),與換代無關 —— 之後拿 7 月數字當 baseline
要先重跑校正。

**H2O KV(ma-slam 本地 patch)re-vendor 後驗證:** `vendor/opense3r/stac_kv/test_smoke.py` 12/12 +
`test_causal_branch.py` 7/7 通過;端到端 TUM fr1_xyz(`--kv_policy h2o`,新 loader,`semconf`)
h2o-kv tokens/layer 固定 15096 / 708 MB / prune 3.7 ms,Sim3-ATE 0.0103 / SE3 0.0251 / loops=2
與 re-vendor 前的 run(`outputs/sem_lc_shadow/unkfail_full/fr1_xyz`)逐位一致。

# ma_slam_semantic — Query-Instance Semantic Mapping on ma_slam

OpenSe3r 的 query-instance 語意建圖（DINOv2+TIPSv2-L14 雙 encoder → 12-block causal
transformer with KV-cache → Mask2Former query head，FastSAM 外部 query seeding）移植到
ma_slam 線上 SLAM 幾何上。參考設定 = OpenSe3r `005_..._geo_default_conf032_gt` run
（pipeline 旋鈕在 `config.SemConfig`;模型架構由 checkpoint 自述）。完整計畫與 gate 紀錄見
[docs/semantic_branch_plan.md](../../docs/semantic_branch_plan.md)。

> **模型換代(2026-08-25)— 完成並通過驗收**(golden 零回歸;SceneNN-12 `semconf_pq`
> p_top25q 29.65/49.56/67.14,同 cluster obj 27.92/46.35/62.12;A1 smoke 005 27.55/43.56/60.65)。 架構不再寫死在 config
> (`FROZEN_MODEL_ARGS` 已刪除):`sem_model.build_model_from_ckpt` 從 checkpoint 自帶的
> args 建模、再用權重校正(`vendor/semantic_mapping/model.py::_infer_arch_from_weights`
> —— descfix ckpt 的 args 寫 descriptor MLP=0 但檔案有 3 層;`pq_head` 有無同理),
> trainable key 有任何 missing/unexpected 直接 raise。`--ckpt {semconf|semconf_pq|a1|path}`
> 切換模型(預設 `semconf_pq`);`--ranker p_top25q`(P/Q 排序,無 P/Q head 自動退回
> objectness 並在 export 印警告);`--sem_agg auto`(A1 無 dense head → 走 frozen-TIPS
> textregion pool)。細節與坑見 **[docs/opense3r_model_handoff.md](../../docs/opense3r_model_handoff.md)**。
> vendor/ 已同步 live repo 的 12 個檔案 + 新增 `opense3r/pq_head.py`(注意:H2O KV 的
> `stac_kv/` + `causal_branch.register_kv_mgr` + `init_kv_cache` 的 `reset_kv_mgr()` 是
> ma-slam 本地 patch,live repo 沒有 —— 之後 re-vendor 時要保留)。

## 架構

```
ma_slam (MaSlam, hooks=SemanticHooks)          ← S1: 僅 3 hooks + 2 accessors
  └─ on_submap(sm, gmap, graph)   每個 submap 優化後
       SubmapBridge.chunk()       幾何→張量合約：解析度重採樣(518x336)、MA-conf→validity、
                                  gauge 錨定(序列 frame0=I)、重疊幀去重、[-1,1] 影像
       split_chunk(8)             KV-causal ⇒ batching 不變；8 幀 sub-batch 控 VRAM
       SemanticModel.forward_chunk()  OpenSe3r forward_semantic + KV 串接 + box_scale
       make_proposals()           鏡射 update_instances 前處理 → process_chunk_to_proposals
       InstanceTable.match_chunk()    provenance 版 instance matching（見下）
  └─ on_loop(k_new, k_old)        只記事件；癒合由 M-signature 機制自動處理
  └─ on_finish(gmap, graph)       全量 re-lift → final merge → noise filter → export
```

## Instance matching（與 OpenSe3r 同演算法 + 同配分 kernel）

逐幀：

1. **KNN gate**：proposal center 的 point-to-AABB 距離 ≤ `knn_radius=1.0m`，最多
   `max_candidates=32`（取最近）。候選若 M-signature 失效先 re-lift（LC 癒合點）。
2. **配分**（import `opense3r.instance_post._proposal_pair_score_dict`）：
   `score = 0.3·cos(desc) + 0.3·vox-IoU + 0.25·exp(−d/0.25)`，`w_sem=0`，
   vox-IoU = `|A∩B|/min(|A|,|B|)`，d = point-to-AABB。
3. **Hungarian**（`linear_sum_assignment`）一對一指派。
4. `score ≥ 0.32` → merge（descriptor 累積平均+renorm、center running mean、bbox
   union、voxel votes、附加 provenance Obs）；否則 spawn。

Export 時：`rekey_all(force)`（最終 graph 全量 re-lift）→ union-find final merge
（AABB 相交且 score ≥ 0.75）→ noise filter（`filter_*` knobs）→ 參考格式輸出。

## LC-safe provenance 設計（與參考實作的唯一行為差異）

- 每次 merge 附加一筆 **`Obs`(frame key, slot, head-res mask packbits, objectness)** —
  pose-free、~1-3KB。
- Record 平常維護與參考完全相同的增量快取，快取掛 **M-signature**（owning frames 的
  `M_i = G_i @ inv(L_i)` 量化 hash）。幾何不動 ⇒ 行為與參考逐位元一致、零開銷。
- Loop closure 移動幀後 signature 失效 ⇒ 下次 match/export 時**從 mask 重新 lift 過新
  幾何**（走 proposal 出生的同一條路徑：crop/nearest→518x336→bilinear→head-res→
  `lift_proposal_to_3d`；rigid M 與插值可交換，數學上精確相等）。
- 實測（s0231，10 loops）：LC 組 213 instances / ATE 0.099 vs no-loop 240 / 0.212 —
  重訪合併而非重複 spawn。

## 使用

```bash
# Golden 測試模式（GT 幾何，重現 OpenSe3r 005 run；S2/S3 正確性 harness）
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python -m ma_slam_semantic.run --geometry gt \
    --data /mnt/HDD4/ricky/data/instance_eval_data/scenenn/oni/005 \
    --demo_type scenenn_gt --out outputs/ma_sem/005_golden [--save_2d_masks]

# ma_slam 幾何模式（完整線上 SLAM + 語意）
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python -m ma_slam_semantic.run --geometry ma \
    --data data/scene0231_00 --mode rgb+depth+intr --submap_size 20 \
    --out outputs/ma_sem/s0231_ma --gt data/scene0231_00/gt_pose.txt

# 效能剖析（各模組時間 + VRAM + KV 成長）
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python -m ma_slam_semantic.profile_sem --max_frames 64

# 文字查詢標籤（export 產物上離線跑）
python - <<'EOF'
from ma_slam_semantic.query_text import label_instances
label_instances("outputs/ma_sem/005_golden", "005", benchmark=20)
EOF

# Instance AP eval（scenenn；GT npz 用 vendor/build_scenenn_gt.py 產）
python vendor/eval_instance_npz_scannet.py \
    --hdbscan_labels_npz outputs/ma_sem/005_golden/005_instance_labels.npz \
    --gt_scenenn_npz /tmp/scenenn_gt/005_scenenn_gt.npz \
    --benchmark_void --segment_snap --segs_json <segs> --snap_mesh_ply <mesh>

# 機器人部署成品（traj json + 多層 instance table + 上色 per-point PLY）
# 用法/參數/座標系規則詳見 ../ma_slam_stream/SEMANTIC_README.md §3
python -m ma_slam_semantic.deploy --dir outputs/ma_sem/<scene> --name <scene>
```

### 主要旋鈕

| 旋鈕 | 預設 | 說明 |
|---|---|---|
| `--conf_gate_coef` | 1.0 | 幾何 validity gate（× submap 平均 conf）。**實務 inference 留 1.0**（擋不準的 pred geo）；**benchmark 報表用 0.5**（GT-mesh 覆蓋率計分下 5-scene mean AP 20.8→25.3）。效果 scene-specific：幾何乾淨的場景放鬆只會進雜訊。 |
| `chunk_size` / `--sem_chunk_size` | 4 | 每次 causal forward 幀數，**純 VRAM 旋鈕**（結果不變）：4≈15GB、8≈26GB 峰值 |
| `--kv_policy` | 離線 `none` / 串流 `h2o` | flat KV 成長 62MB/幀（1000 幀=61GB）；`h2o`（STAC：pinned frame0 + `--h2o_recent` 8 + heavy-hitter `--h2o_hh` 8、`--h2o_subsample 0.25` 已驗證 AP 持平）恆定 ~0.7GB |
| `--no_loop` | off | LC 開著 instance map 會自癒（s0231：213 vs 240 instances） |

### ScanNet 5-scene benchmark（inst SN200 + instsem）

```bash
# 每場景：--geometry ma --gt <gt_pose.txt>（輸出自動 Umeyama 對齊 GT-mesh 座標）
# inst eval（唯一跑 SN200；protocol 固定帶這兩旗標，舊 summary 不可比）：
python vendor/eval_instance_npz_scannet.py --hdbscan_labels_npz <out>/<sc>_instance_labels.npz \
    --gt_ply scannet_200/<sc>/gt_scannet200_rawid_<sc>.ply \
    --aggregation_json scannet_200/<sc>/<sc>.aggregation.json \
    --segs_json scannet_200/<sc>/<sc>_vh_clean_2.0.010000.segs.json \
    --benchmark 200 --benchmark_void --segment_snap
# instsem（SN20+SN200，5 場景一次聚合；在 OpenSe3r repo 跑）：
python query_instance_semantic.py --instance_labels_npz ... --instance_sem_queries_npz ... \
    --benchmark 20 --backbone tips-v2-l14 --output_dir <out> --demo_name <sc>   # 逐場景廣播
python tools/ovo_style_semantic_eval.py --dir_pattern '<outs>/{sc}' \
    --backbone tips-v2-l14 --tasks instsem --benches sn20 sn200                  # 一次聚合
```
一鍵 driver 範本：`outputs/ma_sem/scannet5*/run_all.sh`。

**結果（SN200 inst mean AP/AP50/AP25）**：MA(rgb+d+i) gate1.0 20.8/43.2/67.1、
gate0.5 **25.3/50.5/73.9**；DA3(rgb) gate1.0 17.9/41.9/59.5、gate0.5 21.8/47.5/71.3。
instsem 聚合 SN20 mIoU ~38-40（四配置持平——semantic eval 對覆蓋鈍感：無界 5-NN 眾數
會把洞補成通常正確的 class；inst AP 則被覆蓋罰兩次，這是 gate 敏感度差異的機制）。
0231 缺口分解：覆蓋 >> 漂移碎裂 >> eval bleed（詳見 git log `4b61284`）。

## OpenSe3r 依賴：vendor/（自帶版）

`ensure_opense3r()` 解析順序：**`$OPENSE3R_ROOT` env → `<repo>/vendor/`（預設）**。
`vendor/` 鏡像 OpenSe3r 根目錄結構（runtime 閉包 121 檔實測後整包搬核心套件 +
thirdparty 子集，~15MB code），`slam_opense3r/run.py` 的檔案相對 sys.path 因此原樣可用。

權重在 `vendor/weights/`（**gitignored**，共 ~14GB，已重新命名）：

| 檔案 | 原始來源 |
|---|---|
| `opense3r_fastsam_query_semconf.pth` (7.3G) | OpenSe3r `outputs/opense3r_fastsam_query_semconf/checkpoint-last.pth` |
| `vggt_1b.pt` (4.7G) | OpenSe3r `checkpoints/VGGT.pt`（DINO tokens 來源，frozen） |
| `tips_v2_oss_l14_{vision,text}.npz` (1.2G/0.7G) | `thirdparty/tips/pytorch/checkpoints/`（由該處相對 symlink 指回來） |
| `fastsam_x.pt` (139M) | `/mnt/HDD4/ricky/sam_test/FastSAM-x.pt` |

搬機時：repo（含 vendor code）+ `vendor/weights/` 五個檔案 = 完整可跑。ckpt 是
strict=False 載入——**`FROZEN_MODEL_ARGS` 的架構參數與 ckpt 綁定，不可調**；可調的
在 `SemConfig`（matching/proposal/filter/KV knobs，預設 = 005 run）。

## 驗證結果（gates）

- **Golden**（scenenn 005，同 eval protocol `--benchmark_void --segment_snap`）：
  本移植 AP 26.17 / AP50 41.55 / AP25 53.65 vs 參考 npz 25.12 / 42.35 / 52.97；
  streaming 132 instances == 參考 132。⚠️ 舊 summary 的「AP 14.71」是 2026-07-12 前的
  strict-legacy protocol，不可直接比。
- **s0011 e2e**：SLAM ATE 與無 hooks 基準逐字一致（0.0589）；輸出 71 instances。
- **s0231 LC**：見上。KV reset 閥門（`kv_reset`，預設 off）全程未觸發（pose delta < 0.5m）。

## 效能（profile_sem 實測）

- Forward 僅 **66 ms/frame**（causal 12.4 / FastSAM 11.7 / TIPS 10.9 / M2F 9.6 /
  DPT 5.7 / DINO 4.6）——**FastSAM 不是瓶頸**。
- 原瓶頸是 CPU 後處理 426 ms/frame（`lift_proposal_to_3d` 191：Python tuple-set
  voxel 化 + CPU IQR；dense sem_feat GPU→CPU 搬運 165：712MB/frame。OpenSe3r 原版
  同樣存在——`update_instances` 寫死 `device='cpu'`）。
- **`proposals.py` 已 GPU 化**（NMS/resolve 餵 GPU 張量、sem_query 用 `index_add_`
  批次 pool、lift 用 GPU quantile/median/`torch.unique` 後才落地建 set）：
  A/B 驗證與 CPU 參考路徑**位元級一致**（center/bbox err=0、voxel Jaccard=1、
  sem_query cos=1；`make_proposals_cpu` 保留供回歸），**441 → 18 ms/frame（24×）**。
  Golden 端到端 **1.96 → 10.0 fps**，AP 25.61/40.62/54.70（雜訊範圍內）。
- VRAM：權重 3.75GB；**activation 隨 chunk T 線性 ~2.3GB/frame**（chunk 8 峰值
  ~26GB → chunk 4 ~15GB，M2F dAlloc 精準減半），故 `chunk_size` 預設 4；
  **KV cache 線性成長 62.4MB/frame**（1000 幀 = 61GB ⇒ 長序列需 window/H2O
  policy），與 chunk size 無關。ma 模式（+MapAnything 同卡）chunk 8 時總峰值 49.4GB。

## 待辦（S5）

H2O KV policy（`kv_policies.py` seam 已留；KV 成長是長序列的 VRAM 大頭）、KV-reset
閥門的實測決策（需更大 drift 的場景）、labeled `combined_pcd` 匯出、更多資料集、
下一個效能目標：forward 66ms 與 match_chunk 的 O(P·C) 配分。

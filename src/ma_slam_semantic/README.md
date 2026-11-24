# ma_slam_semantic — Query-Instance Semantic Mapping on ma_slam

OpenSe3r 的 query-instance 語意建圖（DINOv2+TIPSv2-L14 雙 encoder → 12-block causal
transformer with KV-cache → Mask2Former query head，FastSAM 外部 query seeding）移植到
ma_slam 線上 SLAM 幾何上。參考設定 = OpenSe3r `005_..._geo_default_conf032_gt` run
（pipeline 旋鈕在 `config.SemConfig`;模型架構由 checkpoint 自述）。完整計畫與 gate 紀錄見
[docs/semantic_branch_plan.md](../../docs/semantic_branch_plan.md)。

**目前狀態（final）：** 預設值就是最終設定，`--geometry ma` 不需要任何額外旗標。

| 項目 | 預設 | 旗標 |
|---|---|---|
| 幾何 | DA3 rgb、50 px keyframe、每個 submap 32 個 keyframe、SE(3) pose graph | `--backend da3 --mode rgb --keyframe_disparity 50 --submap_size 32` |
| 迴環檢索 | SALAD cos ≥ 0.40、每個 submap 最多 5 個候選 | `--sim_threshold 0.40 --max_per_submap 5` |
| 迴環驗證（IDLV） | 互為最近鄰的 instance descriptor 配對 cos ≥ τ=0.85，配對數 ≥ m=4 才通過；每個 keyframe 取 objectness 前 k=16 個 instance | `--sem_lc veto --sem_lc_mode rigid --rigid_geom off`；τ=`--rigid_tau`、m=`--rigid_min_inliers`、k=`--bd_kbag` |
| 幾何驗證 | 兩幀聯合重推論，相機中心距離 / 深度中位數 ≤ 0.70；通過後以 Huber loop factor 加入 | `--coloc_ratio 0.70` |
| 語意模型 | `a1_pq`（A1 ep28 + P/Q ep14），每次 causal forward 4 幀 | `--ckpt a1_pq --chunk_size 4` |
| KV cache | H2O：frame 0 + 最近 8 幀 + heavy-hitter 8 幀 | `--kv_policy h2o --h2o_recent 8 --h2o_hh 8 --h2o_decay 0.9 --h2o_subsample 0.25` |
| instance matching | `0.70·cos(desc) + 0.30·exp(−d/0.25)`，門檻 0.65 | `--assoc new`（預設） |

`--help` 裡標 `[legacy]` 的旗標屬於早期的驗證設計（Bi-Desc S1、layout S2、tier-B mining），只為了重現舊的
run 而保留，不影響預設路徑。模型架構由 checkpoint 自述（`sem_model.build_model_from_ckpt`），換模型的細節與
歷史見 [docs/opense3r_model_handoff.md](../../docs/opense3r_model_handoff.md)。

## 架構

```
ma_slam (MaSlam, hooks=SemanticHooks)          ← S1: 僅 3 hooks + 2 accessors
  └─ on_submap(sm, gmap, graph)   每個 submap 優化後
       SubmapBridge.chunk()       幾何→張量合約：解析度重採樣(518x336)、MA-conf→validity、
                                  gauge 錨定(序列 frame0=I)、重疊幀去重、[-1,1] 影像
       split_chunk(4)             KV-causal ⇒ batching 不變；4 幀 sub-batch 控 VRAM
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
   `score = 0.70·cos(desc) + 0.30·exp(−d/0.25)`，d = point-to-AABB（`w_vox=0`、`w_sem=0`；
   `--assoc old` 還原 2026-09-04 以前的 0.3/0.3/0.25、門檻 0.32，見
   [docs/config_change_2026-09-04.md](../../docs/config_change_2026-09-04.md)）。
3. **Hungarian**（`linear_sum_assignment`）一對一指派。
4. `score ≥ 0.65` → merge（descriptor 累積平均+renorm、center running mean、bbox
   union、voxel votes、附加 provenance Obs）；否則 spawn。

Export 時：`rekey_all(force)`（最終 graph 全量 re-lift）→ union-find final merge
（AABB 相交且累積 descriptor cos ≥ 0.85）→ noise filter（`filter_*` knobs）→ 參考格式輸出。

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
# Inference（完整線上 SLAM + 語意）— 預設值 = 最終設定（見最上方的表），只需要資料與輸出路徑。
# <scene_dir> 底下要有 rgb/；--gt 可省略（有的話會印 ATE，並把 instance 輸出對齊到 GT 座標）
CUDA_VISIBLE_DEVICES=0 python src/ma_slam_semantic/run.py \
    --data example_data/scene0378_00 --out outputs/ident3r_s0378 \
    --gt example_data/scene0378_00/gt_pose.txt

# 對照組：關掉 IDLV（只剩幾何驗證）／完全關掉迴環
python src/ma_slam_semantic/run.py --data <scene_dir> --out <out> --sem_lc off
python src/ma_slam_semantic/run.py --data <scene_dir> --out <out> --no_loop

# Golden 測試模式（GT 幾何，重現 OpenSe3r 參考 run；正確性 harness）
python src/ma_slam_semantic/run.py --geometry gt --data <scenenn_scene_dir> \
    --demo_type scenenn_gt --out outputs/ma_sem/005_golden [--save_2d_masks]

# 效能剖析：語意模型各模組（GT 幾何）／整條 pipeline 各階段時間 + VRAM
PYTHONPATH=src python -m ma_slam_semantic.profile_sem --data <scenenn_scene_dir> --max_frames 64
PYTHONPATH=src python -m ma_slam_semantic.profile_pipeline --prof_out prof.json -- --data <scene_dir> --out <out>

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
| `--chunk_size`（串流：`--sem_chunk_size`） | 4 | 每次 causal forward 幀數，**純 VRAM 旋鈕**（結果不變）：4≈15GB、8≈26GB 峰值 |
| `--kv_policy` | `h2o`（離線、串流皆同；`none` = flat KV 對照） | flat KV 成長 62MB/幀（1000 幀=61GB）；`h2o`（STAC：pinned frame0 + `--h2o_recent` 8 + heavy-hitter `--h2o_hh` 8、`--h2o_subsample 0.25` 已驗證 AP 持平）恆定 ~0.7GB |
| `--no_loop` | off | LC 開著 instance map 會自癒（s0231：213 vs 240 instances） |
| `--sem_lc` / `--sem_lc_mode` / `--rigid_geom` | `veto` / `rigid` / `off` | IDLV（論文方法）：SALAD 候選需 >=4 個 MNN instance 配對（cos>=0.85）才進幾何驗證，否則 REJECT。`bidesc`（舊 S1/S2 規則）、`--rigid_geom on`（layout）、`--bd_s2 on` 皆為 ablation |
| `--sim_threshold` / `--max_per_submap` | 0.40 / 5（solver 預設） | relaxed 檢索 = 最新主表設定；0.50 / 1 = strict 對照 |

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

權重在 `vendor/weights/`（**gitignored**）：Ident3R checkpoint `opense3r_a1ep28_pq_ep14.pth`（`--ckpt a1_pq`）、
`vggt_1b.pt`、`tips_v2_oss_l14_{vision,text}.npz`、`fastsam_x.pt`。打包檔 `weights.zip` 的下載、SHA-256 與
完整清單見 [../ma_slam_stream/SEMANTIC_README.md §2](../ma_slam_stream/SEMANTIC_README.md)。

搬機時：repo（含 vendor code）+ 解壓後的權重 = 完整可跑。模型架構由 checkpoint 自述、載入時逐 key 檢查；
可調的在 `SemConfig`（matching / proposal / filter / KV knobs）。

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

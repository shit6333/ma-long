# Semantic SLAM — 完整流程（ma_slam / ma_slam_stream + ma_slam_semantic）

從資料（離線資料夾或 RealSense 即時串流）到機器人可讀成品的整條 pipeline：

```
幾何：ma_slam (MapAnything / DA3) ─ submap 重建 → SE3 pose graph → loop closure
語意：ma_slam_semantic hooks ─ 每個 submap 優化後 → causal semantic model (KV cache)
      → FastSAM-seeded Mask2Former proposals → provenance InstanceTable（LC 自癒）
輸出：instance npz/table/ply → [--deploy] 機器人成品（traj json + 多層 table + 上色 pcd）
```

架構/設計/實驗細節見 [ma_slam_semantic/README.md](../ma_slam_semantic/README.md)；
本文件只講「怎麼跑」。

## 0. 前置（一次性）

- **環境**：conda env `amb3r_bw`（`/mnt/HDD4/ricky/envs/amb3r_bw/bin/python`），與
  [README.md](README.md) §1 相同；語意端額外用到 `ultralytics`、`scipy`（已裝）。
- **模型 code + 權重（自帶）**：OpenSe3r 語意模型已 vendor 在 `<repo>/vendor/`
  （import 隔離已驗證）。權重放 `vendor/weights/`（gitignored，共 ~14GB）：
  `opense3r_fastsam_query_semconf.pth`、`vggt_1b.pt`、`fastsam_x.pt`、
  `tips_v2_oss_l14_{vision,text}.npz`。搬機 = repo + 這 5 個檔案。
  （設 `OPENSE3R_ROOT=/path/to/OpenSe3r` 環境變數可切回 live repo。）
- 一律從 **repo root** 以 **`PYTHONPATH=src`** 執行。

## 1. 離線：資料夾 → semantic SLAM

資料格式（ScanNet 式）：`<scene>/rgb/*.png`、`depth/*`（16-bit mm PNG 或 float32 公尺
TIFF，自動偵測）、`intrinsic.txt`（4×4，取左上 3×3）、可選 `gt_pose.txt`（TUM 式）。

```bash
# MapAnything 幾何（rgb+depth+intr，最準）
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m ma_slam_semantic.run \
    --geometry ma --data /path/to/scene --mode rgb+depth+intr \
    --submap_size 20 --out outputs/ma_sem/<scene> \
    [--gt /path/to/gt_pose.txt]          # 給了會做 ATE 報告 + 輸出對齊 GT-mesh 座標
    [--deploy]                            # 直接接 §4 的機器人成品匯出

# DA3 幾何（純 RGB，無深度感測器時）
    ... --backend da3 --mode rgb ...
```

主要旋鈕（詳見 `SemConfig`）：

| 旋鈕 | 預設 | 說明 |
|---|---|---|
| `--conf_gate_coef` | 1.0 | 幾何 validity gate（× submap 平均 conf）。**實務 inference 留 1.0**（擋不準的 geo）；跑 benchmark 報表用 **0.5**（覆蓋率↑，5-scene mean AP 20.8→25.3） |
| `--sem_chunk_size` / `chunk_size` | 4 | 每次 causal forward 的幀數，**純 VRAM 旋鈕**（結果不變）：4≈15GB 峰值、8≈26GB |
| `--kv_policy` | 離線 `none` / 串流 `h2o` | KV cache 保留策略。flat 成長 ~62MB/幀（1000 幀=61GB）；`h2o`（STAC：pinned frame0 + recent 8 + heavy-hitter 8）恆定 ~0.7GB |
| `--no_loop` | off | 關 loop closure（LC 開著 instance map 會自癒：s0231 213 vs 240 instances） |

輸出（`--out` 下）：`camera_poses.txt`、`combined_pcd.ply`、`run_stats.*`、`loops.txt`
（ma_slam 原有）+ `<scene>_instance_labels.npz`、`<scene>_instance_sem_queries.npz`、
`<scene>_query_instance_table.json`、`_voxels.ply`、`_boxes.ply`（語意）+
`alignment.json`（帶 `--gt` 時的 T_align，deploy 用）。

## 2. 即時串流：RealSense → semantic SLAM

與 [README.md](README.md) §2 完全相同的三種 transport（`zmq` 推薦 / `local` / `folder`），
**只多加 `--semantic`**。以 laptop→server 為例：

**① Server**（repo root）：
```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m ma_slam_stream.run_stream \
    --source zmq --bind 'tcp://*:5599' --mode rgb+depth+intr --depth_max 6 \
    --out outputs/maslam_live --submap_size 20 --viz web \
    --semantic                            # ← 語意層：viewer 出現 instance 上色 voxel/bbox
```
**② Client**（laptop，照原 README）：
```bash
cd src/ma_slam_stream
python client.py --transport zmq --addr tcp://SERVER_IP:5599 --mode rgb+depth+intr
```
**③** 開 server 印出的 viewer URL → **④** client 按 ENTER 開始；`Ctrl+C` 停止時
server 會把 §1 同一套語意成品寫進 `--out`。

串流注意：
- `--kv_policy` 預設 **h2o**（串流無界，flat cache 會吃光 VRAM）；`--h2o_subsample 0.25`
  已驗證 AP 持平且 colsum 成本近零。
- `--semantic` v0 是同步跑（每 submap ~0.6s/幀 GPU），throughput 由語意端主導；
  相機餵幀速度通常更慢所以無感。
- VRAM：MapAnything + 語意模型同卡，chunk 4 時峰值 ~30GB。

## 3. Eval（ScanNet 5-scene 基準）

```bash
# instance AP（SN200，唯一要跑的 inst 基準；protocol 一定帶這兩旗標）
python vendor/eval_instance_npz_scannet.py \
    --hdbscan_labels_npz <out>/<sc>_instance_labels.npz \
    --gt_ply       scannet_200/<sc>/gt_scannet200_rawid_<sc>.ply \
    --aggregation_json scannet_200/<sc>/<sc>.aggregation.json \
    --segs_json    scannet_200/<sc>/<sc>_vh_clean_2.0.010000.segs.json \
    --benchmark 200 --benchmark_void --segment_snap

# inst-sem（SN20+SN200，5 場景聚合；先廣播再 ovo eval——在 OpenSe3r repo 跑）
python query_instance_semantic.py --instance_labels_npz ... \
    --instance_sem_queries_npz ... --benchmark 20 --backbone tips-v2-l14 \
    --output_dir <out> --demo_name <sc>
python tools/ovo_style_semantic_eval.py --dir_pattern '<outs>/{sc}' \
    --backbone tips-v2-l14 --tasks instsem --benches sn20 sn200
```
一鍵 driver 範本：`outputs/ma_sem/scannet5*/run_all.sh`。
⚠️ eval 需要 `--gt` 跑出的 GT-mesh 對齊輸出；比較數字只能用同 protocol
（2026-07-12 後 = `--benchmark_void --segment_snap`，舊 summary 不可比）。

## 4. 機器人部署成品（deploy）

跑完 SLAM 後（或直接在 §1 加 `--deploy`）：

```bash
PYTHONPATH=src python -m ma_slam_semantic.deploy \
    --dir outputs/ma_sem/<scene> --name <scene> \
    [--max_points 2000000] [--levels coarse,medium,fine]
```

產出（`<dir>/deploy/`，格式同 OpenSe3r `outputs/deploy/` 範例）：

| 檔案 | 內容 |
|---|---|
| `robot_deploy_<sc>_camera_trajectory.json` | 每幀 c2w（timestamp/basename/tx..qw/4×4，ROS bridge 格式）|
| `robot_deploy_<sc>_instance_semantic_table.json` | **機器人主要讀這個**：每 instance 的 class/center/bbox/`filtered` |
| `..._instance_semantic_table_{coarse,medium,fine,all_levels}.json` | 多層 label set（~10/~29/~57 類，內建清單）|
| `robot_deploy_<sc>_{inst,rgb,sem}_color.ply` | per-point 雲（每頂點帶 `inst_id` 指回 table；`--max_points` 取樣）|
| `..._sem_color_<level>.ply` + legend PNG | 各層語意上色 |

座標系規則：**所有成品同一座標系**。機器人跑法（無 `--gt`）全在 ma_slam 世界座標；
benchmark 跑法（有 `--gt`）deploy 會讀 `alignment.json` 把 poses/pcd 一起轉進
GT-mesh 座標，與 instance 成品一致。

## 5. 其他工具

```bash
# Golden 測試（GT 幾何重現 OpenSe3r 005 run；改語意端後的正確性 harness）
PYTHONPATH=src python -m ma_slam_semantic.run --geometry gt \
    --data .../scenenn/oni/005 --demo_type scenenn_gt --out outputs/ma_sem/005_golden

# 效能剖析（各模組時間 + VRAM + KV 成長）
PYTHONPATH=src python -m ma_slam_semantic.profile_sem --max_frames 64
```

參考數字（5-scene，SN200 inst mean AP/AP50/AP25）：MA(rgb+d+i) gate1.0
20.8/43.2/67.1、gate0.5 25.3/50.5/73.9；DA3(rgb) gate1.0 17.9/41.9/59.5、
gate0.5 21.8/47.5/71.3。instsem 聚合 SN20 mIoU ~38-40（四配置持平）。

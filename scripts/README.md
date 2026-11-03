# scripts/

實驗用腳本。**正式流程的兩個進入點留在 repo 根目錄**：

```bash
./run_xplace.sh <design>            # GP + pin-access pre-pass（td9 配方）
./run_innovus_routing.sh <design>   # 上傳 DEF、遠端繞線 + DRC
```

這裡的腳本都是為了掃參數／比對設定而寫的一次性流程，彼此不互相依賴。
除 `analyze.sh` / `exp_def.sh` 外都用絕對路徑 `X`（預設為腳本所在的 Xplace 目錄），
所以從任何目錄執行都可以；習慣上仍從 repo 根目錄呼叫：

```bash
./scripts/run_recipe.sh seed 0
```

| 腳本 | 用途 |
|---|---|
| `run_recipe.sh` | 配方 q97t85p02：GP(td0.85, q0.97, inflate) → PA 2% → 上傳 |
| `run_papath.sh` | 取回 Innovus placed DEF → GP(td0.7 tuned) → PA → LG+DP（L 4x） |
| `run_congfix.sh` | GP(td0.7 + route force, adaptive cong) → LG+DP → PA 5% → 上傳 |
| `run_sc.sh` | scarcity-weighted GP + PA + 上傳（`TD= Q= PA= ALPHA=`） |
| `run_gpsweep.sh` | 只跑 GP，回報 congestion / HPWL trade-off |
| `run_tail.sh` | 從既有 GP def 接 LG+DP → PA 5% → 上傳 |
| `run_dppa.sh` | 既有 GP → LG+DP → PA → 上傳（L 4x） |
| `run_dp2.sh` | PA 之後再跑一次 Xplace LG+DP（尊重 PA 的 FIXED）→ 上傳 |
| `run_pa.sh` | 只跑 PA pre-pass（吃既有 DP def）→ 上傳 |
| `run_xplace_L4x.sh` / `run_xplace_L4x_tag.sh` | L library 4x 的 tuned GP + PA |
| `run_xplace_tuned.sh` | 舊的完整調校配置，AP json 換成 R |
| `run_xplace_u50.sh` / `run_xplace_u60.sh` / `run_xplace_util.sh` | utilization 0.5 / 0.6 / 可調 的實驗 |
| `exp_def.sh` | 把 `exp/phase2_def/<def>` 送上 CAD 機器跑繞線，收回 `exp/phase2_rpt/` |
| `analyze.sh` | 對一個 design 的 ft00000~ft00090 系列 DEF 連續呼叫 `exp_def.sh` |

## 共用輸入

- `data/pin_scarcity/*.txt` — LEF 推導的 pin scarcity 報告（pre-pass 的 `--reports`）
- `xplace_work/*.lef`, `xplace_work/<design>_unplaced.def` — 技術檔與未擺放 DEF
- `xplace_work/contest*.json` — 各腳本執行時自行產生的 Xplace 設定

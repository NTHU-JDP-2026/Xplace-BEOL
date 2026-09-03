# Pin-Access Pre-Pass：把 pin-access 關鍵 cell 釘死在好位置

> 目標：在 global placement **收斂之後、legalization 之前**，把 pin access 最困難的 cell 重新安置到「不壓 M3 power stripe、彼此不相鄰」的合法位置，並把它們標成 `FIXED`，讓後續的 legalizer / detailed placer 不能再把它們擺回去。
>
> 與 [`cell_pin_access.md`](cell_pin_access.md) 的關係：那份文件是在 GP 目標函數裡加**可微分的軟約束**（$L_{\text{access}}$），這份是 GP 之後的**硬約束**。兩者互補——軟約束負責讓 GP 生出的解本來就不差，硬約束負責讓好不容易調好的位置不被 DP 抹掉。

---

## 1. 動機

軟性的力（density blockage、DRV gradient、routing force）都有同一個結構性弱點：

1. GP 階段給的力只是**偏好**，不是保證；
2. legalization / detailed placement 只看 **displacement 與 HPWL**，不看 pin access，會把 cell 擠回原本的壞位置。

實測（design `seed`，17,334 cells，util 40%）也顯示：現有的 M3 stripe density blockage **已經把「避開 stripe」做得差不多了**——GP 之後只有 8.0% 的 cell、5.1% 的 pin-access 關鍵 cell 還壓在 stripe 上（stripe 金屬覆蓋 x 方向 13.9%）。但

> **74% 的關鍵 cell 在 4 sites / ±1 row 內還有另一顆關鍵 cell，是隨機分佈期望值（19%）的 3.9 倍。**

也就是說，真正沒被處理的是**關鍵 cell 彼此聚集**（它們常屬於同一個邏輯群組，GP 自然把它們拉在一起），而不是壓 stripe。本方法就是針對這一點。

---

## 2. 問題形式化

### 2.1 符號

| 符號 | 意義 |
|---|---|
| $\mathcal{M}$ | 所有 movable instance |
| $m(i)$ | instance $i$ 的 master（cell type） |
| $s(\cdot)$ | master 的 pin scarcity 分數（`*_1x_pin_scarcity_report.txt` 的 `score_flip_max`） |
| $w_i, a_i$ | instance $i$ 的寬度、面積 |
| $(x_i^{\text{gp}}, y_i^{\text{gp}})$ | GP 給的位置 |
| $w_s,\ h_r$ | site width、row height |
| $x_0$ | row 起點 x（site grid 原點） |
| $\mathcal{B}=\{[b_j^-, b_j^+]\}$ | M3 垂直 stripe 的 x 區間（由 DEF `SPECIALNETS` 取得） |
| $\delta$ | stripe margin（sites） |
| $\sigma_x,\ \sigma_r$ | 關鍵 cell 之間的最小水平間距（sites）與垂直間距（rows） |
| $\Delta_s,\ \Delta_r$ | 搜尋窗：最大水平位移（sites）、最大垂直位移（rows） |
| $\beta$ | 面積預算（凍結面積 / 總 movable 面積） |

### 2.2 候選集合

$$
\mathcal{C} = \{\, i \in \mathcal{M} \;:\; s(m(i)) \ge \mu_s + k\sigma_s \,\}
$$

其中 $\mu_s, \sigma_s$ 是所有 master 的 scarcity 平均與標準差，$k$ 由 `--score_sigma` 指定（預設 1.0）。候選再依**選擇策略**（§3）排序，依序取用直到面積預算耗盡：

$$
\mathcal{S} = \text{prefix}\Big(\text{sort}(\mathcal{C}),\ \sum_{i\in\mathcal{S}} a_i \le \beta \sum_{i \in \mathcal{M}} a_i \Big)
$$

### 2.3 合法位置與硬約束

候選位置離散化在 site/row 格點上：$x = x_0 + n w_s$、$y = y_0 + r h_r$。對 $i \in \mathcal{S}$，位置 $(x, r)$ 可行 iff

**(C1) 不壓 stripe**：$\forall j$，$[x, x+w_i] \cap [b_j^- - \delta w_s,\ b_j^+ + \delta w_s] = \emptyset$

**(C2) 與已安置的關鍵 cell 保持距離**：$\forall i' \in \mathcal{S}$ 已安置且 $|r - r_{i'}| \le \sigma_r$，
$$[x - \sigma_x w_s,\ x + w_i + \sigma_x w_s] \cap [x_{i'},\ x_{i'} + w_{i'}] = \emptyset$$

**(C3) 在 core 內**：$x \ge x_0$ 且 $x + w_i \le x_0 + N_{\text{site}} w_s$

**(C4) orientation 與 row 相容**：row 為 `N` 時 cell 用 `N`/`FN`，row 為 `FS` 時用 `FS`/`S`。跨 row parity 移動時作 MX 鏡像（`N`↔`FS`、`FN`↔`S`），**pin 的 x offset 不變**，所以 (C1) 的判定在鏡像後仍然成立。

### 2.4 指派

依排序逐一（貪心）指派，每顆取**位移最小的可行位置**：

$$
(x_i^*, r_i^*) = \arg\min_{(x,r)\ \text{可行}} \; (x - \hat{x}_i)^2 + (r - \hat{r}_i)^2 h_r^2
$$

$(\hat{x}_i, \hat{r}_i)$ 是 GP 位置 snap 到最近 site/row 的結果，搜尋範圍限制在 $|n - \hat n| \le \Delta_s$、$|r - \hat r| \le \Delta_r$。找不到可行位置就**放棄該顆**（不凍結，留給 legalizer 處理），這比硬塞到很遠的地方好。

因為是貪心，先安置的 cell 佔位、後面的讓開；分數高的先放，所以最關鍵的 cell 拿到最小位移。

### 2.5 凍結

被安置的 cell 在輸出 DEF 中寫成 `+ FIXED ( x y ) ORIENT`。這一步是整套方法的關鍵，因為：

- **Xplace**：`FIXED` 的 instance 會被 parser 歸到 fixed node 區段，GP 自動把它們算進 `init_density_map`（[`src/initializer.py`](../src/initializer.py) 的 `get_init_density_map`），它們的 pin 仍正確計入 wirelength（`conn_fix_node_pos`）；greedy / abacus legalizer 依 `node_id >= num_movable_nodes` 判定 fixed，所以完全不會動它們。**不需要改任何 C++。**
- **Innovus**：`defIn -components` + `refinePlace` 尊重 `FIXED`。實測 seed paA：繞完線後的 `seed_final.def` 裡 1,903 顆凍結 cell **位置與狀態零變動**。
- Xplace 的 DEF writer 會保留 `FIXED` 狀態（[`cpp_to_py/common/io/file_lefdef_db.cpp:421`](../cpp_to_py/common/io/file_lefdef_db.cpp#L421)），所以可以再餵回 Xplace 做第二輪。

---

## 3. 候選排序

候選一律**依 pin scarcity 分數由高到低**排序，依序取用直到面積預算耗盡。分數來自 LEF/AP 推導的 `score_flip_max`，屬於**先驗資訊**：不需要先繞線，因此可以放在正常的 P&R 流程裡。

> 曾經試過用「上一輪 route 的 M2-short 肇事名單」來排序（同時試過各 master 輪流取與整包凍兩種花預算的方式）。它在部分設計上表現更好，但那等於把繞線結果回饋進 placement，屬於 ECO 迴圈，**本專案不允許**，因此已從工具與流程中移除。相關實驗數據仍保留在 §7 的歷史紀錄中作為對照。

真正影響結果的兩個旋鈕都是幾何參數：

| 旋鈕 | 意義 | 實測 |
|---|---|---|
| `--area_budget` | 凍結面積佔 movable 總面積的上限 | 5% 是低成本操作點；15% 在 pin-access 型違規為主的設計（seed / aes / camellia）能再往下壓 M2 short，但 misty 會反效果 |
| `--min_sep_sites` | 關鍵 cell 之間的最小水平間距 | **8 sites 全面優於 12 sites**：位移更小、WL 更便宜、M2 short 不劣於 12 |

## 4. 資料流

```
xplace_work/<design>_unplaced.def ──┐（PDN：M3 SPECIALNETS）
                                    │
./run_xplace.sh <design>            │
   └─ result/<design>/placement_<design>_gp.def ─┐
                                                  │
*_1x_pin_scarcity_report.txt ────────┐            │
xplace_work/{L,SL}_modified.lef ─────┤            │
（可選）<design>.m2_short.csv ────────┤            │
                                     ▼            ▼
                        tool/pin_access_prepass.py
                                     │
                                     ├─ <design>_paX.def   （關鍵 cell = FIXED）
                                     └─ <design>_paX.json  （凍結清單、位移統計）
                                     │
                    ┌────────────────┴─────────────────┐
                    ▼                                  ▼
        Innovus hybrid flow                 （可選）再餵回 Xplace
        （defIn -components →                 --custom_json 指向 paX.def
          refinePlace → route → DRC）          --gp_init_method given
```

---

## 5. 執行方式

### 5.0 一鍵流程（`run_xplace.sh` 已整合 pre-pass）

```bash
./run_xplace.sh <design>              # GP + pre-pass（scarcity, 10% 面積, sep 8, 位移上限 24/2）
./run_innovus_routing.sh <design>     # 繞線 + DRC
```

`run_xplace.sh` 的第二段會自動執行 pre-pass，產出：

| 檔案 | 內容 |
|---|---|
| `result/<design>/placement_<design>_gp_raw.def` | 純 GP 解（pre-pass 的輸入） |
| `result/<design>/placement_<design>_pa.def` | pre-pass 結果 |
| `result/<design>/placement_<design>_pa.json` | 凍結清單、位移統計 |
| `result/<design>/placement_<design>_gp.def` | 等於 `_pa.def`，讓 `run_innovus_routing.sh` 不用改就吃到 pre-pass 結果 |

pre-pass 在 docker 容器內執行（`result/` 由容器內的 root 寫入；工具只用 python 標準庫），寫檔後會自動跑合法性自檢，**任何一項不為 0 就中止並保留原本的 GP 解**。

環境變數（預設值 = 六個設計 DRC 全降的設定：10% 面積、8 sites 間距、位移上限 24 sites/2 rows）：

| 變數 | 預設 | 說明 |
|---|---|---|
| `PA_PREPASS` | `on` | `on` = GP + pre-pass / `off` = 只跑 GP |
| `PA_AREA_BUDGET` | `0.10` | 凍結面積上限 |
| `PA_MIN_SEP_SITES` / `PA_MIN_SEP_ROWS` | `8` / `1` | 關鍵 cell 互斥距離 |
| `PA_MAX_SHIFT_SITES` / `PA_MAX_SHIFT_ROWS` | `24` / `2` | 位移上限（**最關鍵的旋鈕**，且最佳值在中間，見 §7.1） |
| `PA_SCORE_SIGMA` / `PA_STRIPE_MARGIN` | `1.0` / `1` | 候選門檻、stripe 淨空 |
| `PA_LEFS` / `PA_REPORTS` / `DOCKER_CTR` | 見腳本 | 路徑與容器名 |

例：改用大預算跑 aes、或完全關掉 pre-pass

```bash
PA_AREA_BUDGET=0.15 ./run_xplace.sh aes
PA_PREPASS=off ./run_xplace.sh seed
```

---

### 5.1 Step 1：跑 global placement

```bash
./run_xplace.sh seed
```

一鍵流程會直接做完 GP + pre-pass；純 GP 解留在 `result/seed/placement_seed_gp_raw.def`，下面的手動步驟就以它為輸入。只想要 GP 的話用 `PA_PREPASS=off ./run_xplace.sh seed`。

### 5.2 Step 2：手動跑 pre-pass（想自己掃參數時）

```bash
python3 tool/pin_access_prepass.py \
  --gp_def   result/seed/placement_seed_gp_raw.def \
  --pdn_def  xplace_work/seed_unplaced.def \
  --lefs     xplace_work/L_modified.lef,xplace_work/SL_modified.lef \
  --reports  L_1x_pin_scarcity_report.txt,SL_1x_pin_scarcity_report.txt,R_1x_pin_scarcity_report.txt \
  --area_budget 0.05 --min_sep_sites 12 --min_sep_rows 1 \
  --max_shift_sites 48 --max_shift_rows 4 \
  --out_def  /tmp/seed_sweep.def \
  --out_json /tmp/seed_sweep.json
```

輸出範例（務必看這幾個數字）：

```
[prepass] M3 vertical stripes: 42 bands, width=3744, pitch~5184 (DBU)
[prepass] movable=17334 candidates=2150 chosen=939 area=1447.0/28965.9 um2 (5.00% ...)
[prepass] frozen=929 (kept=0 moved=929) skipped=10 | of chosen, 74 were on a stripe
[prepass] displacement (um): p50=0.502 p90=3.194 p99=8.000 max=10.551
```

- `skipped` 太多（>15%）表示 `--min_sep_sites` 對這個設計太緊，或搜尋窗太小。
- `displacement p90` 超過 ~5um 通常代表 WL 會付出 >3% 的代價。

掃參數時只要改這三個旋鈕（其餘固定）：

```bash
python3 tool/pin_access_prepass.py ... \
  --score_sigma 1.0 --area_budget 0.10 --min_sep_sites 8
```

### 5.3 Step 3：合法性自檢（工具已內建，這裡是手動版）

`pin_access_prepass.py` 寫完 DEF 後會自動重讀並檢查，印出

```
[prepass] self-check: off-grid=0 orient-mismatch=0 on-stripe=0 same-row-overlap=0
```

**四個數字都必須是 0**，否則工具以 exit code 2 結束（`--skip_check` 可略過，不建議）。要獨立檢查既有的 DEF 可以用：

```bash
python3 - << 'EOF'
import sys, bisect; sys.path.insert(0,'tool')
from pin_access_prepass import (parse_def_header, parse_m3_stripes,
                                parse_components, parse_lef_widths)
D='result/seed/placement_seed_pa.def'   # 想檢查的 DEF
sizes=parse_lef_widths(['xplace_work/L_modified.lef','xplace_work/SL_modified.lef'])
bands=parse_m3_stripes('xplace_work/seed_unplaced.def'); lo=[b[0] for b in bands]
units,die,rows=parse_def_header(D); _,_,_,comps=parse_components(D)
site_w=rows[0]['step']; x0=min(r['x0'] for r in rows)
orient={r['y']:r['orient'] for r in rows}
fx=[c for c in comps if c.get('status')=='FIXED']
bad=ori=st=0; byrow={}
for c in fx:
    if (c['x']-x0)%site_w: bad+=1
    ro=orient.get(c['y'])
    if ro is None: bad+=1; continue
    if not ((ro=='N' and c['orient'] in ('N','FN')) or (ro=='FS' and c['orient'] in ('FS','S'))): ori+=1
    w=sizes[c['master']][0]*units; a,b=c['x'],c['x']+w
    i=bisect.bisect_right(lo,b)-1
    for j in (i,i-1):
        if 0<=j<len(bands) and a<bands[j][1] and b>bands[j][0]: st+=1; break
    byrow.setdefault(c['y'],[]).append((a,b))
ov=sum(1 for y,l in byrow.items() for p,q in zip(sorted(l),sorted(l)[1:]) if p[1]>q[0])
print(f"fixed={len(fx)} off-grid={bad} orient-mismatch={ori} on-stripe={st} overlap={ov}")
EOF
```

（不為 0 時 Innovus 的 `checkPlace` 會報錯，`refinePlace` 行為未定義。）

### 5.4 Step 4：送去繞線

一鍵流程已經把結果放在 `placement_<design>_gp.def`，直接：

```bash
./run_innovus_routing.sh seed
```

手動產生的 DEF 要先覆蓋過去（`result/` 由容器內 root 擁有，用 docker 複製）：

```bash
docker exec compassionate_pascal bash -c \
  "cd /workspace && cp <your>.def result/seed/placement_seed_gp.def"
./run_innovus_routing.sh seed
```

`run_innovus_routing.sh` 固定送 `result/<design>/placement_<design>_gp.def`，並把 `<design>.drc.rpt`（其實是 `verify_drc` 的 summary）與 `<design>_summary.rpt` 收回 `exp/`。要跑多組時務必**序列化**：遠端 flow 共用 `exp/exp.def`，兩個 run 併行時若前一個還沒把檔案複製進自己的 work dir 就會被覆蓋。

判讀報告：

```bash
grep "Verification Complete" exp/seed.drc.rpt          # DRC 總數
grep -A10 "Violation Summary" exp/seed.drc.rpt         # 分層/分類
grep -m1 "Total wire length" exp/seed_summary.rpt      # wirelength
```

**看 M2 Short，不要只看總數**：實測顯示 DRC 總數對任何 placement 擾動都很敏感（無關的擾動也能讓 34 → 23），而 M2 Short 才是 pin access 的機制指標。遠端 flow 另外會產生 `save_*/<design>.m2_short.csv`（哪些 cell type 捲進 M2 short），那是最有診斷價值的檔案。

### 5.5 （可選）Step 5：凍結後再跑一輪 Xplace GP

```bash
# contest_paX.json 的 "def" 指向 pre-pass 輸出
docker exec compassionate_pascal bash -c \
 "cd /workspace && python3 main.py --custom_json xplace_work/contest_paA.json \
    --load_from_raw True --gp_init_method given ...（其餘旗標同 run_xplace.sh）"
```

`--gp_init_method given`（[`main.py:32`](../main.py#L32)）讓 GP 沿用 DEF 位置暖啟動，而不是預設的 `randn_center` 重新初始化。

> **實測結論：這一步目前是負面的**。seed 上不跑 GP2 = 7 viols；暖啟動 GP2 = 11；冷啟動 GP2 = 25。Xplace 的 eplace-style GP 會把沒凍結的 cell 全部重新散開（HPWL 與迭代數幾乎不受初始位置影響），無法當作局部微調。要讓 GP2 有用，必須加位移約束（anchor pseudo-net 拉回 legalized 位置、或對位移 clamp），否則建議直接 GP → pre-pass → legalize。

---

## 6. 參數表

| 旗標 | 預設 | 說明 |
|---|---|---|
| `--gp_def` | 必填 | GP 輸出的 DEF |
| `--pdn_def` | 必填 | 帶 `SPECIALNETS` 的 DEF（通常是 `*_unplaced.def`） |
| `--lefs` | 必填 | cell LEF（取 master 寬高） |
| `--reports` | 必填 | pin scarcity 報告，逗號分隔 |
| `--score_sigma` | 1.0 | 候選門檻 $\mu_s + k\sigma_s$ |
| `--area_budget` | 0.05 | 凍結面積上限（佔 movable 總面積） |
| `--stripe_margin_sites` | 1.0 | stripe 兩側額外淨空 |
| `--min_sep_sites` | 4.0 | 關鍵 cell 之間最小水平間距（`run_xplace.sh` 傳 8） |
| `--min_sep_rows` | 1 | 垂直方向影響範圍（±N rows） |
| `--max_shift_sites` | 24 | 水平搜尋窗 |
| `--max_shift_rows` | 3 | 垂直搜尋窗 |
| `--ignore_stripes` | off | 對照組用：不避 stripe |
| `--random_control` | off | 對照組用：改選隨機的非關鍵 cell |
| `--skip_check` | off | 略過寫檔後的合法性自檢（不建議） |
| `--out_def` / `--out_json` | 必填 / 選填 | 輸出 |

---

## 7. 實驗結果

環境：ASAP7 4x、util 40%、Innovus hybrid flow（`defIn -components` → `refinePlace` → CTS → `routeDesign` → `verify_drc`）。**該 flow 對同一份輸入 DEF 是位元級決定性的**（重跑 baseline 得到逐層相同的結果與到小數點相同的 wirelength），所以量測噪音為零。

### 7.1 主結果

設定（= [`run_xplace.sh`](../run_xplace.sh) 的預設值）：**scarcity 排序、面積預算 10%、關鍵 cell 間距 8 sites / ±1 row、位移上限 24 sites / 2 rows**。

| design | #cells | #nets | **DRC** baseline → pre-pass | 降幅 | **Wirelength (um)** baseline → pre-pass | ΔWL | **Avg WL/net (um)** baseline → pre-pass |
|---|---|---|---|---|---|---|---|
| seed | 17,334 | 17,647 | 34 → **12** | **−64.7%** | 282,329.728 → 284,701.488 | +0.84% | 15.8612 → 15.9945 |
| sha256 | 9,708 | 9,799 | 22 → **13** | **−40.9%** | 216,668.244 → 218,108.296 | +0.66% | 22.0214 → 22.1677 |
| misty | 10,558 | 10,693 | 40 → **24** | **−40.0%** | 204,127.428 → 210,068.420 | +2.91% | 19.0151 → 19.5686 |
| cast | 24,450 | 24,683 | 45 → **24** | **−46.7%** | 606,628.240 → 609,342.972 | +0.45% | 24.5569 → 24.6668 |
| camellia | 10,101 | 10,577 | 33 → **24** | **−27.3%** | 207,143.996 → 210,059.072 | +1.41% | 19.5844 → 19.8600 |
| aes | 263,618 | 264,155 | 519 → **294** | **−43.4%** | 4,863,389.488 → 4,915,201.404 | +1.07% | 18.1978 → 18.3917 |
| **合計** | | | **693 → 391** | **−43.6%** | | +0.45 ~ +2.91% | |

- **六個設計的 DRC 全部下降（−27.3% ~ −64.7%），WL 代價 +0.45% ~ +2.91%。** 機制指標 M2 Short 為 9→3、16→5、19→16、15→6、10→15、275→148（五降一增，camellia 反增但總量仍降）。
- **DRC** = Innovus `verify_drc` 的違規總數（全部層別），取自 `<design>.drc.rpt` 的 `Verification Complete : N Viols.`；**WL / Avg WL per net** 取自 `<design>_summary.rpt`。pre-pass 不改網表，net 數不變，ΔWL 與 Δ(avg WL) 相同。
- baseline 為專案既有的 ISPD2023 流程結果（`Ours (M2 OBS, Dynamic Dimension FCN)`）：seed 34 / sha256 22 / misty 40 / cast 45 / camellia 33 / aes 519。`cast` / `aes` 的 baseline gp.def 與本表輸入不同（我方較新），其餘四個為同一份 GP 解的乾淨比較。
- 流程對同一份輸入 DEF 是位元級決定性的（重跑 baseline 得到逐層相同的違規與到小數點相同的 WL），故表中差異不含量測噪音；但 DRC 總數對任何 placement 擾動都敏感（§7.2），低違規設計（camellia 33）的效應大小已接近該敏感度。

參數掃描（六個設計的 DRC 總和，baseline = 693；括號為通過的設計數）：

| 面積預算 | 位移上限 48 sites/4 rows | 24/2 | 12/1 |
|---|---|---|---|
| 3% | 600（5/6） | — | — |
| 5% | 495（5/6） | — | — |
| 10% | 379（5/6，camellia ✗） | **391（6/6）** | 423（5/6，misty ✗） |
| 15% | 283（4/6，camellia + misty ✗） | — | — |

- **位移上限是決定性旋鈕**，而且最佳值在中間：48/4 太寬（camellia 因大位移失效）、12/1 太緊（misty 因大位移候選被放棄而失效）、**24/2 同時避開兩者**。
- **總和最低的不是最佳設定**：15% 預算的總和 283 最低，但它在 camellia（54）與 misty（42）比 baseline 差。
- **間距 8 sites 優於 12 sites**：六個設計 WL 一律更便宜，因為可行位置變多、貪心指派的位移反而更小。
- **預算對設計的最佳值不同**：cast 在 3% 就有 22（且 WL −0.02%）、sha256 在 3% 反而輸 baseline 1 個（干預不足）、seed 在 15% 有最低的 7。per-design 選檔位可把總和壓到 187（−73%），但需要先驗判準（見 §8）。

### 7.2 對照實驗（seed，證明機制）

> 這兩節是較早的實驗，使用當時的參數（`paB` = 5% / sep 12、`paA` = 15% / sep 8，兩者都沒有位移上限）。結論不受影響，保留作為機制證據與負面結果紀錄。

| 設定 | DRC | M2 short | wirelength (um) |
|---|---|---|---|
| baseline | 34 | 9 | 282,329.728 |
| 重跑 baseline（決定性檢查） | 34 | 9 | 282,329.728（逐位元相同） |
| **ctrlSnap**：同一批關鍵 cell，只 snap + FIXED，不避 stripe 不散開 | 23 | 11 | 283,260.752 (+0.33%) |
| **ctrlRand**：隨機非關鍵 cell，同樣的散開規則 | 24 | 8 | 285,115.100 (+0.99%) |
| paB | 20 | 5 | 284,650.492 (+0.82%) |
| paA | 7 | 2 | 290,209.392 (+2.79%) |

兩個對照組都複製不出效果 → 效果來自「**選對 cell**」×「**重新定位**」的組合，不是「把 cell 釘住」或「隨便散開」。

機制證據：paA 之後，**所有被凍結的 master 都從 `m2_short.csv` 消失**，殘留的 short 正好是門檻外沒選到的 master（NAND5xp2 rank 221、OA211x2 rank 158、INVx1 rank 409）。

### 7.3 GP2（凍結後再跑 global placement）

| seed | DRC | M2 short | wirelength (um) |
|---|---|---|---|
| paA（不跑 GP2） | **7** | **2** | 290,209.392 (+2.79%) |
| paA + GP2 暖啟動（`--gp_init_method given`） | 11 | 3 | 289,407.376 (+2.51%) |
| paA + GP2 冷啟動（預設 `randn_center`） | 25 | 11 | 287,753.140 (+1.92%) |

---

## 8. 已知限制

1. **選擇器只到 master 粒度**。scarcity 分數只能說「這型 cell 危險」，不能指出「這顆 instance 危險」，所以預算會花在同型但其實安全的 instance 上。→ 下一步應改用 **instance 級的風險預測**（DRV predictor 的 per-bin 輸出 × cell 的 scarcity）。
2. **參數的最佳值仍是設計相依的**。預設值（10% / sep 8 / 位移上限 24-2）在六個設計上都改善了 DRC，但每個設計各自的最佳點不同（cast 在 3% 就有 22、seed 在 15% 有 7、camellia 在 10%/12-1 有 21）；per-design 選檔位可把六個設計的總和從 391 再壓到 187。要再往上推，需要一個**先驗**的判準來 per-design 選參數——DRV predictor 在 GP 階段預測的 M2 熱點密度是最合適的候選（不需要繞線結果，不屬於 ECO）。
3. **GP2 目前無用**（見 §7.3），需要位移受限的 GP 才可能把兩階段串成迴圈。
4. **只在 40% util 的 ASAP7 設計上驗證過**。高 util 下可行位置變少，`skipped` 會急遽上升，貪心指派可能要換成 min-cost matching。
5. **flip/orientation 沒有被利用**。scarcity 報告的排序鍵是 `score_flip_max`（最壞朝向），但本方法只在跨 row 時被動鏡像，沒有主動挑更好的朝向。Xplace 內建的 `run_dp_route_opt`（PA-Refine，[`src/detail_placement.py`](../src/detail_placement.py) 的 `dp_route_opt`）可作為互補的最後一步。

---

## 9. 檔案

| 路徑 | 內容 |
|---|---|
| [`run_xplace.sh`](../run_xplace.sh) | GP + pre-pass 一鍵流程（預設 = §7.1 的設定：scarcity / 10% / sep 8 / 位移上限 24-2） |
| [`tool/pin_access_prepass.py`](../tool/pin_access_prepass.py) | 本方法的完整實作 |
| `exp/pa_pilot/<design>_paJ.json` | 六個設計最終版的凍結清單（instance 名稱、原始/最終座標、位移、是否原本壓 stripe） |
| `exp/<design>_paJ.drc.rpt`、`exp/<design>_paJ_summary.rpt` | 最終版的 DRC 與 wirelength 報告；`base_r1` / `base_rep` / `ctrlSnap` / `ctrlRand` / `camellia_base` 為 §7.2 對照實驗與 matched baseline |
| `main.py --gp_init_method` | GP 暖啟動旗標（預設行為不變） |
| 遠端 `designs/<design>/save_*/` | Innovus 各次 run 的完整報告，含 `m2_short.csv` |

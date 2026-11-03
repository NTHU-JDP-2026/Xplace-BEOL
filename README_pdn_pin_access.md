# PDN-aware placement for pin access: methodology

This note describes what this branch (`pin-access`) adds to Xplace and to the
Innovus flow (now in `i3dPDK_NS3K`), why each part is there, and what it measured on
the 13 designs of the i3dPDK_NS3K benchmark set. It is written as the
methodology and results sections of a paper; reproduction notes are at the end.

**In one paragraph.** On a 3 nm-class library with unidirectional M2-M4 routing,
cells next to the M3 power stripes lose pin access: the stripes and the via
stacks that tie them to the M0 rails block the M2/M3 tracks a pin needs, and
the router leaves signal-to-PG and signal-to-signal violations there. We attack
this at two levels. **Global placement (GP):** a PDN-aware density target keeps
the M3-stripe regions sparse (density 0.3 instead of 0.85), and the flow keeps
that placement intact up to routing. Innovus' pre-CTS optimization re-places
the cells and undid it in our earlier runs. **Detailed placement (DP):** just
before routing, a learned per-pin model chooses the x-mirror of every cell near
the PDN, which removes 89% of the signal-to-PG violations while flipping 0.4% of
the cells. Together, against Innovus placement through the same downstream flow,
routed DRC drops by 47% and wirelength by 7% (13 designs).

---

## 1. Setting

| | |
|---|---|
| Library | i3dPDK_NS3K, 5-track cells (`NS3K_5T`), LEFs scaled 4x (`lef_x4/`); DEF 2000 dbu/um |
| Geometry (4x) | site 352 dbu (0.176 um), row 960 dbu, M2 track pitch 192 dbu |
| Cells | signal pins are full-height M1 bars; VDD/VSS rails on M0 |
| Routing | M2-M4 unidirectional, on-grid (`ROUTE_UNIDIR=1`: `RIGHTWAYONGRIDONLY`) |
| PDN (stripe) | M0 rails; VDD and VSS M3 stripes, 384 dbu wide, VSS 880 dbu (2.5 sites) right of VDD, a pair every 45-46 sites; tied to the rails by VIA12-VIA45 stacks; upper straps M5-M9 |
| Flow | Innovus 21.13, "fast" flow (`pnr/flow_fast.tcl`): floorplan, macros and PDN, placement, CTS, routing; timing is not optimized after placement |
| Designs | gcd, serv, riscv32i, aes, dynamic_node, tinyRocket, bp_fe_top, ethmac, bp_be_top, jpeg, swerv, swerv_wrapper, ariane133 |
| Metrics | `verify_drc` violations after `routeDesign` (total, and "signal-PG": markers between a regular wire and a special wire), routed wirelength |

Two failure modes are visible in the routed DRC:

- **signal-to-PG**: the M2 access wire or via of a pin ends too close to a
  stripe or to a pad of a rail via stack (spacing, short). It depends on where
  the pin sits relative to the PDN, so it can be predicted from the placement.
- **signal-to-signal**: competition for the few tracks left near a stripe.
  These are most of the violations, and they depend on routing, not only on
  the pin (our per-pin model cannot predict them; average precision 0.04).

## 2. Flow

```
Innovus (Innovus host)                      GPU host (Xplace docker)
------------------------------------------  --------------------------------------------
floorplan, macro placement, PDN
defOut <top>.in.def  ---------------------> Xplace GP with the PDN-aware density (Sec. 4.1)
defIn  <top>.gp.def  <---------------------   (+ optional keep-out bands, Sec. 4.3)
refinePlace  (legalization, DP)
[optDesign -preCTS]  skipped (Sec. 4.2)
refinePlace; ccopt_design (CTS)
defOut <top>.preroute.def ----------------> pinaccess/flip.py: model-driven x-mirror (Sec. 5)
apply <top>.flip (dbSet orient) <----------
routeDesign; verify_drc
```

Where things live: the experiments, the PDK and the flow are all in the
`i3dPDK_NS3K` checkout. Its `innovus_enablement/` holds:

- **PDK settings:** `pdk_config.tcl/.sh`, the 4x LEFs in `lef_x4/`, and the
  model card `pinaccess/pa_pg_v1.pt`.
- **The flow:** synthesis (`syn/`), the Innovus flow (`pnr/run_pnr.sh`,
  `flow.tcl`, `flow_fast.tcl`), the two Innovus-side hand-off points
  (`pnr/xplace_place.tcl`, `pnr/pa_flip.tcl`), and the GPU-host driver
  `xplace/run_xplace_flow.sh`.
- **Results:** every run, in `pnr/out/`.

The flow started as a copy of `i3dPDK_A16K`'s and now runs without it. The
results in Sec. 7 were produced while NS3K was still run through A16K's
flow; the standalone copy reproduces them exactly (identical DRC and
wirelength on the test runs).

## 3. Pin-access risk model

Both levels use one learned model. It answers a single question: given where a
cell is placed and in which orientation, what is the probability that each of
its signal pins ends up in a signal-to-PG violation after routing?

### 3.1 Samples and labels

One sample per signal pin that has a PDN shape within its window (below), from
52 routed runs (13 designs x 4 setups: Innovus GP and two Xplace settings with
M3 stripes, Innovus GP with M5 straps + via pillars). That gives about 2.0 M
pins, of which about 12.5 k are positive. A pin is labelled 1 when a
`verify_drc` signal-to-PG marker lies on its access region (`extract_grid.py`,
`extract_extra.py`).

### 3.2 Inputs (318)

- **Access (6):** how many M2 tracks a via can land on the pin; which of the
  row's tracks 1-4 those are, after the cell's orientation; pin width.
  Normalized with the training mean and standard deviation.
- **PDN occupancy (234):** a window of 13 columns x 6 levels around the pin,
  for each of M2, M3 and M4.
  - Columns are half a site (176 dbu, one M3 pitch) wide, centred on the pin,
    from -3 to +3 sites.
  - Levels are the 6 track heights of the pin's row (rail tracks included).
  - Each entry is the fraction of that bin covered by VDD/VSS shapes on that
    layer: stripe wires plus the pads of the rail via stacks (VIA12/VIA23 on
    M2, VIA23/VIA34 on M3, VIA34/VIA45 on M4).
- **Neighbour-pin grid (78):** the same window for the other cells' pins. It is
  kept in the input layout but set to zero (`--no_pin_grid`); the model does
  not look at the neighbours.

No PDN type is an input. The PDN is described only by what it occupies near
the pin, so one model serves M3 stripes, via pillars or a mix.

### 3.3 Network and calibration

- **Network:** MLP 318 -> 256 -> 128 -> 1 with ReLU (114,689 parameters),
  trained with a positive class weight $w_+ = 150.98$.
- **Output:** $p = \sigma(z - \log w_+)$, which undoes the class weight, so $p$
  is a probability and sums of $p$ are expected violation counts.
- **Training:** 12 epochs, Adam 1e-3, batch 4096 (`train_grid.py --model mlp
  --train_pdn both --target pg --no_pin_grid`).
- **Checkpoint:** `pinaccess/runs/grid/mlp_both_pg_nopin.pt`. The same model is
  kept with a model card as `i3dPDK_NS3K/innovus_enablement/pinaccess/pa_pg_v1.pt`.

### 3.4 Accuracy

The test set is held out by design: tinyRocket, ethmac, swerv_wrapper and aes
are never seen in training.

| | |
|---|---|
| average precision (signal-to-PG) | 0.895 (M3 stripes 0.898, pillars 0.912) |
| ROC-AUC | 0.998 |
| predicted vs actual violating pins per run, correlation | 0.962 |

On the final placements below, the expected count after the flip matches the
routed signal-to-PG count closely. For example, on jpeg (stripe PDN) the 249
cells with residual risk correspond to 249 signal-to-PG markers.

### 3.5 Evaluating the model inside placement (`src/pin_access.py`)

`PinAccessModel` rebuilds the 318 inputs on the GPU for any lower-left $x$ and
row of any cell, for all pins at once.

- **Coverage function:** for every layer and every (row, track level), the
  covered x-intervals of the PDN are merged and turned into a cumulative
  covered length $C(x)$. $C$ is piecewise linear, so the coverage of a bin
  $[a, b]$ is $(C(b) - C(a))/(b - a)$ and is differentiable in $x$. That
  makes the model's risk differentiable with respect to the cell position.
- **Orientations:** the pin offsets, row offsets and access features are
  tabulated for all four orientations N, FS, FN, S.
- **Row orientation:** the row's orientation (N or FS) follows from the rail
  parity. The x-mirror (N vs FN, FS vs S) is a free choice per cell.
- **Per-cell risk:** with $r_a$ and $r_b$ the summed pin probabilities of a
  cell in its row orientation and in its x-mirror:
  - `cell_risk_minflip` returns $r_a$, $r_b$ and their soft minimum
    $-\tau\log(e^{-r_a/\tau}+e^{-r_b/\tau})$ with $\tau = 0.05$.
  - $\min(r_a, r_b)$ is the risk the pre-route flip (Sec. 5) **cannot** remove.
    We call it the *residual* risk.

## 4. Global placement

### 4.1 PDN-aware density target (the GP strategy)

Xplace's electrostatic density term pushes cells until every bin is filled to
at most the target density $\tau$ (0.85 here). We lower that target in the
bins under the M3 stripes (`src/initializer.py`,
`_apply_m3_snet_density_reduction`).

1. Rasterize the M3 VDD/VSS special wires on the density grid. A bin is an
   *M3 bin* when any stripe overlaps it: 6-21% of the bins, depending on the
   design.
2. Add a virtual blockage of $\tau - d_{M3}$ to every M3 bin of the fixed
   (initial) density map, clamped at $\tau$. Only $d_{M3} = 0.3$ of the bin
   is then left for movable cells.

The rest of Xplace's GP is unchanged: Nesterov, cell inflation for
routability, stop at overflow 0.07. The settings are the user's GP flags from
`run_xplace.sh`: `--target_density 0.85 --use_m3_snet_density True
--m3_stripe_density 0.3 --use_cell_inflate True --route_cong_quantile 0.93
--stop_overflow 0.07 --seed 1 --deterministic True`.

The effect is a band of low cell density along every stripe pair. The cells
that would sit on blocked tracks are spread into the neighbouring columns
before legalization. The only knob is $d_{M3}$. The mechanism also supports
lowering $d_{M3}$ further in hot bins during GP (`--drv_checkpoint`, or
`--pa_density` with this model); those variants have not been re-evaluated in
the final flow yet (Sec. 8).

### 4.2 Keeping the placement until routing

A GP objective only matters if the placement reaches the router. Tracking the
same cells through the Innovus steps showed that it did not.

**Stage analysis (jpeg, M3 stripes).** The residual risk is $\min(r_a, r_b)$,
summed over the cells (Sec. 3.5). A cell is "risky" when it sits on a
dangerous offset next to a stripe (Sec. 7.3).

| stage | risky cells | residual risk |
|---|---|---|
| Xplace GP (snapped to sites) | 250 | 51.3 |
| after `refinePlace` | 396 | 69.4 |
| after `optDesign -preCTS` | 987 | 234.4 |
| before routing (after CTS) | 953 | 228.8 |

**What happens in `optDesign -preCTS`.** It runs `place_opt_design -opt`,
whose GigaOpt step performs an incremental global placement
(`incrementalPlace`). Between legalization and the end of `optDesign`,
10,695 of the 10,802 XNOR2_X1 cells moved, many by 6 or more sites or rows.
The incremental placer fills the sparse M3 bands back up. `setOptMode -moveInst
false` does not stop it.

We therefore skip `optDesign -preCTS` after Xplace (`XPLACE_PRESERVE=1`, or
`PLACE_OPT=0` in `flow_fast.tcl`). The fast flow does not close timing, so
nothing else depends on it. Legalization, the second `refinePlace` and CTS
still run. They move cells by about a site, which is what the DP step (Sec. 5)
is designed for.

For a fair comparison, the Innovus baseline drops the optimization in the same
way: bare `place_design` (global placement and legalization) instead of
`place_opt_design`, then the same `refinePlace`, CTS, flip and routing.

### 4.3 Model-derived keep-out bands (optional, `--pa_keepout`)

After the flip, the remaining signal-to-PG risk is concentrated in a few cell
types at a few exact offsets from a stripe (Sec. 7.3). We turn that into a
type-specific keep-out band in GP.

1. **Profile per cell type (`keepout_bands`).** Place the type at x offsets
   $u \in [-12, 12]$ sites from a stripe centre, over 16 sample stripes x 16
   sample rows. Average its residual risk $\min(r_a, r_b)$ at each $u$.
   Offsets with a mean above 0.05 are dangerous.
   - The band is the dangerous range widened by 2 sites on each side. That
     absorbs the 1-2 site moves of legalization and CTS.
   - On jpeg, two types get a band: XNOR2_X1 [-5, 1.5] and MUX2_X1
     [-7, 0.5]. Every other type there is safe at every offset once flipped.
     The bands are recomputed per design from its own PDN.
2. **Potential in GP.** Each cell of a banded type gets a parabolic bump over
   its band around the nearest stripe:
   $\phi(u) = \max\left(0,\ 1 - ((u - c)/h)^2\right)$, with $c$ the band
   centre and $h$ half its width + 0.5 site. Its gradient always points out of
   the band. A smoothed exact-position penalty, by contrast, has safe "valleys"
   between dangerous offsets that legalization undoes.
3. **Weight.** The weight is set once, when the overflow first falls below 0.2,
   so that the mean |gradient| equals `--pa_weight` (1) x the mean |wirelength
   gradient| of the affected cells. It ramps in over 100 iterations. Each
   cell's push is clipped at `--pa_cell_cap` (3) x its own wirelength gradient.

With the GP alone (jpeg), the residual at the GP output drops from 51.3 to 22.8
for +1.1% HPWL. In the routed results (Sec. 7.2) it lowers signal-to-PG by a
further 5% overall (27% on jpeg) and leaves total DRC unchanged. Legalization
gives part of the gain back (jpeg: 22.8 -> 53.2).

## 5. Detailed placement: model-driven orientation (pre-route flip)

Mirroring a cell in x keeps it legal (same site, same row, same row
orientation) but moves every pin to $W - x_{pin}$. Next to a stripe that often
decides whether a pin's access tracks are blocked. Innovus chooses the mirror
for wirelength and changes it for many cells between legalization and
routing. We choose it last, from the model (`pinaccess/flip.py`,
`pnr/pa_flip.tcl`).

1. **When.** After CTS, right before `routeDesign`, Innovus writes the
   placement (`<top>.preroute.def`) and waits.
2. **Scoring.** On the GPU host, for every cell with a signal pin near the PDN,
   compute $r_a$ (row orientation N/FS) and $r_b$ (its x-mirror FN/S) at the
   cell's actual position.
3. **Decision.** Switch to the lower-risk option only when the current option
   is worse than the best one by at least `--min_gain` 0.05. Write the
   changes as `instance orientation` lines.
4. **Applying.** Innovus sets them (`dbSet .orient`, DEF N/S/FN/FS -> R0/R180/MY/MX),
   checks the placement and routes.

The step changes no positions and no row orientations. It is placer-agnostic:
the same step follows Innovus placement in the baseline. Each run logs the
expected signal-to-PG count before and after
(`pa flip: N cells, F flipped; expected signal-PG violations A -> B`). $B$ is
the residual risk the GP is meant to reduce.

## 6. Experimental setup

- **Designs and flow.** All 13 designs, M3 stripe PDN, unidirectional routing,
  fast flow, one seed (`--seed 1 --deterministic True`; Innovus is
  deterministic for a given placement).
- **Same floorplan.** Every run of a design uses the same floorplan, macro
  placement and PDN. Xplace places only the standard cells.

| run (OUT_TAG) | GP | `optDesign -preCTS` | flip |
|---|---|---|---|
| `ns3k_uni3_fast` | Innovus `place_opt_design` | yes (inside) | no |
| `ns3k_innf_fast` | Innovus `place_design` | **no** | yes |
| `ns3k_xpdrv_fast` | Xplace, $d_{M3}$ 0.3 | yes | no |
| `ns3k_xpf_fast` | Xplace, $d_{M3}$ 0.3 | yes | yes |
| `ns3k_xpfp_fast` | Xplace, $d_{M3}$ 0.3 | **no** | yes |
| `ns3k_xpfpn_fast` | Xplace, no M3 density reduction | **no** | yes |
| `ns3k_xpfkp_fast` | Xplace, $d_{M3}$ 0.3 + keep-out bands | **no** | yes |

## 7. Results

### 7.1 DP: the flip alone

The two runs below have the same Xplace GP, and their GP outputs are
byte-identical. Only the flip differs. 13 designs:

| | total DRC | signal-PG | cells flipped |
|---|---|---|---|
| no flip (`ns3k_xpdrv_fast`) | 13,457 | 5,245 | - |
| flip (`ns3k_xpf_fast`) | 8,777 (-35%) | 573 (-89%) | 2,690 of 655,508 (0.41%) |

With M5 straps and via pillars instead of M3 stripes (`PG_STYLE=pillar`), the
violations left are almost all signal-to-PG, and the flip removes them. On the
6 designs run both ways with the same Xplace GP (tinyRocket, aes, riscv32i,
ethmac, jpeg, swerv), DRC goes from 1,229 to 1 (a signal-to-signal marker on
ethmac). Over all 13, Xplace + flip leaves 3 violations, against 2,533 for
Innovus `place_opt_design` without flip.

### 7.2 GP: PDN-aware density with the placement kept

All runs below are flipped and use the same downstream flow (no pre-CTS
optimization), except where noted. Totals are over the 13 designs:

| | total DRC | signal-PG | routed WL (um) | residual after flip (expected) |
|---|---|---|---|---|
| Innovus `place_opt_design`, no flip (original baseline) | 13,378 | 5,339 | 2,248,508 | - |
| Innovus `place_design` + flip (**fair baseline**) | 9,614 | 600 | 2,382,729 | 537.2 |
| Xplace $d_{M3}$ 0.3, *with* `optDesign` + flip | 8,777 | 573 | 2,196,995 | 512.7 |
| Xplace without M3 density reduction + flip | 10,421 (+8%) | 566 | 2,197,529 (-7.8%) | 500.2 |
| **Xplace $d_{M3}$ 0.3 + flip** | **5,056 (-47%)** | **271 (-55%)** | **2,206,840 (-7.4%)** | 238.6 |
| Xplace $d_{M3}$ 0.3 + keep-out bands + flip | 5,083 (-47%) | 257 (-57%) | 2,218,122 (-6.9%) | 228.6 |

Percentages are against the fair baseline.

Per design, total DRC / signal-PG / routed WL (um):

| design | Innovus `place_design` + flip | Xplace, no M3 density + flip | **Xplace $d_{M3}$ 0.3 + flip** | + keep-out |
|---|---|---|---|---|
| gcd | 12 / 2 / 370 | 7 / 1 / 360 | 1 / 0 / 365 | 1 / 0 / 368 |
| serv | 18 / 0 / 1,696 | 12 / 1 / 1,610 | 6 / 0 / 1,618 | 9 / 0 / 1,641 |
| riscv32i | 97 / 4 / 21,701 | 113 / 0 / 21,142 | 75 / 1 / 21,168 | 57 / 0 / 21,440 |
| aes | 241 / 17 / 26,815 | 288 / 14 / 23,332 | 115 / 8 / 24,157 | 94 / 7 / 23,790 |
| dynamic_node | 116 / 2 / 32,876 | 152 / 4 / 33,457 | 25 / 0 / 32,304 | 29 / 0 / 32,462 |
| tinyRocket | 288 / 15 / 63,510 | 335 / 17 / 63,090 | 155 / 7 / 63,128 | 174 / 7 / 63,227 |
| bp_fe_top | 536 / 4 / 189,853 | 577 / 5 / 166,947 | 380 / 3 / 163,812 | 303 / 4 / 165,109 |
| ethmac | 446 / 2 / 95,924 | 587 / 4 / 89,451 | 337 / 1 / 89,899 | 347 / 1 / 90,461 |
| bp_be_top | 835 / 12 / 310,926 | 889 / 6 / 290,462 | 350 / 10 / 290,481 | 436 / 6 / 291,687 |
| jpeg | 470 / 267 / 88,507 | 515 / 242 / 81,630 | 180 / 104 / 84,414 | 175 / 76 / 85,560 |
| swerv | 2,060 / 65 / 425,518 | 2,225 / 50 / 390,049 | 1,030 / 23 / 365,487 | 1,029 / 26 / 366,140 |
| swerv_wrapper | 1,859 / 59 / 524,537 | 2,043 / 67 / 461,501 | 803 / 26 / 459,323 | 802 / 29 / 467,304 |
| ariane133 | 2,636 / 151 / 600,496 | 2,678 / 155 / 574,498 | 1,599 / 88 / 610,684 | 1,627 / 101 / 608,933 |

What the table supports:

- **The gain comes from the GP strategy.** With the M3 density reduction,
  Xplace beats the fair Innovus baseline on all 13 designs (DRC -23% to -92%).
  Without it, DRC is 8% *above* the baseline, and it is higher than with the
  reduction on every design.
- **Keeping the placement is required.** With `optDesign -preCTS` in the flow,
  the same GP ends at 8,777 DRC, close to the run without the density
  reduction. The incremental placement erases most of the effect.
- **Innovus does not gain from dropping the optimization.** `place_design` +
  flip (9,614) is not better than Xplace with the optimization (8,777), and its
  wirelength is the longest of all. So the improvement is not a by-product of
  skipping `optDesign`.
- **The two levels are complementary.** The density reduction removes
  signal-to-signal congestion along the stripes, which the per-pin model cannot
  see. The flip removes the pin-level signal-to-PG risk.
- **Keep-out bands are a small addition.** Signal-PG drops 271 -> 257 (jpeg
  104 -> 76). Total DRC and WL are about unchanged.

### 7.3 What the flip cannot fix

Profiling every cell type against the M3 stripes (residual risk vs x offset of
the lower-left corner from the stripe centre, in sites) explains the residual
risk:

- **One cell type.** In jpeg, swerv and tinyRocket (Xplace with optDesign,
  before routing), 100% of the residual comes from XNOR2_X1 overlapping a
  stripe.
- **Specific offsets.** XNOR2_X1 is dangerous at offsets -3, -1 and 0 (mean
  residual 0.2-0.5 even with the better mirror). MUX2_X1 is dangerous from -5
  to -1. Every other type is safe at every offset once flipped.
- **Distance to safety.** Every one of these cells is safe again one site to
  the left or right.

This is why the GP strategies that aim at exact positions failed (Sec. 8), and
why a keep-out band with a margin is the GP form that survives legalization.

## 8. Explored alternatives

These were tried earlier, with `optDesign -preCTS` still in the flow. They gave
-4% to +3% on signal-PG or total DRC, within run-to-run noise. Except for the
gradient penalty, they have not yet been re-run with the placement kept.

| variant | idea | outcome |
|---|---|---|
| model risk as a GP penalty (`--pa_model`, `--pa_norm cell/align`, `--pa_cell_cap`) | gradient of $\sum p$ on cells near the PDN, weight relative to the wirelength gradient | +-1-7%. Exact positions are lost in legalization; the smoothed version (`--pa_smooth_sites`) pulls cells into the safe offset *between* two dangerous ones |
| soft-min over the two mirrors (`--pa_flip_mode min`) | penalize only what the flip cannot fix | -4% to -6% residual |
| model-guided dynamic density (`--pa_density`) | lower $d_{M3}$ step by step (0.5 -> 0.2, 0.3 -> 0.1, 0.5 -> 0.15) in bins whose cells carry > 0.05 expected residual violations | -11% to +3%, no trend; all with optDesign in the flow |
| uniform $d_{M3}$ 0.2 | lower every M3 bin | no gain over 0.3 (with optDesign) |

These results all predate Sec. 4.2, so their negative outcome is not conclusive.
The density variants in particular should be re-run with the placement kept.

## 9. Limitations

- **Single seed and single PDK.** One seed per configuration; the per-design
  spread between configurations can be +-20-50% on small designs. One PDK
  (NS3K, 4x LEFs), M3 stripe PDN for the GP results.
- **No timing.** Skipping `optDesign -preCTS` removes pre-CTS buffering,
  sizing and DRV fixing. The fast flow does not close timing anyway, but a
  timing-closed flow would need an optimization that does not re-place cells
  (or that respects the density map).
- **Signal-to-signal violations are not modelled.** The per-pin model predicts
  only signal-to-PG violations. Signal-to-signal violations are reduced only
  indirectly, by the density target.
- **Test designs were in the data.** The model's held-out designs are unseen
  in training. The placements evaluated here, however, come from the same 13
  designs the training runs used.

## 10. Reproduction

### Code

| file | role |
|---|---|
| `src/pin_access.py` | `PinAccessModel` (coverage function, 4 orientations, `score`, `cell_risk_minflip`, `keepout_bands`); `PinAccessContext` (GP penalty, `--pa_density` map, keep-out potential) |
| `src/initializer.py` | M3 stripe density reduction (`--use_m3_snet_density`, `--m3_stripe_density`); M2/M3 layer index from `--m1_rindex` |
| `src/run_placement_nesterov.py` | builds the context; dynamic M3 density update (FCN or `--pa_density`) |
| `src/calculator.py` | adds the pin-access gradient to the placement gradient |
| `main.py` | options `--pa_*`, `--m1_rindex` |
| `pinaccess/extract_grid.py`, `extract_extra.py`, `train_grid.py` | dataset and training |
| `pinaccess/flip.py` | pre-route flip |
| `pinaccess/check_module.py`, `stage_check.py` | model vs dataset features; risk at each flow stage |
| `xplace_work/ana/` (not tracked) | stage and keep-out analysis used in Sec. 4.2 and 7.3 (`stage_band.py`, `type_profile.py`, `residual.py`, `moves.py`) |
| `i3dPDK_NS3K/innovus_enablement/pnr/xplace_place.tcl` | Xplace hand-off; `XPLACE_PRESERVE` / `PLACE_OPT=0` |
| `i3dPDK_NS3K/innovus_enablement/pnr/pa_flip.tcl` | applies the flips before routing |
| `i3dPDK_NS3K/innovus_enablement/pnr/flow_fast.tcl` | `PLACE_OPT`, `PA_FLIP` |
| `i3dPDK_NS3K/innovus_enablement/xplace/run_xplace_flow.sh`, `run_xplace_gp.sh` | GPU-host driver; `PLACER=innovus` for the baseline |

### Runs

From `~/i3dPDK_NS3K/innovus_enablement` (push the checkout to the server
first with `./sync_to_server.sh`, or let `run_xplace_flow.sh` do it):

```bash
# GP flags (the user's run_xplace.sh): density 0.85, M3 bins 0.3, cell inflation
XA="--seed 1 --use_m3_snet_density True --m3_stripe_density 0.3 --m3_stripe_density_min 0.3 \
 --adaptive_congest_rc 1.43 --adaptive_m3_stripe_density_min 0.1 --stop_overflow 0.07 \
 --drv_checkpoint fcn_trainer/results/dynamic_v19_aes2048_dropaoi/best_model.pt --drv_pred_adaptive True \
 --drv_phase1_vis_freq 0 --exp false --drv_density_update_overflow 0.25 --drv_density_update_step 0.05 \
 --drv_density_update_freq 10 --drv_density_hot_threshold 0.08 --use_route_force false \
 --use_cell_inflate true --route_cong_quantile 0.93 --min_area_inc 0.01 --deterministic True \
 --drv_ap_json fcn_trainer/SL_modified_m2_ap.json,fcn_trainer/L_modified_m2_ap.json"

# Xplace + PDN-aware density, placement kept, flip (ns3k_xpfp_fast)
env FLOW=fast OUT_TAG=ns3k_xpfp_fast PNR_ENV="ROUTE_UNIDIR=1 XPLACE_PRESERVE=1" PA_FLIP=1 \
    XPLACE_TD=0.85 XPLACE_ARGS="$XA" xplace/run_xplace_flow.sh jpeg NS3K_5T

# + keep-out bands (ns3k_xpfkp_fast)
XPLACE_ARGS="$XA --pa_model pinaccess/runs/grid/mlp_both_pg_nopin.pt --pa_keepout True \
 --pa_weight 1 --pa_cell_cap 3 --pa_start_overflow 0.2"

# ablation: no M3 density reduction (ns3k_xpfpn_fast)
XPLACE_ARGS="$XA --use_m3_snet_density False"

# fair Innovus baseline: place_design, no pre-CTS optimization, flip (ns3k_innf_fast)
env FLOW=fast OUT_TAG=ns3k_innf_fast PLACER=innovus PNR_ENV="ROUTE_UNIDIR=1 PLACE_OPT=0" PA_FLIP=1 \
    xplace/run_xplace_flow.sh jpeg NS3K_5T
```

Results land in `pnr/out/<design>/NS3K_5T/<tag>/`:

- `rpt/drc.rpt` holds the routed violations.
- `xplace/flip.log` has the expected counts before and after the flip.
- `xplace/<top>.{gp,lg,opt,preroute}.def` are the placement at each stage
  (`opt.def` only with Xplace).

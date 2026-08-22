# README 2 — Reviewer-Revision Work (merge handoff)

Self-contained record of the work done to address the reviewer comments, written
so it can be merged into [`README.md`](README.md) **without needing the
conversation that produced it**. Every change is explained together with the
reasoning behind it.

**Baseline for this document:** `origin/main` at commit `714272c`. All changes
below are relative to that. Everything is additive and defaults-preserving —
every script run without the new environment variables reproduces the published
v3 behaviour exactly.

---

## 1. What changed, at a glance

### New files (not on `origin/main`)

- **`run_pipeline.sh`** — single entry point for all 7 stages
  (`dns, tec, msis, merge, train, eval, tune, ontrack`) in three named modes:
  `old` (published setup, unchanged), `new` (full mission 2002–2017, train
  2003–2015), `storm` (as `new` plus the March-2015 holdout and its 4-regime
  evaluation). *Why:* the pipeline spans 3 directories and was configured by
  **editing constants at the top of each file**, so running two configurations
  meant editing code in between, and nothing recorded which configuration
  produced which result. Stages are selectable (`./run_pipeline.sh new
  train,eval`) and long ones are wrapped in `caffeinate -is`.

- **`run_storm_pipeline.sh`** — chains `storm train,eval` → `storm ontrack` →
  storm report, with `set -euo pipefail` and per-variant log files. *Why:* the
  middle stage takes hours; chaining lets the whole experiment run unattended
  and guarantees the report reads the run directory just produced. Honours
  `AP_HISTORY` and `USE_TUNED`, and derives the variant name so runs never
  overwrite each other.

- **`CoreModel/tune.py`** — seeded random hyperparameter search over 7
  dimensions (tree shape plus initial LR, decay factor, decay step), exposed as
  an importable `random_search(...)` with a thin env-var wrapper. Writes
  `tuning_trials.csv`, `tuning_stability.png`, `tuning_sensitivity.png` and
  `best_params.json`. *Why:* reviewer comment 11 asks how hyperparameters were
  chosen; there was no reproducible search, the values were hand-tuned. Trial 0
  is always the published configuration, so the search doubles as a stability
  check of the existing choice (§7).

- **`Forecast/make_table_storm.py`** — storm report: the 17–18 March main-phase
  table (core vs. warm-start vs. MSIS), an optional `--daily` per-day trace
  across the whole window, and a control-regime comparison against an earlier
  run. Prompts for the run directory when several exist, or takes `--runs`.
  *Why:* the rolling protocol needs a 45-day window but the storm must be
  reported on the main-phase days only; this slices the **saved predictions**,
  so changing which days are reported never requires re-running anything (§6).

- **`Forecast/on_track_persisted.py`** + **`make_table_persisted.py`** —
  rolling evaluation with drivers inside the forecast window replaced by
  issue-time values (TEC persisted via nearest lat/LST geometry with a k-d
  tree; F10.7 and ap held constant). *Why:* the published "forecast" used
  observed drivers, unknown at issue time, making those numbers a
  perfect-driver upper bound. **Parked for the follow-up paper** — see §12.3.

- **`README2.md`** — this document.

### Modified files

Line counts are `diff` against `origin/main`. Every change is additive with
published values as defaults, except where noted: the second TEC lag
(`vtec_matched_lag2`), the row-shift lag implementation (`TEC_LAG_MODE`), and
the true-forecast scripts have been removed.

| File | Scale | What changed |
|---|---|---|
| `Forecast/on_track.py` | +139 / −46 | (a) env-var overrides for model/scaler/data/output paths; (b) feature lists imported from `CoreModel/config.py` instead of redeclared, so `AP_HISTORY` and the TEC lag cannot drift from training; (c) two new evaluation regimes `y2002` and `storm2015` (§6); (d) `compute_metrics` extended with log-RMSE, Top-5 %, R² and `msis_*` baseline columns; (e) resume support — skip finished runs, write summary CSV after each run; (f) `ONTRACK_FILTERS` to select regimes; (g) memory: in-place column drop, freeing per-window frames; (h) **bug fix** — metrics/return block was inside an `else` branch, returning `None` for runs without a snapshot date (§8). |
| `CoreModel/train.py` | +73 / −6 | (a) `TIME_EXCLUDE` interior-holdout loop, applied *after* lag construction; (b) optional tuned hyperparameters via `TRAIN_PARAMS_JSON`, including a rebuilt LR schedule; (c) memory: column-limited parquet read, in-place column drop, float32 downcast of features, freeing unscaled splits after scaling. |
| `CoreModel/config.py` | +41 / −6 | (a) env-var overrides for all paths and the time window; (b) `TIME_EXCLUDE` parsing multiple `"start,end"` intervals separated by `;`; (c) `TEC_LAG` / `TEC_LAG_COL` — the single t−3 h TEC lag, now the one definition shared by CoreModel and Forecast; (d) `AP_HISTORY` appending 3 (or 5, with `full`) ap features to `FEATURES` and `COLS_TO_SCALE`. |
| `DataPreparation/merge_tec_grace.py` | +104 / −9 | (a) `MERGE_CHUNKED=1` year-by-year merge with ±4 h boundary overlap and streamed concatenation (§3.3); (b) deterministic nearest-in-time TEC epoch selection, replacing an arbitrary tie-break (§3.4); (c) env-var overrides. |
| `DataPreparation/download_dns.py` | +37 / −22 | (a) FTP → HTTPS (the FTP listing is empty, §3.1); (b) **bug fix** — reader dispatch compared the satellite ID (`GA`/`GB`) against `"GRACE"`, routing every GRACE file to the Swarm parser; (c) mission directory map incl. GRACE-FO; (d) env-var overrides. |
| `feature_functions.py` | +29 / −0 | New `add_tec_time_lag_features` — exact time-based TEC lags (§4). Nothing else touched. |
| `DataPreparation/download_tec.py` | +4 / −3 | Env-var overrides for year range and output directory. |
| `DataPreparation/run_pymsis.py` | +5 / −4 | Env-var overrides for input/output files and time range. |
| `README.md` | +2 / −0 | figshare DOI for the processed data. |

`CoreModel/evaluate.py`, `plotting.py`, `losses.py`, `pymsis_utils.py`,
`run_pymsis_swarm.py`, `off_track.py`, `swarm_validation.py` are **unchanged**.
`evaluate.py` inherits the new behaviour through `config.py` and
`train.load_and_engineer`.

### Pre-existing local differences (not from this work)

Two differences against `origin/main` predate this revision and should be
resolved deliberately rather than merged blindly:

- **`workshop/` is missing locally** (8 files on `origin/main`: notebooks
  NB0–NB5, `README.md`, `SETUP.md`, `make_workshop_data.py`, and the saved
  workshop model/scaler). Nothing here touched them.
- **`paths.py` (+0 / −5)** — the local copy has dropped the `GRACE_WORKSHOP`
  constant and its comment block. Consistent with the missing `workshop/`
  directory.

---

## 2. Reviewer comments → responses

| Comment | Response | Section |
|---|---|---|
| **Major 2** — why only 2009–2016, ignoring higher solar activity in 2002–2009? | Full mission 2002–2017 downloaded and merged. Model trains on 2003–2015; 2002 becomes a solar-maximum holdout. | §3, §6 |
| **Major 5** — why 3 h and 24 h TEC lags? | The feature set is now the current TEC map plus a **single exact t−3 h lookup**; the 24 h lag is gone, as is the row-shift implementation that made the old lags ~42 min / 24 h. **Manuscript must drop the 24 h lag.** | §4 |
| **Major 7** — are MSIS log-residuals zero-centred and log-normal? | Measured on 73.7 M samples: mean **−0.149**, median −0.120, skew −0.54, ex. kurtosis +1.75. **Neither zero-centred nor log-normal** — reviewer correct on both. Magnitude ~0.1 as they expected, but *negative*. A constant explains only 20 % of it; the rest tracks the solar cycle and flips sign under storms. Log target still justified (§10.1). | §10, fig. `msis_residuals` |
| **Major 8 / 10** — paper mixes nowcasting and forecasting; drop forecasting and warm-start | Not yet decided — see §12.1. The h=1 protocol is fully causal and is better described as *adaptive nowcasting*. | §12.1 |
| **Major 11** — document the hyperparameter search | Seeded random search with trials table and stability/sensitivity plots. | §7 |
| **Figure 5** — Feb 2016 storm too weak | March 2015 **G4** storm (ap = 179) held out of training and evaluated out-of-sample. | §6, §11 |
| **Minor 3** — negative/fill TEC values? | Verified: **no negative or fill values** anywhere in the merged file. | §3.2 |

---

## 3. Data pipeline

### 3.1 TU Delft download: FTP is dead, and a latent bug hid it

The FTP host in the published `download_dns.py` still accepts connections but
returns an **empty file listing**, so no data can be fetched. Data now lives at
`https://thermosphere.tudelft.nl/data/data/version_02/` — which also hosts
GRACE-FO, relevant for the follow-up paper.

While switching, a second bug surfaced that would have produced **zero rows with
no error message**: `_parse_stem` takes the mission from the filename, which is
the *satellite* ID (`GA`, `GB`), but `read_dns_txt` compared it against
`"GRACE"`. Every GRACE file was routed to the Swarm parser, failed on the
12-column format, and was swallowed by the surrounding `try/except`. The symptom
was "348 files processed, 0 read". Dispatch now recognises `GA`/`GB` (GRACE) and
`GC`/`GD` (GRACE-FO).

### 3.2 Full-mission data

GRACE 2002–2017 (348 monthly files, both satellites) and CODE TEC 2002–2017.
Verified after merge: **82.5 M rows**, TEC match failure ≤ 0.26 % in every year,
zero fallbacks, and median TEC tracking the solar cycle (24.9 TECU in 2002 → 6.5
at the 2008 minimum). No negative or fill TEC values anywhere.

**For the manuscript:** CODE maps are **2-hourly before 2015**, hourly after.
The ±3 h matching window handles both, but `matched_tec_value` has coarser time
resolution in the early era.

### 3.3 The merge cannot run on a 24 GB machine

`merge_tec_grace.py` loads the entire TEC table into pandas — 464 M rows, >40 GB.
`MERGE_CHUNKED=1` merges year by year with ±4 h TEC overlap at year boundaries,
so **every GRACE point still sees exactly the same candidate epochs**, then
stream-concatenates the yearly parts at constant memory.

Validated against the original all-at-once merge on 2016: identical row count,
**100 % identical chord distances**, identical NaN and fallback fractions.

### 3.4 The original merge picked the TEC epoch arbitrarily

That validation exposed something worth knowing. The TEC grid geometry is
identical at every epoch, so when several epochs fall inside the ±3 h window the
chord distances **tie exactly**, and the original code broke the tie by whatever
order the rows happened to be in. `matched_tec_value` was therefore "TEC at the
nearest grid cell, at an arbitrary epoch within ±3 h".

Ties are now broken deterministically by **nearest epoch in time**. Verified
independently: 97.3 % of 1000 sampled points reproduce a manual nearest-epoch
lookup exactly; the remainder are half-hour midpoints where "nearest" is
genuinely ambiguous. Worth one sentence in the data section.

---

## 4. TEC lag features (reviewer comment 5)

The model uses the current TEC map plus **one** lagged map at **t−3 h**
(`vtec_matched_lag`), built by `ff.add_tec_time_lag_features`: an exact lookup
of the nearest sample to t−3 h within the same satellite track (±10 min
tolerance, else NaN, which `dropna` removes). Matching within a track prevents
a GA row from being handed GB's TEC, and a time-based lookup is unaffected by
data gaps.

The lag and its column name are set once in `CoreModel/config.py`
(`TEC_LAG` / `TEC_LAG_COL`) and imported by every training and forecast script,
so the two cannot drift apart.

Earlier revisions carried two lags implemented as **row**-shifts
(`shift(500)` / `shift(17280)`). On 10 s data with GA and GB interleaved those
spanned ~42 min and 24 h rather than the intended values, and a fixed row count
covers a different amount of time whenever data is missing — 24 % of the
nominal 24 h values were more than 10 minutes off. Both the second lag and the
row-shift implementation have been removed.

> **Manuscript note:** the 3 h lag the text describes is now what the code
> actually computes. The 24 h lag is no longer part of the feature set, so any
> reference to it must be removed.

---

## 5. Storm-history ap features

**Motivation.** The feature set gives the model `ap_m3h` and `ap_m6h` — a 6-hour
view of geomagnetic activity. NRLMSIS-2.1 itself consumes a **7-element ap
vector spanning ~57 hours**, because storm-time density depends on *integrated*
Joule heating, not the instantaneous index. The correction model was being asked
to correct a baseline that had strictly more driver information than it did.

March 2015 shows why: `ap_0h` peaks at 179 on 17 March, but `ap_avg12_33h` peaks
at **116.6 on 18 March** and stays elevated through recovery. On 18 March the
model saw `ap_m3h` = 32 — apparently quiet — while the atmosphere was still
responding to a day of accumulated heating.

`AP_HISTORY=1` appends three features:

| Feature | Adds |
|---|---|
| `ap_m9h` | Extends short-range history to 9 h |
| `ap_avg12_33h` | Integrated heating, 12–33 h back |
| `ap_avg36_57h` | Integrated heating, 36–57 h back |

`AP_HISTORY=full` also adds `ap_daily` and `ap_0h`, deliberately excluded from
the default: `ap_daily` is largely redundant with the averaged terms and `ap_0h`
is near-collinear with `ap_m3h`.

All columns already exist in the merged parquet (written by `run_pymsis.py`)
with **zero nulls across 2002–2017**, so no new data preparation is needed. They
were verified to be genuinely lagged rather than duplicates: the 17 March trace
shows ap = 179 propagating through `ap_0h` → `ap_m3h` → `ap_m6h` → `ap_m9h`.

**Caveat.** These are min-max scaled like the existing ap features. Min-max is
outlier-sensitive and ap is heavily skewed (the 2003 maximum of 400 sets the
bound), so ordinary values compress near the low end. This matches how `ap_m3h`
already behaves and tree models are insensitive to monotone rescaling — but do
not expect the scaling itself to help storm sensitivity.

---

## 6. Training-set design and evaluation regimes

Training runs 2003–2015 with two **interior holdouts**, dropped *after* lag
construction so rows following a gap keep physically correct lag values:

| Holdout | Window | Purpose |
|---|---|---|
| quiet-2009 | 2009-01-01 → 2009-06-06 | Deep solar minimum |
| storm-2015 | 2015-03-01 → 2015-04-15 | March G4 storm + recovery |

Four evaluation regimes:

| Regime | Period | Tests |
|---|---|---|
| `y2002` | before 2003-01-01 | Solar maximum (out of training distribution) |
| `quiet2009` | Jan–Jun 2009 | Deep solar minimum |
| `storm2015` | 1 Mar – 15 Apr 2015 | Severe (G4) storm |
| `post2016` | after 2016-01-01 | Declining phase |

**Why these four.** They accumulated one at a time: `post2016` from the original
paper; `quiet2009` replacing the old `pre2009` filter once training extended
back to 2002 (otherwise "everything before June 2009" would swallow 2002–2008
into the test set); `y2002` for reviewer comment 2; `storm2015` for Figure 5.
The coverage argument that emerges is defensible — minimum, maximum, severe
storm, declining phase.

**Why the storm window is 45 days but only 2 days are reported.** The rolling
protocol needs 6 lead-in days before its first forecast, and every day it
fine-tunes on must itself be outside core training.

> **Do not widen the reported storm window.** Pooling 15–20 March mixes three
> physically distinct regimes (pre-storm quiet, main phase, recovery), so a
> pooled number is driven by the surrounding quiet days rather than the storm —
> structurally the same criticism the reviewer made of the Feb 2016 figure. The
> 2-day definition is not cherry-picked: 17–18 March are the only days in the
> window with ap = 179, an order of magnitude above the window median of 18.
> This holds regardless of which way the main-phase result comes out.

---

## 7. Hyperparameter search (reviewer comment 11)

`CoreModel/tune.py` exposes `random_search(...)` taking all settings as
arguments, so it can be imported and driven directly; a thin env-var wrapper
lets the pipeline call it.

**Protocol.**
- Same data, split and scaling as `train.py`. Selection RMSE is computed on the
  **8 temporally separated blocks** training already uses for early stopping,
  spanning the whole window — so a configuration that only works in one solar
  era is penalised automatically.
- **Trial 0 is always the published configuration**, so the search doubles as a
  stability check of the existing choice.
- All trials use a decay schedule of the *published form*
  (`initial_lr * decay^(round // step)`); the search varies initial rate, decay
  factor and step size. Decay ≈ 1.0 with a large step approximates a constant
  rate, so that regime is included.
- Runs on a row-stride subsample for speed. **Confirm top configurations on full
  data before adopting** — the top of the ranking was separated by <0.5 %, which
  a subsample cannot resolve reliably.

**Why random, not grid.** With 7 dimensions a 3-value grid is 2187 runs. At
equal budget random search samples every dimension far more densely — the
standard result (Bergstra & Bengio, 2012, JMLR), and a citable justification.

**Result (38 trials completed).** The published configuration ranked **last**;
the best trial improved selection RMSE by **15.1 %** (0.05747 → 0.04877).
Selection RMSE correlates most strongly with `lr_step_size` (−0.45),
`learning_rate` (−0.43) and `lr_decay_factor` (−0.33) — tree-shape parameters
barely matter. The published schedule decays so fast that training effectively
freezes after ~700 rounds, visible in the logs as train/val RMSE identical to 5
decimals across 60 consecutive rounds.

Adopted with `USE_TUNED=1` (reads `tuning_v5/best_params.json`). **Off by
default** so results stay comparable with earlier runs.

> ⚠️ The search optimises average RMSE over the whole window, dominated by quiet
> conditions. Storm days are a tiny fraction, so it should **not** be expected to
> fix storm-day performance — a flat storm result after tuning is not evidence
> that tuning failed.

---

## 8. Robustness, and one real bug

**Bug fixed in `on_track.py`.** The metrics/return block was indented inside the
`else` branch of a plotting-date check, so any run without a snapshot date
returned `None` and crashed the caller with `TypeError: cannot unpack
non-iterable NoneType object`. It never triggered because every legacy regime
happened to have a snapshot date; the new regimes do not.

**Resume support.** `on_track.py` skips runs whose predictions already exist and
writes the summary CSV after every run, so an interrupted job resumes without
repeating hours of work.

> ⚠️ **This makes a separate output root mandatory when the model or feature set
> changes**, or finished runs are silently reused and the summary looks like a
> fresh evaluation while containing stale predictions. Variant naming handles
> this: `AP_HISTORY=1` writes to `runs_v7_storm_ap/`, otherwise `runs_v6_storm/`.

**Metrics.** `compute_metrics` now also reports log-RMSE, Top-5 %, R² and the
MSIS baseline (as `msis_*` columns) for every run, so no separate baseline
computation is needed downstream.

**Memory.** Several stages exceeded 24 GB on the full mission. `train.py` reads
only required parquet columns, drops intermediates, and downcasts features to
float32 (XGBoost converts internally anyway; `msis_rho`, `rho_obs` and the
target stay float64 since densities are ~1e-13). `on_track.py` avoids two
full-frame copies and frees per-window frames.

> **Lesson worth recording:** column dropping and downcasting must be done **in
> place** (`del df[col]`, per-column `astype`). An earlier attempt used
> `df[keep].copy()`, which duplicates an ~11 GB frame and needs far more headroom
> than it saves — it *caused* OOM kills rather than preventing them. If a stage
> dies with `Killed: 9`, profile before assuming the cause.

---

## 9. How to run

```bash
# Published setup — unchanged reference behaviour
./run_pipeline.sh old

# Full-mission setup: 2002–2017 data, train 2003–2015
./run_pipeline.sh new                      # all stages in order
./run_pipeline.sh new dns,tec              # downloads only (slow; resumable)
./run_pipeline.sh new msis,merge
./run_pipeline.sh new train,eval
./run_pipeline.sh new ontrack

# Hyperparameter search
TUNE_TRIALS=200 ./run_pipeline.sh new tune

# Storm experiment (adds the March-2015 holdout, 4 regimes)
./run_storm_pipeline.sh                    # 14 features
AP_HISTORY=1 ./run_storm_pipeline.sh       # + 3 ap storm-history features (default)
USE_TUNED=1  ./run_storm_pipeline.sh       # tuned hyperparameters

# Reports
python Forecast/make_table_storm.py --daily
python Forecast/make_table_storm.py --runs runs_v7_storm_ap --daily
```

Stages: `dns, tec, msis, merge, train, eval, tune, ontrack`. Long stages are
wrapped in `caffeinate -is` inside the scripts — note `caffeinate` does **not**
prevent sleep when a laptop lid is closed.

Downloads need an Earthdata login in `~/.netrc` (mode 600):

```
machine urs.earthdata.nasa.gov login USERNAME password PASSWORD
```

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `TEC_LAGS` | `3h` | TEC lag set; more lags need a retrained model (§4) |
| `AP_HISTORY` | `1` | `1` = +3 storm-history ap features; `full` = +5 |
| `USE_TUNED` | `0` | `1` = load `tuning_v5/best_params.json` |
| `TRAIN_TIME_EXCLUDE` | unset | `"start,end;start,end"` interior holdouts |
| `ONTRACK_FILTERS` | `quiet2009,storm2015,post2016` | Which evaluation regimes to run |
| `ONTRACK_HORIZONS` | `1,3` | Forecast horizons in days |
| `ONTRACK_LOOKBACK_DAYS` | `3` | Days of history each warm-start step fine-tunes on |
| `ONTRACK_RESET_EVERY` | `4` | Reset the fine-tuned model to base every N steps |
| `ONTRACK_PARAMS_JSON` | `tuning_v13_.../best_params.json` | Tree shape for the trees warm-start adds |
| `MERGE_CHUNKED` | unset | `1` = year-by-year merge (required for full mission) |
| `TUNE_TRIALS` | `32` | Random-search trial count |

Plus per-stage path overrides (`TRAIN_PARQUET_FILE`, `TRAIN_MODEL_OUT`,
`ONTRACK_OUTPUT_ROOT`, …), all defaulting to the published filenames.

---

## 10. Measured data facts for the manuscript

**MSIS log-residual distribution** (reviewer comment 7). Measured over the
full-mission training window (2002–2016, quiet-2009 held out; n = 73,746,824),
`log(rho_obs / msis_rho)` has:

| | value | as a factor |
|---|---|---|
| mean | **−0.1487** | ×0.862 (MSIS **+13.8 %** too high) |
| median | −0.1202 | ×0.887 |
| sd | 0.2974 | |
| skew | −0.542 | |
| excess kurtosis | +1.753 | |

**The reviewer is right on both counts.** The residual is *not* centred at zero,
and it is *not* log-normal: Shapiro–Wilk W = 0.976 (p ≈ 2e−28), D'Agostino
K² = 15336 (p = 0), and the left tail is markedly heavier than normal
(standardised p0.1 at −4.03 against −3.09 expected) while the right tail is
lighter. Their expected magnitude (~0.1) is right; **the sign is opposite** —
MSIS over-estimates density on this dataset, so the offset is negative.

**But the offset is not a calibration constant.** A single constant would remove
only **20 %** of the mean-square residual. The remaining structure is
state-dependent:

- **Solar cycle** — yearly median runs from ≈0 at solar maximum (2002: −0.026;
  2013: +0.023; 2015: +0.041) to **−0.446** in the deep 2009 minimum, where MSIS
  over-estimates by ~36 %.
- **Geomagnetic activity** — median rises monotonically with ap and *changes
  sign*: −0.148 at ap 0–10, −0.031 at ap 20–40, **+0.140 at ap > 80**. MSIS
  over-estimates in quiet conditions and under-estimates during storms.
- **Altitude** — +0.024 below 400 km against −0.192 at 460–500 km.

This is the response to the reviewer's attribution: a fixed pipeline calibration
cannot produce an offset that swings by 0.49 across the solar cycle and flips
sign under storm forcing. That systematic, activity-dependent structure is
exactly what a correction model can learn — an argument for the method in its
own right.

**Consequence for the method.** The residual being skewed and heavy-tailed does
*not* invalidate the log target — see §10.1.

Figure: `figs/msis_residuals.png` / `.pdf`, produced by
`python CoreModel/plot_msis_residuals.py`. Four panels: (a) histogram with the
mean/median marked against a same-moment normal, (b) Q–Q against a normal,
(c) yearly median against F10.7, (d) median against ap with the sign flip.

### 10.1 Does the log transform still make sense?

Yes, and the measurements above are the argument for it rather than against it.

1. **The error is multiplicative, so the log target is the right one.** MSIS is
   wrong by a *factor* (×0.64 in 2009, ×1.15 in storms), not by an additive
   amount. Density itself spans two decades across the mission, so a model
   trained on `rho_obs − rho_msis` would let solar-maximum samples dominate the
   loss entirely and ignore the minimum. In log space a 30 % error costs the
   same wherever it occurs. The reviewer's own framing — deviations quoted in
   **per cent** — is a multiplicative statement.

2. **The log transform was never justified by log-normality, and does not need
   to be.** Least-squares on a log target is consistent under far weaker
   conditions than normal residuals; the Gauss–Markov argument needs only finite
   variance, and gradient boosting makes no distributional assumption at all.
   Log-normality would matter if we were quoting analytic confidence intervals
   from the fitted sd — we are not; the reported metrics are empirical
   (log-RMSE, MAPE, top-5 %).

3. **What must change is the manuscript's wording, not the method.** Drop the
   claims that the residual is zero-centred and log-normal, state the measured
   moments, and note that the metrics are distribution-free. If a spread
   statistic is wanted alongside them, quote the empirical IQR or the median
   absolute deviation rather than anything that assumes normality.

4. **The skew is itself physical, not a defect.** The heavy left tail is the
   deep-minimum population where MSIS is worst (2008–2009 sits near −0.45 with
   its own skew of −0.44 to −0.71). Those samples are the ones the correction
   model has most to fix, so the asymmetry is signal about where the baseline
   fails, not noise to be transformed away.

---

## 11. Results

Model: `xgb_model_v8_storm_ap_2002train.json` — 17 features (single t−3 h TEC
lag plus the storm-history ap drivers; `TEC_LAGS=3h`, `AP_HISTORY=1`), tuned
hyperparameters from `tuning_v13_tec3h_depth3_10/best_params.json`.

Warm-start fine-tunes on the preceding **3 days** at each rolling step
(`ONTRACK_LOOKBACK_DAYS`, §9). Aggregate skill is nearly flat over 3–7 days;
3 is chosen for transition behaviour, where a longer lookback carries more
quiet-day history into a rising storm and overshoots.

**Source of record:** `runs_final_20250821/summary_metrics.csv` — 12 rows
(3 regimes × dr0/dr1 × h ∈ {1, 3}), each carrying its own `msis_*` baseline
columns, plus `table_regimes_h1.tex` / `_h3.tex`. Quote numbers from that file
only; the other `runs_*/` directories hold parameter sweeps.

**All three regimes are held out.** quiet-2009 and storm-2015 are excluded from
training via `TIME_EXCLUDE` (2,672,575 and 773,417 rows — see §6); post-2016
lies past `TRAIN_TIME_MAX`. Internal train/val/test splits of the training
period are not reported here.

**Rolling evaluation** (log-RMSE / MAPE against the MSIS baseline *for the same
period*; dr0 = core model only, dr1 = warm-start with daily fine-tuning):

**h = 1 day**

| Regime | Window | MSIS | dr0 | dr1 |
|---|---|---|---|---|
| quiet-2009 | 2009-01-01 → 06-06 | 0.573 / 72.8 % | 0.271 (−53 %) / 23.5 % | 0.214 (**−63 %**) / 17.2 % |
| storm-2015 | 2015-03-01 → 04-15 | 0.188 / 13.8 % | 0.148 (−21 %) / 10.9 % | 0.131 (**−30 %**) / 9.5 % |
| post-2016 | 2016-01-01 → end | 0.243 / 20.9 % | 0.191 (−22 %) / 15.2 % | 0.159 (**−35 %**) / 12.1 % |

**h = 3 days**

| Regime | Window | MSIS | dr0 | dr1 |
|---|---|---|---|---|
| quiet-2009 | 2009-01-01 → 06-06 | 0.575 / 73.3 % | 0.271 (−53 %) / 23.6 % | 0.224 (**−61 %**) / 18.1 % |
| storm-2015 | 2015-03-01 → 04-15 | 0.188 / 13.7 % | 0.147 (−22 %) / 10.7 % | 0.138 (**−27 %**) / 9.9 % |
| post-2016 | 2016-01-01 → end | 0.243 / 20.9 % | 0.191 (−22 %) / 15.2 % | 0.168 (**−31 %**) / 12.9 % |

Both variants beat MSIS in every regime at both horizons, and warm-start
improves on the static core model throughout — most strongly where the
correction has most to do (quiet-2009, deep solar minimum). This answers the
reviewer's doubt about the value of warm-start.

Extending the horizon costs the core model nothing (dr0 is flat h=1 → h=3: it
does not adapt, so the horizon only changes which rows are scored) and costs
warm-start a few percent, as the forecast reaches further from its last
fine-tune. Warm-start still beats the core model at h = 3 in every regime.

**R² against MSIS.** In quiet-2009 MSIS scores **−0.871** (h = 1) — worse than
predicting the mean — because it runs a systematic ~1.8× high through the deep
minimum. The correction lifts this to **0.873** (dr1). Storm-2015 goes
0.787 → 0.909 and post-2016 0.787 → 0.903. Quote R² as an absolute change, not
a percentage: with a negative baseline the relative figure is meaningless.

A solar-maximum regime (`y2002`) is available via `ONTRACK_FILTERS=y2002` but
is not part of the current results file.

**March 2015 G4 main phase (17–18 March), h = 1.** The main-phase daily trace
is not yet generated; the whole-window storm-2015 figures are in the table
above.

> Do **not** quote R² for the two main-phase days: MSIS's own R² there is near
> zero, so the percentage change is dominated by the denominator and the
> figures the script prints are meaningless. Use RMSE_log.

**Suggested Figure 5 replacement:** plot the daily trace (RMSE_log for MSIS /
dr0 / dr1) with ap on a secondary axis, so the storm spike and the recovery
behaviour appear in a single panel.

---

## 12. Open items

### 12.1 The framing decision (blocks several others)

The reviewer asked to remove the forecasting and warm-start sections and focus
on one task. The h = 1 rolling protocol is **fully causal** — fine-tuning uses
only the 5 preceding days, and prediction at time *t* uses drivers at *t*, which
are observations, not forecasts. It is therefore better described as **adaptive
nowcasting with daily recalibration** than as forecasting.

If the paper is reframed that way, the structural objection is answered without
deleting results, and multi-day horizons move to the follow-up paper. **This
determines whether h = 3 results matter at all**, so decide before spending more
compute.

If the section is kept under a new name, the response letter must say so plainly
— "we have not removed it; we believe the concern was the mixing of two tasks,
which we resolved by…" — and offer to remove it if the editor prefers. Renaming
without disclosure would read as evasion.

One honest caveat for the nowcast claim: it depends on product latency. CODE
final TEC arrives days late, so an operational nowcast would use rapid/NRT
products of slightly lower quality. One sentence covers it.

### 12.2 Remaining work

1. **Re-tune after the feature set is final.** The completed search used the
   15-feature set. A 200-trial run was started and died at 38 trials,
   overwriting the earlier 48-trial CSV; the winning configuration is unchanged,
   but rerun to completion for the manuscript figures.
2. **Adopt tuned hyperparameters.** Every result in §11 uses the *published*
   configuration — the one that ranked last. Nothing has been rerun with
   `USE_TUNED=1`.
3. **Evaluate the ap-history variant.** Features implemented and verified, but
   the `AP_HISTORY=1` storm evaluation had not completed when this was written.
   Open question: do integrated-heating features improve 17–18 March, on top of
   the gain the revised fine-tuning setup already delivers?
4. **Storm-weighted training** — weight samples by ap so extreme conditions are
   not swamped by the quiet majority. Targets storm performance directly, unlike
   the hyperparameter search. Candidate for the follow-up paper.
5. **Manuscript corrections** — TEC lag description, now a single t−3 h lag (§4), log-residual centring
   (§10), the arbitrary-epoch note (§3.4), and the Ap/ap notation inconsistency
   (`ap` is 3-hourly, `Ap` its daily average; the MSIS runs use daily-Ap mode).

### 12.3 The persisted-driver script

`on_track_persisted.py` (plus its table script) addresses a different concern:
the published "forecast" used **observed** drivers inside the forecast window,
which are unknown at issue time, making the numbers a perfect-driver upper
bound. It replaces drivers in the forecast window with issue-time values (TEC
persisted via nearest lat/LST geometry; F10.7 and ap held constant).

A further variant that also re-ran NRLMSIS-2.1 with persisted drivers — so that
both the ML prediction and the baseline were genuine issue-time products — has
been removed. Note the consequence: with the MSIS scaffold still built from
observed drivers, some future driver knowledge leaks in through the baseline
that the correction multiplies.

Findings: under fully operational conditions the model keeps large gains in quiet
periods (−48 to −56 % RMSE_log) and a modest 1-day gain when disturbed (−6 %),
but 3-day storm forecasts converge to the persistence baseline. A per-day
breakdown showed ML beating the baseline on 62 % of calm-driver days but only
52 % of the most storm-affected days.

**These are parked for the follow-up forecasting paper** and are not part of the
nowcast revision. They are documented here so the work is not lost or repeated.

### 12.4 Verification status

Verified by measurement: chunked merge reproduces the original exactly (§3.3);
time-based lags match row-shifts on gap-free data (§4); ap columns are genuinely
lagged and null-free (§5); feature lists and scaler column order match exactly
between `config.py` and `on_track.py` in all three `AP_HISTORY` modes;
`load_and_engineer` completes on the full mission (69.4 M rows, peak 16.5 GB).

**Not** verified end to end: a complete `train,eval` run with `AP_HISTORY=1`, and
the ap-history ontrack evaluation.

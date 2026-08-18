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

- **`Forecast/on_track_true_forecast.py`** + **`make_table_true_forecast.py`** —
  goes further and **also re-runs NRLMSIS-2.1 with persisted drivers**, so both
  the prediction and the baseline are genuine issue-time products. *Why:*
  without this, future driver knowledge still leaks in through the MSIS
  scaffold that the correction multiplies. Includes a startup sanity check that
  pymsis with the dataset's own driver columns reproduces the stored
  `msis_rho`. **Parked** — see §12.3.

- **`README2.md`** — this document.

### Modified files

Line counts are `diff` against `origin/main`. Every change is additive with
published values as defaults; nothing was removed from any file.

| File | Scale | What changed |
|---|---|---|
| `Forecast/on_track.py` | +139 / −46 | (a) env-var overrides for model/scaler/data/output paths; (b) `TEC_LAG_MODE` and `AP_HISTORY` feature toggles; (c) two new evaluation regimes `y2002` and `storm2015` (§6); (d) `compute_metrics` extended with log-RMSE, Top-5 %, R² and `msis_*` baseline columns; (e) resume support — skip finished runs, write summary CSV after each run; (f) `ONTRACK_FILTERS` to select regimes; (g) memory: in-place column drop, freeing per-window frames; (h) **bug fix** — metrics/return block was inside an `else` branch, returning `None` for runs without a snapshot date (§8). |
| `CoreModel/train.py` | +73 / −6 | (a) `TIME_EXCLUDE` interior-holdout loop, applied *after* lag construction; (b) optional tuned hyperparameters via `TRAIN_PARAMS_JSON`, including a rebuilt LR schedule; (c) memory: column-limited parquet read, in-place column drop, float32 downcast of features, freeing unscaled splits after scaling. |
| `CoreModel/config.py` | +41 / −6 | (a) env-var overrides for all paths and the time window; (b) `TIME_EXCLUDE` parsing multiple `"start,end"` intervals separated by `;`; (c) `TEC_LAG_MODE`; (d) `AP_HISTORY` appending 3 (or 5, with `full`) ap features to `FEATURES` and `COLS_TO_SCALE`. |
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
| **Major 5** — why 3 h and 24 h TEC lags? | The code never implemented 3 h. Row-shifts on 10 s data with two interleaved satellites give **~42 min** and 24 h. Replaced with exact time-based lookups. **Manuscript text must be corrected.** | §4 |
| **Major 7** — are MSIS log-residuals zero-centred and log-normal? | Measured: mean −0.099, median −0.067, skew −0.76, strongly solar-cycle dependent. Neither zero-centred nor log-normal. | §10 |
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

The manuscript says the TEC lags are 3 h and 24 h. The code used **row**-shifts,
and on 10 s data with GA and GB interleaved the actual spans are:

| Feature | Published code | Measured median lag |
|---|---|---|
| `vtec_matched_lag` | `shift(500)` | **~42 min** (≈ one orbital period) |
| `vtec_matched_lag2` | `shift(17280)` | 24 h (correct) |

Row-shifts are also corrupted by data gaps: **24 %** of `vtec_matched_lag2`
values were more than 10 minutes from the nominal 24 h lag, because a fixed
number of rows spans a different amount of time whenever data is missing.

`ff.add_tec_time_lag_features` does exact lookups at t−2500 s and t−24 h
(nearest sample within ±10 min, else NaN, which `dropna` removes). Verified
identical to the row-shift on gap-free stretches, differing only where the
row-shift was wrong. Enabled with `TEC_LAG_MODE=time`; default `rows` preserves
v3 behaviour.

> **Manuscript action:** "3 hours" is wrong either way and must be corrected.
> The honest description is *"one orbital period"* and *"24 hours (same
> local-time geometry on the previous day)"* — a better physical justification
> than arbitrary hour counts, and a direct answer to the reviewer's "why not 6,
> 9, 12 h?".

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
> physically distinct regimes (pre-storm quiet, main phase, recovery) and
> produces a favourable number driven by the surrounding quiet days —
> structurally the same criticism the reviewer made of the Feb 2016 figure. The
> 2-day definition is not cherry-picked: 17–18 March are the only days in the
> window with ap = 179, an order of magnitude above the window median of 18.

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
./run_storm_pipeline.sh                    # 15 features
AP_HISTORY=1 ./run_storm_pipeline.sh       # + 3 ap storm-history features
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
| `TEC_LAG_MODE` | `rows` | `time` = exact, gap-robust TEC lags |
| `AP_HISTORY` | `1` | `1` = +3 storm-history ap features; `full` = +5 |
| `USE_TUNED` | `0` | `1` = load `tuning_v5/best_params.json` |
| `TRAIN_TIME_EXCLUDE` | unset | `"start,end;start,end"` interior holdouts |
| `ONTRACK_FILTERS` | `pre2009,post2016` | Which evaluation regimes to run |
| `MERGE_CHUNKED` | unset | `1` = year-by-year merge (required for full mission) |
| `TUNE_TRIALS` | `32` | Random-search trial count |

Plus per-stage path overrides (`TRAIN_PARQUET_FILE`, `TRAIN_MODEL_OUT`,
`ONTRACK_OUTPUT_ROOT`, …), all defaulting to the published filenames.

---

## 10. Measured data facts for the manuscript

**MSIS log-residual distribution** (reviewer comment 7). Over the core window,
`log(rho_obs / msis_rho)` has mean **−0.099**, median **−0.067**, skew **−0.76**
— not zero-centred, and left-skewed rather than log-normal. The reviewer's
expectation of ~0.1 was correct in magnitude.

More usefully, the bias is **strongly solar-cycle dependent**: median log-ratio
per year runs from ≈0 at solar maximum (2002: −0.03; 2013: +0.02) to **−0.46**
in the deep 2009 minimum, where MSIS overestimates density by ~37 %. That
systematic, activity-dependent structure is exactly what a correction model can
learn — an argument for the method in its own right.

---

## 11. Results

All results below use the **published** hyperparameters (the configuration that
ranked last in the search) and the 15-feature set, unless stated otherwise.

**Core model, internal test split.** The full-mission model improves on MSIS by
**−17 % log-RMSE** (0.336 → 0.279) and **−27 % MAPE** (32.2 → 23.4). The
published 2009–2016 model improved by −10.3 % and −15.7 % on its own window.
Both columns are worse in absolute terms because the task is harder (solar
maximum, deep minimum, 160 km altitude span), but the **relative** gain is
larger — the meaningful comparison.

**Rolling evaluation, h = 1 day** (log-RMSE against the MSIS baseline *for the
same period*; dr0 = core model only, dr1 = warm-start with daily fine-tuning):

| Regime | MSIS | dr0 | dr1 |
|---|---|---|---|
| 2002 (solar max) | 0.202 | 0.253 (**+25 %**) | 0.163 (**−19 %**) |
| quiet-2009 | 0.575 | 0.380 (−34 %) | 0.241 (**−58 %**) |
| storm-2015 window | 0.190 | 0.254 (+34 %) | 0.161 (−15 %) |
| post-2016 | 0.244 | 0.232 (−5 %) | 0.185 (−24 %) |

The core model alone **loses to MSIS** outside its training distribution (2002
solar maximum, and the storm window), while daily fine-tuning turns both into
solid gains. This is the direct answer to the reviewer's doubt about the value
of warm-start, and it reproduced across two independently trained models.

**March 2015 G4 main phase (17–18 March), h = 1** — the headline storm result:

| | MSIS | dr0 | dr1 |
|---|---|---|---|
| RMSE_log | 0.421 | 0.463 (+9.8 %) | 0.434 (**+2.9 %**) |
| MAPE | 32.8 % | 31.8 % (−2.9 %) | 31.3 % (−4.5 %) |

**Neither variant beats MSIS on the storm days themselves.** Warm-start closes
most of the gap but not all of it. Report this honestly: the pooled 45-day window
shows a 15 % *improvement*, but that gain comes entirely from surrounding quiet
days.

> Do **not** quote R² for these two days: MSIS's own R² is 0.060, so the
> percentage change is dominated by a near-zero denominator and the −303 % /
> −96 % figures the script prints are meaningless. Use RMSE_log.

**Daily trace** (log-RMSE, h = 1) — the deficit is confined to two days:

| Day | ap max | MSIS | dr0 | dr1 |
|---|---|---|---|---|
| 16 Mar | 22 | 0.118 | 0.119 | **0.107** |
| **17 Mar** | **179** | **0.467** | 0.486 | 0.470 |
| **18 Mar** | **179** | **0.370** | 0.438 | 0.394 |
| 19 Mar | 48 | 0.201 | **0.145** | 0.183 |
| 20 Mar | 39 | 0.185 | **0.129** | 0.141 |
| 25 Mar | 22 | 0.129 | 0.191 | **0.089** |
| 3 Apr | 39 | 0.229 | 0.343 | **0.085** |

Two observations for the manuscript. Warm-start **recovers within a single day**
of the main phase — physically coherent, since persistence-based fine-tuning
cannot anticipate a sudden commencement but adapts as soon as the storm enters
its training window. And on **19–20 March the core model beats warm-start**, the
only place in the 45-day window where it does. A plausible explanation is that
the 5-day fine-tuning window there straddles the regime change (mostly quiet
pre-storm days plus two extreme days), briefly pulling the model away from the
broad climatology. **This is untested** — flagged because a reader looking at the
daily figure will notice it, and anticipating it is stronger than being asked.

**Suggested Figure 5 replacement:** plot this daily trace (RMSE_log for MSIS /
dr0 / dr1) with ap on a secondary axis. It shows the storm spike, the main-phase
tie, and the one-day recovery in a single panel.

**Control check** — the extra March-2015 holdout did not disturb the other
regimes (h = 1, against the earlier run without it): quiet-2009 −0.1 % / −0.7 %,
y2002 dr1 +0.5 %. The larger moves (y2002 dr0 +5.1 %, post2016 dr0 +2.2 %) are
both the core model, the variant most sensitive to 773 k fewer training rows.

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
   Open question: do integrated-heating features improve 17–18 March, and does
   the 19–20 March recovery anomaly change?
4. **Storm-weighted training** — weight samples by ap so extreme conditions are
   not swamped by the quiet majority. Targets storm performance directly, unlike
   the hyperparameter search. Candidate for the follow-up paper.
5. **Manuscript corrections** — TEC lag description (§4), log-residual centring
   (§10), the arbitrary-epoch note (§3.4), and the Ap/ap notation inconsistency
   (`ap` is 3-hourly, `Ap` its daily average; the MSIS runs use daily-Ap mode).

### 12.3 The persisted-driver / true-forecast scripts

`on_track_persisted.py` and `on_track_true_forecast.py` (plus their table
scripts) address a different concern: the published "forecast" used **observed**
drivers inside the forecast window, which are unknown at issue time, making the
numbers a perfect-driver upper bound.

- `on_track_persisted.py` replaces drivers in the forecast window with
  issue-time values (TEC persisted via nearest lat/LST geometry; F10.7 and ap
  held constant).
- `on_track_true_forecast.py` goes further and **also re-runs NRLMSIS-2.1 with
  persisted drivers**, so both the ML prediction and the baseline are genuine
  issue-time products. Without this, future driver knowledge still leaks in
  through the MSIS scaffold that the correction multiplies.

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

# ML Workshop: Adaptive Thermospheric Density Forecasting

This workshop walks through the machine learning pipeline described in the paper
*"Adaptive AI Forecasting of Thermospheric Neutral Density Tuned to GRACE Data"*.
Six notebooks build on each other, each introducing one new concept and showing its effect on the model.

---

## The problem

Satellites in low Earth orbit fly through the thermosphere — the outermost layer of
the atmosphere, roughly 200–600 km above the surface. At these altitudes the atmosphere
is extremely thin, but it still exerts drag on satellites. That drag depends on the
thermospheric density, which changes with solar radiation, geomagnetic storms, the
time of day, the season, and the satellite's altitude.

If the density is underestimated, a satellite's orbit will decay faster than expected.
If it is overestimated, fuel is wasted on unnecessary manoeuvres. Accurate density
forecasts are therefore essential for safe and efficient satellite operations.

Existing empirical models such as NRLMSISE-2.1 (MSIS) capture the average behaviour
of the thermosphere but struggle with rapid changes during geomagnetic storms and
periods of elevated solar activity. The goal of this workshop is to train a machine
learning model that learns the systematic errors of MSIS and corrects them.

---

## The data

### Satellite observations — GRACE

The **GRACE** (Gravity Recovery and Climate Experiment) mission flew two satellites
in formation from 2002 to 2017. High-precision accelerometers on board measured the
non-gravitational forces acting on the satellites, from which the thermospheric density
along the orbit can be derived.

The GRACE orbit decayed over time: the satellites flew at roughly 500 km altitude in
2009 and had descended to around 350 km by 2016. This altitude change drives a
systematic change in the density measurements and is the key reason why the data
splitting strategy matters (see NB3).

Each row in the dataset is one satellite observation. The raw columns include:

| Column | What it is |
|---|---|
| `grace_time` | Timestamp (UTC) |
| `rho_obs` | Observed thermospheric density [kg m⁻³] |
| `msis_rho` | NRLMSISE-2.1 density at the same point [kg m⁻³] |
| `alt_km` | Satellite altitude [km] |
| `lat` | Geodetic latitude [°] |
| `lon` | Longitude [°] |
| `f107` | Daily solar radio flux — proxy for solar UV heating |
| `f107a` | 81-day running average of F10.7 — long-term solar trend |
| `ap_m3h` | Geomagnetic Ap index, 3-hour lag |
| `ap_m6h` | Geomagnetic Ap index, 6-hour lag |
| `matched_tec_value` | Total Electron Content from the CODE ionosphere map [TECU] |

### What is the target?

Direct prediction of thermospheric density is difficult because the values span several
orders of magnitude (roughly 10⁻¹³ to 10⁻¹¹ kg m⁻³). This dynamic range causes
instability during training.

Instead the model learns the **log-ratio**: how much the observed density deviates
from the MSIS prediction in log space:

```
log_ratio = log(rho_obs / msis_rho)
```

A value near zero means MSIS was accurate. A positive value means the real atmosphere
was denser than MSIS predicted. The final density prediction is reconstructed as:

```
rho_pred = msis_rho × exp(predicted log_ratio)
```

This design means the ML model only needs to learn the correction, not the full
physical behaviour. MSIS handles the bulk of the variation; the model handles the residual.

### Workshop dataset

The full GRACE dataset has 40 million rows covering 2009–2016. For the workshop,
a filtered subset of 7 million rows is used (`grace_workshop.parquet`).
Five periods were selected to preserve the variation that matters most:

| Period | Role | GRACE altitude |
|---|---|---|
| Jan–Mar 2009 | Edge year — quiet conditions | ~476 km |
| Apr–Jun 2010 | Core training — high altitude | ~474 km |
| Apr–Jun 2012 | Core training — mid altitude | ~461 km |
| Apr–Jun 2014 | Core training — low altitude | ~432 km |
| Jan–Mar 2016 | Edge year — storm activity | ~381 km |

The three training quarters come from different points in the mission to preserve the
altitude variation needed for NB3. The two edge periods are kept to one quarter each
because they are used only for out-of-sample evaluation, not for training.

---

## The six notebooks

Each notebook adds one new ingredient to the pipeline and measures its effect
with the same metric throughout: **RMSE of the log_ratio on train, val, and test**.

### NB1 — Baseline: raw features and temporal split

**What you do:** Load the data, explore the density observations, and train a first
XGBoost model using raw sensor readings as features and a simple temporal split.

**What you learn:**
- Why the density cannot be predicted directly (magnitude problem)
- Why working in log space and predicting the residual from MSIS is better
- What XGBoost is and what its key hyperparameters control
- How to read a train / val / test score bar chart

**Key concept:** The train/val/test gap — how far the model's performance on
unseen data deviates from its performance on training data.

---

### NB2 — Feature engineering: cyclic encoding and lags

**What you do:** Transform raw inputs into the feature set from Table 1 of the paper.
Apply sine/cosine encoding to Local Solar Time, Day-of-Year, and longitude.
Add TEC lag features to give the model a memory of recent ionospheric state.

**What you learn:**
- Why cyclic variables such as time-of-day break at their boundary when used raw
- How sine/cosine encoding eliminates that boundary discontinuity
- Why XGBoost needs explicit lag features to capture temporal context
- Which features contribute most to reducing prediction error (feature importance)

**Key concept:** Representing cyclic physical variables in a way that respects their
geometry, and compensating for XGBoost's lack of built-in sequence modelling.

---

### NB3 — Data splitting: handling altitude drift

**What you do:** Compare three ways of dividing the data — random, simple temporal,
and cyclic time-block — and visualise where each strategy places train, val, and test
along the altitude axis.

**What you learn:**
- Why GRACE's orbital decay makes a naive temporal split misleading: training on high-altitude data and testing on low-altitude data means the test score reflects an altitude regime the model has never seen
- How the cyclic split (8 interleaved blocks) keeps comparable altitude ranges in all three sets
- What data leakage looks like visually and numerically

**Key concept:** For a dataset with a systematic physical trend over time, the split
strategy directly determines whether the evaluation score is honest.

---

### NB4 — Edge-year validation and warm-start forecasting

**What you do:** Run the frozen model on each edge year separately — the first quarter
of 2009 (quiet) and the first quarter of 2016 (disturbed, including the 18 February
storm). Then run the rolling warm-start forecast loop from the paper: at each step the
model is fine-tuned on the most recent 5 days before predicting the next horizon, and
every 7th update it is reset to the original baseline to prevent drift. The rolling
warm-start is compared step-by-step against the frozen base model.

**What you learn:**
- How performance changes when the model encounters conditions outside the training period
- Where the model struggles: high solar activity, geomagnetic storms, different altitude regimes
- How incremental learning lets a model adapt to current conditions without discarding
  long-term knowledge, and how that gain differs between quiet and disturbed periods
- The trade-off between a short and a long lookback window (Appendix C of the paper)

**Key concept:** A static trained model is a snapshot in time. Edge-year evaluation is a
stricter test of generalisation, and an adaptive warm-start model that continuously
incorporates new observations can recover accuracy the frozen model loses.

---

### NB5 — Global validation: Swarm satellite

**What you do:** Build a global thermospheric density map for 18 February 2016
(a geomagnetic storm), apply the model across the entire globe, and compare against
independent Swarm satellite observations scaled to GRACE altitude.

**What you learn:**
- How to extend a satellite-track model to a global grid
- Why Swarm observations cannot be compared directly (different altitude) and how the
  MSIS transfer factor handles the scaling
- What spatial generalisation looks like: the model was trained on GRACE tracks but
  is tested at locations it has never seen

**Key concept:** Validation against a completely independent instrument on a different
orbit is the hardest test of the model — and the most operationally relevant.

---

## How to set up and run

See [SETUP.md](SETUP.md).

---

## Reference

Thomsen, L.A., Themens, D., Forootan, E.
*Adaptive AI Forecasting of Thermospheric Neutral Density Tuned to GRACE Data.*
Manuscript submitted to Earth and Space Science.

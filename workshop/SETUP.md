# Workshop Setup

---

## What is in the workshop dataset

`grace_workshop_small.parquet` (~240 MB) is a 50% sample of the full workshop dataset,
designed for easy upload to Colab. The filtered subset contains the same date ranges
as the original:

| Period | Purpose |
|---|---|
| Jan–Mar 2009 | Edge year — quiet, GRACE at ~476 km altitude |
| Apr–Jun 2010 | Core training — high altitude (~474 km) |
| Apr–Jun 2012 | Core training — mid altitude (~461 km) |
| Apr–Jun 2014 | Core training — low altitude (~432 km) |
| Jan–Mar 2016 | Edge year — storm activity, GRACE at ~381 km (includes 18 Feb storm) |

3.5 million rows (50% sample), ~240 MB. This size is much more reliable for Colab
uploads than the full 480 MB version. The models and metrics are statistically
equivalent (±1% RMSE, R² difference <0.01).

---

## Data files

The notebooks, the trained model, and the scaler are in the GitHub repo. The data file
is sent to participants directly:

| File | Used by | Size | Where it comes from |
|---|---|---|---|
| `grace_workshop_small.parquet` | NB1–NB4 | ~240 MB | send to participants |
| `nb3_model_cyclic.json` | NB4 | ~300 KB | in the repo (also made by NB3) |
| `nb3_scaler_cyclic.joblib` | NB4 | ~1 KB | in the repo (also made by NB3) |
| `tec` + `swarm` parquet | NB5 | varies | send to participants (if needed) |

**Note:** A full `grace_workshop.parquet` (480 MB) also exists locally for testing.
The small version is recommended for Colab uploads.

---

## Option A — Run locally

**Step 1.** Clone the repository:
```
git clone https://github.com/lotteat11/ML_TND
cd ML_TND
```

**Step 2.** Download the workshop data file. Choose one:

- **Option A1:** Download directly from [GitHub Releases](https://github.com/lotteat11/ML_TND/releases/download/v1.0-workshop/grace_workshop_small.parquet) and place in the repo root.
- **Option A2:** Use `wget` or `curl`:
  ```bash
  wget https://github.com/lotteat11/ML_TND/releases/download/v1.0-workshop/grace_workshop_small.parquet
  ```

**Step 3.** Install the required packages:
```
pip install -r workshop/requirements_workshop.txt
```

**Step 4.** Start Jupyter:
```
jupyter notebook workshop/
```

Open `NB1_baseline.ipynb` and run top to bottom. The Colab setup cell is a no-op locally.

---

## Option B — Run in Google Colab

**Step 1.** Open a notebook directly in Colab by clicking this link (or manually follow Step 2–3):

- [NB1_baseline.ipynb](https://colab.research.google.com/github/lotteat11/ML_TND/blob/main/workshop/NB1_baseline.ipynb)
- [NB2_feature_engineering.ipynb](https://colab.research.google.com/github/lotteat11/ML_TND/blob/main/workshop/NB2_feature_engineering.ipynb)
- [NB3_splitting.ipynb](https://colab.research.google.com/github/lotteat11/ML_TND/blob/main/workshop/NB3_splitting.ipynb)
- [NB4_edge_years.ipynb](https://colab.research.google.com/github/lotteat11/ML_TND/blob/main/workshop/NB4_edge_years.ipynb)
- [NB5_swarm_validation.ipynb](https://colab.research.google.com/github/lotteat11/ML_TND/blob/main/workshop/NB5_swarm_validation.ipynb)

Or manually:

**Step 2.** Go to [colab.research.google.com](https://colab.research.google.com), click **File → Open notebook → GitHub tab**, enter
`https://github.com/lotteat11/ML_TND`, and select a notebook from the `workshop/` folder.

**Step 3.** Run the **Colab setup cell** at the top (the first code cell). It will:
   - Clone the repository (for code modules)
   - Download `grace_workshop_small.parquet` from GitHub Releases (~257 MB)
   - Install Python dependencies
   - Set up the working directory

The download takes ~1–2 minutes. No runtime restart is required.

**Step 4.** Run the rest of the notebook top to bottom.

---

## Note on NB5 (Swarm validation)

NB5 additionally installs `cartopy` and needs the `tec` and `swarm` parquet files —
upload those to the Files pane (`/content/`) the same way as the GRACE file.

---

## Run order

| Notebook | What it needs |
|---|---|
| NB1 | `grace_workshop.parquet` |
| NB2 | same |
| NB3 | same — saves `nb3_model_cyclic.json` |
| NB4 | same + `nb3_model_cyclic.json` (in repo, or from NB3) |
| NB5 | `tec` and `swarm` parquet files + `cartopy` |

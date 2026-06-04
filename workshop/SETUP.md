# Workshop Setup

---

## What is in the workshop dataset

`grace_workshop.parquet` (~480 MB) is **too large for GitHub** and is hosted on
Google Drive instead (see the data-hosting section below). It is a filtered subset
of the full GRACE mission data:

| Period | Purpose |
|---|---|
| Jan–Mar 2009 | Edge year — quiet, GRACE at ~476 km altitude |
| Apr–Jun 2010 | Core training — high altitude (~474 km) |
| Apr–Jun 2012 | Core training — mid altitude (~461 km) |
| Apr–Jun 2014 | Core training — low altitude (~432 km) |
| Jan–Mar 2016 | Edge year — storm activity, GRACE at ~381 km (includes 18 Feb storm) |

7 million rows, ~500 MB.

---

## Data hosting (one-time, by the maintainer)

The data and model files are **gitignored** (`*.parquet`, `*.json`, `*.joblib`), so
they are not in the GitHub repo. They live on Google Drive:

| File | Used by | Size |
|---|---|---|
| `grace_workshop.parquet` | NB1–NB4 | ~480 MB |
| `nb3_model_cyclic.json` | NB4 | ~300 KB |
| `nb3_scaler_cyclic.joblib` | NB4 | ~1 KB |
| `tec` + `swarm` parquet | NB5 | varies |

For each file: upload to Drive → **Share → Anyone with the link (Viewer)** → copy the
ID from the link `https://drive.google.com/file/d/<FILE_ID>/view`. Paste each `<FILE_ID>`
into the **Colab setup cell** at the top of the relevant notebook (placeholders marked
`REPLACE_WITH_..._FILE_ID`).

---

## Option A — Run locally

**Step 1.** Clone the repository and place `grace_workshop.parquet` in the repo root
(and the `nb3_*` files in `workshop/` if you want to run NB4 without first running NB3):
```
git clone https://github.com/lotteat11/ML_TND
cd ML_TND
```

**Step 2.** Install the required packages (needs scikit-learn ≥ 1.4 for
`root_mean_squared_error`):
```
pip install -r workshop/requirements_workshop.txt gdown
```

**Step 3.** Start Jupyter:
```
jupyter notebook workshop/
```

Open `NB1_baseline.ipynb` and run top to bottom. The Colab setup cell is a no-op locally.

---

## Option B — Run in Google Colab

**Step 1.** Go to [colab.research.google.com](https://colab.research.google.com)

**Step 2.** Click **File → Open notebook → GitHub tab**, enter
`https://github.com/lotteat11/ML_TND`, and select a notebook from the `workshop/` folder.

**Step 3.** Run the **Colab setup cell** at the top (the first code cell). It clones the
repo, downloads the data from Google Drive via `gdown`, and `chdir`s into `workshop/`.
The Drive `FILE_ID` placeholders in that cell must already be filled in (see *Data
hosting* above). No runtime restart is required.

**Step 4.** Run the rest of the notebook top to bottom.

---

## Note on NB5 (Swarm validation)

NB5 additionally installs `cartopy` and downloads the `tec` and `swarm` parquet files —
this is already handled by NB5's Colab setup cell once you fill in its `TEC_ID` and
`SWARM_ID` placeholders.

---

## Run order

| Notebook | What it needs |
|---|---|
| NB1 | `grace_workshop.parquet` (in repo) |
| NB2 | same |
| NB3 | same — saves `nb3_model_cyclic.json` |
| NB4 | same + `nb3_model_cyclic.json` from NB3 |
| NB5 | `tec` and `swarm` parquet files + `cartopy` |

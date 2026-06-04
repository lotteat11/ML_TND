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

## Data files

The notebooks, the small trained model (`nb3_model_cyclic.json`), and the scaler are in
the GitHub repo. The large data file is **not** — it is sent to participants directly:

| File | Used by | Size | Where it comes from |
|---|---|---|---|
| `grace_workshop.parquet` | NB1–NB4 | ~480 MB | sent to you; you upload it |
| `nb3_model_cyclic.json` | NB4 | ~300 KB | in the repo (also made by NB3) |
| `nb3_scaler_cyclic.joblib` | NB4 | ~1 KB | in the repo (also made by NB3) |
| `tec` + `swarm` parquet | NB5 | varies | sent to you; you upload them |

---

## Option A — Run locally

**Step 1.** Clone the repository and place `grace_workshop.parquet` in the repo root:
```
git clone https://github.com/lotteat11/ML_TND
cd ML_TND
```

**Step 2.** Install the required packages (needs scikit-learn ≥ 1.4 for
`root_mean_squared_error`):
```
pip install -r workshop/requirements_workshop.txt
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

**Step 3.** Upload `grace_workshop.parquet` using the **Files pane** (the folder icon on
the left sidebar → upload button). It lands at `/content/grace_workshop.parquet`.

**Step 4.** Run the **Colab setup cell** at the top (the first code cell). It clones the
repo for the code modules, moves your uploaded parquet into place, and `chdir`s into
`workshop/`. No runtime restart is required.

**Step 5.** Run the rest of the notebook top to bottom.

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

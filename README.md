Adaptive Thermospheric Density Modeling with TEC Coupling
This repository contains the full data processing, modeling, validation, and forecasting pipeline used in the study of adaptive thermospheric neutral density modeling using satellite observations and ionospheric Total Electron Content (TEC).
The code supports:

reproducible data preparation from satellite products,
machine‑learning–based density correction using XGBoost,
rigorous time‑aware validation,
adaptive on‑track and off‑track forecasting experiments.


Repository Structure
.
├── data_preparation/
│   ├── get_grace_swarm.py
│   ├── import_tec.py
│   ├── merge_grace_tec.py
│   └── README.md
│
├── modeling/
│   └── train.py
│
├── validation/
│   └── evaluate.py
│
├── forecasting/
│   ├── run_ontrack_forecast.py
│   └── run_offtrack_forecast.py
│
├── config.py
├── Feature_functions.py
├── environment.yml   (or requirements.txt)
└── README.md


Data Preparation (data_preparation/)
This folder contains the three‑step pipeline used to construct the machine‑learning input dataset.


Satellite density data
get_grace_swarm.py
Downloads and parses thermospheric neutral density (DNS) products from the TU Delft Thermosphere archive (GRACE or Swarm).


TEC data
import_tec.py
Downloads and parses CODE Global Ionosphere Maps (IONEX TEC data) from NASA CDDIS.


GRACE × TEC matching
merge_grace_tec.py
Temporally matches TEC to GRACE observations (±3 hours) and spatially collocates TEC values using a spherical nearest‑neighbor (K‑D tree) approach.


Output:
A merged Parquet dataset used by all downstream modeling steps.

Model Training (modeling/train.py)
This script trains the core XGBoost density‑correction model.
Key characteristics:

Feature engineering (LST, DOY, trigonometric encodings)
Target defined as
log⁡(ρobs/ρMSIS)\log(\rho_{\text{obs}} / \rho_{\text{MSIS}})log(ρobs​/ρMSIS​)

Time‑aware splitting using a repeated time‑block strategy with 7 cycles
Feature and target scaling
Model and scalers saved to disk

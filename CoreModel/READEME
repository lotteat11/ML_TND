# XGBoost Thermospheric Density Correction

Trains an XGBoost model to correct MSIS atmospheric density predictions using GRACE observations and CODE TEC maps. The model predicts `log(ρ_obs / ρ_MSIS)` and recovers corrected density as `ρ_MSIS × exp(prediction)`.

## Files

- `config.py` — paths, feature list, time bounds. Edit here first.
- `losses.py` — custom XGBoost objective functions and learning-rate scheduler.
- `plotting.py` — all plot functions.
- `train.py` — loads data, engineers features, trains and saves the model.
- `evaluate.py` — loads saved model, runs predictions, produces diagnostic plots.

## Usage

```bash
python train.py       # trains and saves model + scalers
python evaluate.py    # evaluates saved model on validation/test set
```

## Dependencies

```
xgboost, scikit-learn, pandas, numpy, matplotlib, seaborn, scipy, joblib, Feature_functions
```

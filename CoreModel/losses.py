# Author: Lotte Ansgaard Thomsen
# Aalborg University
"""
Custom XGBoost loss functions and learning-rate scheduler.
"""

import numpy as np


def mse_extreme_obj(preds, dtrain, threshold: float = 0.5, lambda_extreme: float = 0.6):
    """MSE loss with an additional penalty on large residuals."""
    y = dtrain.get_label()
    r = preds - y
    extreme_mask = np.abs(r) > threshold
    grad = r + lambda_extreme * np.sign(r) * extreme_mask
    hess = np.ones_like(r) * 2
    return grad, hess


def pseudo_huber_obj(preds, dtrain, delta: float = 0.08):
    """Pseudo-Huber loss for XGBoost."""
    y = dtrain.get_label()
    z = preds - y
    scale      = 1.0 + (z / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = z / scale_sqrt
    hess = 1.0 / (scale * scale_sqrt)
    return grad, hess


def pseudo_huber_extreme_obj(preds, dtrain, delta: float = 0.1,
                              threshold: float = 0.3, lambda_extreme: float = 15.0):
    """Pseudo-Huber loss with additional extreme-residual penalty."""
    y = dtrain.get_label()
    r = preds - y
    scale      = 1.0 + (r / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    extreme    = np.abs(r) > threshold
    grad = r / scale_sqrt + lambda_extreme * np.sign(r) * extreme
    hess = 1.0 / (scale * scale_sqrt) + lambda_extreme * extreme.astype(float)
    return grad, hess


def lr_scheduler(current_round: int,
                 initial_lr: float = 5e-3,
                 decay_factor: float = 0.8,
                 step_size: int = 60) -> float:
    """Step-decay learning-rate schedule for XGBoost callbacks."""
    lr = initial_lr * (decay_factor ** (current_round // step_size))
    if current_round % 10 == 0:
        print(f"Round {current_round}: lr = {lr:.8f}", flush=True)
    return lr

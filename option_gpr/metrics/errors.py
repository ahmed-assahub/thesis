"""Pricing error metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def mae(pred: ArrayLike, ref: ArrayLike) -> float:
    """Return the mean absolute error between predictions and references."""

    pred_arr, ref_arr = _validate_pair(pred, ref)
    return float(np.mean(np.abs(pred_arr - ref_arr)))


def max_abs_error(pred: ArrayLike, ref: ArrayLike) -> float:
    """Return the maximum absolute error between predictions and references."""

    pred_arr, ref_arr = _validate_pair(pred, ref)
    return float(np.max(np.abs(pred_arr - ref_arr)))


def mean_relative_error(pred: ArrayLike, ref: ArrayLike, eps: float = 1e-8) -> float:
    """Return mean ``abs(pred - ref) / max(abs(ref), eps)``."""

    pred_arr, ref_arr = _validate_pair(pred, ref)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be positive and finite, got {eps!r}.")
    denominator = np.maximum(np.abs(ref_arr), eps)
    return float(np.mean(np.abs(pred_arr - ref_arr) / denominator))


def _validate_pair(pred: ArrayLike, ref: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    pred_arr = np.asarray(pred, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    if pred_arr.ndim != 1:
        raise ValueError(f"pred must be a 1D array, got shape {pred_arr.shape}.")
    if ref_arr.ndim != 1:
        raise ValueError(f"ref must be a 1D array, got shape {ref_arr.shape}.")
    if pred_arr.shape != ref_arr.shape:
        raise ValueError(
            f"pred and ref must have matching shapes, got {pred_arr.shape} "
            f"and {ref_arr.shape}."
        )
    if not np.all(np.isfinite(pred_arr)):
        raise ValueError("pred must contain only finite values.")
    if not np.all(np.isfinite(ref_arr)):
        raise ValueError("ref must contain only finite values.")
    return pred_arr, ref_arr

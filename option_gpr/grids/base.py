"""Shared grid data containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GridSet:
    """Interior collocation points and boundary observations."""

    X_int: NDArray[np.float64]
    X_bd: NDArray[np.float64]
    y_bd: NDArray[np.float64]

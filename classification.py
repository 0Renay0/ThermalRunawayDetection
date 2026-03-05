import numpy as np
import pandas as pd


# Helpers
def _to_numpy(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype=float)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    window = int(window)
    if window < 1:
        return x
    # limiter les effets de bords
    pad = window // 2
    xpad = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(window, dtype=float) / float(window)

    return np.convolve(xpad, kernel, mode="valid")

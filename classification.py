import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


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


def _safe_gradient(y: np.gradient, x: np.ndarray) -> np.ndarray:
    if len(y) < 3:
        return np.zeros_like(y)

    if np.any(np.diff(x) <= 0):
        idx = np.argsort(x)
        x2 = x[idx]
        y2 = y[idx]
        dy = np.gradient(y2, x2)

        out = np.empty_like(dy)
        out[idx] = dy
        return out

    return np.gradient(y, x)


# --------------------------- critères ---------------------------
@dataclass
class CriterionResult:
    triggered: bool
    score: float  # points ou la condition est vraie
    details: Dict[str, float]


def _get_temperature(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """
    Retourne T en Kelvin et le nom de la colonne source
    """
    if "Tr_K" in df.columns:
        return _to_numpy(df["Tr_K"]), "Tr_K"
    if "Tr_C" in df.columns:
        return _to_numpy(["Tr_C"]) + 273.15, "Tr_C"
    if "Tr_K_meas" in df.columns:
        return _to_numpy(df["Tr_K_meas"]), "Tr_K_meas"
    if "Tr_C_meas" in df.columns:
        return _to_numpy(df["Tr_C_meas"]) + 273.15, "Tr_C_meas"
    raise KeyError(
        "Aucune colonne température trouvée. Attendu: Tr_C / Tr_K (ou *_meas)."
    )


def _get_time(df: pd.DataFrame) -> np.ndarray:
    for col in ("Time", "t", "time", "temps"):
        if col in df.columns:
            return _to_numpy(df[col])
    raise KeyError('Aucune colonne temps trouvée. Attendu: "Time".')


def _infer_conversion(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Essaie d'inférer une conversion x.

    Priorité:
    - colonne "X" ou "Conversion" si présente
    - à partir de CA: x = 1 - CA/CA0
    - à partir de HP: x = 1 - HP/HP0

    Retourne None si impossible.
    """
    for col in ("X", "Conversion", "conversion", "x"):
        if col in df.columns:
            x = _to_numpy(df[col])
            return np.clip(x, 0.0, 1.0)

    if "CA" in df.columns:
        CA = _to_numpy(df["CA"])
        CA0 = CA[0] if CA[0] != 0 else np.nan
        if np.isfinite(CA0) and CA0 > 0:
            x = 1.0 - CA / CA0
            return np.clip(x, 0.0, 1.0)

    if "HP" in df.columns:
        HP = _to_numpy(df["HP"])
        HP0 = HP[0] if HP[0] != 0 else np.nan
        if np.isfinite(HP0) and HP0 > 0:
            x = 1.0 - HP / HP0
            return np.clip(x, 0.0, 1.0)

    return None


def criterion_thomas_bowes(
    t: np.ndarray,
    T: np.ndarray,
    smooth_window: int = 1,
    min_frac: float = 0.001,
) -> CriterionResult:
    """Thomas & Bowes: runaway si dT/dt>0 ET d²T/dt²>0."""
    T_s = _moving_average(T, smooth_window)
    dTdt = _safe_gradient(T_s, t)
    d2Tdt2 = _safe_gradient(dTdt, t)

    mask = (dTdt > 0) & (d2Tdt2 > 0)
    frac = float(np.mean(mask)) if len(mask) else 0.0
    triggered = frac >= float(min_frac)

    return CriterionResult(
        triggered=triggered,
        score=frac,
        details={
            "min_frac": float(min_frac),
            "max_dTdt": float(np.nanmax(dTdt)) if len(dTdt) else 0.0,
            "max_d2Tdt2": float(np.nanmax(d2Tdt2)) if len(d2Tdt2) else 0.0,
        },
    )

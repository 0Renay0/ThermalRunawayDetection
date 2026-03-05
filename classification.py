import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import Config as cfg


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


def criterion_alder_enig(
    x: Optional[np.ndarray],
    T: np.ndarray,
    smooth_window: int = 1,
    min_frac: float = 0.001,
) -> CriterionResult:
    """Adler & Enig: runaway si dT/dx>0 ET d²T/dx²>0."""
    if x is None or len(x) < 3 or np.allclose(x, x[0]):
        return CriterionResult(
            triggered=False,
            score=0.0,
            details={"available": 0.0, "reason": 1.0},
        )

    # S'assurer que x est croissant pour dériver dans le plan T-x
    idx = np.argsort(x)
    x2 = x[idx]
    T2 = _moving_average(T[idx], smooth_window)

    # Retirer les doublons (gradient peut diverger)
    uniq = np.concatenate(([True], np.diff(x2) > 1e-12))
    x2 = x2[uniq]
    T2 = T2[uniq]

    if len(x2) < 3:
        return CriterionResult(
            triggered=False,
            score=0.0,
            details={"available": 0.0, "reason": 2.0},
        )

    dTdx = np.gradient(T2, x2)
    d2Tdx2 = np.gradient(dTdx, x2)

    mask = (dTdx > 0) & (d2Tdx2 > 0)
    frac = float(np.mean(mask)) if len(mask) else 0.0
    triggered = frac >= float(min_frac)

    return CriterionResult(
        triggered=triggered,
        score=frac,
        details={
            "available": 1.0,
            "min_frac": float(min_frac),
            "max_dTdx": float(np.nanmax(dTdx)) if len(dTdx) else 0.0,
            "max_d2Tdx2": float(np.nanmax(d2Tdx2)) if len(d2Tdx2) else 0.0,
        },
    )


def _get_Tw(df: pd.DataFrame, T: np.ndarray) -> np.ndarray:
    """Température de jaquette Tw (K).

    - si Tw_K/Tw_C existe dans le CSV, l'utiliser
    - sinon, essayer cfg.Tj (K) si Config est importable
    - sinon, approx: Tw = T[0]
    """
    if "Tw_K" in df.columns:
        return _to_numpy(df["Tw_K"])
    if "Tw_C" in df.columns:
        return _to_numpy(df["Tw_C"]) + 273.15

    if cfg is not None and hasattr(cfg, "Tj"):
        return np.full_like(T, float(getattr(cfg, "Tj")))

    return np.full_like(T, float(T[0]))


def criterion_hub_jones(
    t: np.ndarray,
    T: np.ndarray,
    Tw: np.ndarray,
    smooth_window: int = 1,
    min_frac: float = 0.001,
) -> CriterionResult:
    """Hub & Jones: runaway si d²T/dt²>0 ET d(T-Tw)/dt>0."""
    T_s = _moving_average(T, smooth_window)
    Tw_s = _moving_average(Tw, smooth_window)

    dTdt = _safe_gradient(T_s, t)
    d2Tdt2 = _safe_gradient(dTdt, t)
    dDelta_dt = _safe_gradient(T_s - Tw_s, t)

    mask = (d2Tdt2 > 0) & (dDelta_dt > 0)
    frac = float(np.mean(mask)) if len(mask) else 0.0
    triggered = frac >= float(min_frac)

    return CriterionResult(
        triggered=triggered,
        score=frac,
        details={
            "min_frac": float(min_frac),
            "max_d2Tdt2": float(np.nanmax(d2Tdt2)) if len(d2Tdt2) else 0.0,
            "max_dDelta_dt": float(np.nanmax(dDelta_dt)) if len(dDelta_dt) else 0.0,
        },
    )

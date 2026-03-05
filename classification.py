import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import Config as cfg
import os
import shutil


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


def criterion_adler_enig(
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


# --------------------------- pipeline scénario ---------------------------


@dataclass
class ScenarioClassification:
    file: str
    label: int
    votes: int
    thomas_bowes: int
    adler_enig: int
    hub_jones: int
    tb_score: float
    ae_score: float
    hj_score: float
    p_max_bar: float
    p_over_100: int


def classify_file(
    path: str,
    smooth_window: int = 1,
    min_frac: float = 0.001,
    p_fault_bar: float = 100.0,
) -> ScenarioClassification:
    df = pd.read_csv(path)

    t = _get_time(df)
    T, _ = _get_temperature(df)
    Tw = _get_Tw(df, T)
    x = _infer_conversion(df)

    # --- Critères thermiques ---
    tb = criterion_thomas_bowes(t, T, smooth_window=smooth_window, min_frac=min_frac)
    ae = criterion_adler_enig(x, T, smooth_window=smooth_window, min_frac=min_frac)
    hj = criterion_hub_jones(t, T, Tw, smooth_window=smooth_window, min_frac=min_frac)

    votes = int(tb.triggered) + int(ae.triggered) + int(hj.triggered)
    label = 1 if votes >= 2 else 0

    # --- Règle pression (override) ---
    try:
        P_bar, _ = _get_pressure_bar(df)
        p_max = float(np.nanmax(P_bar)) if len(P_bar) else float("nan")
        p_over_100 = int(np.isfinite(p_max) and (p_max > p_fault_bar))
        if p_over_100:
            label = 1  # override -> Fault
    except Exception:
        # si pas de colonne pression, on n'applique pas la règle
        p_max = float("nan")
        p_over_100 = 0

    return ScenarioClassification(
        file=os.path.basename(path),
        label=label,
        votes=votes,
        thomas_bowes=int(tb.triggered),
        adler_enig=int(ae.triggered),
        hub_jones=int(hj.triggered),
        tb_score=float(tb.score),
        ae_score=float(ae.score),
        hj_score=float(hj.score),
        p_max_bar=p_max,
        p_over_100=p_over_100,
    )

    def move_file(src: str, dst_dir: str, dry_run: bool = False) -> str:
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src))
        if dry_run:
            return dst
        # Si un fichier du même nom existe déjà, on suffixe
        if os.path.exists(dst):
            base, ext = os.path.splitext(os.path.basename(src))
            i = 1
            while True:
                cand = os.path.join(dst_dir, f"{base}__{i}{ext}")
                if not os.path.exists(cand):
                    dst = cand
                    break
                i += 1
        shutil.move(src, dst)
        return dst


def _get_temperature_C(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """Retourne T en °C (pour le plot) et le nom de la colonne source."""
    if "Tr_C" in df.columns:
        return _to_numpy(df["Tr_C"]), "Tr_C"
    if "Tr_K" in df.columns:
        return _to_numpy(df["Tr_K"]) - 273.15, "Tr_K"
    if "Tr_C_meas" in df.columns:
        return _to_numpy(df["Tr_C_meas"]), "Tr_C_meas"
    if "Tr_K_meas" in df.columns:
        return _to_numpy(df["Tr_K_meas"]) - 273.15, "Tr_K_meas"
    raise KeyError("Aucune colonne température trouvée (Tr_C/Tr_K ou *_meas).")


def _get_pressure_bar(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """Retourne la pression en bar et le nom de la colonne source.

    Supporte plusieurs conventions usuelles.
    """
    # Cas direct en bar (ton cas: Pression_ideal_bar)
    bar_cols = (
        "Pression_ideal_bar",
        "Pression_bar",
        "Pressure_bar",
        "P_bar",
        "Pbar",
        "pressure_bar",
        "pression_bar",
    )
    for col in bar_cols:
        if col in df.columns:
            return _to_numpy(df[col]), col

    # Cas en Pa -> conversion bar
    pa_cols = (
        "Pression_ideal_Pa",
        "Pression_Pa",
        "Pressure_Pa",
        "P_Pa",
        "Ppa",
        "pressure_Pa",
        "pression_Pa",
        "P",
        "Pressure",
        "Pression",
        "pressure",
        "pression",
    )
    for col in pa_cols:
        if col in df.columns:
            p = _to_numpy(df[col])
            # heuristique: si valeurs typiquement > 1e3, on suppose Pa
            # (évite de convertir à tort des données déjà en bar)
            if np.nanmax(p) > 1e3:
                return p / 1e5, col  # 1 bar = 1e5 Pa
            # sinon on suppose déjà en bar
            return p, col

    raise KeyError("Aucune colonne pression trouvée (ex: Pression_ideal_bar).")

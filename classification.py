from __future__ import annotations

import numpy as np
import pandas as pd
import Config as cfg
import os
import shutil
import glob
import argparse

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


def _safe_gradient(y: np.ndarray, x: np.ndarray) -> np.ndarray:
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
    """Retourne T en Kelvin et le nom de la colonne source."""
    if "Tr_K" in df.columns:
        return _to_numpy(df["Tr_K"]), "Tr_K"
    if "Tr_C" in df.columns:
        return _to_numpy(df["Tr_C"]) + 273.15, "Tr_C"
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

    raise KeyError("Aucune colonne pression trouvée.")


def plot_phase_plane_TP(
    nominal_dir: str = "./Nominal",
    fault_dir: str = "./Faults",
    pattern: str = "*.csv",
    out_path: Optional[str] = None,
    show: bool = True,
    max_files_per_class: Optional[int] = None,
    alpha: float = 0.6,
    lw: float = 1.0,
) -> None:
    """Plot du plan de phase Température (°C) vs Pression (bar).

    - Nominal en bleu
    - Faults en rouge
    - Sans legend / label
    """
    import matplotlib.pyplot as plt

    def _list_csv(d: str) -> list[str]:
        paths = sorted(glob.glob(os.path.join(d, pattern)))
        if max_files_per_class is not None:
            paths = paths[: int(max_files_per_class)]
        return paths

    nom_files = _list_csv(nominal_dir)
    flt_files = _list_csv(fault_dir)

    if not nom_files and not flt_files:
        print(
            f"[WARN] Aucun fichier trouvé dans {nominal_dir} ni {fault_dir} (pattern={pattern})"
        )
        return

    fig, ax = plt.subplots()

    # Nominal -> bleu
    for path in nom_files:
        try:
            df = pd.read_csv(path)
            T_C, _ = _get_temperature_C(df)
            P_bar, _ = _get_pressure_bar(df)
            n = min(len(T_C), len(P_bar))
            if n >= 2:
                ax.plot(P_bar[:n], T_C[:n], color="blue", alpha=alpha, linewidth=lw)
        except Exception as e:
            print(f"[WARN] Plot nominal skip {os.path.basename(path)}: {e}")

    # Faults -> rouge
    for path in flt_files:
        try:
            df = pd.read_csv(path)
            T_C, _ = _get_temperature_C(df)
            P_bar, _ = _get_pressure_bar(df)
            n = min(len(T_C), len(P_bar))
            if n >= 2:
                ax.plot(P_bar[:n], T_C[:n], color="red", alpha=alpha, linewidth=lw)
        except Exception as e:
            print(f"[WARN] Plot faults skip {os.path.basename(path)}: {e}")

    ax.set_xlabel("Pression (bar)")
    ax.set_ylabel("Température (°C)")
    ax.set_title("Plan de phase T–P (Nominal=bleu, Faults=rouge)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200)
        print(f"[OK] Figure sauvegardée: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def reset_outputs_to_data_dir(
    data_dir: str, nominal_dir: str, fault_dir: str, pattern: str = "*.csv"
) -> None:
    os.makedirs(data_dir, exist_ok=True)

    for src_dir in (nominal_dir, fault_dir):
        if not os.path.isdir(src_dir):
            continue

        for f in glob.glob(os.path.join(src_dir, pattern)):
            base = os.path.basename(f)

            # Évite de remonter des fichiers résultats éventuels
            if base in {"classification_summary.csv", "index.csv"}:
                continue

            dst = os.path.join(data_dir, base)

            # si conflit de nom dans data_dir, on suffixe
            if os.path.exists(dst):
                b, ext = os.path.splitext(base)
                i = 1
                while True:
                    cand = os.path.join(data_dir, f"{b}__reset{i}{ext}")
                    if not os.path.exists(cand):
                        dst = cand
                        break
                    i += 1

            shutil.move(f, dst)

    for d in (nominal_dir, fault_dir):
        try:
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
        except Exception:
            pass


def main() -> int:
    p = argparse.ArgumentParser(
        description="Classification des scénarios (runaway) avec vote 2/3."
    )
    p.add_argument(
        "--data_dir",
        default="./Data/Simulated",
        help="Dossier contenant les scénarios CSV",
    )
    p.add_argument(
        "--nominal_dir",
        default="./Data/Simulated/Nominal",
        help="Dossier de sortie pour label=0",
    )
    p.add_argument(
        "--fault_dir",
        default="./Data/Simulated/Faults",
        help="Dossier de sortie pour label=1",
    )
    p.add_argument("--pattern", default="*.csv", help="Motif glob des fichiers")
    p.add_argument(
        "--p_fault_bar", type=float, default=100.0, help="Seuil pression (bar) => Fault"
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Avant classification: remonte les CSV de Nominal/Faults vers data_dir (reclassement propre)",
    )
    p.add_argument(
        "--smooth_window",
        type=int,
        default=21,
        help="Fenêtre de lissage (moyenne glissante) en # d'échantillons (>=1)",
    )

    p.add_argument(
        "--min_frac",
        type=float,
        default=0.05,
        help="Fraction minimale de points satisfaisant la condition pour déclencher un critère",
    )

    p.add_argument("--dry_run", action="store_true", help="Ne déplace pas les fichiers")

    p.add_argument(
        "--summary",
        default="classification_summary.csv",
        help="Nom du fichier récapitulatif CSV",
    )

    # ---- Options pour le plot ----
    p.add_argument(
        "--plot_tp",
        action="store_true",
        help="Génère le plot du plan de phase Température-Pression",
    )

    p.add_argument(
        "--plot_out",
        default="phase_plane_TP.png",
        help="Nom du fichier image du plot",
    )

    args = p.parse_args()

    if args.reset and not args.dry_run:
        reset_outputs_to_data_dir(
            data_dir=args.data_dir,
            nominal_dir=args.nominal_dir,
            fault_dir=args.fault_dir,
            pattern=args.pattern,
        )

    pattern = os.path.join(args.data_dir, args.pattern)
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[WARN] Aucun fichier trouvé: {pattern}")
        return 1

    rows = []

    for f in files:
        base = os.path.basename(f)

        # ignorer les fichiers de sortie
        if base in {args.summary, "index.csv"}:
            print(f"[SKIP] {base} (fichier résultat)")
            continue

        # Skip rapide si pas de colonne temps
        try:
            df_test = pd.read_csv(f, nrows=1)
            if not any(c in df_test.columns for c in ["Time", "time", "t", "temps"]):
                print(f"[SKIP] {base} (pas de colonne temps)")
                continue
        except Exception:
            print(f"[SKIP] {base} (lecture impossible)")
            continue

        # Classification
        try:
            res = classify_file(
                f,
                smooth_window=args.smooth_window,
                min_frac=args.min_frac,
                p_fault_bar=args.p_fault_bar,  # règle P>100bar => Fault, cas extreme
            )
        except Exception as e:
            print(f"[ERROR] {base}: {e}")
            continue

        rows.append(res.__dict__)

        target_dir = args.fault_dir if res.label == 1 else args.nominal_dir
        _ = move_file(f, target_dir, dry_run=args.dry_run)

        print(
            f"[{'FAULT' if res.label == 1 else 'NOMINAL'}] {res.file} "
            f"votes={res.votes} (TB={res.thomas_bowes}, AE={res.adler_enig}, HJ={res.hub_jones}) "
            f"Pmax={res.p_max_bar:.2f} bar (P>{args.p_fault_bar:g}={res.p_over_100})"
        )

    # ---- Sauvegarde du résumé ----
    if rows:
        out = pd.DataFrame(rows)
        out_path = os.path.join(args.data_dir, args.summary)

        if args.dry_run:
            print(f"[DRY_RUN] Récapitulatif non écrit: {out_path}")
        else:
            out.to_csv(out_path, index=False)
            print(f"[OK] Récapitulatif écrit: {out_path}")

    # ---- Plot du plan de phase T-P ----
    if args.plot_tp:
        try:
            plot_phase_plane_TP(
                nominal_dir=args.nominal_dir,
                fault_dir=args.fault_dir,
                pattern=args.pattern,
                out_path=args.plot_out,
                show=True,
            )
        except Exception as e:
            print(f"[WARN] Plot T-P impossible: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

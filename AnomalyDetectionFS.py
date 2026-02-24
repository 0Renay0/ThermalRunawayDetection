import pandas as pd
from AnomalyDetection import compute_RCI, make_windows_features


def build_features(
    data: pd.DataFrame,
    time_col="Time",
    T_col="Tr_K",
    P_col="Pression_ideal_bar",
    win=15,
):
    """
    Build features for anomaly detection
    """

    df = data.copy()
    t = df[time_col].to_numpy()
    T = df[T_col].to_numpy()
    P = df[P_col].to_numpy()

    sig = compute_RCI(t=t, T=T, P_bar=P)

    df["dTdt"] = sig["dTdt"]
    df["d2Tdt2"] = sig["d2Tdt2"]
    df["dPdt"] = sig["dPdt"]
    df["d2Pdt2"] = sig["d2Pdt2"]
    df["kappa_TP"] = sig["kappa"]
    df["v_TP"] = sig["v"]
    df["RCI"] = sig["RCI"]

    base_cols = [
        T_col,
        P_col,
        "dTdt",
        "d2Tdt2",
        "dPdt",
        "d2Pdt2",
        "kappa_TP",
        "v_TP",
        "RCI",
    ]
    X = make_windows_features(df, base_cols, win=win)

    tp_corr = df[T_col].rolling(win, center=True, min_periods=win).corr(df[P_col])
    X["TP_corr"] = tp_corr.loc[X.index]

    return df, X

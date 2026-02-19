import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


def compute_RCI(t, T, P_bar, eps=1e-12):
    """
    We compute dT/dt, d2T/dt2, dP/dt, d2P/dt2, curvature, the speed v and RCI
    In firther work we will add filtering and smoothing of the data to reduce noise.
    """

    t = np.asarray(t, dtype=float)
    T = np.asarray(T, dtype=float)
    P = np.asarray(P_bar, dtype=float)

    # Compute derivatives using gradient
    dT = np.gradient(T, t)
    d2T = np.gradient(dT, t)
    dP = np.gradient(P, t)
    d2P = np.gradient(dP, t)

    # Curvature of (T(t), P(t))
    num = np.abs(dT * d2P - dP * d2T)
    den = (dT**2 + dP**2) ** 1.5 + eps
    kappa = num / den

    # Speed in (T, P) space
    v = np.sqrt(dT**2 + dP**2 + eps)

    # RCI
    pos_d2T = np.maximum(d2T, 0)
    RCI = kappa * v * pos_d2T

    return {
        "dTdt": dT,
        "d2Tdt2": d2T,
        "dPdt": dP,
        "d2Pdt2": d2P,
        "kappa": kappa,
        "v": v,
        "RCI": RCI,
    }


def make_window_features(df, cols, win=15):
    """
    Features fenetre glissante
    """

    feats = {}
    for c in cols:
        s = df[c].astype(float)

        feats[f"{c}_mean"] = s.rolling(win, center=True, min_periods=win).mean()
        feats[f"{c}_std"] = s.rolling(win, center=True, min_periods=win).std()
        feats[f"{c}_max"] = s.rolling(win, center=True, min_periods=win).max()
        feats[f"{c}_min"] = s.rolling(win, center=True, min_periods=win).min()

        feats[f"{c}_sclope"] = (s.shift(-win // 2) - s.shift(win // 2)) / win

    return pd.DataFrame(feats, index=df.index).dropna()


def detect_anomalies(
    data: pd.DataFrame,
    time_col="Time",
    T_col="Tr_K",
    P_col="Pression_ideal_bar",
    win=15,
    baseline_end_s=3000,
    Contamination=0.1,
    persist_k=3,
    random_state=0,
):
    """
    Anomalies detection
     - Derivatives via np.gradient
     - curvature, speed and RCI
     - Features sliding window
     - RobustScaler + IsolationForest trained on baseline data
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

    # Features based on T, P + derivatives + RCI
    # (T and P)

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
    X = make_window_features(df, base_cols, win=win)

    # Correlation locale T-P
    tp_corr = df[T_col].rolling(win, center=True, min_periods=win).corr(df[P_col])
    X["TP_corr"] = tp_corr.loc[X.index]

    # Baseline data
    mask_base = (df.loc[X.index, time_col] <= baseline_end_s).to_numpy()

    scaler = RobustScaler()
    X_base = scaler.fit_transform(X[mask_base].to_numpy())
    X_all = scaler.transform(X.to_numpy())

    iso = IsolationForest(
        n_estimators=500, contamination=Contamination, random_state=random_state
    )
    iso.fit(X_base)

    normal_score = iso.score_samples(X_all)
    anomaly_score = -normal_score

    # Thresholding for baseline
    thr = np.quantile(anomaly_score[mask_base], 0.95)

    flag = (anomaly_score > thr).astype(int)
    flag_persist = (
        pd.Series(flag, index=X.index).rolling(persist_k).sum() >= persist_k
    ).astype(int)

    df["anomaly_score"] = np.nan
    df.loc[X.index, "anomaly_score"] = anomaly_score

    df["anomaly_flag"] = 0
    df.loc[X.index, "anomaly_flag"] = flag_persist.values

    t_detect = df.loc[df["anomaly_flag"] == 1, time_col].min()
    t_detect = None if pd.isna(t_detect) else float(t_detect)

    results = {
        "threshold": float(thr),
        "t_detect_s": t_detect,
        "model": iso,
        "scaler": scaler,
        "X_index": X.index,
        "X": X,
    }
    return df, results

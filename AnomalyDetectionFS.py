import numpy as np
import pandas as pd
from AnomalyDetection import compute_RCI, make_window_features
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


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
    X = make_window_features(df, base_cols, win=win)

    tp_corr = df[T_col].rolling(win, center=True, min_periods=win).corr(df[P_col])
    X["TP_corr"] = tp_corr.loc[X.index]

    return df, X


def fit_nominal_model(
    nominal_datasets: list[pd.DataFrame],
    time_col="Time",
    T_col="Tr_K",
    P_col="Pression_ideal_bar",
    win=15,
    baseline_end_s=None,
    contamination=0.0001,
    random_state=0,
    thr_quantile=0.95,
):
    """
    Fit the nominal model (IsolationForest) on the nominal datasets
    """

    X_list = []
    for data in nominal_datasets:
        df, X = build_features(
            data, time_col=time_col, T_col=T_col, P_col=P_col, win=win
        )
        if baseline_end_s is None:
            X_list.append(X)
        else:
            mask = (df.loc[X.index, time_col] <= baseline_end_s).to_numpy()
            X_list.append(X.loc[X.index[mask]])

    X_train = pd.concat(X_list, axis=0).dropna()

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X_train.to_numpy())

    iso = IsolationForest(
        n_estimators=500, contamination=contamination, random_state=random_state
    )
    iso.fit(Xs)

    # seuil global sur les données nominales
    score_train = -iso.score_samples(Xs)  # anomaly_score
    thr = float(np.quantile(score_train, thr_quantile))

    return {
        "model": iso,
        "scaler": scaler,
        "threshold": thr,
        "win": win,
        "cols": {"time_col": time_col, "T_col": T_col, "P_col": P_col},
    }


def detect_with_pretrained(
    data: pd.DataFrame,
    pretrained: dict,
    persist_k=3,
):
    time_col = pretrained["cols"]["time_col"]
    T_col = pretrained["cols"]["T_col"]
    P_col = pretrained["cols"]["P_col"]
    win = pretrained["win"]
    iso = pretrained["model"]
    scaler = pretrained["scaler"]
    thr = pretrained["threshold"]

    df, X = build_features(data, time_col=time_col, T_col=T_col, P_col=P_col, win=win)
    X_all = scaler.transform(X.to_numpy())
    anomaly_score = -iso.score_samples(X_all)

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

    return df, {"threshold": thr, "t_detect_s": t_detect, "X": X}

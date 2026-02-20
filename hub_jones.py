import numpy as np


def detect_hub_jones(df, time_col="Time", T_col="Tr_K", Tw_col=None, Tw_value=None):
    """_summary_

    Args:
        df (_type_): _description_
        time_col (str, optional): _description_. Defaults to "Time".
        T_col (str, optional): _description_. Defaults to "Tr_K".
        Tw_col (_type_, optional): _description_. Defaults to None.
        Tw_value (_type_, optional): _description_. Defaults to None.

    Condition runaway:
        d2T/dt2 > 0 and d(T-Tw)/dt > 0)
    """

    if Tw_col is None and Tw_value is None:
        raise ValueError("Either Tw_col or Tw_value must be provided.")

    t = df[time_col].to_numpy()
    T = df[T_col].to_numpy()

    if Tw_col is not None:
        Tw = df[Tw_col].to_numpy()
    else:
        Tw = np.full_like(T, Tw_value)

    dTdt = np.gradient(T, t)
    d2Tdt2 = np.gradient(dTdt, t)
    dDeltaTdt = np.gradient(T - Tw, t)

    flag = (d2Tdt2 > 0) & (dDeltaTdt > 0)

    df["HJ_flag"] = flag

    if np.any(flag):
        idx = np.argmax(flag)
        t_detect = t[idx]
    else:
        idx = None
        t_detect = None

    return df, t_detect, idx

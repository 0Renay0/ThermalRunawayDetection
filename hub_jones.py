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

    # ---- Patch for the begning points ----
    n = len(flag)
    if n > 0:
        # bords (np.gradient unilatéral)
        flag[0] = False
        flag[-1] = False

        # warm-up : ignorer les 5 premiers pas de temps pour éviter les faux positifs liés au démarrage
        if n >= 2:
            dt = np.median(np.diff(t))
            # 5 pas de temps => 5*dt en secondes; on masque via temps
            warmup_end = t[0] + 5 * dt
            flag[t <= warmup_end] = False
        else:
            flag[:] = False
    # ------------------------------------

    df["HJ_flag"] = flag

    if np.any(flag):
        idx = int(np.argmax(flag))
        t_detect = float(t[idx])
    else:
        idx = None
        t_detect = None

    return df, t_detect, idx

import numpy as np


def detect_strozzi_zaldivar(
    df,
    time_col="Time",
    T_col="Tr_K",
    X_col=None,
    reactant_col="CA",
    reactant_initial=None,
):
    """_summary_

    Args:
        df (_type_): _description_
        time_col (str, optional): _description_. Defaults to "Time".
        T_col (str, optional): _description_. Defaults to "Tr_K".
        x_col (_type_, optional): _description_. Defaults to None.
        reactant_col (str, optional): _description_. Defaults to "CA".
        reactant_initial (_type_, optional): _description_. Defaults to None.

    Condition runaway:
        div = (d2X/dt2) / (dX/dt) + (d2T/dt2) / (dT/dt) > 0

        condition : div > 0
    """

    t = df[time_col].to_numpy()
    T = df[T_col].to_numpy()

    # Définition de X (conversion)
    if X_col is not None:
        X = df[X_col].to_numpy()
    else:
        if reactant_initial is None:
            raise ValueError("reactant_initial must be provided.")
        CA = df[reactant_col].to_numpy()
        X = 1.0 - CA / reactant_initial

    dTdt = np.gradient(T, t)
    d2Tdt2 = np.gradient(dTdt, t)

    dXdt = np.gradient(X, t)
    d2Xdt2 = np.gradient(dXdt, t)

    div = (d2Xdt2 / dXdt) + (d2Tdt2 / dTdt)

    flag = div > 0

    df["div_SZ"] = div
    df["SZ_flag"] = flag

    if np.any(flag):
        idx = np.argmax(flag)
        t_detect = t[idx]
    else:
        idx = None
        t_detect = None

    return df, t_detect, idx

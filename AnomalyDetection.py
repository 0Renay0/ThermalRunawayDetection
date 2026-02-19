import numpy as np


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

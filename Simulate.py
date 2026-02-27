import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import Config as cfg
from ode_model import rhs
from AnomalyDetection import detect_anomalies
from hub_jones import detect_hub_jones
from strozzi_zaldivar import detect_strozzi_zaldivar


def postprocess(sol):
    t = sol.t
    y = sol.y

    CA, HP, Ep, RO, W = y[0], y[1], y[2], y[3], y[4]
    O2_liq = y[5]
    nO2_gas, nCA_gas, nHP_gas, nEp_gas, nW_gas = y[6], y[7], y[8], y[9], y[10]
    mr, Tr, P = y[11], y[12], y[13]

    P_bar = P / 100000.0

    # N2 constant at t=0 (comme ton script)
    nN2_gas_0 = (cfg.PN2 * cfg.Vheadspace) / (cfg.Rg * cfg.Tr0)

    # Pression (bar) (Ideal gas law)
    P_ideal = (
        (nO2_gas + nCA_gas + nHP_gas + nEp_gas + nW_gas + nN2_gas_0)
        * cfg.Rg
        * Tr
        / cfg.Vheadspace
    ) / 100000.0

    # VP (Pa) -> bar
    VP_W = 10 ** (11.008 - (2239.7 / Tr))
    VP_HP = 10 ** (9.9669 - (2175.0 / Tr))
    VP_CA = 10 ** (9.7621 - (1511.9 / Tr))
    VP_Ep = 10 ** (10.671 - (2182.2 / Tr))
    VP_Diol = 10 ** (12.266 - (3455.4 / Tr))

    VP_mix = (
        (W * VP_W + HP * VP_HP + CA * VP_CA + Ep * VP_Ep + RO * VP_Diol)
        / (W + HP + CA + Ep + RO + O2_liq)
    ) / 100000.0

    # Rates
    k1_T = cfg.k01_100 * np.exp((-cfg.Ea1 / cfg.Rg) * ((1.0 / Tr) - (1.0 / 373.15)))
    k2_T = cfg.k02_100 * np.exp((-cfg.Ea2 / cfg.Rg) * ((1.0 / Tr) - (1.0 / 373.15)))
    k3_T = cfg.k03_100 * np.exp((-cfg.Ea3 / cfg.Rg) * ((1.0 / Tr) - (1.0 / 373.15)))

    rate1 = k1_T * CA * HP
    rate2 = k2_T * (HP**2)
    rate3 = k3_T * Ep * W

    qrx = (
        -rate1 * cfg.H1 * (mr / cfg.rho)
        - rate2 * cfg.H2 * (mr / cfg.rho)
        - rate3 * cfg.H3 * (mr / cfg.rho)
    )

    # qexch = cfg.UA * (cfg.Tj - Tr)
    UA_t = np.array([cfg.UA_eff(tt) for tt in t], dtype=float)
    qexch = UA_t * (cfg.Tj - Tr)

    df = pd.DataFrame(
        {
            "Time": t,
            "CA": CA,
            "HP": HP,
            "Ep": Ep,
            "RO": RO,
            "W": W,
            "O2_liq_mol/L": O2_liq,
            "O2_gas_mol": nO2_gas,
            "CA_gas_mol": nCA_gas,
            "HP_gas_mol": nHP_gas,
            "Ep_gas_mol": nEp_gas,
            "W_gas_mol": nW_gas,
            "N2_gas_mol": nN2_gas_0,
            "m_kg": mr,
            "Pression_ODE_bar": P_bar,
            "Pression_ideal_bar": P_ideal,
            "VP_mix_bar": VP_mix,
            "Rate1": rate1,
            "Rate2": rate2,
            "Rate3": rate3,
            "qrx_W": qrx,
            "qexch_W": qexch,
            "Tr_K": Tr,
            "Tr_C": Tr - cfg.Tconv,
            "Ratio_Ep_CA0": Ep / 7.26,
        }
    )
    return df


def run():
    t_span, t_eval = cfg.time_grid()
    y0 = cfg.initial_state()

    sol = solve_ivp(rhs, t_span, y0, method="BDF", t_eval=t_eval)

    data = postprocess(sol)

    # export data
    """out_xlsx = "Ver2_Question6.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        data.to_excel(writer, sheet_name="Sheet1", index=False)"""
    data.to_csv("simulated_data.csv", index=False)

    # print(data.head())
    t = data["Time"].to_numpy()
    # plot Results

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(t, data["CA"].to_numpy(), "b+", label="CA")
    axs[0, 0].plot(t, data["HP"].to_numpy(), "r+", label="HP")
    axs[0, 0].plot(t, data["RO"].to_numpy(), "y+", label="RO")
    axs[0, 0].plot(t, data["Ep"].to_numpy(), "g+", label="Ep")
    # axs[0, 0].plot(t, data["W"].to_numpy(), "m+", label="W")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Concentration (mol/L)")
    axs[0, 0].set_title("Concentrations over time")
    axs[0, 0].legend(loc="upper right")

    axs[0, 1].plot(t, data["Tr_C"].to_numpy(), "b+", label="Tr")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Temperature (°C)")
    axs[0, 1].legend(loc="upper right")
    axs[0, 1].set_title("Temperature over time")

    axs[1, 0].plot(t, data["Pression_ideal_bar"].to_numpy(), "r2", label="Pressure")
    axs[1, 0].plot(t, data["VP_mix_bar"].to_numpy(), "b2", label="Vapor Pressure")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Pressure (bar)")
    axs[1, 0].legend(loc="upper right")
    axs[1, 0].set_title("Pressure over time")

    axs[1, 1].plot(t, data["qrx_W"].to_numpy(), "m2", label="Heat from reactions")
    axs[1, 1].plot(t, data["qexch_W"].to_numpy(), "m2", label="Heat exchanged")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Heat (W)")
    axs[1, 1].legend(loc="upper right")
    axs[1, 1].set_title("Heat over time")

    plt.tight_layout()
    plt.show()

    # Plan de phase  P over T
    plt.figure(figsize=(7, 6))
    plt.scatter(data["Tr_C"].to_numpy(), data["Pression_ideal_bar"].to_numpy(), lw=2)
    plt.xlabel("Reactor Tempreature (°C)")
    plt.ylabel("Pressure (bar)")
    plt.title("Phase plane: Pressure over Temperature")
    plt.grid(True)
    plt.show()

    # Anomalies detection
    print("Time head:", data["Time"].head(10).to_list())
    print("Time min/max:", data["Time"].min(), data["Time"].max())
    print("dt median:", data["Time"].diff().median())

    data, det = detect_anomalies(
        data,
        time_col="Time",
        T_col="Tr_K",  # colonne température en Kelvin dans ton DataFrame
        P_col="Pression_ideal_bar",  # pression idéale calculée
        baseline_end_s=1500,
        Contamination=0.01,
        persist_k=1,
        win=15,
    )

    print("Seuil anomaly_score =", det["threshold"])
    print("Temps détection anomalie =", det["t_detect_s"])

    fig_det, axs_det = plt.subplots(2, 1, figsize=(8, 8))

    # --- subplot 1 : anomaly score ---
    axs_det[0].plot(
        data["Time"].to_numpy(), data["anomaly_score"].to_numpy(), label="Anomaly score"
    )

    axs_det[0].axhline(det["threshold"], linestyle="--", label="Threshold")

    axs_det[0].set_xlabel("Time (s)")
    axs_det[0].set_ylabel("Score")
    axs_det[0].set_title("Anomaly score over time")
    axs_det[0].legend()
    axs_det[0].grid(True)

    # --- subplot 2 : phase plane coloré par anomalies ---
    axs_det[1].scatter = axs_det[1].scatter(
        data["Tr_C"].to_numpy(),
        data["Pression_ideal_bar"].to_numpy(),
        c=data["anomaly_flag"].to_numpy(),
        s=15,
    )

    axs_det[1].set_xlabel("Reactor Temperature (°C)")
    axs_det[1].set_ylabel("Pressure (bar)")
    axs_det[1].set_title("Phase plane with anomaly detection")
    axs_det[1].grid(True)

    plt.tight_layout()
    plt.show()

    # ===================================
    # Detection by Hub & Jones criterion
    # ===================================
    data, t_hj, idx_hj = detect_hub_jones(
        data, time_col="Time", T_col="Tr_K", Tw_value=cfg.Tj
    )

    print("Hub & Jones detection time =", t_hj)

    # ===============================
    # Strozzi & Zaldívar criterion
    # ===============================
    data, t_sz, idx_sz = detect_strozzi_zaldivar(
        data, time_col="Time", T_col="Tr_K", reactant_col="CA", reactant_initial=7.26
    )

    print("Strozzi & Zaldivar detection time =", t_sz)

    return data, sol


if __name__ == "__main__":
    run()

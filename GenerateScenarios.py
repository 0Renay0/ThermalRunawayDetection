import re
import Config as cfg
import os
from scipy.integrate import solve_ivp
from Simulate import postprocess
from ode_model import rhs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob


# Helpers for file name
def _safe_float_str(x: float) -> str:
    """Format court et sûr pour noms de fichiers (pas de . ni d'espaces)."""
    s = f"{float(x):.6g}"
    s = s.replace(".", "p").replace("-", "m")
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    return s


def set_params(
    *,
    CA0=None,
    HP0=None,
    Tr0_C=None,
    PN2_Pa=None,
    nO2_gas=None,
    cooling_stop_s=None,
    UA_deg=None,
):
    """
    Applique les paramètres dans config.py pour générer différents scénarios.
    CA0: Concentration initiale de CA (mol/L)
    HP0: Concentration initiale de HP (mol/L)
    Tr0_C: Température initiale en °C
    PN2_Pa: Pression partielle de N2 dans le gaz (Pa)
    nO2_gas: Moles initiales de O2 dans le gaz (mol)
    """

    cfg.CF_CA0 = CA0
    cfg.CF_HP0 = HP0
    cfg.Tr0_fault = Tr0_C
    if PN2_Pa is not None:
        cfg.PN2 = PN2_Pa
        cfg.CF_nO2_gas = nO2_gas
    # défaut refroidissement
    if cooling_stop_s is not None:
        cfg.cooling_stop_s = float(cooling_stop_s)
    if UA_deg is not None:
        cfg.UA_deg = float(UA_deg)


def reset_params(*, PN2_nominal=10 * 100000.0):
    """Réinitialise les paramètres à leurs valeurs nominales."""
    cfg.CF_CA0 = None
    cfg.CF_HP0 = None
    cfg.Tr0_fault = None
    cfg.PN2 = float(PN2_nominal)
    cfg.CF_nO2_gas = None
    cfg.cooling_stop_s = None
    cfg.UA_deg = 0.0


def run_one_scenario(ovverides: dict, out_dir, filename_tag: str, noisy: bool = False):
    """Lance une simulation pour un scénario donné et exporte les données."""
    os.makedirs(out_dir, exist_ok=True)

    # Appliquer les paramètres du scénario
    set_params(
        CA0=ovverides.get("CA0", None),
        HP0=ovverides.get("HP0", None),
        Tr0_C=ovverides.get("Tr0_C", None),
        PN2_Pa=ovverides.get("PN2_Pa", None),
        nO2_gas=ovverides.get("nO2_gas", None),
    )

    # Integration ODE
    t_span, t_eval = cfg.time_grid()
    y0 = cfg.initial_state()
    sol = solve_ivp(rhs, t_span, y0, method="BDF", t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(f"Echec ODE pour scenario {filename_tag}: {sol.message}")

    df = postprocess(sol)
    if noisy:
        df_noisy = cfg.add_measurement_noise(df.copy())
        # mais on remet CA/HP propres pour la conversion
        for c in ["CA", "HP"]:
            if c in df.columns and c in df_noisy.columns:
                df_noisy[c] = df[c]
        df = df_noisy

    wanted = [
        "Time",
        "Tr_C",
        "Pression_ideal_bar",
        "CA",
        "HP",
    ]

    # Garde seulement celles qui existent (robuste)
    cols = [c for c in wanted if c in df.columns]
    out = df[cols].copy()

    out_path = os.path.join(out_dir, f"scenario_{filename_tag}.csv")
    out.to_csv(out_path, index=False)
    return out_path


# Genérer tous les scénarios
def main():
    out_dir = "./Data/Simulated"

    # plage de variations [7.26, 9.15, 0, 0, 7.41, 0, 0, 0,0,0,0,90,Tr0, PN2]
    CA0_range = np.linspace(6.5, 9.5, 30)  # mol/L
    HP0_range = np.linspace(8.5, 11.5, 30)  # mol/L
    Tr0_C_range = np.linspace(65.0, 105, 50)  # °C
    PN2_bar_range = np.linspace(8.0, 20, 30)  # bar
    nO2_gas_range = np.linspace(0.0, 1.5, 15)  # mol

    PN2_Pa_range = PN2_bar_range * 1e5
    t_fault_values = np.arange(0, 50000 + 1, 5000)  # 0..50000 step 5000
    UA_deg_values = np.arange(10, -1, -1)  # 10..0 step 1

    mode = "one_at_a_time"

    PN2_nominal = 10.0 * 1e5

    paths = []

    try:
        if mode == "one_at_a_time":
            for v in CA0_range:
                reset_params(PN2_nominal=PN2_nominal)
                tag = f"CA0_{_safe_float_str(v)}"
                paths.append(
                    run_one_scenario({"CA0": float(v)}, out_dir, tag, noisy=True)
                )

            for v in HP0_range:
                reset_params(PN2_nominal=PN2_nominal)
                tag = f"HP0_{_safe_float_str(v)}"
                paths.append(
                    run_one_scenario({"HP0": float(v)}, out_dir, tag, noisy=True)
                )

            for v in Tr0_C_range:
                reset_params(PN2_nominal=PN2_nominal)
                tag = f"Tr0C_{_safe_float_str(v)}"
                paths.append(
                    run_one_scenario({"Tr0_C": float(v)}, out_dir, tag, noisy=True)
                )

            for v in PN2_Pa_range:
                reset_params(PN2_nominal=PN2_nominal)
                tag = f"PN2Pa_{_safe_float_str(v)}"
                paths.append(
                    run_one_scenario({"PN2_Pa": float(v)}, out_dir, tag, noisy=True)
                )

            for v in nO2_gas_range:
                reset_params(PN2_nominal=PN2_nominal)
                tag = f"nO2gas_{_safe_float_str(v)}"
                paths.append(
                    run_one_scenario({"nO2_gas": float(v)}, out_dir, tag, noisy=True)
                )

            for t_fault in t_fault_values:
                for UA_d in UA_deg_values:
                    reset_params(PN2_nominal=PN2_nominal)

                    tag = f"Cooling_t{int(t_fault)}_UAdeg{int(UA_d)}"
                    overrides = {
                        "cooling_stop_s": float(t_fault),
                        "UA_deg": float(UA_d),
                    }
                    paths.append(run_one_scenario(overrides, out_dir, tag, noisy=True))

        else:
            raise ValueError("Only one mode exists! One at a time")
    finally:
        reset_params(PN2_nominal=PN2_nominal)

    print(f"[OK] Création de scenarios - {len(paths)} fichiers écrit dans {out_dir}")

    pd.DataFrame({"file": paths}).to_csv(
        os.path.join(out_dir, "index.csv"), index=False
    )


def plot_scenario(files):
    if not files:
        print("Aucun fichier trouvé avec le pattern scenario_*.csv")
        return

    # --- Pression ---
    plt.figure()
    for f in files:
        df = pd.read_csv(f)
        plt.plot(df["Time"].to_numpy(), df["Pression_ideal_bar"].to_numpy())

    plt.xlabel("Time")
    plt.ylabel("Pression (bar)")
    plt.title("Comparaison des scénarios - Pression")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Température ---
    plt.figure()
    for f in files:
        df = pd.read_csv(f)
        plt.plot(df["Time"].to_numpy(), df["Tr_C"].to_numpy())

    plt.xlabel("Time")
    plt.ylabel("Température (°C)")
    plt.title("Comparaison des scénarios - Température")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

    # plot scenario
    files = glob.glob("Data/Simulated/scenario_*.csv")
    plot_scenario(files=files)

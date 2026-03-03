import re
import Config as cfg
import os
from scipy.integrate import solve_ivp
from Simulate import postprocess
from ode_model import rhs

# Helpers for file name


def _safe_float_str(x: float) -> str:
    """Format court et sûr pour noms de fichiers (pas de . ni d'espaces)."""
    s = f"{float(x):.6g}"
    s = s.replace(".", "p").replace("-", "m")
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    return s


def set_params(*, CA0=None, HP0=None, Tr0_C=None, PN2_Pa=None, nO2_gas=None):
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


def reset_params(*, PN2_nominal=10 * 100000.0):
    """Réinitialise les paramètres à leurs valeurs nominales."""
    cfg.CF_CA0 = None
    cfg.CF_HP0 = None
    cfg.Tr0_fault = None
    cfg.PN2 = float(PN2_nominal)
    cfg.CF_nO2_gas = None


def run_one_sceario(ovverides: dict, out_dir, filename_tag: str, noisy: bool = False):
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
        df = cfg.add_measurement_noise(df)

    cols = ["Time"]
    if "Tr_C" in df.columns:
        cols.append("Tr_C")
    if "Pression_ideal_bar" in df.columns:
        cols.append("Pression_ideal_bar")

    out = df[cols].copy()

    # Sauvegarder les données
    out_path = os.path.join(out_dir, f"scenario_{filename_tag}.csv")
    out.to_csv(out_path, index=False)

    return out_path

import re
import Config as cfg

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

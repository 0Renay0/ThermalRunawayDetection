import numpy as np


# ======== Cinétique ========
k01_100 = 5.25e-5  # L/mol/s
k02_100 = 7e-8
k03_100 = 0.3e-8

Ea1 = 57000.0  # J/mol
Ea2 = 220000.0
Ea3 = 250000.0

H1 = -10000.0  # J/mol
H2 = -95000.0
H3 = -5000.0

# ======== CI & paramètres physiques ========
Rg = 8.314  # J/mol/K
UA = 10.0  # W/K
Cpr = 2500.0  # J/kg/K
mr0 = 90.0  # kg
rho = 1.0  # kg/L (tel que utilisé dans ton code)

Tconv = 273.15
Tr0 = 95.0 + Tconv  # K
Tj = Tr0  # K

kla = 0.23  # s^-1 (O2 liquide)
kla2 = 3.5  # s^-1 (gaz pour espèces =\ O2)

Vheadspace = 50.0 / 1000.0  # m^3
PN2 = 10.0 * 100000.0  # Pa

# ======== Enthalpies de vaporisation (J/mol) ========
Hvap_CA = 28943.0
Hvap_W = 42876.0
Hvap_Epi = 41775.0
Hvap_HP = 41638.0
Hvap_diol = 66149.0

# ======== Masses molaires (kg/mol) ========
M_o2 = 16e-3
M_CA = 76.53e-3
M_HP = 34e-3
M_Ep = 92.53e-3
M_W = 18e-3


# ======== Conditions initiales ========
def initial_state():
    # [CA, HP, Ep, RO, W, O2_liq, nO2_gas, nCA_gas, nHP_gas, nEp_gas, nW_gas, mr, Tr, P]
    return np.array(
        [7.26, 9.15, 0.0, 0.0, 7.41, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, mr0, Tr0, PN2],
        dtype=float,
    )


# ---------- Grille de simulation ----------
def time_grid():
    t_span = (0.0, 50000.0)
    t_eval = np.linspace(t_span[0], t_span[1], 501)
    return t_span, t_eval

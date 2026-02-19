import numpy as np 



# ======== Cinétique ========
k01_100 = 5.25e-5   # L/mol/s
k02_100 = 7e-8
k03_100 = 0.3e-8

Ea1 = 57000.0       # J/mol
Ea2 = 220000.0
Ea3 = 250000.0

H1 = -10000.0       # J/mol
H2 = -95000.0
H3 = -5000.0

# ======== CI & paramètres physiques ========
Rg = 8.314          # J/mol/K
UA = 10.0           # W/K
Cpr = 2500.0        # J/kg/K
mr0 = 90.0          # kg
rho = 1.0           # kg/L (tel que utilisé dans ton code)

Tconv = 273.15
Tr0 = 65.0 + Tconv  # K
Tj = Tr0            # K

kla = 0.23          # s^-1 (O2 liquide)
kla2 = 3.5          # s^-1 (gaz pour espèces ≠ O2)

Vheadspace = 50.0 / 1000.0   # m^3


# ======== Enthalpies de vaporisation (J/mol) ========
Hvap_CA = 28943.0
Hvap_W  = 42876.0
Hvap_Epi = 41775.0
Hvap_HP  = 41638.0
Hvap_diol = 66149.0

# ======== Masses molaires (kg/mol) ========
M_o2 = 16e-3
M_CA = 76.53e-3
M_HP = 34e-3
M_Ep = 92.53e-3
M_W  = 18e-3


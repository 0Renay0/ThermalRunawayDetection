import numpy as np
import Config as cfg


def rhs(t, c):
    CA, HP, Ep, RO, W, O2_liq, n02_gas, nCA_gas, nHP_gas, nEp_gas, nW_gas, mr, Tr, P = c

    # Loi d'Arrhenius
    k1 = cfg.k01_100 * np.exp((-cfg.Ea1 / cfg.Rg) * ((1.0 / Tr) - (1.0 / 373.15)))
    k2 = cfg.k02_100 * np.exp((-cfg.Ea2 / cfg.Rg) * ((1.0 / Tr) - (1.0 / 373.15)))
    k3 = cfg.k03_100 * np.exp((-cfg.Ea3 / cfg.Rg) * ((1.0 / Tr) - (1.0 / 373.15)))

    # Vitesses
    R1 = k1 * CA * HP
    R2 = k2 * (HP**2)
    R3 = k3 * Ep * W

    # Solubilité O2 (mol/L)
    O2_star = 0.001473 * np.exp(-0.008492 * (Tr - cfg.Tconv))

    # Équilibres gaz (mol/m^3)
    W_star_g = (10 ** (11.008 - (2239.7 / Tr))) / (cfg.Rg * Tr)
    HP_star_g = (10 ** (9.9669 - (2175.0 / Tr))) / (cfg.Rg * Tr)
    CA_star_g = (10 ** (9.7621 - (1511.9 / Tr))) / (cfg.Rg * Tr)
    Ep_star_g = (10 ** (10.671 - (2182.2 / Tr))) / (cfg.Rg * Tr)
    # Diol_star_g = (10 ** (12.266 - (3455.4 / Tr))) / (cfg.Rg * Tr)  # Negligé

    # O2 transfert
    kla_effective = cfg.kla if (O2_liq - O2_star) >= 0 else 0.0

    # O2 liquide / gaz
    dO2_liqdt = R2 - kla_effective * (O2_liq - O2_star)
    dn02_gasdt = kla_effective * (mr / cfg.rho) * (O2_liq - O2_star)

    # Variation de masse (kg/s)
    dmrdt = -(dn02_gasdt * cfg.M_o2) - cfg.Vheadspace * (
        (cfg.kla2 * (CA_star_g - nCA_gas / cfg.Vheadspace)) * cfg.M_CA
        + (cfg.kla2 * (HP_star_g - nHP_gas / cfg.Vheadspace)) * cfg.M_HP
        + (cfg.kla2 * (Ep_star_g - nEp_gas / cfg.Vheadspace)) * cfg.M_Ep
        + (cfg.kla2 * (W_star_g - nW_gas / cfg.Vheadspace)) * cfg.M_W
    )

    # Phase liquide (mol/L/s)
    vol_liq_L = mr / cfg.rho
    gas2liq_factor = cfg.Vheadspace / vol_liq_L

    dCAdt = (
        -R1
        - (cfg.kla2 * (CA_star_g - nCA_gas / cfg.Vheadspace) / 1000.0) * gas2liq_factor
        - (CA / mr) * dmrdt
    )
    dHPdt = (
        -R1
        - 2.0 * R2
        - (cfg.kla2 * (HP_star_g - nHP_gas / cfg.Vheadspace) / 1000.0) * gas2liq_factor
        - (HP / mr) * dmrdt
    )
    dEpdt = (
        R1
        - R3
        - (cfg.kla2 * (Ep_star_g - nEp_gas / cfg.Vheadspace) / 1000.0) * gas2liq_factor
        - (Ep / mr) * dmrdt
    )
    dROdt = R3 - (RO / mr) * dmrdt
    dWdt = (
        R1
        + 2.0 * R2
        - R3
        - (cfg.kla2 * (W_star_g - nW_gas / cfg.Vheadspace) / 1000.0) * gas2liq_factor
        - (W / mr) * dmrdt
    )

    # Phase gazeuse (mol/s)
    dnCA_gasdt = cfg.Vheadspace * cfg.kla2 * (CA_star_g - nCA_gas / cfg.Vheadspace)
    dnHP_gasdt = cfg.Vheadspace * cfg.kla2 * (HP_star_g - nHP_gas / cfg.Vheadspace)
    dnEp_gasdt = cfg.Vheadspace * cfg.kla2 * (Ep_star_g - nEp_gas / cfg.Vheadspace)
    dnW_gasdt = cfg.Vheadspace * cfg.kla2 * (W_star_g - nW_gas / cfg.Vheadspace)

    # Evaporation
    qevapo = (
        dnCA_gasdt * cfg.Hvap_CA
        + dnHP_gasdt * cfg.Hvap_HP
        + dnEp_gasdt * cfg.Hvap_Epi
        + dnW_gasdt * cfg.Hvap_W
    )

    # Bilan thermique
    dTrdt = (
        (-R1 * vol_liq_L * cfg.H1)
        + (-R2 * vol_liq_L * cfg.H2)
        + (-R3 * vol_liq_L * cfg.H3)
        + cfg.UA_eff(t) * (cfg.Tj - Tr)
        - qevapo
    ) / (mr * cfg.Cpr)

    # Pression (Pa/s)
    nN2_gas = (cfg.PN2 * cfg.Vheadspace) / (cfg.Rg * Tr)
    n_tot = n02_gas + nCA_gas + nHP_gas + nEp_gas + nW_gas + nN2_gas

    dn_tot_dt = dn02_gasdt + dnCA_gasdt + dnHP_gasdt + dnEp_gasdt + dnW_gasdt
    dPdt = (cfg.Rg / cfg.Vheadspace) * (dn_tot_dt * Tr + n_tot * dTrdt)

    return [
        dCAdt,
        dHPdt,
        dEpdt,
        dROdt,
        dWdt,
        dO2_liqdt,
        dn02_gasdt,
        dnCA_gasdt,
        dnHP_gasdt,
        dnEp_gasdt,
        dnW_gasdt,
        dmrdt,
        dTrdt,
        dPdt,
    ]

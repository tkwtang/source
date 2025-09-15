import numpy as np
import os, sys
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from PARAMETER_INPUT import PHI_0, x_c

# PHI_0 = 2.067833848 * 1e-15
# x_c0 = PHI_0 / (2 * np.pi)
# default parmas and the domain
coupled_fq_default_param = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x_c]
[phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound] = [4, 4, 4, 4]
coupled_fq_domain = [[-phi_1_bound, -phi_2_bound, -phi_1dc_bound, -phi_2dc_bound], [phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound]]


[ 
  ("U0_1", 0), ("U0_2", 1), ("gamma_1", 2), ("gamma_2", 3), ("beta_1", 4), ("beta_2", 5),
  ("d_beta_1", 6), ("d_beta_2", 7), ("phi_1x", 8), ("phi_2x", 9), ("phi_1xdc", 10), ("phi_2xdc", 11),
  ("mu_12", 12), ('x_c', 13)
] 

def coupled_flux_qubit_non_linear_approx_pot(phi_1, phi_2, phi_1dc, phi_2dc, params):
    """
    4D 4-well potential.

    Parmeters
    -------------
    phi: ndaray of dimension [N, 2]
    phi_dc: ndaray of dimension [N, 2]

    params: list / tuple
    - [U_0, g, beta, delta_beta, phi_x, phi_xdc ]: correspond to the energy scale, gamma
    - phi_x: associated with asymmetry in the informational subspace, and will only take a nonzero value to help
      offset asymmetry from the delta_beta term in U'
     - scale factor is x_c used in the simulation
    """

    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, delta_beta_1, delta_beta_2, phi_1x, phi_2x , phi_1xdc, phi_2xdc, mu_12, x_c = params
    U0_kBT_ratio = U0_1
    

    U0_2 = U0_2 / U0_1
    U0_1 = 1
    xi = 1 / (1 - mu_12**2)

    u1_1 = 1/2 * xi * (phi_1 - phi_1x)**2
    u2_1 = 1/2 * g_1 * (phi_1dc - phi_1xdc)**2
    u3_1 = beta_1 * np.cos(phi_1) * np.cos(phi_1dc/2)
    u4_1 = -delta_beta_1 * np.sin(phi_1) * np.sin(phi_1dc/2)
    
    u1_2 = 1/2 * xi * (phi_2 - phi_2x)**2    
    u2_2 = 1/2 * g_2 * (phi_2dc - phi_2xdc)**2
    u3_2 = beta_2 * np.cos(phi_2) * np.cos(phi_2dc/2)
    u4_2 = -delta_beta_2 * np.sin(phi_2) * np.sin(phi_2dc/2)
    u5 = mu_12 * xi * (phi_1 - phi_1x) * (phi_2 - phi_2x)

    U = U0_1 * ( u1_1 + u2_1 + u3_1 + u4_1 ) + U0_2 * ( u1_2 + u2_2 + u3_2 + u4_2 ) + np.sqrt(U0_1 * U0_2) * u5

    return U * U0_kBT_ratio

def coupled_flux_qubit_non_linear_approx_force(phi_1, phi_2, phi_1dc, phi_2dc, params):
    """
    2D 4-well potential. Note that mu_12 is already normalized with L_1 and L_2.

    Parmeters
    -------------
    phi: ndaray of dimension [N, ]
    phi_dc: ndaray of dimension [N, ]

    params: list / tuple
    - [U_0, g, beta, delta_beta, phi_x, phi_xdc ]: correspond to the energy scale, gamma
    """

    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, delta_beta_1, delta_beta_2, phi_1x, phi_2x , phi_1xdc, phi_2xdc, mu_12, x_c = params # mu_12 is already normalized with L_1 and L_2
    U0_kBT_ratio = U0_1
    U0_2 = U0_2 / U0_1
    U0_1 = 1
    xi = 1 / (1 - mu_12**2)

    U_dp1 = U0_1* (
        xi * (phi_1 - phi_1x)
        - beta_1 * np.sin(phi_1) * np.cos(phi_1dc / 2)
        - delta_beta_1 * np.cos(phi_1) * np.sin(phi_1dc/2)
    )  + np.sqrt(U0_1 * U0_2) * mu_12 * xi * (phi_2 - phi_2x)

    U_dp2 = U0_2* (
        xi * (phi_2 - phi_2x)
        - beta_2 * np.sin(phi_2) * np.cos(phi_2dc / 2)
        - delta_beta_2 * np.cos(phi_2) * np.sin(phi_2dc/2)
    )  + np.sqrt(U0_1 * U0_2) * mu_12 * xi * (phi_1 - phi_1x)

    U_dp1dc = U0_1* (
        g_1 * (phi_1dc - phi_1xdc)
        - 1/2 * beta_1 * np.cos(phi_1) * np.sin(phi_1dc / 2)
        - 1/2 * delta_beta_1 * np.sin(phi_1) * np.cos(phi_1dc/2)
    )
    # print(params)

    U_dp2dc = U0_2* (
        g_2 * (phi_2dc -  phi_2xdc)
        - 1/2 * beta_2 * np.cos(phi_2) * np.sin(phi_2dc / 2)
        - 1/2 * delta_beta_2 * np.sin(phi_2) * np.cos(phi_2dc/2)
    )

    return -1 *  np.array([U_dp1, U_dp2, U_dp1dc, U_dp2dc])
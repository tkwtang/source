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
  ("d_beta_1", 6), ("d_beta_2", 7), ("phi_1_x", 8), ("phi_2_x", 9), ("phi_1_dcx", 10), ("phi_2_dcx", 11),
  ("M_12", 12), ('x_c', 13)
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

    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, delta_beta_1, delta_beta_2, phi_1x, phi_2x , phi_1dcx, phi_2dcx, M_12, x_c = params
    U0_kBT_ratio = U0_1
    

    U0_2 = U0_2 / U0_1
    U0_1 = 1
    xi = 1 / (1 - M_12**2)

    u1_1 = 1/2 * xi * (phi_1 - phi_1x)**2
    u2_1 = 1/2 * g_1 * (phi_1dc - phi_1dcx)**2
    u3_1 = beta_1 * np.cos(phi_1) * np.cos(phi_1dc/2)
    u4_1 = -delta_beta_1 * np.sin(phi_1) * np.sin(phi_1dc/2)
    
    u1_2 = 1/2 * xi * (phi_2 - phi_2x)**2    
    u2_2 = 1/2 * g_2 * (phi_2dc - phi_2dcx)**2
    u3_2 = beta_2 * np.cos(phi_2) * np.cos(phi_2dc/2)
    u4_2 = -delta_beta_2 * np.sin(phi_2) * np.sin(phi_2dc/2)
    u5 = M_12 * xi * (phi_1 - phi_1x) * (phi_2 - phi_2x)

    U = U0_1 * ( u1_1 + u2_1 + u3_1 + u4_1 ) + U0_2 * ( u1_2 + u2_2 + u3_2 + u4_2 ) + np.sqrt(U0_1 * U0_2) * u5

    return U * U0_kBT_ratio

def coupled_flux_qubit_non_linear_approx_force(phi_1, phi_2, phi_1dc, phi_2dc, params):
    """
    2D 4-well potential. Note that M_12 is already normalized with L_1 and L_2.

    Parmeters
    -------------
    phi: ndaray of dimension [N, ]
    phi_dc: ndaray of dimension [N, ]

    params: list / tuple
    - [U_0, g, beta, delta_beta, phi_x, phi_xdc ]: correspond to the energy scale, gamma
    """

    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, delta_beta_1, delta_beta_2, phi_1x, phi_2x , phi_1dcx, phi_2dcx, M_12, x_c = params # M_12 is already normalized with L_1 and L_2
    U0_kBT_ratio = U0_1
    U0_2 = U0_2 / U0_1
    U0_1 = 1
    xi = 1 / (1 - M_12**2)

    U_dp1 = U0_1* (
        xi * (phi_1 - phi_1x)
        - beta_1 * np.sin(phi_1) * np.cos(phi_1dc / 2)
        - delta_beta_1 * np.cos(phi_1) * np.sin(phi_1dc/2)
    )  + np.sqrt(U0_1 * U0_2) * M_12 * xi * (phi_2 - phi_2x)

    U_dp2 = U0_2* (
        xi * (phi_2 - phi_2x)
        - beta_2 * np.sin(phi_2) * np.cos(phi_2dc / 2)
        - delta_beta_2 * np.cos(phi_2) * np.sin(phi_2dc/2)
    )  + np.sqrt(U0_1 * U0_2) * M_12 * xi * (phi_1 - phi_1x)

    U_dp1dc = U0_1* (
        g_1 * (phi_1dc - phi_1dcx)
        - 1/2 * beta_1 * np.cos(phi_1) * np.sin(phi_1dc / 2)
        - 1/2 * delta_beta_1 * np.sin(phi_1) * np.cos(phi_1dc/2)
    )
    # print(params)

    U_dp2dc = U0_2* (
        g_2 * (phi_2dc -  phi_2dcx)
        - 1/2 * beta_2 * np.cos(phi_2) * np.sin(phi_2dc / 2)
        - 1/2 * delta_beta_2 * np.sin(phi_2) * np.cos(phi_2dc/2)
    )

    return -1 *  np.array([U_dp1, U_dp2, U_dp1dc, U_dp2dc])



def coupled_flux_qubit_non_linear_approx_pot_special(phi_1, phi_2, phi_1dc, phi_2dc, params):
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

    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, delta_beta_1, delta_beta_2, phi_1x, phi_2x , phi_1dcx, phi_2dcx, M_12, x_c = params
    U0_kBT_ratio = U0_1
    

    U0_2 = U0_2 / U0_1
    U0_1 = 1
    xi = 1 / (1 - M_12**2)

    u1_1 = 1/2 * xi * (phi_1 - phi_1x)**2 * 0
    u2_1 = 1/2 * g_1 * (phi_1dc - phi_1dcx)**2 * 0
    u3_1 = beta_1 * np.cos(phi_1) * np.cos(phi_1dc/2) * 0
    u4_1 = -delta_beta_1 * np.sin(phi_1) * np.sin(phi_1dc/2) * 0
    
    u1_2 = 1/2 * xi * (phi_2 - phi_2x)**2    
    u2_2 = 1/2 * g_2 * (phi_2dc - phi_2dcx)**2 * 0
    u3_2 = beta_2 * np.cos(phi_2) * np.cos(phi_2dc/2) * 0
    u4_2 = -delta_beta_2 * np.sin(phi_2) * np.sin(phi_2dc/2) * 0
    u5 = M_12 * xi * (phi_1 - phi_1x) * (phi_2 - phi_2x) * 0

    U = U0_1 * ( u1_1 + u2_1 + u3_1 + u4_1 ) + U0_2 * ( u1_2 + u2_2 + u3_2 + u4_2 ) + np.sqrt(U0_1 * U0_2) * u5

    return U * U0_kBT_ratio

def coupled_flux_qubit_non_linear_approx_force_special(phi_1, phi_2, phi_1dc, phi_2dc, params):
    """
    2D 4-well potential. Note that M_12 is already normalized with L_1 and L_2.

    Parmeters
    -------------
    phi: ndaray of dimension [N, ]
    phi_dc: ndaray of dimension [N, ]

    params: list / tuple
    - [U_0, g, beta, delta_beta, phi_x, phi_xdc ]: correspond to the energy scale, gamma
    """

    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, delta_beta_1, delta_beta_2, phi_1x, phi_2x , phi_1dcx, phi_2dcx, M_12, x_c = params # M_12 is already normalized with L_1 and L_2
    U0_kBT_ratio = U0_1
    U0_2 = U0_2 / U0_1
    U0_1 = 1
    xi = 1 / (1 - M_12**2)

    scale_factor = x_c / x_c
    phi_1 = phi_1 * scale_factor
    phi_2 = phi_2 * scale_factor
    phi_1dc = phi_1dc * scale_factor
    phi_2dc = phi_2dc  * scale_factor
    phi_1x = phi_1x * scale_factor
    phi_2x = phi_2x * scale_factor
    phi_1dcx = phi_1dcx * scale_factor
    phi_2dcx = phi_2dcx * scale_factor

    U_dp1 = U0_1* (
        xi * (phi_1 - phi_1x)
        - beta_1 * np.sin(phi_1) * np.cos(phi_1dc / 2) 
        - delta_beta_1 * np.cos(phi_1) * np.sin(phi_1dc/2)
    ) * 0

    U_dp2 = U0_2* (
        xi * (phi_2 - phi_2x)
        
    )

    U_dp1dc = U0_1* (
        g_1 * (phi_1dc - phi_1dcx)
        - 1/2 * beta_1 * np.cos(phi_1) * np.sin(phi_1dc / 2)
        - 1/2 * delta_beta_1 * np.sin(phi_1) * np.cos(phi_1dc/2)
    ) * 0
    # print(params)

    U_dp2dc = U0_2* (
        g_2 * (phi_2dc -  phi_2dcx)
        - 1/2 * beta_2 * np.cos(phi_2) * np.sin(phi_2dc / 2)
        - 1/2 * delta_beta_2 * np.sin(phi_2) * np.cos(phi_2dc/2)
    ) * 0

    return -1 * U0_kBT_ratio *  np.array([U_dp1, U_dp2, U_dp1dc, U_dp2dc])




def coupled_flux_qubit_non_linear_approx_pot_break_down(phi_1, phi_2, phi_1dc, phi_2dc, params):
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

    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, delta_beta_1, delta_beta_2, phi_1x, phi_2x , phi_1dcx, phi_2dcx, M_12, x_c = params
    U0_kBT_ratio = U0_1 / k_BT

    U0_2 = U0_2 / U0_1
    U0_1 = 1
    xi = 1 / (1 - M_12**2)

    scale_factor = x_c / x_c
    phi_1 = phi_1 * scale_factor
    phi_2 = phi_2 * scale_factor
    phi_1dc = phi_1dc * scale_factor
    phi_2dc = phi_2dc  * scale_factor
    phi_1x = phi_1x * scale_factor
    phi_2x = phi_2x * scale_factor
    phi_1dcx = phi_1dcx * scale_factor
    phi_2dcx = phi_2dcx * scale_factor

    u1_1 = 1/2 * xi * (phi_1 - phi_1x)**2
    u1_2 = 1/2 * xi * (phi_2 - phi_2x)**2
    u2_1 = 1/2 * g_1 * (phi_1dc - phi_1dcx)**2
    u2_2 = 1/2 * g_2 * (phi_2dc - phi_2dcx)**2
    u3_1 = beta_1 * np.cos(phi_1) * np.cos(phi_1dc/2)
    u3_2 = beta_2 * np.cos(phi_2) * np.cos(phi_2dc/2)
    u4_1 = -delta_beta_1 * np.sin(phi_1) * np.sin(phi_1dc/2)
    u4_2 = -delta_beta_2 * np.sin(phi_2) * np.sin(phi_2dc/2)
    u5 = M_12 * xi * (phi_1 - phi_1x) * (phi_2 - phi_2x)

    U = U0_1 * ( u1_1 + u2_1 + u3_1 + u4_1 ) + U0_2 * ( u1_2 + u2_2 + u3_2 + u4_2 ) + np.sqrt(U0_1 * U0_2) * u5

    return {
        "u1_1": U0_kBT_ratio * u1_1, "u2_1": U0_kBT_ratio * u2_1, 
        "u3_1": U0_kBT_ratio * u3_1, "u4_1": U0_kBT_ratio * u4_1,
        
        "u1_2": U0_kBT_ratio * u1_2, "u2_2": U0_kBT_ratio * u2_2, 
        "u3_2": U0_kBT_ratio * u3_2, "u4_2": U0_kBT_ratio * u4_2,
        "u5":   U0_kBT_ratio * u5
    }


def coupled_flux_qubit_non_linear_approx_force_break_down(phi_1, phi_2, phi_1dc, phi_2dc, params):
    """
    2D 4-well potential. Note that M_12 is already normalized with L_1 and L_2.

    Parmeters
    -------------
    phi: ndaray of dimension [N, ]
    phi_dc: ndaray of dimension [N, ]

    params: list / tuple
    - [U_0, g, beta, delta_beta, phi_x, phi_xdc ]: correspond to the energy scale, gamma
    """

    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, delta_beta_1, delta_beta_2, phi_1x, phi_2x , phi_1dcx, phi_2dcx, M_12, x_c = params # M_12 is already normalized with L_1 and L_2
    U0_kBT_ratio = U0_1 / k_BT
    U0_2 = U0_2 / U0_1
    U0_1 = 1
    xi = 1 / (1 - M_12**2)


    scale_factor = x_c / x_c
    phi_1 = phi_1 * scale_factor
    phi_2 = phi_2 * scale_factor
    phi_1dc = phi_1dc * scale_factor
    phi_2dc = phi_2dc  * scale_factor
    phi_1x = phi_1x * scale_factor
    phi_2x = phi_2x * scale_factor
    phi_1dcx = phi_1dcx * scale_factor
    phi_2dcx = phi_2dcx * scale_factor

    U_dp1_1 = xi * (phi_1 - phi_1x)
    U_dp1_2 = - beta_1 * np.sin(phi_1) * np.cos(phi_1dc / 2)
    U_dp1_3 = - delta_beta_1 * np.cos(phi_1) * np.sin(phi_1dc/2)
    U_dp1_4 = + np.sqrt(U0_1 * U0_2)  * M_12  * xi  *  (phi_2 - phi_2x)

    U_dp2_1 = xi * (phi_2 - phi_2x)
    U_dp2_2 = - beta_2 * np.sin(phi_2) * np.cos(phi_2dc / 2)
    U_dp2_3 = - delta_beta_2 * np.cos(phi_2) * np.sin(phi_2dc/2)
    U_dp2_4 = np.sqrt(U0_1 * U0_2)  * M_12 * xi *   (phi_1 - phi_1x)

    U_dp1dc_1 = g_1 * (phi_1dc - phi_1dcx)
    U_dp1dc_2 = - 1/2 * beta_1 * np.cos(phi_1) * np.sin(phi_1dc / 2)
    U_dp1dc_3 = - 1/2 * delta_beta_1 * np.sin(phi_1) * np.cos(phi_1dc/2)
    U_dp1dc_4 = 0

    U_dp2dc_1 = g_2 * (phi_2dc -  phi_2dcx)
    U_dp2dc_2 = - 1/2 * beta_2 * np.cos(phi_2) * np.sin(phi_2dc / 2)
    U_dp2dc_3 = - 1/2 * delta_beta_2 * np.sin(phi_2) * np.cos(phi_2dc/2)
    U_dp2dc_4 = 0
        

    return -1 * U0_kBT_ratio *  np.array([
        [U_dp1_1,   U_dp1_2,   U_dp1_3,   U_dp1_4],
        [U_dp2_1,   U_dp2_2,   U_dp2_3,   U_dp2_4], 
        [U_dp1dc_1, U_dp1dc_2, U_dp1dc_3, U_dp1dc_4],
        [U_dp2dc_1, U_dp2dc_2, U_dp2dc_3, U_dp2dc_4]
    ])


def coupled_flux_qubit_non_linear_approx_pot_only_d_beta(phi_1, phi_2, phi_1dc, phi_2dc, params):
    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, d_beta_1, d_beta_2, phi_1x, phi_2x , phi_1dcx, phi_2dcx, M_12, x_c = params
    U0_kBT_ratio = U0_1 / k_BT

    U0_2 = U0_2 / U0_1
    U0_1 = 1
    
    u4_1 = -d_beta_1 * np.sin(phi_1) * np.sin(phi_1dc/2)
    u4_2 = -d_beta_2 * np.sin(phi_2) * np.sin(phi_2dc/2)
    U = U0_1 * u4_1 + U0_2 * u4_2

    return U * U0_kBT_ratio

def coupled_flux_qubit_non_linear_approx_force_only_d_beta(phi_1, phi_2, phi_1dc, phi_2dc, params):

    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, delta_beta_1, delta_beta_2, phi_1x, phi_2x , phi_1dcx, phi_2dcx, M_12, x_c = params # M_12 is already normalized with L_1 and L_2
    U0_kBT_ratio = U0_1 / k_BT
    U0_2 = U0_2 / U0_1
    U0_1 = 1
    xi = 1 / (1 - M_12**2)

    U_dp1 = U0_1* (
        - delta_beta_1 * np.cos(phi_1) * np.sin(phi_1dc/2)
    )

    U_dp2 = U0_2* (
        - delta_beta_2 * np.cos(phi_2) * np.sin(phi_2dc/2)
    )

    U_dp1dc = U0_1* (
        - 1/2 * delta_beta_1 * np.sin(phi_1) * np.cos(phi_1dc/2)
    )
    # print(params)

    U_dp2dc = U0_2* (
        - 1/2 * delta_beta_2 * np.sin(phi_2) * np.cos(phi_2dc/2)
    )

    return -1 * U0_kBT_ratio *  np.array([U_dp1, U_dp2, U_dp1dc, U_dp2dc])

def coupled_flux_qubit_non_linear_approx_pot_only_beta(phi_1, phi_2, phi_1dc, phi_2dc, params):
    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, delta_beta_1, delta_beta_2, phi_1x, phi_2x , phi_1dcx, phi_2dcx, M_12, x_c = params
    U0_kBT_ratio = U0_1 / k_BT

    U0_2 = U0_2 / U0_1
    U0_1 = 1
    xi = 1 / (1 - M_12**2)

    u3_1 = beta_1 * np.cos(phi_1) * np.cos(phi_1dc/2)

    u3_2 = beta_2 * np.cos(phi_2) * np.cos(phi_2dc/2)

    U = U0_1 * u3_1 + U0_2 * u3_2

    return U * U0_kBT_ratio



def coupled_flux_qubit_non_linear_approx_pot_only_phi_12(phi_1, phi_2, phi_1dc, phi_2dc, params):
    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, d_beta_1, d_beta_2, phi_1x, phi_2x , phi_1dcx, phi_2dcx, M_12, x_c = params
    U0_kBT_ratio = U0_1 / k_BT

    U0_2 = U0_2 / U0_1
    U0_1 = 1
    xi = 1 / (1 - M_12**2)
    
    u1_1 = 1/2 * xi * (phi_1 - phi_1x)**2
    u1_2 = 1/2 * xi * (phi_2 - phi_2x)**2
    
    U = U0_1 * u1_1 + U0_2 * u1_2

    return U * U0_kBT_ratio



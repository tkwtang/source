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


def simple_harmonic_pot(phi_2, params):

    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, delta_beta_1, delta_beta_2, phi_1x, phi_2x , phi_1dcx, phi_2dcx, M_12, x_c = params
    U0_kBT_ratio = U0_1
    U0_2 = U0_2 / U0_1
    U0_1 = 1


    return U0_kBT_ratio * 1/2 * M_12 * (phi_2 - phi_2x)**2 + phi_2dcx

def simple_harmonic_force(phi_2, params):
    
    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, delta_beta_1, delta_beta_2, phi_1x, phi_2x , phi_1dcx, phi_2dcx, M_12, x_c = params # M_12 is already normalized with L_1 and L_2
    U0_kBT_ratio = U0_1
    U0_2 = U0_2 / U0_1
    U0_1 = 1

    return -1 * M_12 * (phi_2 - phi_2x)



def flux_qubit_1D_non_linear_approx_pot(phi_2, params):

    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, delta_beta_1, delta_beta_2, phi_1x, phi_2x , phi_1dcx, phi_2dcx, M_12, x_c = params
    U0_kBT_ratio = U0_1
    U0_2 = U0_2 / U0_1
    U0_1 = 1

    return U0_kBT_ratio * (1/2 * (phi_2 - phi_2x)**2 + beta_2 * np.cos(phi_2) * np.cos(phi_2dcx/2))
    

def flux_qubit_1D_non_linear_approx_force(phi_2, params):
    
    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, delta_beta_1, delta_beta_2, phi_1x, phi_2x , phi_1dcx, phi_2dcx, M_12, x_c = params # M_12 is already normalized with L_1 and L_2
    U0_kBT_ratio = U0_1
    U0_2 = U0_2 / U0_1
    U0_1 = 1
    
    U_dp2 = (phi_2 - phi_2x) - beta_2 * np.sin(phi_2) * np.cos(phi_2dcx/2)


    return -1 * U_dp2


def simple_1D_potential(x, params):
    U0_kBT_ratio, m, c = params
    return 1/2 * U0_kBT_ratio * x**2

def simple_1D_force(x, params):
    U0_kBT_ratio, m, c = params
    return -x

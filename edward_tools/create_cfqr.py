import sys
import os
source_path = os.path.expanduser('~/Project/source/')

import numpy as np
import matplotlib.pyplot as plt
import importlib, os, datetime
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from sus.protocol_designer.protocol import sequential_protocol
from IPython import display
from IPython.display import HTML
from quick_sim import setup_sim

from edward_tools.coupled_fq_potential import coupled_flux_qubit_non_linear_approx_pot, coupled_flux_qubit_non_linear_approx_force, coupled_flux_qubit_non_linear_approx_pot_break_down

from edward_tools.visualization import animate_sim_flux_qubit

import kyle_tools as kt


from edward_tools import coupled_fq_protocol_library, cfq_runner

import edward_tools.cfq_batch_sweep as cfq_batch_sweep
import edward_tools.Analysis_tool.general_analysis_tools as general_analysis_tool
import edward_tools.Analysis_tool.minimum_value_of_potential as minimum_value_of_potential
from edward_tools.couple_flux_qubit_metrics import fidelityEvaluation



coupled_fq_protocol_library = importlib.reload(coupled_fq_protocol_library)
create_system = coupled_fq_protocol_library.create_system
get_potential_shot_at_different_t = coupled_fq_protocol_library.get_potential_shot_at_different_t
get_potential_shot_at_different_t_1D = coupled_fq_protocol_library.get_potential_shot_at_different_t_1D
create_simple_protocol_parameter_dict = coupled_fq_protocol_library.create_simple_protocol_parameter_dict
coupled_fq_runner = importlib.reload(cfq_runner)
coupled_fq_protocol_library = importlib.reload(coupled_fq_protocol_library)
create_system = coupled_fq_protocol_library.create_system
get_potential_along_a_1D_cutline = coupled_fq_protocol_library.get_potential_along_a_1D_cutline
plotCutlines = coupled_fq_protocol_library.plotCutlines


PHI_0 = 2.067833848 * 1e-15
k_B = 1.38e-23
T = 4.2
# T = 7
k_BT = k_B * T

# I_m_factor = 50
# I_m_factor = 15
I_m_factor = 0
time_scale = 1.0


R_1, R_2 = 1 * R_factor, 1 * R_factor                                         # ohm
C_1, C_2 = 500e-15 * C_factor, 500e-15 * C_factor                             # F

L_1, L_2 = 140e-12 * L_factor, 140e-12 * L_factor                             # H 
L_1, L_2 = 5e-12 * L_factor, 5e-12 * L_factor                             # H 
freq = 1/np.sqrt(C_1 * L_1)
characteristic_time = np.sqrt(C_1 * C_factor * L_1 * L_factor)


kwargs['C_factor'] 
{
    "L": 5e-12, "C": 500e-15, "R": 1, 
    "L_factor": 1, "C_factor": 1, "R_factor": 1,
    "I_p": 5e-6, "I_m": 7e-9, "I_m_factor": 0,
    "phi_1_x_on": 0, "phi_2_x_on": 0, "M_12_on": 0, "phi_1_dcx_on": 0, "phi_2_dcx_on": 0,
    "T": 0.5, "gamma": 10
}


def generateParameters(kwargs):
    C_factor = kwargs['C_factor'] 
    L_factor = kwargs['L_factor']
    I_m_factor = kwargs['I_m_factor']
    T = kwargs['T']
    I_p_1, I_p_2 = kwargs['I_p'], kwargs['I_p']
    I_m_1, I_m_2 = kwargs['I_m'], kwargs['I_m']
    R_1, R_2 = kwargs["R"], kwargs["R"]
    C_1, C_2 = kwargs["C"], kwargs["C"]
    L_1, L_2 = kwargs["L"], kwargs["L"] 
    phi_1_x_on, phi_2_x_on = kwargs["phi_1_x_on"], kwargs["phi_2_x_on"] 
    phi_1_dcx_on, phi_2_dcx_on = kwargs["phi_1_dcx_on"], kwargs["phi_2_dcx_on"] 
    M_12_on =  kwargs["M_12_on"]
    gamma = kwargs["gamma"]
    
    PHI_0 = 2.067833848 * 1e-15
    k_B = 1.38e-23
    k_BT = k_B * T
    
    # calculated parameters
    m_c = C_1
    m_1 = C_1
    m_2 = C_2
    x_c = PHI_0 / (2 * np.pi)
    time_scale_factor = 1
    t_c = np.sqrt(L_1 * C_1)


    U0_1 = m_c * x_c**2 / t_c**2
    U0_2 = m_2 * x_c**2 / t_c**2
    kappa_1, kappa_2, kappa_3, kappa_4 = k_BT/U0_1, k_BT/U0_1, k_BT/U0_1, k_BT/U0_1

    lambda_1 = 2 * np.sqrt(L_1 * C_1) / (C_1 * R_1)
    theta_1  = 1
    eta_1    = np.sqrt(np.sqrt(L_1 * C_1)/ (R_1 * C_1)) * np.sqrt(2 * kappa_1 / 1**2)

    lambda_2 = 2 * np.sqrt(L_1 * C_1) / (C_2 * R_2)
    theta_2  = 1 / (C_2/C_1)
    eta_2    = np.sqrt(np.sqrt(L_1 * C_1)/ (R_1 * C_1)) * np.sqrt(2 * kappa_2 * (R_1 * C_1**2) / (R_2 * C_2**2))

    lambda_3 = 2 * np.sqrt(L_1 * C_1) / (C_1 * R_1)
    theta_3  = 4
    eta_3    = np.sqrt(np.sqrt(L_1 * C_1)/ (R_1 * C_1)) * np.sqrt(8 * kappa_3)

    lambda_4 = 2 * np.sqrt(L_1 * C_1) / (C_2 * R_2)
    theta_4  = 4 / (C_2/C_1)
    eta_4    = np.sqrt(np.sqrt(L_1 * C_1)/ (R_1 * C_1)) * np.sqrt(8 * kappa_4 * (R_1 * C_1**2) / (R_2 * C_2**2))

    beta_1 = 2 * np.pi * L_1 * I_p_1 / PHI_0; 
    beta_2 = 2 * np.pi * L_2 * I_p_2 / PHI_0;

    d_beta_1 = 2 * np.pi * L_1 * I_m_1 / PHI_0; 
    d_beta_2 = 2 * np.pi * L_2 * I_m_2 / PHI_0;

    _lambda = np.array([lambda_1, lambda_2, lambda_3, lambda_4])
    _theta  = np.array([theta_1, theta_2, theta_3, theta_4])
    _eta  =   np.array([eta_1, eta_2, eta_3, eta_4])
    
    return {
        "U0_1": U0_1, "U0_2": U0_2, 
        "lambda_1": lambda_1, "theta_1": theta_1, "eta_1": eta_1,
        "lambda_2": lambda_2, "theta_2": theta_2, "eta_2": eta_2,
        "lambda_3": lambda_3, "theta_3": theta_3, "eta_3": eta_3,
        "lambda_4": lambda_4, "theta_4": theta_4, "eta_4": eta_4,
        "beta_1": beta_1, "beta_2": beta_2, "d_beta_1": d_beta_1, "d_beta_2": d_beta_2,
    }


def createCFQR(params, protocol_list, initial_parameter_dict, default_init_state = "../default_init_state.npy", **kwargs):
    C_factor = kwargs['C_factor'] 
    L_factor = kwargs['L_factor']
    I_m_factor = kwargs['I_m_factor']
    T = kwargs['T']
    I_p_1, I_p_2 = kwargs['I_p'], kwargs['I_p']
    I_m_1, I_m_2 = kwargs['I_m'], kwargs['I_m']
    R_1, R_2 = kwargs["R"], kwargs["R"]
    C_1, C_2 = kwargs["C"], kwargs["C"]
    L_1, L_2 = kwargs["L"], kwargs["L"] 
    phi_1_x_on, phi_2_x_on = kwargs["phi_1_x_on"], kwargs["phi_2_x_on"] 
    phi_1_dcx_on, phi_2_dcx_on = kwargs["phi_1_dcx_on"], kwargs["phi_2_dcx_on"] 
    M_12_on =  kwargs["M_12_on"]
    gamma = kwargs["gamma"]
    
    PHI_0 = 2.067833848 * 1e-15
    k_B = 1.38e-23
    k_BT = k_B * T
    
    # calculated parameters
    m_c = C_1
    m_1, m_2 = C_1, C_2
    x_c = PHI_0 / (2 * np.pi)
    time_scale_factor = 1
    t_c = np.sqrt(L_1 * C_1)


    U0_1 = m_c * x_c**2 / t_c**2
    U0_2 = m_2 * x_c**2 / t_c**2
    kappa_1, kappa_2, kappa_3, kappa_4 = k_BT/U0_1, k_BT/U0_1, k_BT/U0_1, k_BT/U0_1

    lambda_1 = 2 * np.sqrt(L_1 * C_1) / (C_1 * R_1)
    theta_1  = 1
    eta_1    = np.sqrt(np.sqrt(L_1 * C_1)/ (R_1 * C_1)) * np.sqrt(2 * kappa_1 / 1**2)

    lambda_2 = 2 * np.sqrt(L_1 * C_1) / (C_2 * R_2)
    theta_2  = 1 / (C_2/C_1)
    eta_2    = np.sqrt(np.sqrt(L_1 * C_1)/ (R_1 * C_1)) * np.sqrt(2 * kappa_2 * (R_1 * C_1**2) / (R_2 * C_2**2))

    lambda_3 = 2 * np.sqrt(L_1 * C_1) / (C_1 * R_1)
    theta_3  = 4
    eta_3    = np.sqrt(np.sqrt(L_1 * C_1)/ (R_1 * C_1)) * np.sqrt(8 * kappa_3)

    lambda_4 = 2 * np.sqrt(L_1 * C_1) / (C_2 * R_2)
    theta_4  = 4 / (C_2/C_1)
    eta_4    = np.sqrt(np.sqrt(L_1 * C_1)/ (R_1 * C_1)) * np.sqrt(8 * kappa_4 * (R_1 * C_1**2) / (R_2 * C_2**2))

    beta_1 = 2 * np.pi * L_1 * I_p_1 / PHI_0; 
    beta_2 = 2 * np.pi * L_2 * I_p_2 / PHI_0;

    d_beta_1 = 2 * np.pi * L_1 * I_m_1 / PHI_0; 
    d_beta_2 = 2 * np.pi * L_2 * I_m_2 / PHI_0;

    _lambda = np.array([lambda_1, lambda_2, lambda_3, lambda_4])
    _theta  = np.array([theta_1, theta_2, theta_3, theta_4])
    _eta  =   np.array([eta_1, eta_2, eta_3, eta_4])
    
    """
    # step 1: Define potential
    """
    coupled_fq_default_param = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x_c]
    [phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound] = np.array([4, 4, 4, 4])/time_scale_factor

    coupled_fq_domain = [[-phi_1_bound, -phi_2_bound, -phi_1dc_bound, -phi_2dc_bound], \
                         [phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound]]

    # coupled_fq_pot = Potential(coupled_flux_qubit_pot_with_offset_at_00_xy, coupled_flux_qubit_force, 14, 4,\
    #                            default_params = coupled_fq_default_param,  relevant_domain = coupled_fq_domain)

    coupled_fq_pot = Potential(coupled_flux_qubit_non_linear_approx_pot, coupled_flux_qubit_non_linear_approx_force, 14, 4, default_params = coupled_fq_default_param,  relevant_domain = coupled_fq_domain)
    
    """
    # step 2: Define initial condition and protocol
    """
    manual_domain=[np.array([-5, -5]), np.array([5, 5])]
    phi_1_dcx , phi_2_dcx = 0, 0
    phi_1_dc, phi_2_dc = phi_1_dcx, phi_2_dcx

    phi_1_x_off, phi_2_x_off = 0, 0
    phi_1_dcx_off, phi_2_dcx_off, M_12_off = 0, 0, 0

    zeroDissipation = False
    params['sim_params'] = [_lambda, _theta, _eta]
    
    if zeroDissipation:
        params['sim_params'] = [_lambda * 0, _theta, _eta * 0]

    params['circuit_parameter'] = {
        "C_factor":C_factor, "L_factor": L_factor, "I_m_factor": I_m_factor, "T": T, 
        "I_p_1": I_p_1, "I_p_2": I_p_2, "I_m_1": I_m_1, "I_m_2": I_m_2,
        "R_1": R_1, "R_2": R_2, "C_1": C_1, "C_2": C_2, "L_1": L_1, "L_2": L_2, 
        "phi_1_x_on": phi_1_x_on, "phi_2_x_on": phi_1_x_on,
        "phi_1_dcx_on": phi_1_dcx_on, "phi_2_dcx_on": phi_2_dcx_on, "M_12_on": M_12_on,
        "gamma": gamma
    }

    protocol_time_array = [item["duration"] for item in protocol_list]
    protocol_time_array.insert(0, 0)
    protocol_time_array = np.cumsum(protocol_time_array)
    protocol_time_index_array = protocol_time_array / params['dt'] - 1
    protocol_time_index_array[0] = 0
    protocol_time_index_array = protocol_time_index_array.astype(int)
    
    
    """
    # step 3: create the relevant storage protocol and computation protocol
    """
    computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, \
                                                                        protocol_list)
    storage_protocol, comp_protocol = create_system(computation_protocol_parameter_dict)

    """
    # step 4: create the coupled_fq_runner
    """
    cfqr = cfq_runner.coupledFluxQubitRunner(potential = coupled_fq_pot, params = params, \
                                                    storage_protocol= storage_protocol, \
                                                    computation_protocol= comp_protocol, measure_all_states=True)
    cfqr.initialize_sim()
    # cfqr.set_sim_attributes(extra_constraint=extra_constraint_00_and_11_only)
    cfqr.set_sim_attributes()
    init_state_saved = cfqr.init_state
    
    return cfqr
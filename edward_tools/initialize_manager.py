import sys, os, importlib
source_path = os.path.expanduser('~/Project/source/')
import numpy as np
import importlib, os, datetime
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from sus.protocol_designer.protocol import sequential_protocol
from IPython import display
from IPython.display import HTML, Image
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter

from quick_sim import setup_sim

# import edward_tools.fq_runner as fq_runner
from edward_tools.coupled_fq_potential import coupled_flux_qubit_non_linear_approx_pot, coupled_flux_qubit_non_linear_approx_force
from edward_tools.visualization import animate_sim_flux_qubit, plotFidelityBarChart, separate_by_state_2
from edward_tools.initial_state_sampling import extra_constraint_00_and_11_only
from NAND_PARAMETERS import *

import kyle_tools as kt
import matplotlib.pyplot as plt

from edward_tools import coupled_fq_protocol_library, cfq_runner, visualization
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


has_velocity = True

PHI_0 = 2.067833848 * 1e-15
k_B = 1.38e-23
T = 4.2
# T = 7
k_BT = k_B * T

C_factor = 1
L_factor = 5
R_factor = 500
# I_m_factor = 50
# I_m_factor = 15
I_m_factor = 0
time_scale = 1.0


I_p_1, I_p_2 = 5e-6 , 5e-6  # Amp
I_m_1, I_m_2 = 7e-9 * I_m_factor, 7e-9 * I_m_factor                           # Amp
R_1, R_2 = 1 * R_factor, 1 * R_factor                                         # ohm
C_1, C_2 = 500e-15 * C_factor, 500e-15 * C_factor                             # F

L_1, L_2 = 140e-12 * L_factor, 140e-12 * L_factor                             # H 
L_1, L_2 = 5e-12 * L_factor, 5e-12 * L_factor                             # H 
freq = 1/np.sqrt(C_1 * L_1)
characteristic_time = np.sqrt(C_1 * C_factor * L_1 * L_factor)


m_c = C_1
m_1 = C_1
m_2 = C_2
x_c = PHI_0 / (2 * np.pi)
time_scale_factor = 1
t_c = np.sqrt(L_1 * C_1)
v_c = x_c / t_c


U0_1 = m_c * x_c**2 / t_c**2 / k_BT
U0_2 = m_2 * x_c**2 / t_c**2 / k_BT
kappa_1, kappa_2, kappa_3, kappa_4 = 1/U0_1, 1/U0_1, 1/U0_1, 1/U0_1

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

gamma = 10


beta_1 = 2 * np.pi * L_1 * I_p_1 / PHI_0; 
beta_2 = 2 * np.pi * L_2 * I_p_2 / PHI_0;

beta_1 = 2.3
beta_2 = 2.3

d_beta_1 = 2 * np.pi * L_1 * I_m_1 / PHI_0; 
d_beta_2 = 2 * np.pi * L_2 * I_m_2 / PHI_0;


_damping_factor = 1
_lambda = np.array([lambda_1, lambda_2, lambda_3, lambda_4])
_theta  = np.array([theta_1, theta_2, theta_3, theta_4])
_eta  =   np.array([eta_1, eta_2, eta_3, eta_4])

v_1 = np.random.normal(0, np.sqrt(k_BT/m_1)) / v_c
v_2 = np.random.normal(0, np.sqrt(k_BT/m_2)) / v_c
v_3 = np.random.normal(0, np.sqrt(k_BT/(m_1/4))) / v_c
v_4 = np.random.normal(0, np.sqrt(k_BT/(m_2/4))) / v_c


"""
# step 0: modify parameters
- All the parameters are stored in a separate file PARAMETER_INPUT
- You can override some of the parameters here.
"""
params = {}
params['N'] = 1000
params['dt'] = 1/100
params['lambda'] = 1
params['beta'] = 1
params['sim_params'] = [_lambda, _theta, _eta]
params['target_work'] = None
params['applyOffset'] = False
params['monitor_work_dist_in_whole_process'] = True # To monitor the work process
params['comment'] = "Experiment 8 (2024/3/17): 4 well, with no compensation for asym, 1/5000"
params['capacitance'] = [C_1, C_2, C_1/4, C_2/4]
params['mass_special'] = [1, 1, 1/4, 1/4]
params['v_c'] = x_c/t_c
params['k_BT'] = k_BT
params['U0'] = U0_1


"""
# step 2: Define initial condition and protocol
"""
manual_domain=[np.array([-5, -5]), np.array([5, 5])]

# µ = 0.06, φ2xdc = 1.79, φ1x = 0.59, and φ2x = 0.10.
    
initial_parameter_dict = {
        "U0_1": U0_1,     "U0_2": U0_2,     "gamma_1": gamma,  "gamma_2": gamma,
        "beta_1": beta_1,   "beta_2": beta_2,   "d_beta_1": d_beta_1 ,   "d_beta_2": d_beta_2,
        "phi_1_x": phi_1_x_off,  "phi_2_x": phi_2_x_off,  "phi_1_dcx": phi_1_dcx_off,  "phi_2_dcx": phi_1_dcx_off,
        "M_12": M_12_off, 'x_c': x_c
}

"""
# step 1: Define potential
"""
coupled_fq_default_param = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x_c]
[phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound] = np.array([4, 4, 4, 4])/time_scale_factor
contour_range = [300, 2000]
    
coupled_fq_domain = [[-phi_1_bound, -phi_2_bound, -phi_1dc_bound, -phi_2dc_bound], \
                     [phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound]]

coupled_fq_pot = Potential(coupled_flux_qubit_non_linear_approx_pot, coupled_flux_qubit_non_linear_approx_force, 14, 4,
                           default_params = initial_parameter_dict,  relevant_domain = coupled_fq_domain)


zeroDissipation = False
# zeroDissipation = True

saveAllStates = True

params['sim_params'] = [_lambda, _theta, _eta]
params['capacitance'] = [C_1, C_2, C_1/4, C_2/4]
params['mass_special'] = [1, 1, 1/4, 1/4]
params['v_c'] = x_c/t_c
params['k_BT'] = k_BT
params['as_step'] = np.s_[::100] # the time step to skep for the all_state
params['percentage'] = 1 # For what percentage of the total sample do you want to keep in the output all_state

if zeroDissipation:
    params['sim_params'] = [_lambda * 0, _theta, _eta * 0]

params['circuit_parameters'] = {
    "C_factor":C_factor, "L_factor": L_factor, "I_m_factor": I_m_factor, "T": T, 
    "I_p_1": I_p_1, "I_p_2": I_p_2, "I_m_1": I_m_1, "I_m_2": I_m_2,
    "R_1": R_1, "R_2": R_2, "C_1": C_1, "C_2": C_2, "L_1": L_1, "L_2": L_2, 
    "characteristic_time": np.sqrt(C_1 * C_factor * L_1 * L_factor),
    "phi_1_x_on": phi_1_x_on_12, "phi_2_x_on": phi_2_x_on_12,
    "phi_1_dcx_on": phi_1_dcx_on_12, "phi_2_dcx_on": phi_2_dcx_on_12, "M_12_on": M_12_on,
    "gamma": gamma
}


# bookmark
initial_parameter_dict["phi_1_dcx"], initial_parameter_dict["phi_2_dcx"], initial_parameter_dict["M_12"] = \
phi_1_dcx_off, phi_2_dcx_off, M_12_off


initial_parameter_dict["phi_1_dcx"] = phi_1_dcx_off
initial_parameter_dict["phi_2_dcx"] = phi_2_dcx_off
initial_parameter_dict["phi_1_x"]   = phi_1_x_off
initial_parameter_dict["phi_2_x"]   = phi_2_x_off
initial_parameter_dict["M_12"]      = M_12_off

m_12_factor = 1
percentage_factor = 0.99

# phi_2_x_on_12 = 0.1 
M_12_on = 0.06
phi_2_x_on_12 = 0.1

CE_1 = {
    "phi_1_x": phi_1_x_on_12, "phi_2_x": phi_2_x_on_12, "M_12": M_12_on, \
    "phi_1_dcx": phi_1_dcx_off, "phi_2_dcx": phi_2_dcx_on_12, "name":"CE_1"
}



# longer CE
t_duration = 100

ratio_array = [0.998] * 5
# ratio_array = [0.9, 0.9, 0.9, 0.9, 0.9]
protocol_list_variable_duration = [
    # forward
    
    # create_CE_Protocol(50, CE_break_1),
    create_CE_Protocol(100, CE_1, ratio_array),
    create_CE_Protocol(20, CE_1),
    # create_CE_Protocol(150, CE_1, np.array([1.3, 0.3, 0.3, 0.3, 1])),
    # create_CE_Protocol(150, CE_1),
    # create_CE_Protocol(30, CE_1, [0.8, 1.2, -0.8, 0.8, 0.8]),
    create_CE_Protocol(400, four_well)

]
# params['as_step'] = np.s_[::] # the time step to skep for the all_state
protocol_list = protocol_list_variable_duration


phi_1_dcx_on = 3.0
phi_2_dcx_on = 3.0
M_12_on = -0.6

regenerate_init_state = False
# regenerate_init_state = True
if regenerate_init_state:
    initial_potential_protocol_list = [create_CE_Protocol(10, four_well)]
    init_state_saved = cfq_batch_sweep.create_initial_state(initial_parameter_dict, initial_potential_protocol_list, coupled_fq_pot, params)
else:
    init_state_saved = np.load("../four_well_default_init_state_N_1000.npy")
    pass

init_state_used = init_state_saved


"""
# step 3: create the relevant storage protocol and computation protocol
"""
def modifiedFunction(t, result):
    return result


computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, \
                                                                    protocol_list)
storage_protocol, comp_protocol = create_system(computation_protocol_parameter_dict, modifiedFunction = None)

def create_cfqr():
    cfqr = cfq_runner.coupledFluxQubitRunner(potential = coupled_fq_pot, params = params, \
                                                    storage_protocol= storage_protocol, \
                                                    computation_protocol= comp_protocol, \
                                             protocol_list = protocol_list, \
                                            has_velocity=has_velocity)

    protocol_time_array, protocol_time_index_array, protocol_all_time_array, protocol_time_all_index_array = cfqr.createProtocolTimeArray(protocol_list, params)
    cfqr.initialize_sim()
    # cfqr.set_sim_attributes(init_state=init_state_saved)
    cfqr.set_sim_attributes(init_state=init_state_used)
    return cfqr

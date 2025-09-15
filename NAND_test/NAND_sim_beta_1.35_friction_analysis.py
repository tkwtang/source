from mpi4py import MPI
comm = MPI.COMM_WORLD
size, rank = comm.Get_size(), comm.Get_rank()


import sys, os
sys.path.append(os.path.expanduser('~/source'))
import numpy as np
import importlib, os, datetime, pickle
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
import importlib

import kyle_tools as kt
import matplotlib.pyplot as plt


from edward_tools import coupled_fq_protocol_library, cfq_runner
from edward_tools import coupled_fq_protocol_library

import edward_tools.cfq_batch_sweep as cfq_batch_sweep
import edward_tools.Analysis_tool.general_analysis_tools as general_analysis_tool
# from edward_tools.Analysis_tool.general_analysis_tools import show_phi_dc_with_time
import edward_tools.Analysis_tool.minimum_value_of_potential as minimum_value_of_potential
from edward_tools.couple_flux_qubit_metrics import fidelityEvaluation
from edward_tools import visualization

from edward_tools.pot_analysis_helper_functions import get_XYU, pot_function, find_all_critical_points_for_all_potential

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


index_array = [0, 1, 2, 3, 4, 5, 6]
index_choice = 4
zeroDissipation = False
# zeroDissipation = True
experiment_idenitfication = f"Experiment {index_array[index_choice]} (2025-08-25)"
experiment_comment = f"{experiment_idenitfication}: projectile speed and fidelity analysis of harmonic potential with beta = 1.35 and with friction"




def create_parameter_list(cfqr, _protocol):
    params_list_used = cfqr.protocol.get_params(0)
    params_list_used[8] = _protocol['phi_1x'] # phi_1x
    params_list_used[9] = _protocol['phi_2x'] # phi_2x"
    params_list_used[10] = _protocol['phi_1xdc'] # phi_1xdc
    params_list_used[11] = _protocol['phi_2xdc'] # phi_2xdc
    params_list_used[12] = _protocol['mu_12'] # mu
    return params_list_used

# control swap protocol
def createProtocol(duration, protocol, name):
    protocol_new = {
        "phi_1xdc": protocol["phi_1xdc"], "phi_2xdc": protocol["phi_2xdc"], 
        "phi_1x": protocol["phi_1x"], "phi_2x": protocol["phi_2x"], 
        "mu_12": protocol["mu_12"], "duration": duration, "name": name
    }
    return protocol_new

has_velocity = True

PHI_0 = 2.067833848 * 1e-15
k_B = 1.38e-23
T = 4.2
k_BT = k_B * T

C_factor = 1
L_factor = 1
R_factor = 75
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

m_c, m_1, m_2 = C_1, C_1, C_2
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

gamma = 9

beta_1 = 2 * np.pi * L_1 * I_p_1 / PHI_0; 
beta_2 = 2 * np.pi * L_2 * I_p_2 / PHI_0;

beta_1, beta_2 = [1.35]*2 # barrier kBT = 50
# beta_1, beta_2 = [1.5]*2 # barrier kBT = 100
# beta_1, beta_2 = [2.3]*2 # barrier KBT = 400
d_beta_1, d_beta_2 = 2 * np.pi * L_1 * I_m_1 / PHI_0, 2 * np.pi * L_2 * I_m_2 / PHI_0


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
params['dt'] = 1/1000
params['lambda'] = 1
params['beta'] = 1
params['sim_params'] = [_lambda, _theta, _eta]
params['target_work'] = None
params['applyOffset'] = False
params['measureWorkWithOffset'] = True
params['monitor_work_dist_in_whole_process'] = True # To monitor the work process
params['comment'] = experiment_comment
params['capacitance'] = np.array([C_1, C_2, C_1/4, C_2/4])
params['mass'] = np.array([1, 1, 1/4, 1/4])
params['v_c'] = x_c/t_c
params['k_BT'] = k_BT
params['U0'] = U0_1
params['as_step'] = np.s_[::10] # the time step to skep for the all_state
params['percentage'] = 1 # For what percentage of the total sample do you want to keep in the output all_state


"""
# step 2: Define initial condition and protocol
"""
"""
# step 2: Define initial condition and protocol
"""
manual_domain=[np.array([-10, -10]), np.array([10, 10])]

initial_parameter_dict = {
        "U0_1": U0_1,     "U0_2": U0_2,     "gamma_1": gamma,  "gamma_2": gamma,
        "beta_1": beta_1,   "beta_2": beta_2,   "d_beta_1": d_beta_1 ,   "d_beta_2": d_beta_2,
        "phi_1x": phi_1x_off,  "phi_2x": phi_2x_off,  "phi_1xdc": phi_1xdc_off,  "phi_2xdc": 1.8,
        "mu_12": mu_12_off, 'x_c': x_c
}



"""
# step 1: Define potential
"""
"""
# step 1: Define potential
"""
coupled_fq_default_param = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x_c]
[phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound] = np.array([4, 4, 4, 4])/time_scale_factor
contour_range = [300, 2000]
    
coupled_fq_domain = [[-phi_1_bound, -phi_2_bound, -phi_1dc_bound, -phi_2dc_bound], \
                     [phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound]]

coupled_fq_pot = Potential(coupled_flux_qubit_non_linear_approx_pot, coupled_flux_qubit_non_linear_approx_force, 14, 4,\
                           default_params = initial_parameter_dict,  relevant_domain = coupled_fq_domain)
    
    


saveAllStates = True

params['sim_params'] = [_lambda, _theta, _eta]

if zeroDissipation:
    params['sim_params'] = [_lambda * 0, _theta, _eta * 0]

params['circuit_parameters'] = {
    "C_factor":C_factor, "L_factor": L_factor, "R_factor": R_factor, "I_m_factor": I_m_factor, "T": T, 
    "I_p_1": I_p_1, "I_p_2": I_p_2, "I_m_1": I_m_1, "I_m_2": I_m_2,
    "R_1": R_1, "R_2": R_2, "C_1": C_1, "C_2": C_2, "L_1": L_1, "L_2": L_2, 
    "characteristic_time": np.sqrt(C_1 * C_factor * L_1 * L_factor),
    "gamma": gamma
}


# harmonic potential
initial_parameter_dict["phi_1xdc"] = 0
initial_parameter_dict["phi_2xdc"] = 0
initial_parameter_dict["phi_1x"]   = 0
initial_parameter_dict["phi_2x"]   = 0
initial_parameter_dict["mu_12"]      = 0



mapping_state_1_to_state_2_dict_SWAP = {'00': ['01'], '01': ['00'], '10': ['10'], '11': ['11']}
mapping_state_1_to_state_2_dict_CE = {'00': ['00'], '01': ['01'], '10': ['00'], '11': ['11']}
mapping_state_1_to_state_2_dict_storage = {'00': ['00'], '01': ['01'], '10': ['10'], '11': ['11']}
mapping_state_1_to_state_2_erasure_flip = {'00': ['10'], '01': ['11'], '10': ['10'], '11': ['01']}
mapping_dict = mapping_state_1_to_state_2_erasure_flip


params["mapping_state_1_to_state_2_dict"] = mapping_dict



# beta = 2.3
# mu_factor_h0 = 0.25
# mu_factor_h1 = 0.35

if beta_1 == 1.5:
    mu_factor_h0 = 0.30
    mu_factor_h1 = 0.45
    # for beta = 1.5
    protocol_harmonic_h0 = {"phi_1x": 0.0, "phi_2x": 0.0, "mu_12": 0.0, "phi_1xdc": 0, "phi_2xdc": 0, "name": "four_well", "color": "r", "linestyle": "-"} 
    protocol_harmonic_h1 = {"phi_1x": 0.1 * mu_factor_h0, "phi_2x": 0.0, "mu_12": 0.068 * mu_factor_h0, "phi_1xdc": 1.1, "phi_2xdc": 0, "name": "h1", "color": "r", "linestyle": "-"} 
    protocol_harmonic_h2 = {"phi_1x": 0.1 * mu_factor_h0, "phi_2x": 0.0, "mu_12": 0.068 * mu_factor_h0, "phi_1xdc": 1.8, "phi_2xdc": 0, "name": "h2", "color": "g", "linestyle": "-"} 
    protocol_harmonic_h3 = {"phi_1x": 0.1 * mu_factor_h1, "phi_2x": 0.0, "mu_12": 0.068 * mu_factor_h1, "phi_1xdc": 1.1, "phi_2xdc": 0, "name": "h3", "color": "b", "linestyle": "-"} 
    # protocol_harmonic_h4 = {"phi_1x": 0. * mu_factor_h0, "phi_2x": 0.0, "mu_12": 0.00 * mu_factor_h0, "phi_1xdc": 1.0, "phi_2xdc": 0, "name": "harmonic", "color": "r", "linestyle": "-"} 
    cutline_value = 1.5

    time_ratio = 1
    duration_fast_set = [20, 15, 20]
    duration_slwo_set = [25, 15, 25]
    duration_set = duration_fast_set

    protocol_list_harmonic = [
        createProtocol(duration_set[0] * time_ratio, protocol_harmonic_h2, name = "flip erasure"),
        createProtocol(duration_set[1] * time_ratio, protocol_harmonic_h3, name = "raise barrier"),
        createProtocol(duration_set[2] * time_ratio, protocol_harmonic_h0, name = "4 well")
    ] 

if beta_1 == 1.35:
    mu_factor_h0 = 0.45
    mu_factor_h1 = 0.9
    protocol_harmonic_h0 = {"phi_1x": 0.0, "phi_2x": 0.0, "mu_12": 0.0, "phi_1xdc": 0, "phi_2xdc": 0, "name": "four_well", "color": "r", "linestyle": "-"} 
    protocol_harmonic_h2 = {"phi_1x": 0.1 * mu_factor_h0, "phi_2x": 0.0, "mu_12": 0.076 * mu_factor_h0, "phi_1xdc": 1.65, "phi_2xdc": 0, "name": "h2", "color": "g", "linestyle": "-"} 
    protocol_harmonic_h3 = {"phi_1x": 0.1 * mu_factor_h1, "phi_2x": 0.0, "mu_12": 0.076 * mu_factor_h1, "phi_1xdc": 0.6, "phi_2xdc": 0, "name": "h3", "color": "b", "linestyle": "-"} 
    # protocol_harmonic_h4 = {"phi_1x": 0. * mu_factor_h0, "phi_2x": 0.0, "mu_12": 0.00 * mu_factor_h0, "phi_1xdc": 1.0, "phi_2xdc": 0, "name": "harmonic", "color": "r", "linestyle": "-"} 
    cutline_value = 1.3

    protocol_list_harmonic = [
        createProtocol(14, protocol_harmonic_h2, name = "flip erasure"),
        createProtocol(2, protocol_harmonic_h2, name = "flip erasure"),
        createProtocol(10, protocol_harmonic_h3, name = "raise barrier"),
        # createProtocol(15, protocol_harmonic_h3, name = "raise barrier"),
        createProtocol(20, protocol_harmonic_h0, name = "four well"),

    ] 



if beta_1 == 1.35:
    mu_factor_h1 = 0.0
    mu_factor_h2 = 0.8
    mu_factor_h3 = 0.8
    protocol_harmonic_h0 = {"phi_1x": 0.0, "phi_2x": 0.0, "mu_12": 0.0, "phi_1xdc": 0, "phi_2xdc": 0, "name": "four_well", "color": "r", "linestyle": "-"} 
    protocol_harmonic_h1 = {"phi_1x": 0.1 * mu_factor_h1, "phi_2x": 0.0, "mu_12": 0.0765 * mu_factor_h1, "phi_1xdc": 1.2, "phi_2xdc": 0, "name": "h1", "color": "r", "linestyle": "-"} 
    protocol_harmonic_h2 = {"phi_1x": 0.1 * mu_factor_h2, "phi_2x": 0.0, "mu_12": 0.0765 * mu_factor_h2, "phi_1xdc": 1.7, "phi_2xdc": 0, "name": "h2", "color": "g", "linestyle": "-"} 
    protocol_harmonic_h3 = {"phi_1x": 0.1 * mu_factor_h3, "phi_2x": 0.0, "mu_12": 0.0765 * mu_factor_h3, "phi_1xdc": 0.6, "phi_2xdc": 0, "name": "h3", "color": "b", "linestyle": "-"} 
    # protocol_harmonic_h4 = {"phi_1x": 0. * mu_factor_h0, "phi_2x": 0.0, "mu_12": 0.00 * mu_factor_h0, "phi_1xdc": 1.0, "phi_2xdc": 0, "name": "harmonic", "color": "r", "linestyle": "-"} 
    cutline_value = 1.3

    protocol_list_harmonic = [
        createProtocol(1, protocol_harmonic_h1, name = "flip erasure"),
        createProtocol(1, protocol_harmonic_h2, name = "flip erasure"),
        createProtocol(2, protocol_harmonic_h2, name = "flip erasure"),
        createProtocol(5, protocol_harmonic_h3, name = "raise barrier"),
        # createProtocol(15, protocol_harmonic_h3, name = "raise barrier"),
        # createProtocol(20, protocol_harmonic_h0, name = "four well"),
        createProtocol(30, protocol_harmonic_h0, name = "four well"),

    ] 

# params_array = [(0.5, -0.9), [(0.76, -0.65), (0.89, -0.55), (1.03, -0.4), (1.1, -0.3)], (1.18, -0.25)]



# if beta_1 == 1.35:
#     mu_factor_h1 = 0.0
#     mu_factor_h2 = 0.4
#     mu_factor_h3 = 0.4
#     protocol_harmonic_h0 = {"phi_1x": 0.0, "phi_2x": 0.0, "mu_12": 0.0, "phi_1xdc": 0, "phi_2xdc": 0, "name": "four_well", "color": "pink", "linestyle": "-"} 
#     # protocol_harmonic_h1 = {"phi_1x": 0.1 * mu_factor_h1, "phi_2x": 0.0, "mu_12": 0.0765 * mu_factor_h1, "phi_1xdc": 0.8, "phi_2xdc": 0, "name": "h1", "color": "r", "linestyle": "-"} 
#     protocol_harmonic_h1 = {"phi_1x": 0.1 * mu_factor_h1, "phi_2x": 0.0, "mu_12": 0.0765 * mu_factor_h1, "phi_1xdc": 1.1, "phi_2xdc": 0, "name": "h1", "color": "r", "linestyle": "-"} 
#     protocol_harmonic_h2 = {"phi_1x": 0.1 * mu_factor_h2, "phi_2x": 0.0, "mu_12": 0.0765 * mu_factor_h2, "phi_1xdc": 2.2554232466933932, "phi_2xdc": 0, "name": "h2", "color": "g", "linestyle": "-"} 
#     protocol_harmonic_h3 = {"phi_1x": 0.1 * mu_factor_h3, "phi_2x": 0.0, "mu_12": 0.0765 * mu_factor_h3, "phi_1xdc": 1.1, "phi_2xdc": 0, "name": "h3", "color": "b", "linestyle": "-"} 
#     # protocol_harmonic_h4 = {"phi_1x": 0. * mu_factor_h0, "phi_2x": 0.0, "mu_12": 0.00 * mu_factor_h0, "phi_1xdc": 1.0, "phi_2xdc": 0, "name": "harmonic", "color": "r", "linestyle": "-"} 
#     cutline_value = 1.3

#     protocol_list_harmonic = [
#         createProtocol(15, protocol_harmonic_h1, name = "flip erasure"),
#         createProtocol(6, protocol_harmonic_h2, name = "flip erasure"),
#         createProtocol(6, protocol_harmonic_h3, name = "raise barrier"),
#         createProtocol(15, protocol_harmonic_h0, name = "four well"),
#     ] 

# params_array = [(0.5, -0.9), [(0.76, -0.65), (0.89, -0.55), (1.03, -0.4), (1.1, -0.3)], (1.18, -0.25)]
parameter_index = int(sys.argv[1])
params_array = [(4, 0), (4.5, 0), (5, 0), (5.5, 0), (6, 0)]
params_array = [(4, 1), (4.5, 1), (5, 1), (5.5, 1), (6, 0)]
t_2, t_2_5 = params_array[parameter_index]

if beta_1 == 1.35:
    mu_factor_h1 = 0.0
    mu_factor_h2 = 0.5
    mu_factor_h3 = 0.5
    protocol_harmonic_h0 = {"phi_1x": 0.0, "phi_2x": 0.0, "mu_12": 0.0, "phi_1xdc": 0, "phi_2xdc": 0, "name": "four_well", "color": "pink", "linestyle": "-"} 
    # protocol_harmonic_h1 = {"phi_1x": 0.1 * mu_factor_h1, "phi_2x": 0.0, "mu_12": 0.0765 * mu_factor_h1, "phi_1xdc": 0.8, "phi_2xdc": 0, "name": "h1", "color": "r", "linestyle": "-"} 
    protocol_harmonic_h1 = {"phi_1x": 0.1 * mu_factor_h1, "phi_2x": 0.0, "mu_12": 0.0765 * mu_factor_h1, "phi_1xdc": 1.1, "phi_2xdc": 0, "name": "h1", "color": "r", "linestyle": "-"} 
    protocol_harmonic_h2 = {"phi_1x": 0.1 * mu_factor_h2, "phi_2x": 0.0, "mu_12": 0.0765 * mu_factor_h2, "phi_1xdc": 1.7, "phi_2xdc": 0, "name": "h2", "color": "g", "linestyle": "-"} 
    protocol_harmonic_h3 = {"phi_1x": 0.1 * mu_factor_h3, "phi_2x": 0.0, "mu_12": 0.0765 * mu_factor_h3, "phi_1xdc": 1.1, "phi_2xdc": 0, "name": "h3", "color": "b", "linestyle": "-"} 
    # protocol_harmonic_h4 = {"phi_1x": 0. * mu_factor_h0, "phi_2x": 0.0, "mu_12": 0.00 * mu_factor_h0, "phi_1xdc": 1.0, "phi_2xdc": 0, "name": "harmonic", "color": "r", "linestyle": "-"} 
    cutline_value = 1.3
    
    
    protocol_list_harmonic = [
        createProtocol(15, protocol_harmonic_h1, name = "flip erasure"),
        createProtocol(t_2, protocol_harmonic_h2, name = "flip erasure"),
        createProtocol(t_2, protocol_harmonic_h3, name = "raise barrier"),
        createProtocol(15, protocol_harmonic_h0, name = "four well"),
    ] 

    if t_2_5 > 0:
        protocol_list_harmonic = protocol_list_harmonic[0:2] + [createProtocol(t_2_5, protocol_harmonic_h2, name = "flip erasure")] + protocol_list_harmonic[2:]

protocol_list = protocol_list_harmonic


initial_state_array = [f"../saved_init_state/init_state_database/T=4.2K, N = 1000, beta = {beta_1}"]

import random

initial_state_folder = initial_state_array[0]
initial_state_folder_path = os.listdir(initial_state_folder)
initial_state_chosen = os.path.join(initial_state_folder, random.choice(initial_state_folder_path))


regenerate_init_state = False
init_state_saved = None
# regenerate_init_state = True
if regenerate_init_state:
    four_well = {'phi_1x': 0, 'phi_2x': 0, 'mu_12': 0, 'phi_1xdc': 0, 'phi_2xdc': 0, 'name': 'four well'}
    initial_potential_protocol_list = [create_CE_Protocol(100, four_well)]
    init_params = params.copy()
    init_params['sim_params'] = [_lambda, _theta, _eta]
    init_params['dt'] = 1/500
    init_state_saved = cfq_batch_sweep.create_initial_state(initial_parameter_dict, initial_potential_protocol_list, coupled_fq_pot, init_params)
    
    # np.save(f"../saved_init_state/four_well_default_init_state_N_1000_T={T}, L={L_1}, beta = {beta_1}, phi_1xdc = {phi_1xdc}.npy", init_state_saved)
else:
    init_state_saved = np.load(initial_state_chosen)

init_state_used = init_state_saved


"""
# step 3: create the relevant storage protocol and computation protocol
"""
def modifiedFunction(t, result):
    return result


computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, \
                                                                    protocol_list)
storage_protocol, comp_protocol = create_system(computation_protocol_parameter_dict, modifiedFunction = None)

cfqr = cfq_runner.coupledFluxQubitRunner(potential = coupled_fq_pot, params = params, \
                                                storage_protocol= storage_protocol, \
                                                computation_protocol= comp_protocol, \
                                        protocol_list = protocol_list, \
                                        has_velocity=has_velocity)

protocol_time_array, protocol_time_index_array, protocol_all_time_array, protocol_time_all_index_array = cfqr.createProtocolTimeArray(protocol_list, params)
cfqr.initialize_sim()
# cfqr.set_sim_attributes(init_state=init_state_saved)
cfqr.set_sim_attributes(init_state=init_state_used)
print(f"mean KE = {np.mean(cfqr.system.get_kinetic_energy(init_state_used))}")


"""
actual simulations
"""

RESET = "\033[0m"
RED = "\033[91m"
print(f"{RED} {params['comment']} {RESET}")
print(f"{RED} {protocol_list} {RESET}")
pickle_save_path = os.path.join("..", "coupled_flux_qubit_protocol", "simulation_protocol_history")

manual_domain=[np.array([-5, -5]), np.array([5, 5])]

print(f"characteristic_time = {characteristic_time * 10e9:.3g}ns, total speed = {1/(characteristic_time * protocol_time_array[-1]) / 1e9:.3g}GHz")

simResult = cfq_batch_sweep.simulateSingleCoupledFluxQubit(params, initial_parameter_dict, protocol_list, \
                        potential = coupled_fq_pot, potential_default_param = coupled_fq_default_param, mapping_state_1_to_state_2_dict = mapping_dict,         
                        initial_state = init_state_used, manual_domain = manual_domain, \
                        phi_1xdc = phi_1xdc,  phi_2xdc = phi_2xdc, measure_all_states=saveAllStates, \
                        has_velocity = has_velocity, pickle_save_path=pickle_save_path)




"""
# save results
"""
if regenerate_init_state:
    np.save(f"../saved_init_state/init_state_database/T=4.2K, N = 1000, beta = {beta_1}/four_well_default_init_state_{simResult["simulation_data"]["simulation_id"]}.npy", init_state_saved)

cfq_batch_sweep.saveSimulationResult(simResult, U0_1, timeOrStep = 'step', save = True, save_final_state = True, saveFolderPath = "../coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery")


failure_path_file_path = f"../coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{experiment_idenitfication.replace("/", "-")}_failure_path.npy"
failure_particles = simResult['cfqr'].get_failure_particles(mapping_dict)

if not os.path.isfile(failure_path_file_path):
    np.save(failure_path_file_path, np.array([]))


failure_particles_array = np.load(failure_path_file_path)


if len(failure_particles) > 0 and len(failure_particles_array) == 0:
    np.save(failure_path_file_path, failure_particles)

if len(failure_particles) > 0 and len(failure_particles_array) > 0:
    failure_particles_new_array = np.vstack([failure_particles_array, failure_particles])
    np.save(failure_path_file_path, failure_particles_new_array)

    


step_time_array = np.cumsum(np.array([x["duration"] for x in protocol_list]))/params['dt']
name_array = [x["name"] for x in protocol_list]
mean_work = np.mean(simResult["work_distribution"])
jarzyn_term = np.mean(np.exp(-simResult["work_distribution"]))
print("The simulation id is", simResult["simulation_data"]["simulation_id"])
print(f"{RED} {params['comment']} {RESET}")
print(f"N = {params['N']}, dt = {params['dt']}")
print(f"L_factor = {L_factor}, C_factor = {C_factor}, I_m_factor = {I_m_factor}")
print([x["duration"] for x in protocol_list])
print(f'jarzyn = {jarzyn_term}, mean work = {mean_work}')
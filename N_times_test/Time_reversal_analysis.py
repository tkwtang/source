import sys, os
sys.path.append(os.path.expanduser('~/source'))


import numpy as np
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from sus.protocol_designer.protocol import sequential_protocol
from IPython import display
from IPython.display import HTML
from quick_sim import setup_sim
from edward_tools.coupled_fq_potential import coupled_flux_qubit_non_linear_approx_pot, coupled_flux_qubit_non_linear_approx_force
import edward_tools.fq_runner as fq_runner
from edward_tools.visualization import animate_sim_flux_qubit
from PARAMETER_INPUT import *
from PARAMETER_INPUT import _lambda, _theta, _eta

import kyle_tools as kt
import matplotlib.pyplot as plt

import importlib, os, datetime
from edward_tools import coupled_fq_protocol_library, cfq_runner
from edward_tools import coupled_fq_protocol_library
import edward_tools.cfq_batch_sweep as cfq_batch_sweep

coupled_fq_protocol_library = importlib.reload(coupled_fq_protocol_library)
create_system = coupled_fq_protocol_library.create_system
get_potential_shot_at_different_t = coupled_fq_protocol_library.get_potential_shot_at_different_t
get_potential_shot_at_different_t_1D = coupled_fq_protocol_library.get_potential_shot_at_different_t_1D
create_simple_protocol_parameter_dict = coupled_fq_protocol_library.create_simple_protocol_parameter_dict
coupled_fq_runner = importlib.reload(cfq_runner)
coupled_fq_protocol_library = importlib.reload(coupled_fq_protocol_library)
create_system = coupled_fq_protocol_library.create_system

"""
# step 0: modify parameters
- All the parameters are stored in a separate file PARAMETER_INPUT
- You can override some of the parameters here.
"""
params = {}
params['N'] = 10_000
params['dt'] = 1/5000
params['lambda'] = 1
params['beta'] = 1
params['sim_params'] = [_lambda, _theta, _eta]
params['target_work'] = None
params['comment'] = "Experiment 7 (2024/7/3): time reverse prtocol 1 and 8"
as_step_parameter = np.s_[::100]

insert_four_well = True
compensation_for_asym = True
time_scale = 1

I_m_factor = 15
I_m_factor = 0
C_factor = 100
L_factor = 0.35

# override parameters
PHI_0 = 2.067833848 * 1e-15
k_B = 1.38e-23
T = 0.5
k_BT = k_B * T

I_p_1, I_p_2 = 2e-6 , 2e-6  # Amp
I_m_1, I_m_2 = 7e-9 * I_m_factor, 7e-9 * I_m_factor                                # Amp
R_1, R_2 = 371, 371                                # ohm
C_1, C_2 = 4e-9 * C_factor, 4e-9 * C_factor                              # F
L_1, L_2 = 1e-9 * L_factor, 1e-9 * L_factor                              # H 

quick_doubler = lambda x1, x2: np.hstack([np.array([x1] * 2), np.array([x2]*2)])
I_p, I_m = quick_doubler(I_p_1, I_p_2), quick_doubler(I_m_1, I_m_2)

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
eta_1    = np.sqrt(np.sqrt(L_1 * C_1)/ (R_1 * C_1)) * np.sqrt(2 * kappa_1)

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

coupled_fq_pot = Potential(coupled_flux_qubit_non_linear_approx_pot, coupled_flux_qubit_non_linear_approx_force, 14, 4,\
                           default_params = coupled_fq_default_param,  relevant_domain = coupled_fq_domain)


"""
# step 2: Define initial condition and protocol
"""
manual_domain=[np.array([-5, -5]), np.array([5, 5])]
# phi_1_dcx, phi_2_dcx = 3, 3
phi_1_dcx, phi_2_dcx = 0, 0
phi_1_dc, phi_2_dc = phi_1_dcx, phi_2_dcx

phi_1_x_off = 0
phi_2_x_off = 0
phi_1_dcx_off = 0
phi_2_dcx_off = 0
M_12_off = 0

# bookmark
# phi_1_dcx_on = 2.8
# phi_2_dcx_on = 2.8
# M_12_on = -0.6
# phi_1_x_on = -0.0075
# phi_2_x_on = -0.005
# phi_1_x_on = -0.009142857142857001
# phi_2_x_on = -0.009142857142857001
if compensation_for_asym:
    if I_m_factor == 1:        
        phi_1_x_on = -0.00934
        phi_2_x_on = -0.00916
    if I_m_factor == 5:
        phi_1_x_on = -0.046710526315789
        phi_2_x_on = -0.046842105263157005
    if I_m_factor == 23:
        phi_1_x_on = -0.337
        phi_2_x_on = -0.337
else:
    phi_1_x_on = 0
    phi_2_x_on = 0

phi_1_x_on = 0
phi_2_x_on = 0  

### for I_m = 23
# phi_1_x_on = -0.337
# phi_2_x_on = -0.337

### for I_m = 15
#phi_1_x_on = -0.1938
#phi_2_x_on = -0.1938

phi_1_dcx_on = 3.0
phi_2_dcx_on = 3.0
M_12_on = -0.6



initial_parameter_dict = {
        "U0_1": U0_1,     "U0_2": U0_2,     "gamma_1": gamma,  "gamma_2": gamma,
        "beta_1": beta_1,   "beta_2": beta_2,   "d_beta_1": d_beta_1 ,   "d_beta_2": d_beta_2,
        "phi_1_x": phi_1_x_off,  "phi_2_x": phi_2_x_off,  "phi_1_dcx": phi_1_dcx_off,  "phi_2_dcx": phi_1_dcx_off,
        "M_12": M_12_off, 'x_c': x_c
}



zeroDissipation = False

params['capacitance'] = [C_1, C_2, C_1/4, C_2/4]
params['mass_special'] = [1, 1, 1/4, 1/4]
params['v_c'] = x_c/t_c
params['k_BT'] = k_BT
params['U0'] = U0_1
params['sim_params'] = [_lambda, _theta, _eta]
if zeroDissipation:
    params['sim_params'] = [_lambda * 0, _theta, _eta * 0]

params['circuit_parameters'] = {
    "C_factor":C_factor, "L_factor": L_factor, "I_m_factor": I_m_factor, "T": T, 
    "I_p_1": I_p_1, "I_p_2": I_p_2, "I_m_1": I_m_1, "I_m_2": I_m_2,
    "R_1": R_1, "R_2": R_2, "C_1": C_1, "C_2": C_2, "L_1": L_1, "L_2": L_2, 
    "phi_1_x_on": phi_1_x_on, "phi_2_x_on": phi_2_x_on, 
    "phi_1_dcx_on": phi_1_dcx_on, "phi_2_dcx_on": phi_2_dcx_on, "M_12_on": M_12_on
}

    
TR_initial_condition = [
    (phi_1_dcx_off, phi_2_dcx_off, M_12_off), 
    (phi_1_dcx_off, phi_2_dcx_on,  M_12_off), 
    (phi_1_dcx_off, phi_2_dcx_on,  M_12_on), 
    (phi_1_dcx_off, phi_2_dcx_off, M_12_on), 
    (phi_1_dcx_off, phi_2_dcx_off, M_12_off), 
    (phi_1_dcx_on,  phi_2_dcx_off, M_12_off)]

protocol_index = 0

# bookmark
# initial_parameter_dict["phi_1_dcx"], initial_parameter_dict["phi_2_dcx"], initial_parameter_dict["M_12"] = TR_initial_condition[protocol_index-1]
initial_parameter_dict["phi_1_dcx"], initial_parameter_dict["phi_2_dcx"], initial_parameter_dict["M_12"] = \
TR_initial_condition[0]


zeroDissipation = False
saveAllStates = True
params['sim_params'] = [_lambda, _theta, _eta]
if zeroDissipation:
    params['sim_params'] = [_lambda * 0, _theta, _eta * 0]

params['circuit_parameters'] = {
    "C_factor":C_factor, "L_factor": L_factor, "I_m_factor": I_m_factor, "T": T, 
    "I_p_1": I_p_1, "I_p_2": I_p_2, "I_m_1": I_m_1, "I_m_2": I_m_2,
    "R_1": R_1, "R_2": R_2, "C_1": C_1, "C_2": C_2, "L_1": L_1, "L_2": L_2, 
    "phi_1_x_on": phi_1_x_on, "phi_2_x_on": phi_1_x_on,
    "phi_1_dcx_on": phi_1_dcx_on, "phi_2_dcx_on": phi_2_dcx_on, "M_12_on": M_12_on,
    "gamma": gamma
}
    
TR_initial_condition = [
    (phi_1_dcx_off, phi_2_dcx_off, M_12_off), 
    (phi_1_dcx_off, phi_2_dcx_on,  M_12_off), 
    (phi_1_dcx_off, phi_2_dcx_on,  M_12_on), 
    (phi_1_dcx_off, phi_2_dcx_off, M_12_on), 
    (phi_1_dcx_off, phi_2_dcx_off, M_12_off), 
    (phi_1_dcx_on,  phi_2_dcx_off, M_12_off)]

protocol_index = 0

# bookmark
initial_parameter_dict["phi_1_dcx"], initial_parameter_dict["phi_2_dcx"], initial_parameter_dict["M_12"] = \
TR_initial_condition[0]



def protocol_maker(time, _p):
    return {
        "duration": time, "name": _p["name"], "phi_1_x": _p["phi_1_x"], "phi_2_x": _p["phi_2_x"],
        "M_12": _p["M_12"], "phi_1_dcx": _p["phi_1_dcx"], "phi_2_dcx": _p["phi_2_dcx"]
    }





_P = [
    {"duration":50 * time_scale, "name":"return to 4 well potential (a)",
        "phi_1_x": phi_1_x_off, "phi_2_x": phi_2_x_off , "M_12": M_12_off,  
        "phi_1_dcx": phi_1_dcx_off, "phi_2_dcx": phi_2_dcx_off,  
    },
    
    # forward
    {"duration": 50 * time_scale, "name":"(t_1) mix in y direction (a)",
        "phi_1_x": phi_1_x_off, "phi_2_x": phi_2_x_on * 0.1, "M_12": M_12_off,  
        "phi_1_dcx": phi_1_dcx_off, "phi_2_dcx": phi_2_dcx_on * 0.5,  
    },
    {"duration":200 * time_scale, "name":"(t_2) mix in y direction (b)",
        "phi_1_x": phi_1_x_off, "phi_2_x": phi_2_x_on, "M_12": M_12_off,  
        "phi_1_dcx": phi_1_dcx_off, "phi_2_dcx": phi_2_dcx_on,  
    },
    {"duration":50 * time_scale, "name":"(t_3) conditional tilt",
        "phi_1_x": phi_1_x_off, "phi_2_x": phi_2_x_on, "M_12": M_12_on,  
        "phi_1_dcx": phi_1_dcx_off, "phi_2_dcx": phi_2_dcx_on,  
    },
    {"duration":50 * time_scale, "name":"(t_4) return to 4 well",
        "phi_1_x": phi_1_x_off, "phi_2_x": phi_2_x_off, "M_12": M_12_off,  
        "phi_1_dcx": phi_1_dcx_off, "phi_2_dcx": phi_2_dcx_off,  
    },
    {"duration":50 * time_scale, "name":"(t_5) conditional tilt",
        "phi_1_x": phi_1_x_on, "phi_2_x": phi_2_x_off, "M_12": M_12_on,  
        "phi_1_dcx": phi_1_dcx_on, "phi_2_dcx": phi_2_dcx_off,  
    },
    {"duration":50 * time_scale, "name":"(t_6) mix in x direction",
        "phi_1_x": phi_1_x_on, "phi_2_x": phi_2_x_off, "M_12": M_12_off,  
        "phi_1_dcx": phi_1_dcx_on, "phi_2_dcx": phi_2_dcx_off,  
    },
    {"duration":200 * time_scale, "name":"return to 4 well potential (a)",
        "phi_1_x": phi_1_x_on * 0.1, "phi_2_x": phi_2_x_off , "M_12": M_12_off,  
        "phi_1_dcx": phi_1_dcx_on * 0.5, "phi_2_dcx": phi_2_dcx_off,  
    },
    {"duration":50 * time_scale, "name":"return to 4 well potential (a)",
        "phi_1_x": phi_1_x_off, "phi_2_x": phi_2_x_off , "M_12": M_12_off,  
        "phi_1_dcx": phi_1_dcx_off, "phi_2_dcx": phi_2_dcx_off,  
    }
]


protocol_list_1 = [
    protocol_maker(50,  _P[1]), protocol_maker(50,  _P[0])
]

protocol_list_2 = [
    protocol_maker(50,  _P[1]), protocol_maker(200, _P[2]), 
    protocol_maker(200, _P[1]), protocol_maker(50,  _P[0])
]

protocol_list_3 = [
    protocol_maker(50, _P[1]), protocol_maker(200, _P[2]), protocol_maker(50, _P[3]), 
    protocol_maker(50, _P[2]), protocol_maker(200, _P[1]), protocol_maker(50,  _P[0])
]

protocol_list_4 = [
    protocol_maker(50, _P[1]), protocol_maker(200, _P[2]), protocol_maker(50, _P[3]), protocol_maker(50, _P[4]), 
    protocol_maker(50, _P[3]), protocol_maker(50, _P[2]), protocol_maker(200, _P[1]), protocol_maker(50,  _P[0])
]

protocol_list_5 = [
    protocol_maker(50, _P[1]), protocol_maker(200, _P[2]), protocol_maker(50, _P[3]), protocol_maker(50, _P[4]), 
    protocol_maker(50, _P[5]), 
    protocol_maker(50, _P[4]), protocol_maker(50, _P[3]), protocol_maker(50, _P[2]), protocol_maker(200, _P[1]), 
    protocol_maker(50,  _P[0])
]

protocol_list_6 = [
    protocol_maker(50, _P[1]), protocol_maker(200, _P[2]), protocol_maker(50, _P[3]), protocol_maker(50, _P[4]), 
    protocol_maker(50, _P[5]), protocol_maker(50, _P[6]),
    protocol_maker(50, _P[5]), protocol_maker(50, _P[4]), protocol_maker(50, _P[3]), protocol_maker(50, _P[2]), 
    protocol_maker(200, _P[1]), protocol_maker(50,  _P[0])
]

protocol_list_7 = [
    protocol_maker(50, _P[1]), protocol_maker(200, _P[2]), protocol_maker(50, _P[3]), protocol_maker(50, _P[4]), 
    protocol_maker(50, _P[5]), protocol_maker(50, _P[6]), protocol_maker(200, _P[7]),
    protocol_maker(200, _P[6]), protocol_maker(50, _P[5]), protocol_maker(50, _P[4]), protocol_maker(50, _P[3]), 
    protocol_maker(50, _P[2]), protocol_maker(200, _P[1]), protocol_maker(50,  _P[0])
]

protocol_list_8 = [
    protocol_maker(50, _P[1]), protocol_maker(200, _P[2]), protocol_maker(50, _P[3]), protocol_maker(50, _P[4]), 
    protocol_maker(50, _P[5]), protocol_maker(50, _P[6]), protocol_maker(200, _P[7]), protocol_maker(50, _P[8]),
    
    protocol_maker(50, _P[7]), protocol_maker(200, _P[6]), protocol_maker(50, _P[5]), protocol_maker(50, _P[4]), 
    protocol_maker(50, _P[3]), protocol_maker(50, _P[2]), protocol_maker(200, _P[1]), protocol_maker(50,  _P[0])
]


array_1 = [protocol_list_1, protocol_list_8]
array_2 = [protocol_list_2, protocol_list_7]
array_3 = [protocol_list_3, protocol_list_6]
array_4 = [protocol_list_4, protocol_list_5]


# params['comment'] = "Experiment 8 (2024/7/3): time reverse protocol 1 and 8"
# params['comment'] = "Experiment 9 (2024/7/3): time reverse protocol 2 and 7"
params['comment'] = "Experiment 10 (2024/7/3): time reverse protocol 3 and 6"
# params['comment'] = "Experiment 11 (2024/7/3): time reverse protocol 4 and 5"


for protocol_list in array_3:
    protocol_time_array = [item["duration"] for item in protocol_list]
    protocol_time_array.insert(0, 0)

    print(params['sim_params'])
    print(initial_parameter_dict["phi_1_dcx"], initial_parameter_dict["phi_2_dcx"], initial_parameter_dict["M_12"])
    for x in protocol_list:
        print(x)
    print(params['comment'])

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
                                                    computation_protocol= comp_protocol, protocol_list = protocol_list)
    cfqr.createProtocolTimeArray(protocol_list, params)
    cfqr.initialize_sim()
    cfqr.set_sim_attributes()
    init_state_saved = cfqr.init_state

    # step 5: perform simulations

    # for _ in range(0, 3):
    simResult = cfq_batch_sweep.simulateSingleCoupledFluxQubit(params, initial_parameter_dict, protocol_list, \
                            potential = coupled_fq_pot, potential_default_param = coupled_fq_default_param,            
                            initial_state = init_state_saved, manual_domain = manual_domain, \
                            phi_1_dcx = phi_1_dcx,  phi_2_dcx = phi_2_dcx, \
                            percentage = 1, as_step = as_step_parameter)


    cfq_batch_sweep.saveSimulationResult(simResult, U0_1, timeOrStep = 'step', save = True, save_final_state = False, saveFolderPath = "../coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery")



    step_time_array = np.cumsum(np.array([x["duration"] for x in protocol_list]))/params['dt']
    name_array = [x["name"] for x in protocol_list]
    # plt.hist(simResult["work_distribution"], bins = 45)
    # plt.show()
    mean_work = np.mean(simResult["work_distribution"])
    jarzyn_term = np.mean(np.exp(-simResult["work_distribution"]))
    print("The simulation id is", simResult["simulation_data"]["simulation_id"])
    print(params['comment'])
    print(f"N = {params['N']}, dt = {params['dt']}")
    print(f"phi_1dcx_on: {phi_1_dcx_on}, phi_2dcx_on: {phi_2_dcx_on}, M_12_on: {M_12_on}")
    print(f"L_factor = {L_factor}, C_factor = {C_factor}, I_m_factor = {I_m_factor}")
    print([x["duration"] for x in protocol_list])
    print(f'jarzyn = {jarzyn_term}, mean work = {mean_work}')

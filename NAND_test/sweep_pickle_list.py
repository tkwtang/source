import sys, os, importlib, datetime, json, pickle
sys.path.append(os.path.expanduser('~/source'))


import numpy as np
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from sus.protocol_designer.protocol import sequential_protocol
from IPython import display
from IPython.display import HTML, Image
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter


from quick_sim import setup_sim
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

# only change these three things


pickle_prefix = "Experiment 7 (2024-11-12)"
print(pickle_prefix)
comment = "Experiment 7 (2024-11-12) ratio of KEEP: 200, 0"
param_dt = 1/100
loopTime = 100
protocol_type = "CE"






####################
# no need to change the following 
####################

pickle_file_base_folder = "../coupled_flux_qubit_protocol/simulation_protocol_history"

all_pickle_file = os.listdir(pickle_file_base_folder)
target_pickle_file_list = [os.path.join(pickle_file_base_folder, p) for p in all_pickle_file if pickle_prefix in p]


for pickle_file in target_pickle_file_list:
    for _ in range(0, loopTime):
        with open(pickle_file, "rb") as f:
            sim_info = pickle.load(f)
        # do not need to change below
        if protocol_type == "CE":
            mapping_dict = mapping_state_1_to_state_2_dict_CE
        if protocol_type == "NAND":
            mapping_dict = mapping_state_1_to_state_2_dict_NAND

        coupled_fq_default_param = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        [phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound] = np.array([4, 4, 4, 4])

        coupled_fq_domain = [[-phi_1_bound, -phi_2_bound, -phi_1dc_bound, -phi_2dc_bound], [phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound]]

        coupled_fq_pot = Potential(coupled_flux_qubit_non_linear_approx_pot, coupled_flux_qubit_non_linear_approx_force, 14, 4, default_params = coupled_fq_default_param,  relevant_domain = coupled_fq_domain)


        params = sim_info['params']
        params["comment"] = comment
        params["dt"] = param_dt


        initial_parameter_dict = sim_info['initial_external_parameter_dict']
        protocol_list = sim_info['protocol_list']
        manual_domain = sim_info['manual_domain']
        U0_1 = params['U0']

        phi_1_dcx, phi_2_dcx = 0, 0

        if "initial_state" in sim_info["params"]:
            init_state_used = np.load("../" + sim_info["params"]["initial_state"])
        else:
            init_state_used = None
        has_velocity = True
        saveAllStates = False

        print(protocol_list)

        pickle_folder = os.path.join("../coupled_flux_qubit_protocol", "simulation_protocol_history")

        print(comment)
        simResult = cfq_batch_sweep.simulateSingleCoupledFluxQubit(params, initial_parameter_dict, protocol_list, potential = coupled_fq_pot, potential_default_param = coupled_fq_default_param, initial_state = init_state_used, manual_domain = manual_domain, phi_1_dcx = phi_1_dcx,  phi_2_dcx = phi_2_dcx, measure_all_states=saveAllStates, has_velocity = has_velocity, pickle_save_path = pickle_folder, mapping_state_1_to_state_2_dict = mapping_dict)


        """
        # save results
        """
        cfq_batch_sweep.saveSimulationResult(simResult, U0_1, timeOrStep = 'step', save = True, save_final_state = False, saveFolderPath = "../coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery")

        step_time_array = np.cumsum(np.array([x["duration"] for x in protocol_list]))/params['dt']
        name_array = [x["name"] for x in protocol_list]
        mean_work = np.mean(simResult["work_distribution"])
        jarzyn_term = np.mean(np.exp(-simResult["work_distribution"]))
        print("The simulation id is", simResult["simulation_data"]["simulation_id"])
        print(params['comment'])
        print(f"N = {params['N']}, dt = {params['dt']}")
        print([x["duration"] for x in protocol_list])
        print(f'jarzyn = {jarzyn_term}, mean work = {mean_work}')
# %%notify
import matplotlib.pyplot as plt
import numpy as np
from edward_tools.visualization import plotFidelityBarChart

def work_analysis(simResult, params, protocol_list):
    characteristic_t = np.sqrt(params['circuit_parameters']['L_1'] * params['circuit_parameters']['L_factor'] * params['circuit_parameters']['C_1'] * params['circuit_parameters']['C_factor'])
    
    work_distribution_array = np.repeat(simResult["work_distribution"], 10, axis=0)
    step_time_array = np.cumsum(np.array([x["duration"] for x in protocol_list]))/params['dt']
    name_array = [x["name"] for x in protocol_list]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(work_distribution_array, bins = 100)
    mean_work = np.mean(simResult["work_distribution"])
    jarzyn_term = np.mean(np.exp(-simResult["work_distribution"]))
    
    
    # average work done time plot
    step_array = simResult['cfqr'].sim.target_step_index
    skip_step = int(len(step_array) * 0.05)
    work_mean = simResult["work_statistic"][:,0]
    work_std = simResult["work_statistic"][:,1]

    ax[1].errorbar(step_array[::skip_step], work_mean[::skip_step], yerr = work_std[::skip_step])
    # plt.yscale('log')

    for i, t in enumerate(step_time_array):
        ax[1].axvline(x = t, color = 'b', linestyle ='--', label = 'axvline - full height')
        # plt.annotate(f"{i+1}", xy=(t, 4.5),  color='red', ha='center', size=16)

    print("sim_id = ", simResult['simulation_data'])
    print(f"N = {params['N']}, dt = {params['dt']}")
    print(f"phi_1dcx_on: {params['circuit_parameters']['phi_1xdc_on']}, phi_2dcx_on: {params['circuit_parameters']['phi_2xdc_on']}, mu_12_on: {params['circuit_parameters']['mu_12_on']}")
    print(f"characteristic_time = {characteristic_t * 1e9:.3g} ns")
    print(f"L_factor = {params['circuit_parameters']['L_factor']}, C_factor = {params['circuit_parameters']['C_factor']}, I_m_factor = {params['circuit_parameters']['I_m_factor']}, gamma = {params['circuit_parameters']['gamma']}")
    print([x["duration"] for x in protocol_list])
    print(f'jarzyn = {jarzyn_term}, mean work = {mean_work}')

def fidelity_analysis(simResult, params, initial_parameter_dict, protocol_list):
    step_time_array = np.cumsum(np.array([x["duration"] for x in protocol_list]))/params['dt']
    name_array = [x["name"] for x in protocol_list]
    # 

    """ barrier height """
    critical_points_of_four_well = simResult['cfqr'].find_critical_point_for_all_potential(0, initial_parameter_dict, guess = [[-2, 2], [-2, 0], [-2, 2]], showPE=True)
    four_well_barrier_height = critical_points_of_four_well[1][1] - critical_points_of_four_well[1][0]

    """ work, jarzyn term """
    mean_work = np.mean(simResult["work_distribution"])
    mean_work_stand_error = np.std(simResult["work_distribution"]) / np.sqrt(params['N'])
    average_KE = np.mean(simResult['cfqr'].system.get_kinetic_energy(simResult['cfqr'].sim.output.final_state))
    jarzyn_term = np.mean(np.exp(-simResult["work_distribution"]))
    N_00 = len(simResult["work_distribution"])

    """ print results """
    print("sim_id = ", simResult['simulation_data'])
    print(f"N = {params['N']}, dt = {params['dt']}, T = {params['circuit_parameters']['T']}K, N_00 = {N_00}")
    print(f"lambda = {params['sim_params'][0]}")
    print(f"theta = {params['sim_params'][1]}")
    print(f"eta = {params['sim_params'][2]}")
    # print(f"phi_1dcx_on: {phi_1xdc_on}, phi_2dcx_on: {phi_2xdc_on}, mu_12_on: {mu_12_on}")
    print(f"L_factor = {params['circuit_parameters']['L_factor']}, C_factor = {params['circuit_parameters']['C_factor']}, R_factor = {params['circuit_parameters']['R_factor']}, I_m_factor = {params['circuit_parameters']['I_m_factor']}, gamma = {params['circuit_parameters']['gamma']}")
    print([x["duration"] for x in protocol_list])
    
    

    fig = plt.figure(figsize=[7, 6])
    gs = fig.add_gridspec(4,4,width_ratios=[1,1, 1,1])
    ax_WD = fig.add_subplot(gs[0:2,0:2])
    ax_WD_avg = fig.add_subplot(gs[0:2,2:4])
    fidelity_ax_00 = fig.add_subplot(gs[3,0])
    fidelity_ax_01 = fig.add_subplot(gs[2,0])
    fidelity_ax_10 = fig.add_subplot(gs[3,1])
    fidelity_ax_11 = fig.add_subplot(gs[2,1])
    
    ax_error = fig.add_subplot(gs[2:4,2:4])
    
    fidelity_ax_flatten = [fidelity_ax_01, fidelity_ax_11, fidelity_ax_00, fidelity_ax_10]

    ax_WD.hist(simResult["work_distribution"], bins = 45)
    ax_WD_avg

    time_array = simResult['cfqr'].getReducedTimeArray()
    skip_step = int(len(time_array) * 0.05)
    work_mean = simResult["work_statistic"][:,0]
    work_std = simResult["work_statistic"][:,1]

    ax_WD_avg.errorbar(time_array[::skip_step], work_mean[::skip_step], yerr = work_std[::skip_step])
    # plt.yscale('log')

    # for i, t in enumerate(time_array):
    #     ax_WD_avg.axvline(x = t, color = 'b', linestyle ='--', label = 'axvline - full height')

    ax_WD.set_xlabel(r"work done ($k_BT$)")
    ax_WD_avg.set_ylabel(r"avg WD ($k_BT$)")
    ax_WD_avg.set_xlabel(r"t ($t_c$)")

    # fidelity bar chart
    plotFidelityBarChart(simResult['cfqr'].sim.fidelity_time_array, ax_flatten = fidelity_ax_flatten)
    
    # error_bar_chart
    error_information = simResult['error_rate_array']
    bit_array = ["00", "01", "10", "11"]
    pColor = {"00": "#061DF7", "01": "#FCEF51", "10": "#3FC7F2", "11": "#F7B6D2"}
    total_number = np.sum([x['initial'] for x in error_information])
    total_wrong_number = np.sum([x['wrong_number'] for x in error_information])
    error_rate = np.hstack([x['wrong_number'] for x in error_information]) / total_number * 100
    total_error_rate = np.sum(error_rate)
    
    ax_error.bar(bit_array, error_rate, color = pColor.values())
    ax_error.set_xlabel("particle category")
    ax_error.set_ylabel("error rate in percentage")
    ax_error.set_title(f"total_error_rate: {total_error_rate:.3g}%, N = {total_number}")
    
    print(f"total_time = {time_array[-1]}")
    print(f'jarzyn = {jarzyn_term:.3g}, mean work = {mean_work:.3g} ± {3 * mean_work_stand_error:.3g}')
    print(f"total_error_rate: {total_error_rate:.3g}%, N = {total_number}")
    print(f"barrier_height = {four_well_barrier_height:.3g}")
    print(f"average KE = {average_KE}")
    
    fig.tight_layout()

#     fig = plt.figure(figsize=[9, 3])
#     gs = fig.add_gridspec(2,6,width_ratios=[1,1, 1,1, 1, 1])
#     ax_WD = fig.add_subplot(gs[0:2,0:2])
#     ax_WD_avg = fig.add_subplot(gs[0:2,2:4])
#     ax_00 = fig.add_subplot(gs[1,4])
#     ax_01 = fig.add_subplot(gs[0,4])
#     ax_10 = fig.add_subplot(gs[1,5])
#     ax_11 = fig.add_subplot(gs[0,5])
#     ax_flatten = [ax_01, ax_11, ax_00, ax_10]

#     ax_WD.hist(simResult["work_distribution"], bins = 45)
#     ax_WD_avg

#     step_array = simResult['cfqr'].sim.target_step_index 
#     skip_step = int(len(step_array) * 0.05)
#     work_mean = simResult["work_statistic"][:,0]
#     work_std = simResult["work_statistic"][:,1]

#     ax_WD_avg.errorbar(step_array[::skip_step], work_mean[::skip_step], yerr = work_std[::skip_step])
#     # plt.yscale('log')

#     for i, t in enumerate(step_time_array):
#         ax_WD_avg.axvline(x = t, color = 'b', linestyle ='--', label = 'axvline - full height')



#     ax_WD.set_xlabel(r"work done ($k_BT$)")
#     plotFidelityBarChart(simResult['cfqr'].sim.fidelity_time_array, ax_flatten = ax_flatten)
#     fig.tight_layout()

    
def KE_analysis(simResult):
    """To investigate the average KE of each type of particles"""
    all_state = simResult['cfqr'].sim.output.all_state['states']
    protocal_key_time_array = simResult['cfqr'].protocol_all_time_array[simResult['cfqr'].sim.output.all_state['step_indices']]
    index_of_00 = simResult['cfqr'].getIndexOfParticles()["00"]
    index_of_01 = simResult['cfqr'].getIndexOfParticles()["01"]
    index_of_10 = simResult['cfqr'].getIndexOfParticles()["10"]
    index_of_11 = simResult['cfqr'].getIndexOfParticles()["11"]
    index_of_not_00 = ~simResult['cfqr'].getIndexOfParticles()["00"]

    fig, ax = plt.subplots(1, 4, figsize = [16, 3])
    KE_of_00 = np.mean(simResult['cfqr'].system.get_kinetic_energy(all_state[index_of_00]), axis = 0)
    KE_of_01 = np.mean(simResult['cfqr'].system.get_kinetic_energy(all_state[index_of_01]), axis = 0)
    KE_of_10 = np.mean(simResult['cfqr'].system.get_kinetic_energy(all_state[index_of_10]), axis = 0)
    KE_of_11 = np.mean(simResult['cfqr'].system.get_kinetic_energy(all_state[index_of_11]), axis = 0)
    # KE_of_10 = np.mean(simResult['cfqr'].system.get_kinetic_energy(all_state[index_of_10]), axis = 0)
    ax[0].plot(protocal_key_time_array, KE_of_00)
    ax[0].set_title("average KE of 00")
    ax[0].set_xlabel("time")
    ax[0].set_ylabel(r"KE $(k_BT)$")
    # ax[0].set_yscale("log")

    ax[1].set_title("average KE of 01")
    ax[1].plot(protocal_key_time_array, KE_of_01)
    ax[1].set_xlabel("time")
    # ax[1].set_ylim(0, 6)

    ax[2].set_title("average KE of 10")
    ax[2].plot(protocal_key_time_array, KE_of_10)
    ax[2].set_xlabel("time")

    ax[3].set_title("average KE of 10")
    ax[3].plot(protocal_key_time_array, KE_of_11)
    ax[3].set_xlabel("time")
    # ax[1].set_ylim(0, 6)

    # fig.text(0.5, -0.15, f'L = {L_1 * 1e12:.3g}pH, T = {T}K')

def search_protocol_time_array_index(array, start, end):
    x = np.sort(np.append(array, start))
    y = np.sort(np.append(array, end))
    
    start_index = np.where(x == start)[0][0]
    if start != round(array[start_index]):
        start_index += 1
    
    
    final_index = np.where(y == end)[0][0] 
    if end != round(array[final_index]):
        final_index -= 1

    return slice(start_index, final_index)    


def single_particle_analysis(simResult, params, protocol_list, plot_ax = None, particle_index = 0, particle_category = "00", time_range = None, selected_quantity = ['PE', 'KE', 'KE + PE', 'WD', 'E_loss']):
    all_states = simResult['cfqr'].sim.output.all_state['states']
    index_of_00 = simResult['cfqr'].getIndexOfParticles()["00"]
    index_of_01 = simResult['cfqr'].getIndexOfParticles()["01"]
    index_of_10 = simResult['cfqr'].getIndexOfParticles()["10"]
    index_of_11 = simResult['cfqr'].getIndexOfParticles()["11"]

    index_choice = simResult['cfqr'].getIndexOfParticles()[particle_category]

    all_states_00_i1 = all_states[index_choice][particle_index, ...]

    # _, _, protocol_time_array, protocol_time_index_array = simResult['cfqr'].createProtocolTimeArray(protocol_list, params)
    # protocol_all_time_index_array = np.array(range(0, int(np.array(protocol_time_array[-1]) / params['dt']) + 1))
    # protocol_all_time_array  = protocol_all_time_index_array * params['dt']
    reduced_protocal_time_array = simResult['cfqr'].getReducedTimeArray()

    KE_of_the_particle = simResult['cfqr'].system.get_kinetic_energy(all_states_00_i1)
    PE_of_the_particle = [simResult['cfqr'].system.get_potential(_s, _t ) for _t, _s in  zip(reduced_protocal_time_array, all_states_00_i1)]
    WD_of_the_particle = simResult['cfqr'].sim.work_dist_time_array_whole_process[index_choice][particle_index][::params['as_step'].step]
    adjusted_PE = PE_of_the_particle - PE_of_the_particle[0]
    adjusted_total_energy = adjusted_PE + KE_of_the_particle
    characteristic_t = np.sqrt(params['circuit_parameters']['L_1'] * params['circuit_parameters']['L_factor'] * params['circuit_parameters']['C_1'] * params['circuit_parameters']['C_factor'])

    print(f"max KE: {max(KE_of_the_particle):.3g}, max PE: {max(PE_of_the_particle):.3g}")
    if time_range:
        index_slice = search_protocol_time_array_index(reduced_protocal_time_array, time_range[0], time_range[1])
    else:
        index_slice = slice(None, None, None)
    
    
    # KE_of_the_particle = KE_of_the_particle[index_slice]
    if 'PE' in selected_quantity:
        plot_ax.plot(reduced_protocal_time_array[index_slice], adjusted_PE[index_slice], label = "PE", color = "#CC79A7")
    if 'KE' in selected_quantity:
        plot_ax.plot(reduced_protocal_time_array[index_slice], KE_of_the_particle[index_slice], label = "KE")
    if 'KE + PE' in selected_quantity:
        plot_ax.plot(reduced_protocal_time_array[index_slice], adjusted_total_energy[index_slice], label = "KE + PE")
    if 'WD' in selected_quantity:
        plot_ax.plot(reduced_protocal_time_array[index_slice], WD_of_the_particle[index_slice], label = "WD", color = "#000000")
    if 'E_loss' in selected_quantity:
        plot_ax.plot(reduced_protocal_time_array[index_slice], WD_of_the_particle[index_slice] - adjusted_total_energy[index_slice], label = "energy loss")
    plot_ax.set_ylabel(r"energy $(k_BT)$")
    plot_ax.set_xlabel(f"time ({characteristic_t * 1e9:.3g} ns)")
    plot_ax.legend(bbox_to_anchor=(1.25, 1.05))
    
    return {"index_slice": index_slice, "reduced_protocal_time_array": reduced_protocal_time_array}
# def single_particle_analysis(simResult, params, protocol_list, plot_ax = None, particle_index = 0, particle_category = "00", time_range = None):
    

#     index_choice = simResult['cfqr'].getIndexOfParticles()[particle_category]

#     all_states = simResult['cfqr'].sim.output.all_state['states']
    
#     reduced_protocal_time_array = simResult['cfqr'].getKeyTime()

#     index_slice = None
#     if time_range:
#         start_index, final_index = time_range
#         index_slice = search_protocol_time_array_index(reduced_protocal_time_array, start_index, final_index)
        
            
#     all_states_00_i1 = all_states[index_choice][particle_index, index_slice, ...]
#     # reduced_protocal_time_array = reduced_protocal_time_array
    
    
#     KE_of_the_particle = simResult['cfqr'].system.get_kinetic_energy(all_states_00_i1)[index_slice]
    
#     print(reduced_protocal_time_array.shape)
#     # print()
    
#     PE_of_the_particle = [simResult['cfqr'].system.get_potential(_s, _t ) for _t, _s in  zip(reduced_protocal_time_array, all_states_00_i1[:, index_slice, ...])]
# #     WD_of_the_particle = simResult['cfqr'].sim.work_dist_time_array_whole_process[index_choice][particle_index]
    
#     keyState_choosen_particle = [ s[index_choice][particle_index, ...] for s in simResult['cfqr'].sim.keyStep_all_states]
#     total_energy = PE_of_the_particle - min(PE_of_the_particle) + KE_of_the_particle
#     adjusted_PE = PE_of_the_particle - PE_of_the_particle[0]
#     adjusted_total_energy = adjusted_PE + KE_of_the_particle
# #     characteristic_t = np.sqrt(params['circuit_parameters']['L_1'] * params['circuit_parameters']['L_factor'] * params['circuit_parameters']['C_1'] * params['circuit_parameters']['C_factor'])
    

#     if plot_ax == None:
#         plot_ax = plt.figure(figsize=(8, 5))
#     plot_ax.plot(reduced_protocal_time_array, adjusted_PE , label = "PE", color = "#CC79A7")
#     plot_ax.plot(reduced_protocal_time_array, KE_of_the_particle, label = "KE")
#     plot_ax.plot(protocal_key_time_array, adjusted_total_energy, label = "KE + PE")
#     # plot_ax.plot(protocal_key_time_array, WD_of_the_particle[::100], label = "WD", color = "#000000")
#     # plot_ax.plot(protocal_key_time_array, WD_of_the_particle[::100] - adjusted_total_energy, label = "energy loss")
#     plot_ax.set_ylabel(r"energy $(k_BT)$")
#     plot_ax.set_xlabel(f"time ({characteristic_t * 1e9:.3g} ns)")
#     plot_ax.legend(bbox_to_anchor=(1.25, 1.05))

from edward_tools import coupled_fq_protocol_library

def state_plot_potential_KE_animation_anlaysis(simResult, params, _t, cutlineValue, index_label = ['00', '01', '10', '11']):
    cutlineDirection = "v"
    contour_range = [0, 450]
    graph_title = " "
    x_range = None
    y_range = None
    mark_critical_points = None
    showGraph = True
    saveGraph = False

    fig, ax = plt.subplots(1, 2, figsize = [9, 4.5])
    ax[0].set_box_aspect(aspect=1)

    X, Y, U, cutlineDirection, _targetAxis, _plotAxis = coupled_fq_protocol_library.get_potential_along_a_1D_cutline(simResult['cfqr'], t = _t, cutlineDirection ='v', cutlineValue = cutlineValue, resolution = 100, contours = 5, manual_domain = [[-5, -5], [5, 5]])

    plotLineData = coupled_fq_protocol_library.plotCutlines(X, Y, U, cutlineDirection = cutlineDirection, cutlineValue = cutlineValue, x_range = x_range, y_range = y_range, contour_range = contour_range, cutlineColor = 'red' , contour_plt = ax[0], cutline_plt = ax[1], contours = 5, time = None, graph_title = graph_title, showGraph = showGraph, saveGraph = saveGraph)

    all_states = simResult['cfqr'].get_all_state()
    index_dict = simResult['cfqr'].getIndexOfParticles()

    protocol_all_time_array = simResult['cfqr'].protocol_all_time_array
    _index_of_t = np.min(np.where(protocol_all_time_array[params['as_step']]==_t)) 

    U_on_cutline = plotLineData['cutline_plot']['targetU']
    offset_U = min(U_on_cutline)
    print(offset_U)
    
    state_potential = simResult['cfqr'].system.get_potential(all_states[:, _index_of_t, ...], _t)
    

    pColor = {"00": "#061DF7", "01": "#FCEF51", "10": "#3FC7F2", "11": "#F187F4"}
    for _label in index_label:
        states_x_data = all_states[index_dict[_label], _index_of_t, 0,0]
        states_y_data = all_states[index_dict[_label], _index_of_t, 1,0]
        ax[0].scatter(states_x_data, states_y_data, color = pColor[_label])
        ax[1].scatter(states_y_data, state_potential[index_dict[_label]], color = pColor[_label])
        
        
def work_done_of_each_type(simResult):
    index_of_00 = simResult['cfqr'].getIndexOfParticles()["00"]
    index_of_01 = simResult['cfqr'].getIndexOfParticles()["01"]
    index_of_10 = simResult['cfqr'].getIndexOfParticles()["10"]
    index_of_11 = simResult['cfqr'].getIndexOfParticles()["11"]

    pColor = simResult['cfqr'].pColor
    fig, ax = plt.subplots(2, 2, figsize=[6, 4])
    fig.suptitle("Work done of all the four type particles")
    fig.tight_layout()
    ax_flatten = ax.flatten()
    ["01", "11", "00", "10"]
    ax_flatten[0].hist(simResult["work_distribution"][index_of_01], bins = 30, label = "01", color = pColor["01"])
    ax_flatten[0].set_title(f"mean of 01 = {np.mean(simResult['work_distribution'][index_of_01]):.3g}")
    ax_flatten[1].hist(simResult["work_distribution"][index_of_11], bins = 30, label = "11", color = pColor["11"])
    ax_flatten[1].set_title(f"mean of 11 = {np.mean(simResult['work_distribution'][index_of_11]):.3g}")
    ax_flatten[2].hist(simResult["work_distribution"][index_of_00], bins = 30, label = "00", color = pColor["00"])
    ax_flatten[2].set_title(f"mean of 00 = {np.mean(simResult['work_distribution'][index_of_00]):.3g}")
    ax_flatten[3].hist(simResult["work_distribution"][index_of_10], bins = 30, label = "10", color = pColor["10"])
    ax_flatten[3].set_title(f"mean of 10 = {np.mean(simResult['work_distribution'][index_of_10]):.3g}")

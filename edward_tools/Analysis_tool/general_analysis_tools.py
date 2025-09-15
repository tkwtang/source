import numpy as np
import matplotlib.pyplot as plt
import importlib, os, hashlib, json, sys
source_path = os.path.expanduser('~/source/')
sys.path.append(source_path)
import copy
from edward_tools.couple_flux_qubit_metrics import fidelityEvaluation

"""
fidelityAnalysis

getFidelityTimeGraph

show_phi_dc_with_time

get_work_done_from_time_array

crookAnalysis_method_1

crookAnalysis_method_2
"""

def fidelityAnalysis():
    # final_percentage_matrix = np.array([d["final_percentage"] for d in simResult["fidelity"]]).T
    # text_box_array = ["\n".join(row) for row in final_percentage_matrix]

    fig, ax = plt.subplots(1, 2, figsize = [10, 4])
    # plt.figure(figsize=[4, 4])
    # plt.title(f"N = {item['N']}, dt = {item['dt']}, t1 = {item['protocol_list'][-1]['duration']} ")
    # _ax = ax[1]
    # _ax.set_xlim(-5, 5);plt.ylim(-5, 5);
    # _ax.text(-3.25, -3.5, text_box_array[0]); _ax.text(-3.25, 1.5, text_box_array[1])
    # _ax.text(1.75, -3.5, text_box_array[2]); _ax.text(1.75, 1.5, text_box_array[3])
    # _ax.vlines(0, ymin = -5, ymax = 5); _ax.hlines(0, xmin = -5, xmax = 5)

    ideal_fidelity = np.array([['0.500', '0.000', '0.500', '0.000'], ['0.500', '0.000', '0.500', '0.000'],
                              ['0.000', '0.500', '0.000', '0.500'], ['0.000', '0.500', '0.000', '0.500']]).T
    text_box_array = ["\n".join(row) for row in ideal_fidelity]
    # plt.figure(figsize=[4, 4])
    # plt.title(f"N = {item['N']}, dt = {item['dt']}, t1 = {item['protocol_list'][-1]['duration']} ")
    _ax = ax[0]
    _ax.set_xlim(-5, 5);plt.ylim(-5, 5);
    _ax.text(-3.25, -3.5, text_box_array[0]); _ax.text(-3.25, 1.5, text_box_array[1])
    _ax.text(1.75, -3.5, text_box_array[2]); _ax.text(1.75, 1.5, text_box_array[3])
    _ax.vlines(0, ymin = -5, ymax = 5); _ax.hlines(0, xmin = -5, xmax = 5)
    
    ideal_fidelity = copy.deepcopy(final_percentage_matrix)
    ideal_fidelity[0]["final_percentage"] = ['0.500', '0.000', '0.500', '0.000']
    ideal_fidelity[1]["final_percentage"] = ['0.500', '0.000', '0.500', '0.000']
    ideal_fidelity[2]["final_percentage"] = ['0.000', '0.500', '0.000', '0.500']
    ideal_fidelity[3]["final_percentage"] = ['0.000', '0.500', '0.000', '0.500']


def getFidelityTimeGraph(fidelity_time_array):
    """
    trace the fidelity of the systems inside a region at different time as 4 different plots
    """
    metastable_state_array = ["00", "01", "10", "11"]
    fidelity_time_array = np.array(fidelity_time_array, dtype = float)
    
    fig, ax = plt.subplots(2, 2, figsize = [10, 8])
    ax_flatten = ax.flatten()

    for i, _ax in enumerate(ax_flatten):
        _ax.plot(simResult['cfqr'].sim.target_step_index, fidelity_time_array[:, 0, i], label = "00")
        _ax.plot(simResult['cfqr'].sim.target_step_index, fidelity_time_array[:, 1, i], label = "01")
        _ax.plot(simResult['cfqr'].sim.target_step_index, fidelity_time_array[:, 2, i], label = "10")
        _ax.plot(simResult['cfqr'].sim.target_step_index, fidelity_time_array[:, 3, i], label = "11")
        _ax.set_title(f"time evolution of fidelity inside {metastable_state_array[i]}")
        _ax.legend()
    plt.show()
    
    
def show_phi_dc_with_time(cfqr, params):
    fig, ax = plt.subplots(1, 2, figsize = [12, 4])
    
    _time_step_array = np.array(cfqr.sim.output.all_state['step_indices']) * params['dt']
    _phi_1dcx_time_array = []
    _phi_2dcx_time_array = []
    for _t in _time_step_array:
        params_at_t = cfqr.system.protocol.get_params(_t)
        _phi_1dcx_time_array.append(params_at_t[10])
        _phi_2dcx_time_array.append(params_at_t[11])

    all_state = cfqr.sim.output.all_state['states']
    _phi_1dc_array = all_state[:, :, 2, 0]
    _phi_2dc_array = all_state[:, :, 3, 0]
    _time_array = range(len(_phi_1dc_array))
    
    

    for line in _phi_1dc_array[0:10]:
        ax[0].plot(line)
    ax[0].plot(_phi_1dcx_time_array, "--")
    
    
    for line in _phi_2dc_array[0:10]:
        ax[1].plot(line)
    ax[1].plot(_phi_2dcx_time_array, "--")
    # ax[1].title(r"$\varphi_{2dc}$ vs step ($\gamma$ " + f"= {gamma})"  )

    
def showParametersVsTime(cfqr, protocol_time_array):
    """
    To show the values of phi_1dcx, phi_2dcx and M_12 
    """
    plt.figure(plt.figure(figsize=(10,6)))
    t_array = np.linspace(0, protocol_time_array[-1], 1000)
    phi_1_dc_array = [cfqr.protocol.get_params(t)[10] for t in t_array]
    phi_2_dc_array = [cfqr.protocol.get_params(t)[11] for t in t_array]
    M_12_array = [cfqr.protocol.get_params(t)[12] for t in t_array]
    # protocol_params_at_t = cfq_runner.protocol.get_params(t)

    for t in protocol_time_array:
        print(t)
        plt.vlines(x = t, ymin = -0.8, ymax = 3.5, linestyles="--", color="grey")

    plt.ylim(-1.24, 3.8)
    plt.plot(t_array, phi_1_dc_array, "--r", label = r"$\phi_{1dcx}$")
    plt.plot(t_array, phi_2_dc_array, "-.g", label = r"$\phi_{2dcx}$")
    plt.plot(t_array, M_12_array, label = r"$M_{12}$")
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel("time", fontsize = 23)
    plt.ylabel("value", fontsize = 23)

    plt.legend(fontsize = 23, bbox_to_anchor= (1.15, 1.02))
    
def get_work_done_from_time_array(params, work_statistic_array, protocol_time_array, offset_value):
    target_time_index = np.array(protocol_time_array)/params['dt']
    target_time_index = target_time_index.astype("int") - 1
    
    work_done_array = [work_statistic_array[i] for i in target_time_index[1:]]
    work_done_array.insert(0, 0)
    work_done_of_each_step = np.array(work_done_array)[1:] - np.array(work_done_array)[:-1]

    work_done_after_offset = work_done_of_each_step - offset_value
    print(f"work_done = {work_done_of_each_step}" )
    print(f"work_offset = {work_done_after_offset}")
    print("measurement + reset = ", work_done_after_offset[0] + work_done_after_offset[1] + work_done_after_offset[2] )
    print("control = ", work_done_after_offset[3] + work_done_after_offset[4] + work_done_after_offset[5] + work_done_after_offset[6])
    print(sum(work_done_after_offset))
    
    
    
    
    return work_done_after_offset



def crookAnalysis_method_1(simResult, params):
    test_w = np.linspace(-0.025, 0.0025, 21)
    dw = 0.001
    crook_analysis_result = []

    for _w in test_w:
        array = simResult["work_distribution"]

        # The 0 index is to remove the single element array
        number_of_plus_W = len(np.where(np.logical_and(array <= _w + dw, array >= _w - dw))[0])
        number_of_minus_W = len(np.where(np.logical_and(array <= -_w + dw, array >= -_w - dw))[0])
        crook_analysis_result.append(np.array([_w, number_of_plus_W, number_of_minus_W * np.exp(_w)]))

    crook_analysis_result = np.array(crook_analysis_result)    
    print(f"dw = {dw}, dt = {params['dt']}, N = {params['N']}")
    print("LHS: number of +W, RHS = number of -W * np.exp(+W)]")
    all_w = crook_analysis_result[:, 0]
    all_lhs = crook_analysis_result[:, 1]
    all_rhs = crook_analysis_result[:, 2]
    plt.xlabel("W")
    plt.ylim([0, 2])
    plt.plot(all_w, all_lhs/all_rhs, "*", label = "LHS")
    # plt.plot(all_w, all_rhs, "-", label = "RHS")
    plt.legend()
    plt.show()
    # print("p(W) = p(-W) exp(W)")
    # print(f"LHS = {number_of_plus_W}, RHS = {number_of_minus_W * np.exp(test_w)}")
    # print(f"mean work = {avg_W:.3f} and the std error of W = {error_W:.5f}")
    # print(f'jarzyn = {jarzyn_term:.3f}, std error= {jarzyn_term_error:.5f}')
    # print([x["duration"] for x in protocol_list])
    # print(time_length_1, time_length_2, time_length_3, time_length_4, time_length_5, time_length_6)


    
def crookAnalysis_method_2(simResult, params):
    counts, bin_edges = np.histogram(simResult["work_distribution"], bins = np.linspace(-2, 2, 50),density=True)
    counts, bin_edges = np.histogram(simResult["work_distribution"], bins = np.linspace(-0.01, 0.01, 20),density=True)
    log_P_ratio = np.log(counts / counts[::-1])
    plt.figure(figsize=(5, 5))
    plt.plot(bin_edges[1:], bin_edges[1:], "--")
    plt.scatter(bin_edges[:-1], log_P_ratio)
    plt.xlabel("W")
    plt.ylabel("log ratio")
    
    # crook analysis
    # np.linspace(0.005, 0.2, 100)
    test_w = np.linspace(-0.02, 0.02, 20)
    dw = 0.001
    crook_analysis_result = []

    for _w in test_w:
        array = simResult["work_distribution"]

        # The 0 index is to remove the single element array
        number_of_plus_W = len(np.where(np.logical_and(array <= _w + dw, array >= _w - dw))[0])
        number_of_minus_W = len(np.where(np.logical_and(array <= -_w + dw, array >= -_w - dw))[0])
        crook_analysis_result.append(np.array([_w, number_of_plus_W, number_of_minus_W * np.exp(_w)]))

    crook_analysis_result = np.array(crook_analysis_result)    
    print(f"dw = {dw}, dt = {params['dt']}, N = {params['N']}")
    print("LHS: number of +W, RHS = number of -W * np.exp(+W)]")
    all_w = crook_analysis_result[:, 0]
    all_lhs = crook_analysis_result[:, 1]
    all_rhs = crook_analysis_result[:, 2]
    plt.xlabel("W")
    plt.plot(all_w, all_lhs, "x", label = "LHS")
    plt.plot(all_w, all_rhs, "x", label = "RHS")
    plt.legend()
    plt.show()
    # print("p(W) = p(-W) exp(W)")
    # print(f"LHS = {number_of_plus_W}, RHS = {number_of_minus_W * np.exp(test_w)}")
    print(f"mean work = {avg_W:.3f} and the std error of W = {error_W:.5f}")
    print(f'jarzyn = {jarzyn_term:.3f}, std error= {jarzyn_term_error:.5f}')
    print([x["duration"] for x in protocol_list])
    print(time_length_1, time_length_2, time_length_3, time_length_4, time_length_5, time_length_6)
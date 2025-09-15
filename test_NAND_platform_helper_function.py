import pandas as pd
import json, importlib, os
import matplotlib as mpl
import numpy as np
import matplotlib.animation as animation
import NAND_PARAMETERS
from IPython.display import HTML
from IPython import display
import matplotlib.pyplot as plt
from PIL import Image

def getSimulationID(df):
    return [item["simulation_id"] for item in df["simulation_data"]] 

def getDataByComment(df, comment):
    if type(comment) == str:
        target = df[df["comment"].str.find(comment) == 0]
    else:
        target = pd.concat(df[df["comment"].str.find(_c) == 0] for _c in comment)
    return target

def getDataByField(df, field):
    return list(target[field])


def showGraph(sim_id):
    work_distribution_img = Image.open(f"coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{sim_id}_work_distribution.png")
    work_distribution_data = np.load(f"coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{sim_id}_work_distribution.npy")
    work_statistic_img = Image.open(f"coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{sim_id}_work_statistic.png")
    work_statistic_data = np.load(f"coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{sim_id}_work_statistic.npy")
    display(work_distribution_img)
    display(work_statistic_img)
    # work_distribution_img.show()
    print(sim_id)
    return work_distribution_data, work_statistic_data

def getSimulationWorkStatistics(df, sim_id, folderPath = "coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery"):
    """
    'params', 'initial_parameter_dict', 'protocol_list_item',
       'simulation_data', 'circuit_parameters', 'sim_params', 'jarzynski_term',
       'mean_work', 'fidelity', 'keyStep_work_statistic',
       'fidleity_time_array', 'target_step_index_info', 'comment', 'script',
       'simulation_id'
    """
    
    target = getDataByID(df, sim_id)
    dt = list(target['params'])[0]["dt"]
    N =  list(target['params'])[0]["N"]
    
    work_distribution_data = np.load(f"{folderPath}/{sim_id}_work_distribution.npy")
    work_statistic_data = np.load(f"{folderPath}/{sim_id}_work_statistic.npy")
    fidelity_time_array = np.load(f"{folderPath}/{sim_id}_fidelity_time_array.npy")
    jarzyn_term = np.mean(np.exp(-work_distribution_data))
    jarzyn_term_error = 3 * np.std(np.exp(-work_distribution_data)) / np.sqrt(N)
    keyStep_work_statistic = target['keyStep_work_statistic'].values[0]
    error_rate_array = target['error_rate_array'].values[0]
    target_step_index_info = list(target['target_step_index_info'])[0]
    target_step_index = list(range(target_step_index_info['start'], target_step_index_info['end']+1, target_step_index_info['step']))

    return {
        "work_distribution_data": work_distribution_data,
        "work_statistic_data": work_statistic_data,
        "fidelity_time_array": fidelity_time_array,
        "target_time_index": target_step_index,
        "error_rate_array": error_rate_array,
        "keyStep_work_statistic": keyStep_work_statistic, 
        "N": N, "dt": dt, "mean_W": np.mean(work_distribution_data), "mean_W_error": np.std(work_distribution_data) / np.sqrt(len(work_distribution_data)),
        "jarzyn_term": jarzyn_term, "jarzyn_term_error": jarzyn_term_error
    }

def loadDataFrame(gallery_path = NAND_PARAMETERS.GALLERY_JSON_FILE):
    df = pd.read_json(gallery_path)
    df['comment'] = [item['comment'] for item in df["params"]]
    df["simulation_id"] = [item["simulation_id"] for item in df["simulation_data"]]
    return df


def getDataByID(df, sim_id):
    identifier = "Experiment 5"
    target = df[df["simulation_id"] == sim_id]
    return target


def getFidelityTimeGraph(fidelity_time_array, target_step_index):
    """
    trace the fidelity of the systems  at different time as 4 different plots
    """
    metastable_state_array = ["00", "01", "10", "11"]
    fidelity_time_array = np.array(fidelity_time_array, dtype = float)
    
    fig, ax = plt.subplots(2, 2, figsize = [10, 9])
    ax_flatten = ax.flatten()

    for i, _ax in enumerate(ax_flatten):
        _ax.plot(target_step_index, fidelity_time_array[:, 0, i], label = "00")
        _ax.plot(target_step_index, fidelity_time_array[:, 1, i], label = "01")
        _ax.plot(target_step_index, fidelity_time_array[:, 2, i], label = "10")
        _ax.plot(target_step_index, fidelity_time_array[:, 3, i], label = "11")
        _ax.set_title(f"time evolution of fidelity inside {metastable_state_array[i]}")
        _ax.legend()

    
    ax_flatten[0].set_ylim(0, 1)
    ax_flatten[2].set_ylim(0, 1)
    
    ax_flatten[0].set_xticks([])
    ax_flatten[1].set_xticks([])
    ax_flatten[1].set_yticks([])
    ax_flatten[3].set_yticks([])
    plt.show()
    
def drawFidelityBarChart(final_fidelity):
    final_fidelity = final_fidelity.astype("float")

    fig, ax = plt.subplots(2, 2, figsize=[4, 4])
    fig.tight_layout()
    pColor = {"00": "#061DF7", "01": "#FCEF51", "10": "#3FC7F2", "11": "#F7B6D2"}

    # 00
    ax[1][0].bar(["00", "01", "10", "11"], final_fidelity[:,0], color = pColor.values())
    ax[1][0].set_ylim(0, 1)
    ax[1][0].axhline(0.5, linestyle='--')
    ax[1][0].set_yticks([0, 0.5, 1])

    # 01
    ax[0][0].bar(["00", "01", "10", "11"], final_fidelity[:,1], color = pColor.values())
    ax[0][0].set_ylim(0, 1)
    ax[0][0].axhline(0.5, linestyle='--')
    ax[0][0].set_ylim(0, 1)
    ax[0][0].set_yticks([0, 0.5, 1])

    # 10
    ax[1][1].bar(["00", "01", "10", "11"], final_fidelity[:,2], color = pColor.values())
    ax[1][1].set_ylim(0, 1)
    ax[1][1].axhline(0.5, linestyle='--')
    ax[1][1].set_ylim(0, 1)
    ax[1][1].set_yticks([0, 0.5, 1])
    
    # 11
    ax[0][1].bar(["00", "01", "10", "11"], final_fidelity[:,3], color = pColor.values())
    ax[0][1].set_ylim(0, 1)
    ax[0][1].axhline(0.5, linestyle='--')
    ax[0][1].set_ylim(0, 1)
    ax[0][1].set_yticks([0, 0.5, 1])



def showFidelity(target):

    final_percentage_matrix = np.array([d["final_percentage"] for d in list(target["fidelity"])[0]]).T
    text_box_array = ["\n".join(row) for row in final_percentage_matrix]
    params_list = list(target["params"])[0]
    fig, ax = plt.subplots(1, 2, figsize = [10, 4])
    # plt.figure(figsize=[4, 4])
    # plt.title(f"N = {params_list['N']}, dt = {params_list['dt']}")
    _ax = ax[1]
    _ax.set_xlim(-5, 5);plt.ylim(-5, 5);
    _ax.text(-3.25, -3.5, text_box_array[0]); _ax.text(-3.25, 1.5, text_box_array[1])
    _ax.text(1.75, -3.5, text_box_array[2]); _ax.text(1.75, 1.5, text_box_array[3])
    _ax.vlines(0, ymin = -5, ymax = 5); _ax.hlines(0, xmin = -5, xmax = 5)



#--------------reprot data --------------#
def get_correct_and_wrong_number(item):
    correct_number = np.sum([x['correct_number'] for x in item])
    wrong_number = np.sum([x['wrong_number'] for x in item])
    # if wrong_number > 0:
    #     print(item)
    return np.array([correct_number, wrong_number])

def pick_out_wrong_item(item_array):
    wrong_number_info = np.array([0, 0, 0, 0])
    for i, item in enumerate(item_array):
        if item['wrong_number'] != 0:
            wrong_number_info[i] += item['wrong_number']
    return wrong_number_info

import NAND_PARAMETERS
def getSimulationWorkStatistics2(target, sim_id, folderPath = NAND_PARAMETERS.GALLERY_DATA_FOLDER):
    """
    'params', 'initial_parameter_dict', 'protocol_list_item',
       'simulation_data', 'circuit_parameters', 'sim_params', 'jarzynski_term',
       'mean_work', 'fidelity', 'keyStep_work_statistic',
       'fidleity_time_array', 'target_step_index_info', 'comment', 'script',
       'simulation_id'
    """
    dt = list(target['params'])[0]["dt"]
    N =  list(target['params'])[0]["N"]
    
    
    work_distribution_data = np.load(f"{folderPath}/{sim_id}_work_distribution.npy")
    work_statistic_data = np.load(f"{folderPath}/{sim_id}_work_statistic.npy")
    fidelity_time_array = np.load(f"{folderPath}/{sim_id}_fidelity_time_array.npy")
    jarzyn_term = np.mean(np.exp(-work_distribution_data))
    jarzyn_term_error = 3 * np.std(np.exp(-work_distribution_data)) / np.sqrt(N)
    keyStep_work_statistic = target['keyStep_work_statistic'].values[0]
    error_rate_array = target['error_rate_array'].values[0]
    target_step_index_info = list(target['target_step_index_info'])[0]
    target_step_index = list(range(target_step_index_info['start'], target_step_index_info['end']+1, target_step_index_info['step']))

    return {
        "work_distribution_data": work_distribution_data,
        "work_statistic_data": work_statistic_data,
        "fidelity_time_array": fidelity_time_array,
        "target_time_index": target_step_index,
        "error_rate_array": error_rate_array,
        "keyStep_work_statistic": keyStep_work_statistic, 
        "N": N, "dt": dt, "mean_W": np.mean(work_distribution_data), "mean_W_error": np.std(work_distribution_data) / np.sqrt(len(work_distribution_data)),
        "jarzyn_term": jarzyn_term, "jarzyn_term_error": jarzyn_term_error
    }

def report_data(target, showGraph = False, preText = None):
    parameter_array = ["T", "dt", "N", "duration", "C", "L", "gamma"]

    beta = target['initial_parameter_dict'].values[0]['beta_1']
    gamma = target['initial_parameter_dict'].values[0]['gamma_1']
    protocol_list = list(target['protocol_list_item'])[0]
    total_duration = [0] + [x['duration'] for x in protocol_list]

    sim_id_array = list(target['simulation_id'].values)
    statistic_data_array = [getSimulationWorkStatistics2(target, _sim_id) for _sim_id in sim_id_array]
    
    saveFolderPath = NAND_PARAMETERS.GALLERY_DATA_FOLDER
    
    index_array = {"00": [], "01": [], "10": [], "11": []}

    # work_analysis
    work_dist_array = [_data['work_distribution_data'] for _data in statistic_data_array]
    mean_work_array = [_data['mean_W'] for _data in statistic_data_array]
    mean_work_ste   = np.array([_data['mean_W_error'] for _data in statistic_data_array])

    # work_dist_array  = np.repeat(work_dist_array, 2)
    work_stat_array = np.array([_data['work_statistic_data'] for _data in statistic_data_array])
    N = len(work_dist_array)
    index = np.mean([np.sum(x['protocol_time_array']) for x in target['params']])
    
    joint_work_dist = np.concatenate(work_dist_array)
    mean_joint_work_dist = np.mean(joint_work_dist)
    std_error_joint_work_dist = np.std(joint_work_dist)/np.sqrt(len(joint_work_dist))* 3

    
    """error analysis"""
    
    total_correct_particle = np.array([0, 0, 0, 0])
    total_wrong_particle = np.array([0, 0, 0, 0])
    for item in target.error_rate_array.values:
        for index, particle_data in enumerate(item):
            total_correct_particle[index] += particle_data['correct_number']
            total_wrong_particle[index] += particle_data['wrong_number']

    correct_number = np.sum(total_correct_particle)
    wrong_number = np.sum(total_wrong_particle)
    total_number = correct_number + wrong_number
    
    print(target["comment"].head(1).values[0])
    print(target["simulation_id"].head(1).values[0])
    print(f"beta = {beta}, gamma = {gamma}")
    print(f"protocol = [")
    for item in protocol_list:
        print(f"{item}, ",)
    print("]")
    print(f"substage duration = {total_duration}")
    print(f"duration = {np.sum(total_duration)}")
    if preText:
        print(preText)
        
    if showGraph:
        fig, ax = plt.subplots(1, 1, figsize=[5, 5])
        ax.set_title(r"$\langle W \rangle$ = " + f"{mean_joint_work_dist:.3g}")
        
        ax.hist(joint_work_dist, bins = 80)
        ymin, ymax = ax.get_ylim()
        ax.vlines(x=mean_joint_work_dist, ymin=ymin, ymax = ymax, linestyle = "--", color = "red")
        ax.set_xlabel(r"work ($k_BT$)")
        ax.set_ylabel("count")
        plt.show()

        pColor = {"00": "#061DF7", "01": "#FCEF51", "10": "#3FC7F2", "11": "#F187F4"}

        for sim_id in target['simulation_id'].values:
            _index = np.load(os.path.join(saveFolderPath, f"{sim_id}_initial_index.npy"), allow_pickle=True)
            
            index_array["00"].append(_index.item()["00"])
            index_array["01"].append(_index.item()["01"])
            index_array["10"].append(_index.item()["10"])
            index_array["11"].append(_index.item()["11"])

        index_00 = np.hstack(index_array['00'])
        index_01 = np.hstack(index_array['01'])
        index_10 = np.hstack(index_array['10'])
        index_11 = np.hstack(index_array['11'])
        fig, ax = plt.subplots(2, 2, figsize=[6, 6])
        ax_01, ax_11, ax_00, ax_10 = ax.flatten()

        
        ax_01.hist(joint_work_dist[index_01], bins = 80, color = pColor['01'])
        ax_01.set_title(r"$\langle W_{01} \rangle$ = " + f"{np.mean(joint_work_dist[index_01]):.3g}")
        ax_11.hist(joint_work_dist[index_11], bins = 80, color = pColor['11'])
        ax_11.set_title(r"$\langle W_{11} \rangle$ = " + f"{np.mean(joint_work_dist[index_11]):.3g}")
        ax_00.hist(joint_work_dist[index_00], bins = 80, color = pColor['00'])
        ax_00.set_title(r"$\langle W_{00} \rangle$ = " + f"{np.mean(joint_work_dist[index_00]):.3g}")
        ax_10.hist(joint_work_dist[index_10], bins = 80, color = pColor['10'])
        ax_10.set_title(r"$\langle W_{10} \rangle$ = " + f"{np.mean(joint_work_dist[index_10]):.3g}")
        # ax_10.set_yticks([0, 20000, 40000])
        fig.supxlabel(r"work ($k_BT$)")
        fig.supylabel(r"counts")

        plt.show()
                    
            
                

    else:
        print(f"N = {len(joint_work_dist):,}, {mean_joint_work_dist:.3g}± { std_error_joint_work_dist:.3g}")
    
    
    print("error rate by old method") 
    print(correct_number, wrong_number, f"{wrong_number/total_number}")
    print(f"N = {total_number:,},\ncorrect = {correct_number:,},\nwrong = {wrong_number:,}")
    print(f"{wrong_number/total_number * 100:.3e}%")
    print(f"{wrong_number/total_number:.3e}")
    print(f"{total_wrong_particle}")

    print("-" * 20)



















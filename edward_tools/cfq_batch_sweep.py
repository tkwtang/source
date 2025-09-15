import copy, datetime, socket, sys, os, pickle
import numpy as np
import edward_tools.couple_flux_qubit_metrics as couple_flux_qubit_metrics
import edward_tools.coupled_fq_protocol_library as coupled_fq_protocol_library
import edward_tools.visualization as visualization
from edward_tools import coupled_fq_protocol_library, cfq_runner
coupled_fq_runner = cfq_runner
from IPython import display
from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import time, datetime, json, hashlib
from PARAMETER_INPUT import *
from edward_tools.visualization import separate_by_state
import NAND_PARAMETERS


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()        
        if isinstance(obj, slice):
            return [obj.start, obj.stop, obj.step]

        return json.JSONEncoder.default(self, obj)

create_system = coupled_fq_protocol_library.create_system
get_potential_shot_at_different_t = coupled_fq_protocol_library.get_potential_shot_at_different_t
create_simple_protocol_parameter_dict = coupled_fq_protocol_library.create_simple_protocol_parameter_dict
create_system_from_storage_and_computation_protocol = coupled_fq_protocol_library.create_system_from_storage_and_computation_protocol
# mapping_state_1_to_state_2_dict = {"00": ["00", "10"], "01": ["00", "10"], "10": ["01", "11"], "11": ["01", "11"]} # this if for sizlard engine


mapping_index = {"00": 0, "01": 1, "10": 2, "11": 3}


mapping_state_1_to_state_2_dict_szilard_engine = {"00": ["00", "10"], "01": ["00", "10"], "10": ["01", "11"], "11": ["01", "11"]}


# mapping_state_1_to_state_2_dict_CE = {"00": ["01"], "01": ["01"], "10": ["10"], "11": ["11"]}

# mapping_state_1_to_state_2_dict_in_index_form_CE = {0: [1], 1: [1], 2: [2], 3: [3]}


# mapping_state_1_to_state_2_dict_NAND = {"00": ["11"], "01": ["11"], "10": ["11"], "11": ["00"]}

# mapping_state_1_to_state_2_dict_in_index_form_NAND = {0: [3], 1: [3], 2: [3], 3: [0]}



# mapping_state_1_to_state_2_dict = mapping_state_1_to_state_2_dict_NAND
# mapping_state_1_to_state_2_dict_in_index_form = mapping_state_1_to_state_2_dict_in_index_form_NAND


U0_kBT_ratio = U0_1 / k_BT


def create_initial_state(initial_parameter_dict, initial_potential_protocol_list, coupled_fq_pot, params, has_velocity = True):
    computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, \
                                                                    initial_potential_protocol_list)
    storage_protocol, comp_protocol = create_system(computation_protocol_parameter_dict, modifiedFunction = None)
    cfqr = cfq_runner.coupledFluxQubitRunner(potential = coupled_fq_pot, params = params, \
                                                storage_protocol= storage_protocol, \
                                                computation_protocol= comp_protocol, \
                                         protocol_list = initial_potential_protocol_list, \
                                        has_velocity=has_velocity)
    cfqr.initialize_sim()
    cfqr.set_sim_attributes()
    cfqr.run_sim(init_state=cfqr.init_state)
    return  cfqr.sim.output.final_state
    

def simulateSingleCoupledFluxQubit(params, initial_parameter_dict, protocol_list, potential = None, potential_default_param = None,
    mapping_state_1_to_state_2_dict = None,
    phi_1xdc = 0, phi_2xdc = 0, initial_state = None, manual_domain = None, comment = "", measure_all_states = False, has_velocity = True, modifiedFunction=None, fidelityTest = True, pickle_save_path = None):
    
    """To create the index mapping dict"""
    keys = mapping_state_1_to_state_2_dict.keys()
    values = mapping_state_1_to_state_2_dict.values()

    new_keys = [mapping_index[key] for key in keys]
    new_values = []

    for value_array in values:
        new_values.append([mapping_index [item] for item in value_array])

    mapping_state_1_to_state_2_dict_in_index_form = dict(zip(new_keys, new_values))

    
    """
    The main object to perform simulations.
    """
    # base information
    start_time = time.time()
    print("started at ", datetime.datetime.now())
    now = str(start_time)
    sim_id = hashlib.sha256(bytes(now, encoding='utf8')).hexdigest()

    computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, protocol_list)

    print(modifiedFunction)
    storage_protocol, comp_protocol = create_system(computation_protocol_parameter_dict, modifiedFunction = modifiedFunction)

    cfqr = cfq_runner.coupledFluxQubitRunner(potential = potential, params = params, potential_default_param = potential_default_param, storage_protocol= storage_protocol,  computation_protocol= comp_protocol, measure_all_states = measure_all_states, protocol_list = protocol_list, has_velocity = has_velocity)
    cfqr.initialize_sim()
    cfqr.run_sim(init_state = initial_state) # in the multism
    cfqr.system.protocol_list = protocol_list

    initial_state = cfqr.init_state
    final_state = cfqr.sim.output.final_state
    failure_particles = cfqr.get_failure_particles(mapping_state_1_to_state_2_dict)


    # fidelity_test
    if cfqr.system.has_velocity and fidelityTest:
        initial_phi_1_phi_2 = initial_state[:, (0, 1), :]
        final_phi_1_phi_2   = final_state[:, (0, 1), :]
        fidelity = couple_flux_qubit_metrics.fidelityEvaluation(initial_phi_1_phi_2, final_phi_1_phi_2, mapping_state_1_to_state_2_dict)
        error_rate = couple_flux_qubit_metrics.errorRateEvaluation(initial_phi_1_phi_2, final_phi_1_phi_2, mapping_state_1_to_state_2_dict_in_index_form)
    else:
        fidelity = None
        error_rate = None
        cfqr.sim.fidelity_time_array = None
        # cfqr.sim.work_statistic_array = None
        cfqr.sim.keyStep_work_statistic = None
    simulation_time = time.time() - start_time
    
    simulation_datestamp = datetime.datetime.now()
    print("finished at ", simulation_datestamp)
    print(f"time_duration: {simulation_time}, sim_id: {sim_id}")
    

    if not hasattr(cfqr, "keyStep_work_statistic"):
        cfqr.keyStep_work_statistic = None
    
    export_cfqr_params = params['circuit_parameters']
    
    export_params = {
        "N": params["N"], "dt": params["dt"], "lambda": params["lambda"], "beta": params["beta"],
        "protocol_time_array": params['protocol_time_array'], "protocol_time_index_array": params['protocol_time_index_array'],
        'target_work': params["target_work"], 'applyOffset': params["applyOffset"], 
        'monitor_work_dist_in_whole_process': params["monitor_work_dist_in_whole_process"], 
        "measureWorkWithOffset": params["measureWorkWithOffset"],
        "comment": params["comment"], "capacitance": params['capacitance'], "v_c": params['v_c'], "k_BT": params['k_BT'],
        "k_BT": params['U0'], 'as_step': params["as_step"], 'percentage': params["percentage"], "mass": params["mass"]
    }
    

    export_object = {
        "cfqr_params": export_cfqr_params,
        "params": export_params,
        "initial_external_parameter_dict": initial_parameter_dict,
        "protocol_list": protocol_list,
        "manual_domain": [np.array([-5, -5]), np.array([5, 5])]
    }
    
    if pickle_save_path == None:
        pickle_file_path = os.path.join("coupled_flux_qubit_protocol", "simulation_protocol_history", f"{simulation_datestamp}.pickle")
    else:
        pickle_file_path = os.path.join(pickle_save_path, f"{simulation_datestamp}.pickle")
    with open(pickle_file_path, "wb") as f:
        pickle.dump(export_object, f)
        


    
    

            
    if has_velocity:
        return {
            "cfqr": cfqr,
            "fidelity": fidelity,
            "work_distribution": cfqr.sim.work_dist_array , # work_dist_array is for get_dW
            "work_statistic": cfqr.sim.work_statistic_array ,
            "keyStep_work_statistic": cfqr.sim.keyStep_work_statistic,
            "fidelity_time_array": cfqr.sim.fidelity_time_array,
            "error_rate_array": error_rate,
            "mapping_state_1_to_state_2_dict": mapping_state_1_to_state_2_dict,
            "params": params,
            "failure_particles": failure_particles,
            "initial_parameter_dict": initial_parameter_dict,
            "protocol_list_item": protocol_list,
            "simulation_data":{
                "simulation_time": simulation_time,
                "simulation_date": str(simulation_datestamp),
                "simulation_id": sim_id,
                "pickle_save_path": pickle_save_path
            },
            "comment": comment
        }
    else:
        return {
            "cfqr": cfqr,
            "fidelity": fidelity,
            "work_distribution": None , # work_dist_array is for get_dW
            "work_statistic": None ,
            "keyStep_work_statistic": None,
            "fidelity_time_array": None,
            "error_rate_array": error_rate,
            "mapping_state_1_to_state_2_dict": mapping_state_1_to_state_2_dict,
            "params": params,
            "failure_particles": failure_particles,
            "initial_parameter_dict": initial_parameter_dict,
            "protocol_list_item": protocol_list,
            "simulation_data":{
                "simulation_time": simulation_time,
                "simulation_date": str(datetime.datetime.now()),
                "simulation_id": sim_id,
                "pickle_save_path": pickle_save_path
            },
            "comment": comment
        }

def generateSweepProtocolArray(protocol_list_wanted_to_sweep):
    """
    To create a list of protocol with a particular parameter you want to sweep.
    """

    sweep_protocol_list = []
    index = 0
    sweepKey = ""
    sweepArray = None

    for i, substep in enumerate(protocol_list_wanted_to_sweep):
        for key, elem in substep.items():
            if type(elem) is np.ndarray:
                index = i
                sweepKey = key
                sweepArray = elem
                substageName = substep["name"]
                # sweepArray = elem
                # sweepKey = substep["name"]

    for x in sweepArray:
        newProtocolList = copy.deepcopy(protocol_list_wanted_to_sweep)
        newProtocolList[index][sweepKey] = x
        sweep_protocol_list.append({"protocol_list": newProtocolList, "substepIndex": i, "sweepKey": key, "sweepParameter": x, "substageName": substageName})
    return sweep_protocol_list


def showResut(resultArray, itemsWantToShow = None, bins = 10):
    """
    To show the results such as fidelity and work_distribution
    """

    _resultArray = []

    if itemsWantToShow:
        for x in itemsWantToShow:
            _resultArray.append(resultArray[x])
    else:
        _resultArray = resultArray

    for item in _resultArray:
        fidelity = item["fidelity"]
        work_distribution = item["work_distribution"]
        jarzyn_term = np.mean(np.exp(-work_distribution))
        ani = item["ani"]
        substepIndex = item["substepIndex"]
        sweepKey = item["sweepKey"]
        sweepParameter = item["sweepParameter"]

        print(f"substep: {substepIndex}, key: {sweepKey}, value: {sweepParameter}")
        video = ani.to_html5_video()
        html = display.HTML(video)
        display.display(html)

    #     fidelity analysis
        for x in fidelity:
            initialLocation = x["initial"]["location"]
            initialCount = x["initial"]["count"]
            summaryText = f"initial: {initialLocation} ({initialCount}), final: "
            rightLocationCount = sum([y["count"] for y in x["final"]])
            goodRatio = rightLocationCount/initialCount * 100
            for y in x["final"]:
                summaryText += f"{y['location']} ({y['count']}/{rightLocationCount},{y['count']/rightLocationCount * 100: .3g}%),"
            summaryText += f" goodRatio:{goodRatio: .3g}%"
            # print(summaryText)

        plt.hist(work_distribution, bins = bins)
        plt.show()
        unmodified_jarzyn = np.mean(np.exp(work_distribution))
        print(f"jarzyn_term: {jarzyn_term}")
        print("-" * 100)


def getProtocolSubstepName(protocol_list, t):
    """
    Return the name of the substep at time t.
    """
    time_array = [item["duration"] for item in protocol_list]
    name_array = [item["name"] for item in protocol_list]
    cumulative_time_array = list(itertools.accumulate(time_array, operator.add))

    targetIndex = 0

    for i, x in enumerate(cumulative_time_array):
        if i == len(cumulative_time_array) - 1:
            targetIndex = i
            break
        elif i == 0 and t < cumulative_time_array[i]:
            print("case 2")
            targetIndex = i
            break
        else:
            if t >= cumulative_time_array[i] and t <= cumulative_time_array[i+1]:
                targetIndex = i + 1
                break

    print(time_array, cumulative_time_array, name_array[targetIndex])


def saveSimulationResult(simResult, U0_1, timeOrStep = "time", save = False, save_final_state = False, comment = "", saveFolderPath = NAND_PARAMETERS.GALLERY_DATA_FOLDER, save_all_state = False, save_work_distribution_time_array = False, save_animation = False):
    """U0_1"""
    #  fidelity
    fidelity = simResult["fidelity"]

 
    # work_distribution
    work_distribution = simResult["work_distribution"]
    mean_work = np.mean(work_distribution)
    unmodified_jarzyn = float(np.mean(np.exp(-work_distribution)))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(work_distribution, bins = 30)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, f"mean work: {mean_work: .3g}, Jarzyn: {unmodified_jarzyn: .3g}", transform=ax.transAxes, fontsize=14, verticalalignment='top',bbox=props)
    # plt.hist(work_distribution)
    if save:
        np.save(f'{saveFolderPath}/{simResult["simulation_data"]["simulation_id"]}_work_distribution.npy', work_distribution)
        plt.savefig(f'{saveFolderPath}/{simResult["simulation_data"]["simulation_id"]}_work_distribution.png')
        pass

        
    # work statistics
    work_statistic = simResult["work_statistic"]
    work_mean, work_std = work_statistic[:, 0], work_statistic[:, 1]

    step_array = simResult['cfqr'].sim.target_step_index
    target_step_index_info = {"start": step_array[0], "end": step_array[-1], "step": step_array[1] - step_array[0]}
    skip_step = int(len(step_array) * 0.05)
    work_mean = simResult["work_statistic"][:,0]
    work_std = simResult["work_statistic"][:,1]
    final_KE = np.mean(simResult['cfqr'].system.get_kinetic_energy(simResult["cfqr"].sim.output.final_state))

    plt.figure(figsize=(10, 7))
    plt.errorbar(step_array[::skip_step], work_mean[::skip_step], yerr = work_std[::skip_step])

    substep_array = np.cumsum([substep["duration"]/simResult["cfqr"].sim.dt for substep in simResult["protocol_list_item"]])
    for _t in substep_array[:-1]:
        plt.vlines(x=_t, ymin = np.min(work_mean), ymax = np.max(work_mean), ls="--", colors = "purple")

    if save:
        np.save(f'{saveFolderPath}/{simResult["simulation_data"]["simulation_id"]}_work_statistic.npy', work_statistic)
        plt.savefig(f'{saveFolderPath}/{simResult["simulation_data"]["simulation_id"]}_work_statistic.png')
        pass

    # index
    initial_index = simResult['cfqr'].getIndexOfParticles()
    final_index =  simResult['cfqr'].getIndexOfParticles(simResult['cfqr'].sim.output.final_state)
    if save:
        np.save(f'{saveFolderPath}/{simResult["simulation_data"]["simulation_id"]}_initial_index.npy', initial_index)
        np.save(f'{saveFolderPath}/{simResult["simulation_data"]["simulation_id"]}_final_index.npy', final_index)


    # to save the fidelity time array
    fidelity_time_array = simResult["fidelity_time_array"]
    if save:
        np.save(f'{saveFolderPath}/{simResult["simulation_data"]["simulation_id"]}_fidelity_time_array.npy', fidelity_time_array)

    if save_final_state:
        final_state = simResult["cfqr"].sim.output.final_state
        initial_state_index = simResult['cfqr'].getIndexOfParticles()

        np.save(f'{saveFolderPath}/{simResult["simulation_data"]["simulation_id"]}_final_state.npy', final_state)
        np.save(f'{saveFolderPath}/{simResult["simulation_data"]["simulation_id"]}_particle_index.npy', initial_state_index)
        np.save(f'{saveFolderPath}/{simResult["simulation_data"]["simulation_id"]}_final_work_distribution.npy', work_distribution)

        plt.figure(figsize=(10, 7))
        plt.scatter(final_state[:, 0, 0], final_state[:, 1, 0])
        plt.savefig(f'{saveFolderPath}/{simResult["simulation_data"]["simulation_id"]}_final_state.png')

    
    # if save_all_state:
        # np.save(f'{saveFolderPath}/{simResult["simulation_data"]["simulation_id"]}_all_states.npy', simResult["cfqr"].sim.output.all_state)
        # number_of_failure_particles = len(simResult['failure_particles'])
        # pass
     


    if save_work_distribution_time_array:
        work_dist_time_array = simResult["cfqr"].sim.work_dist_time_array
        np.save(f'{saveFolderPath}/{simResult["simulation_data"]["simulation_id"]}_work_distribution_time_array.npy', work_dist_time_array[::25])

    if save_animation:
        FFwriter = animation.FFMpegWriter(fps=10)

        pColor = {"00": "#061DF7", "01": "#FCEF51", "10": "#3FC7F2", "11": "#F187F4"}

        ani,_,_ =  coupled_fq_protocol_library.animate_sim_flux_qubit_with_cutline(simResult['cfqr'], frame_skip = int(1/params['dt']), time_array=simResult['cfqr'].sim.target_step_index, params = params, cutlineInformation = [(-1.96, "v","red") , (1.96, "v","orange")], pColor=pColor, opacity = 0, cutlineYLimit = [0, 250])
        ani.save( f'coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{simResult["simulation_data"]["simulation_id"]}_szilard_engine.mp4', writer = FFwriter)


    params_save_field = ["N", "dt", "comment", "protocol_time_array", "protocol_time_index_array", "lambda", "beta", "target_work", "applyOffset", "monitor_work_dist_in_whole_process", 
                         "target_work", "applyOffset", "measureWorkWithOffset", "monitor_work_dist_in_whole_process", "comment", "as_step", "percentage", "mass"
    ]

    saveParams = {field: simResult["params"][field] for field in params_save_field}

    simResult["simulation_data"]["simulation_computer"] = socket.gethostname()
    simResult["simulation_data"]["saveTime"] = str(datetime.datetime.timestamp( datetime.datetime.now()))

    saveData = {
        "params":                       simResult["params"],
        "initial_parameter_dict":       simResult["initial_parameter_dict"],
        "protocol_list_item":           simResult["protocol_list_item"],
        "simulation_data":              simResult["simulation_data"],
        "circuit_parameters":           simResult["params"]['circuit_parameters'],
        "sim_params":                   [list(item) for item in simResult["params"]["sim_params"]],
        "jarzynski_term":               unmodified_jarzyn,
        "mean_work":                    mean_work,
        "number_of_failure_particles":  len(simResult['failure_particles']),
        "final_KE":                     final_KE,
        "fidelity":                     simResult["fidelity"],
        "error_rate_array":             simResult["error_rate_array"],
        "keyStep_work_statistic":       simResult["keyStep_work_statistic"],
        "target_step_index_info":       target_step_index_info,
        "comment":                      simResult["params"]["comment"],
        "script":                       sys.argv[0]
    }

    np.save(f'{saveFolderPath}/{simResult["simulation_data"]["simulation_id"]}_sim_data.npy', saveData)

    
    RESET = "\033[0m"
    RED = "\033[91m"    
    print(f"{RED} The number of failure particles (from batch_sweep): = {len(simResult['failure_particles'])} {RESET}")

    with open(f"../{NAND_PARAMETERS.GALLERY_JSON_FOLDER}/{simResult['simulation_data']['simulation_id']}_waiting_to_be_integrated_to_gallery.json", "w+") as fw:
        json.dump(saveData, fw, cls = NumpyArrayEncoder)


import json

def saveToJSON(file_name, simResult, dt = 1/1000):
    
    params = copy.deepcopy(simResult['params']) 
    params['dt'] = dt
    json_output_path = f"json_input_folder/{file_name}"

    data = {"params": params, "initial_parameter_dict": simResult['initial_parameter_dict'], "protocol_list_item": simResult['protocol_list_item']}
    with open(os.path.join("NAND_test", json_output_path), "w") as f:
        json.dump(data, f, cls = NumpyArrayEncoder, indent = 4)
    
    print(f"bash run_N_time_dummy_json_input.sh -s '{json_output_path}' -n 200")


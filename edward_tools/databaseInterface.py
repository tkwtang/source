import matplotlib.pyplot as plt
import numpy as np
import datetime, itertools, json, operator, os, sys, hashlib

source_path = os.path.expanduser('~/Project/source/')
sys.path.append(source_path)

dataBasePath = "cfq_data/cfq_data_all_expt_results"

keyArrayToSave = ["fidelity", "work_distribution", "work_statistic", "params", "initial_parameter_dict", "protocol_list_item", "simulation_data", "comment"]

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def generate_ID():
    now = str(datetime.datetime.now())
    sim_id = hashlib.sha256(bytes(now, encoding='utf8')).hexdigest()
    return sim_id

def saveToDataBase(simResult, keyArrayToSave = keyArrayToSave):
    """
    keyArrayToSave: the field that you want to save
    """
    print("sim_id is: " + simResult['simulation_data']['simulation_id'])
    with open(f"{dataBasePath}/{simResult['simulation_data']['simulation_id']}.json", "w+") as f:
        filteredJsonData = {key: simResult[key] for key in keyArrayToSave}
        json.dump(filteredJsonData, f, cls = NumpyArrayEncoder)




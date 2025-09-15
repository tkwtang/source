import pandas as pd
import json, sys
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML
from IPython import display
import importlib
import os
import matplotlib.pyplot as plt
from PIL import Image
from PARAMETER_INPUT import *
from PARAMETER_INPUT import _lambda, _theta, _eta
import edward_tools.create_cfqr as create_cfqr
U0_kBT_ratio = U0_1/k_BT
comment = sys.argv[1]

def getDataByField(df, field):
    return list(target[field])

def getSimulationID(df):
    return [item["simulation_id"] for item in df["simulation_data"]] 

def loadDataFrame(folderPath = 
"coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery"):
        df = pd.read_json(f"{folderPath}/gallery.json")
        df['comment'] = [item['comment'] for item in df["params"]]
        df["simulation_id"] = [item["simulation_id"] for item in 
df["simulation_data"]]
        return df

def getDataByComment(df, identifier):
    target = df[df["comment"].str.find(identifier) == 0]
    return target

df = loadDataFrame()
target = getDataByComment(df, comment)

print(list(target['simulation_id'].values))

#for t in target:
#    sim_id = getSimulationID(t)
#    print(sim_id)


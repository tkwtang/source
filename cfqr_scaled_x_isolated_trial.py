#!/usr/bin/env python
# coding: utf-8

# In[110]:


import numpy as np
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from sus.protocol_designer.protocol import sequential_protocol
from IPython import display
from IPython.display import HTML
from quick_sim import setup_sim
from edward_tools.coupled_fq_potential import coupled_flux_qubit_pot, coupled_flux_qubit_force, coupled_fq_pot
import edward_tools.fq_runner as fq_runner
from edward_tools.visualization import animate_sim_flux_qubit

import kyle_tools as kt
import matplotlib.pyplot as plt

import importlib, os
from edward_tools import coupled_fq_protocol_library, cfq_runner
from PARAMETER_INPUT import *
from PARAMETER_INPUT import _lambda, _theta, _eta

from edward_tools import coupled_fq_protocol_library
import edward_tools.cfq_batch_sweep as cfq_batch_sweep

coupled_fq_protocol_library = importlib.reload(coupled_fq_protocol_library)
create_system = coupled_fq_protocol_library.create_system
get_potential_shot_at_different_t = coupled_fq_protocol_library.get_potential_shot_at_different_t
get_potential_shot_at_different_t_1D = coupled_fq_protocol_library.get_potential_shot_at_different_t_1D
create_simple_protocol_parameter_dict = coupled_fq_protocol_library.create_simple_protocol_parameter_dict
create_system_from_storage_and_computation_protocol = coupled_fq_protocol_library.create_system_from_storage_and_computation_protocol
coupled_fq_runner = importlib.reload(cfq_runner)
coupled_fq_protocol_library = importlib.reload(coupled_fq_protocol_library)
create_system = coupled_fq_protocol_library.create_system


# In[111]:


def analysis_protocol_function(var_name, var_value, simResult):
    average_work = simResult["work_statistic"][:,0][-1]
    jarzyn_term = np.mean(np.exp(-simResult["work_distribution"]))
    fidelity_data = simResult["fidelity"]

    saveData = {
        var_name:       var_value,
        "fidelity":     fidelity_data,
        "average_work": average_work,
        "jarzyn_term":  jarzyn_term
    }

    json_path = f"coupled_flux_qubit_protocol/{var_name}_analysis.json"
    
    print(os.path.isfile(json_path))
    
    
    if not os.path.isfile(json_path):
        with open(json_path, "w") as fw:
            jsonData = []
            json.dump(jsonData, fw)
        
    with open(json_path, "r") as fr:
        jsonData = json.load(fr)
        jsonData.append(saveData)
        print(jsonData)
        with open(json_path, "w+") as fw:
            json.dump(jsonData, fw)


# In[ ]:





# # 4. Szilard engine test

# In[112]:


import numpy as np
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from sus.protocol_designer.protocol import sequential_protocol
from IPython.display import HTML
from quick_sim import setup_sim
from edward_tools.coupled_fq_potential import coupled_flux_qubit_pot, coupled_flux_qubit_force, coupled_fq_pot
from edward_tools.visualization import animate_sim_flux_qubit
import kyle_tools as kt
import matplotlib.pyplot as plt
import importlib, os, hashlib, json
from edward_tools import coupled_fq_protocol_library, cfq_runner


# In[113]:


# from edward_tools import coupled_fq_protocol_library
# coupled_fq_protocol_library = importlib.reload(coupled_fq_protocol_library)
# create_system = coupled_fq_protocol_library.create_system
# get_potential_shot_at_different_t = coupled_fq_protocol_library.get_potential_shot_at_different_t
# create_simple_protocol_parameter_dict = coupled_fq_protocol_library.create_simple_protocol_parameter_dict
# create_system_from_storage_and_computation_protocol = coupled_fq_protocol_library.create_system_from_storage_and_computation_protocol
# coupled_fq_runner = importlib.reload(cfq_runner)


# In[114]:


"""
# step 0: modify parameters
- All the parameters are stored in a separate file PARAMETER_INPUT
- You can override some of the parameters here.
"""
params['N'] = 50_00
params['dt'] = 1.0/1_00





# In[115]:


"""
# step 1: Define potential
"""
coupled_fq_default_param = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x_c0]
[phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound] = np.array([4, 4, 4, 4])/time_scale_factor

coupled_fq_domain = [[-phi_1_bound, -phi_2_bound, -phi_1dc_bound, -phi_2dc_bound], \
                     [phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound]]

coupled_fq_pot = Potential(coupled_flux_qubit_pot, coupled_flux_qubit_force, 14, 4,\
                           default_params = coupled_fq_default_param,  relevant_domain = coupled_fq_domain)


# In[116]:


"""
# step 2: Define initial condition and protocol
"""
zeroDissipation = False
params['sim_params'] = [_lambda, _theta, _eta]
if zeroDissipation:
    params['sim_params'] = [_lambda * 0, _theta, _eta * 0]


manual_domain=[np.array([-5, -5]), np.array([5, 5])]
# phi_1_dcx, phi_2_dcx = 3, 3
phi_1_dcx, phi_2_dcx = 0, 0
phi_1_dc, phi_2_dc = phi_1_dcx, phi_2_dcx

# gamma, beta_1, beta_2 = 0, 0, 0

# d_beta_1, d_beta_2 = 0.6, 0.6
# d_beta_1, d_beta_2 = 0, 0
# params['sim_params'] = [_lambda, _theta, _eta]

# this is the initial parameter used
initial_parameter_dict = {
        "U0_1": U0_1,     "U0_2": U0_2,     "gamma_1": gamma,  "gamma_2": gamma,
        "beta_1": beta_1,   "beta_2": beta_2,   "d_beta_1": d_beta_1 ,   "d_beta_2": d_beta_2,
        "phi_1_x": 0,  "phi_2_x": 0,  "phi_1_dcx": phi_1_dcx,  "phi_2_dcx": phi_2_dcx,
        "M_12": 0, 'x_c': x_c
}


# Time symmetric protocol
# THis part is how you change the protocol. You can specify each substep by duration, the variables you want to 
# change and the name of the substep.


#**************#**************
time_length = 50
#**************#**************

# protocol_list = [
#     {"duration":time_length,"phi_1_dcx":0, "phi_2_dcx":3, "M_12": 0,     "name":"mix in y direction"},
#     {"duration":time_length,"phi_1_dcx":0, "phi_2_dcx":3, "M_12": -0.05, "name":"conditional tilt"},
#     {"duration":time_length,"phi_1_dcx":0, "phi_2_dcx":0, "M_12": -0.05, "name":"raise the barrie"},
#     {"duration":time_length,"phi_1_dcx":0, "phi_2_dcx":0, "M_12": 0,      "name":"4 well potential"},    
#     {"duration":time_length,"phi_1_dcx":3, "phi_2_dcx":0, "M_12": 0,      "name":"mix in x direction"}, 
#     {"duration":time_length,"phi_1_dcx":0, "phi_2_dcx":0, "M_12": 0,      "name":"4 well potential"},   
# ]


# In[ ]:





# In[ ]:





# In[283]:


M_12_value = -0.05
time_length_1 = 150
time_length_2 = 50
# time_length_3 = 90
time_length_3 = 50
time_length_4 = 1
protocol_list = [
    # forward
    {"duration":time_length_1,"phi_1_dcx":0, "phi_2_dcx":3, "M_12": 0,     "name":"mix in y direction"},
    {"duration":time_length_2, "phi_1_dcx":0, "phi_2_dcx":3, "M_12": M_12_value, "name":"conditional tilt"},
    {"duration":time_length_3,"phi_1_dcx":0, "phi_2_dcx":0, "M_12": M_12_value, "name":"raise the barrie"},
    {"duration":time_length_4,"phi_1_dcx":0, "phi_2_dcx":0, "M_12": 0, "name":"4 well potential"}, 
#     {"duration":time_length,"phi_1_dcx":3, "phi_2_dcx":0, "M_12": 0,      "name":"mix in x direction"}, 

    # reverse
#     {"duration":time_length,"phi_1_dcx":0, "phi_2_dcx":0, "M_12": 0,"     name":"4 well potential"}, 
    {"duration":time_length_4,"phi_1_dcx":0, "phi_2_dcx":0, "M_12": -0.05, "name":"raise the barrie"},
    {"duration":time_length_3,"phi_1_dcx":0, "phi_2_dcx":3, "M_12": M_12_value, "name":"raise the barrie"},
    {"duration":time_length_2, "phi_1_dcx":0, "phi_2_dcx":3, "M_12": 0,     "name":"conditional tilt"},
    {"duration":time_length_1,"phi_1_dcx":0, "phi_2_dcx":0, "M_12": 0,     "name":"mix in y direction"}
]


# In[ ]:





# In[284]:


protocol_list


# In[285]:


"""
# step 3: create the relevant storage protocol and computation protocol
"""
computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, \
                                                                    protocol_list)
storage_protocol, comp_protocol = create_system(computation_protocol_parameter_dict)


# In[286]:


"""
# step 4: create the coupled_fq_runner
"""
cfqr = cfq_runner.coupledFluxQubitRunner(potential = coupled_fq_pot, params = params, \
                                                storage_protocol= storage_protocol, \
                                                computation_protocol= comp_protocol)
cfqr.initialize_sim()
cfqr.set_sim_attributes()
init_state_saved = cfqr.init_state


# In[287]:


manual_domain=[np.array([-5, -5])/time_scale_factor, np.array([5, 5])/time_scale_factor]
mapping_state_1_to_state_2_dict = {"00": ["00", "10"], "01": ["00", "10"], "10": ["01", "11"], "11": ["01", "11"]}


# In[ ]:





# In[288]:


# step 5: perform simulations

simResult = cfq_batch_sweep.simulateSingleCoupledFluxQubit(params, initial_parameter_dict, protocol_list, \
                                        initial_state = init_state_saved, manual_domain = manual_domain, \
                                        phi_1_dcx = phi_1_dcx,  phi_2_dcx = phi_2_dcx)

cfqr = simResult["cfqr"]



# In[289]:


get_ipython().run_line_magic('load_ext', 'jupyternotify')


# In[290]:


get_ipython().run_line_magic('notify', '')
# display work distribution

plt.hist(simResult["work_distribution"], bins = 45)
plt.show()
jarzyn_term = np.mean(np.exp(-simResult["work_distribution"]))
print(f'jarzyn = {jarzyn_term}')
print(time_length_4)


# In[291]:


step_array = np.arange(simResult["work_statistic"][:,0].shape[0])
skip_step = int(len(step_array) * 0.05)
work_mean = simResult["work_statistic"][:,0]
work_std = simResult["work_statistic"][:,1]


plt.errorbar(step_array[::skip_step], work_mean[::skip_step], yerr = work_std[::skip_step])


# In[ ]:





# In[16]:


work_std


# In[61]:


average_work = simResult["work_statistic"][:,0][-1]
average_work_std = simResult["work_statistic"][:,1][-1]


# In[62]:


jarzyn_term_std = average_work_std * np.exp(-average_work)


# In[63]:


jarzyn_term_std


import numpy as np


# GALLERY_FILE_PATH = "coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery"
GALLERY_DATA_FOLDER  = "coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery_temp"
GALLERY_JSON_FOLDER = "coupled_flux_qubit_protocol"
GALLERY_JSON_FILE = "coupled_flux_qubit_protocol/gallery.json"
GALLERY_JSON_BACKUP_FILE = "coupled_flux_qubit_protocol/gallery_backup.json"
GALLERY_DATABASE_ARRAY =  ["gallery.json", "gallery_2025_08_30.json"]


protocol_key = [ "U0_1", "U0_2", "gamma_1", "gamma_2", "beta_1", "beta_2", "d_beta_1", "d_beta_2", "phi_1x", "phi_2x", "phi_1xdc", "phi_2xdc", "mu_12", "x_c"] 


phi_1xdc , phi_2xdc = 0, 0
phi_1dc, phi_2dc = phi_1xdc, phi_2xdc

phi_1x_off, phi_2x_off = 0, 0
phi_1xdc_off, phi_2xdc_off, mu_12_off = 0, 0, 0




parameter_choice = ["chris", "squeeze_by_phi_2xdc", "squeeze_by_L", "experiment"]
parameter_choosen = parameter_choice[-1]

experiment_circuit_params_0 = {
    "L": 5e-12 * 2, "T": 4.2,
    "beta": 2.3, "d_beta": 0,
    "phi_1x": 0.58,   "phi_2x": 0.095, "phi_1xdc": 0, "phi_2xdc": 1.96,
    "mu_12": 0.055
}


# experiment_circuit_params = {
#     "L": 5e-12, "T": 4.2,
#     "beta": 2.3, "d_beta": 0,
#     "phi_1x": 0.61,   "phi_2x": 0.10, "phi_1xdc": 0, "phi_2xdc": 1.79,
#     "mu_12": 0.06
# }


pColor = {"00": "#061DF7", "01": "#FCEF51", "10": "#3FC7F2", "11": "#F187F4"}
mapping_state_1_to_state_2_dict_SWAP = {'00': ['01'], '01': ['00'], '10': ['10'], '11': ['11']}
mapping_state_1_to_state_2_dict_CE = {'00': ['00'], '01': ['01'], '10': ['00'], '11': ['11']}
mapping_state_1_to_state_2_dict_storage = {'00': ['00'], '01': ['01'], '10': ['10'], '11': ['11']}
mapping_state_1_to_state_2_erasure_flip = {'00': ['10'], '01': ['11'], '10': ['10'], '11': ['01']}


def choose_NAND_Parameter(parameter_choice):
    print(parameter_choosen)
# Chris' parameter
# i, j = 1, 2

    if parameter_choice == "chris":
        phi_1x_on_12 = 0.61
        phi_2x_on_12 = 0.10
        phi_1xdc_on_12 = 0.0
        phi_2xdc_on_12 = 1.79

        # i, j = 2, 1
        phi_1x_on_21 = 0.10
        phi_2x_on_21 = 0.61
        phi_1xdc_on_21 = 1.79
        phi_2xdc_on_21 = 0.0

        mu_12_on = 0.06


    if parameter_choice == "squeeze_by_phi_2xdc":
        # i, j = 1, 2
        phi_1x_on_12 = 0.59
        phi_2x_on_12 = 0.09
        phi_1xdc_on_12 = 0.0
        phi_2xdc_on_12 = 1.9

        # i, j = 2, 1
        phi_1x_on_21 = 0.09
        phi_2x_on_21 = 0.59
        phi_1xdc_on_21 = 1.9
        phi_2xdc_on_21 = 0.0

        mu_12_on = 0.052

    if parameter_choice == "squeeze_by_L":
        # L_factor = 1.55 parameter
        phi_1x_on_12 = 0.58
        phi_2x_on_12 = 0.10
        phi_1xdc_on_12 = 0.0
        phi_2xdc_on_12 = 1.79

        # i, j = 2, 1
        phi_1x_on_21 = 0.10
        phi_2x_on_21 = 0.58
        phi_1xdc_on_21 = 1.79
        phi_2xdc_on_21 = 0.0

        mu_12_on = 0.06


    if parameter_choice == "experiment":
        # L_factor = 1.5 parameter
        print(f"T = {experiment_circuit_params['T']}, L = {experiment_circuit_params['L']}")
        phi_1x_on_12 = experiment_circuit_params['phi_1x']
        phi_2x_on_12 = experiment_circuit_params['phi_2x']
        phi_1xdc_on_12 = experiment_circuit_params['phi_1xdc']
        phi_2xdc_on_12 = experiment_circuit_params['phi_2xdc']

        # i, j = 2, 1
        phi_1x_on_21 = experiment_circuit_params['phi_2x']
        phi_2x_on_21 = experiment_circuit_params['phi_1x']
        phi_1xdc_on_21 = experiment_circuit_params['phi_2xdc']
        phi_2xdc_on_21 = experiment_circuit_params['phi_1xdc']

        mu_12_on = experiment_circuit_params['mu_12']
        
    four_well = {
        "phi_1x": phi_1x_off, "phi_2x": phi_2x_off, "mu_12": mu_12_off, \
        "phi_1xdc": phi_1xdc_off, "phi_2xdc": phi_2xdc_off, "name":"four well"
    }

    CE_1 = {
        "phi_1x": phi_1x_on_12, "phi_2x": phi_2x_on_12, "mu_12": mu_12_on, \
        "phi_1xdc": phi_1xdc_off, "phi_2xdc": phi_2xdc_on_12, "name":"CE_1"
    }

    
    CE_3 = {
        "phi_1x": phi_1x_on_21, "phi_2x": -phi_2x_on_21, "mu_12": -mu_12_on,\
        "phi_1xdc": phi_1xdc_on_21, "phi_2xdc": phi_2xdc_off, "name":"CE_3"
    }

    CE_5 = {
        "phi_1x": -phi_1x_on_12, "phi_2x": -phi_2x_on_12, "mu_12": mu_12_on, \
        "phi_1xdc": phi_1xdc_off, "phi_2xdc": phi_2xdc_on_12, "name":"CE_5"
    }
    
    
    CE_6 = {
        "phi_1x": -phi_1x_on_12, "phi_2x": -phi_2x_on_12, "mu_12": +mu_12_on, \
        "phi_1xdc": phi_1xdc_off, "phi_2xdc": phi_2xdc_on_12, "name":"CE_6"
    }

    CE_7 = {
        "phi_1x": -phi_1x_on_21, "phi_2x": -phi_2x_on_21, "mu_12": +mu_12_on,\
        "phi_1xdc": phi_1xdc_on_21, "phi_2xdc": phi_2xdc_off, "name":"CE_7"
    }

    CE_8 = {
        "phi_1x": -phi_1x_on_21, "phi_2x": phi_2x_on_21, "mu_12": -mu_12_on,\
        "phi_1xdc": phi_1xdc_on_21, "phi_2xdc": phi_2xdc_off, "name":"CE_8"
    }
    
    KE_extraction_protocol_for_CE_1 = {
        "phi_1x": phi_1x_on_12, "phi_2x": 0.00, "mu_12": 0.00, \
        "phi_1xdc": phi_1xdc_on_12, "phi_2xdc": phi_2xdc_on_12, "name":"KE_extraction_protocol"
    }


    KE_extraction_protocol_for_CE_8 = {
        "phi_1x": 0.00, "phi_2x": phi_2x_on_21, "mu_12": 0.00, \
        "phi_1xdc": phi_1xdc_on_21, "phi_2xdc": phi_2xdc_off, "name":"KE_extraction_protocol"
    }



    lower_H_wells = {
        "phi_1x": 0, "phi_2x": 0, "mu_12": 0,  
        "phi_1xdc":  np.pi, "phi_2xdc": 0, "name":"lower_V_wells"
    }

    flip_H= {
        "phi_1x": 0, "phi_2x": 0, "mu_12": -0.6,  
        "phi_1xdc": np.pi, "phi_2xdc": 0, "name":"Flip Vertically"
    }


    lower_V_wells = {
        "phi_1x": 0, "phi_2x": 0, "mu_12": 0,  
        "phi_1xdc": 0, "phi_2xdc": np.pi, "name":"Flip Horizontally"
    }

    flip_V = {
        "phi_1x": 0, "phi_2x": 0, "mu_12": 0.6,  
        "phi_1xdc": 0, "phi_2xdc": np.pi, "name":"Flip Horizontally"
    }

    return phi_1x_on_12, phi_2x_on_12, phi_1xdc_on_12, phi_2xdc_on_12, phi_1x_on_21, phi_2x_on_21, phi_1xdc_on_21, phi_2xdc_on_21, mu_12_on, four_well, CE_1, CE_8, lower_H_wells, flip_H, lower_V_wells, flip_V, KE_extraction_protocol_for_CE_1, KE_extraction_protocol_for_CE_8





def generate_protocols_from_circuit_params(circuit_params):
    phi_1x_on_12 = circuit_params['phi_1x']
    phi_2x_on_12 = circuit_params['phi_2x']
    phi_1xdc_on_12 = circuit_params['phi_1xdc']
    phi_2xdc_on_12 = circuit_params['phi_2xdc']

    # i, j = 2, 1
    phi_1x_on_21 = circuit_params['phi_2x']
    phi_2x_on_21 = circuit_params['phi_1x']
    phi_1xdc_on_21 = circuit_params['phi_2xdc']
    phi_2xdc_on_21 = circuit_params['phi_1xdc']
    mu_12_on = circuit_params['mu_12']
    
    print(f"T = {circuit_params['T']}, L = {circuit_params['L']}")
    print(f"phi_1x_on_12 = {phi_1x_on_12}, phi_2x_on_12 = {phi_2x_on_12}, phi_1xdc_on_12 = {phi_1xdc_on_12}, phi_2xdc_on_12 = {phi_2xdc_on_12}, mu_12_on = {mu_12_on}")

    
    four_well = {
        "phi_1x": phi_1x_off, "phi_2x": phi_2x_off, "mu_12": mu_12_off, \
        "phi_1xdc": phi_1xdc_off, "phi_2xdc": phi_2xdc_off, "name":"four well"
    }

    CE_1 = {
        "phi_1x": phi_1x_on_12, "phi_2x": phi_2x_on_12, "mu_12": mu_12_on, \
        "phi_1xdc": phi_1xdc_off, "phi_2xdc": phi_2xdc_on_12, "name":"CE_1"
    }

    CE_3 = {
        "phi_1x": phi_1x_on_21, "phi_2x": -phi_2x_on_21, "mu_12": -mu_12_on,\
        "phi_1xdc": phi_1xdc_on_21, "phi_2xdc": phi_2xdc_off, "name":"CE_3"
    }

    CE_5 = {
        "phi_1x": phi_1x_on_12, "phi_2x": -phi_2x_on_12, "mu_12": -mu_12_on, \
        "phi_1xdc": phi_1xdc_off, "phi_2xdc": phi_2xdc_on_12, "name":"CE_1"
    }
    
    CE_6 = {
        "phi_1x": -phi_1x_on_12, "phi_2x": -phi_2x_on_12, "mu_12": +mu_12_on, \
        "phi_1xdc": phi_1xdc_off, "phi_2xdc": phi_2xdc_on_12, "name":"CE_6"
    }

    CE_7 = {
        "phi_1x": -phi_1x_on_21, "phi_2x": -phi_2x_on_21, "mu_12": +mu_12_on,\
        "phi_1xdc": phi_1xdc_on_21, "phi_2xdc": phi_2xdc_off, "name":"CE_7"
    }

    CE_8 = {
        "phi_1x": -phi_1x_on_21, "phi_2x": phi_2x_on_21, "mu_12": -mu_12_on,\
        "phi_1xdc": phi_1xdc_on_21, "phi_2xdc": phi_2xdc_off, "name":"CE_8"
    }
    
    KE_extraction_protocol_for_CE_1 = {
        "phi_1x": phi_1x_on_12, "phi_2x": 0.00, "mu_12": 0.00, \
        "phi_1xdc": phi_1xdc_on_12, "phi_2xdc": phi_2xdc_on_12, "name":"KE_extraction_protocol"
    }


    KE_extraction_protocol_for_CE_8 = {
        "phi_1x": 0.00, "phi_2x": phi_2x_on_21, "mu_12": 0.00, \
        "phi_1xdc": phi_1xdc_on_21, "phi_2xdc": phi_2xdc_off, "name":"KE_extraction_protocol"
    }



    lower_H_wells = {
        "phi_1x": 0, "phi_2x": 0, "mu_12": 0,  
        "phi_1xdc":  np.pi, "phi_2xdc": 0, "name":"lower_V_wells"
    }

    flip_H= {
        "phi_1x": 0, "phi_2x": 0, "mu_12": -0.6,  
        "phi_1xdc": np.pi, "phi_2xdc": 0, "name":"Flip Vertically"
    }


    lower_V_wells = {
        "phi_1x": 0, "phi_2x": 0, "mu_12": 0,  
        "phi_1xdc": 0, "phi_2xdc": np.pi, "name":"Flip Horizontally"
    }

    flip_V = {
        "phi_1x": 0, "phi_2x": 0, "mu_12": 0.6,  
        "phi_1xdc": 0, "phi_2xdc": np.pi, "name":"Flip Horizontally"
    }

    return phi_1x_on_12, phi_2x_on_12, phi_1xdc_on_12, phi_2xdc_on_12, phi_1x_on_21, phi_2x_on_21, phi_1xdc_on_21, phi_2xdc_on_21, mu_12_on, four_well, CE_1, CE_5, CE_8, lower_H_wells, flip_H, lower_V_wells, flip_V, KE_extraction_protocol_for_CE_1, KE_extraction_protocol_for_CE_8


def create_CE_Protocol(_t, _CE, ratio = [1, 1, 1, 1, 1]):
    return {
        "duration": _t, 
        "phi_1x": _CE["phi_1x"] * ratio[0] , "phi_2x": _CE["phi_2x"] * ratio[1], 
        "mu_12": _CE["mu_12"] * ratio[2], "phi_1xdc": _CE["phi_1xdc"] * ratio[3], 
        "phi_2xdc": _CE["phi_2xdc"] * ratio[4], "name": _CE["name"]
    }


mapping_index = {"00": 0, "01": 1, "10": 2, "11": 3}

mapping_state_1_to_state_2_dict_CE = {"00": ["00"], "01": ["01"], "10": ["00"], "11": ["11"]}

mapping_state_1_to_state_2_dict_partial_NAND_by_CE= {"00": ["01"], "01": ["01"], "10": ["01"], "11": ["10"]}
mapping_state_1_to_state_2_dict_NAND  = {"00": ["11"], "01": ["11"], "10": ["11"], "11": ["00"]}
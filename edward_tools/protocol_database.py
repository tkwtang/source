



# Control erasure
{'phi_1xdc': 1.8, 'phi_2xdc': 0, 'phi_1x': -0.0, 'phi_2x': 0.0, 'mu_12': -0.0, 'duration': 40, 'name': 'pre-acceleration_protocol'}
{'phi_1xdc': 2.8, 'phi_2xdc': 0, 'phi_1x': -0.09, 'phi_2x': 0.0, 'mu_12': -0.06, 'duration': 30, 'name': 'acceleration_protocol'}
{'phi_1xdc': 2.8, 'phi_2xdc': 0, 'phi_1x': -0.0, 'phi_2x': 0.0, 'mu_12': -0.0, 'duration': 50, 'name': 'harmonic'}
{'phi_1xdc': 2.1, 'phi_2xdc': 0, 'phi_1x': -0.7, 'phi_2x': 0.0, 'mu_12': -0.0, 'duration': 80, 'name': 'transit'}
{'phi_1xdc': 2.1, 'phi_2xdc': 0, 'phi_1x': -0.7, 'phi_2x': 0.0, 'mu_12': 0.33, 'duration': 40, 'name': 'catch'}
{'phi_1xdc': 1.9, 'phi_2xdc': 0, 'phi_1x': -0.7, 'phi_2x': 0.0, 'mu_12': 0.35, 'duration': 1.5, 'name': 'catch_phase_2'}
{'phi_1xdc': 1.9, 'phi_2xdc': 0, 'phi_1x': -0.7, 'phi_2x': 0.0, 'mu_12': 0.35, 'duration': 20, 'name': 'catch_phase_2'}
{'phi_1xdc': 0, 'phi_2xdc': 0, 'phi_1x': -0.0, 'phi_2x': 0.0, 'mu_12': 0.0, 'duration': 100, 'name': 'four well'}

# Control Flip
{'phi_1xdc': 1.8, 'phi_2xdc': 0, 'phi_1x': 0.0, 'phi_2x': 0.0, 'mu_12': 0.0, 'duration': 40, 'name': 'pre-acceleration_protocol'}
{'phi_1xdc': 2.1, 'phi_2xdc': 0, 'phi_1x': 0.0, 'phi_2x': 0.0, 'mu_12': 0.08, 'duration': 10, 'name': 'acceleration_protocol'}
{'phi_1xdc': 2.7, 'phi_2xdc': 0, 'phi_1x': 0.0, 'phi_2x': 0.0, 'mu_12': 0.0, 'duration': 50, 'name': 'harmonic'}
{'phi_1xdc': 1.7, 'phi_2xdc': 0, 'phi_1x': -0.15, 'phi_2x': 0.0, 'mu_12': 0.035, 'duration': 20, 'name': 'catch_1'}
{'phi_1xdc': 1.8, 'phi_2xdc': 0, 'phi_1x': -0.05, 'phi_2x': 0, 'mu_12': 0.02, 'duration': 25, 'name': 'catch_2'}
{'phi_1xdc': 1.7, 'phi_2xdc': 0, 'phi_1x': -0.0, 'phi_2x': 0, 'mu_12': 0.0, 'duration': 20, 'name': 'catch_3'}
{'phi_1xdc': 0, 'phi_2xdc': 0, 'phi_1x': -0.0, 'phi_2x': 0.0, 'mu_12': 0.0, 'duration': 100, 'name': 'four well'}



# harmonic
beta = 1.35
mu_factor_h0 = 0.2
mu_factor_h1 = 0.0
protocol_harmonic_0 = {"phi_1x": 0.0, "phi_2x": 0.0, "mu_12": 0.0, "phi_1xdc": 0, "phi_2xdc": 0.2, "name": "four_well", "color": "r", "linestyle": "-"} 
protocol_harmonic_1 = {"phi_1x": 0.1 * mu_factor_h0, "phi_2x": 0.0, "mu_12": 0.08 * mu_factor_h0, "phi_1xdc": 0.2, "phi_2xdc": 0, "name": "harmonic_1", "color": "b", "linestyle": "-"} 
protocol_harmonic_2 = {"phi_1x": 0.1 * mu_factor_h0, "phi_2x": 0.0, "mu_12": 0.08 * mu_factor_h0, "phi_1xdc": 1.6, "phi_2xdc": 0, "name": "harmonic", "color": "g", "linestyle": "-"} 
protocol_harmonic_3 = {"phi_1x": 0.1 * mu_factor_h1, "phi_2x": 0.0, "mu_12": 0.08 * mu_factor_h1, "phi_1xdc": 3.0, "phi_2xdc": 0, "name": "harmonic", "color": "g", "linestyle": "-"} 

protocol_list_harmonic = [
    createProtocol(12, protocol_harmonic_1, name = "harmonic"),
    createProtocol(0.1, protocol_harmonic_2, name = "harmonic"),
    createProtocol(0.35, protocol_harmonic_2, name = "harmonic"),
    createProtocol(0.05, protocol_harmonic_3, name = "harmonic"),
    createProtocol(30, protocol_harmonic_3, name = "harmonic"),
    # createProtocol(10, protocol_harmonic_2, name = "harmonic"),
    # createProtocol(0.1, protocol_harmonic_1, name = "harmonic"),
    # createProtocol(12, protocol_harmonic_0, name = "four well"),
    # createProtocol(6.5, protocol_fast_erasure_2, name = "acceleration_protocol_hold"),
    # createProtocol(0.01, protocol_fast_erasure_0, name = "acceleration_protocol_hold"),
    # createProtocol(10, protocol_fast_erasure_0, name = "acceleration_protocol_hold"),
    
    
] 


mu_factor_h0 = 0.28
mu_factor_h1 = 0.45
if beta_1 == 1.5:
    # for beta = 1.5
    protocol_harmonic_h0 = {"phi_1x": 0.0, "phi_2x": 0.0, "mu_12": 0.0, "phi_1xdc": 0, "phi_2xdc": 0, "name": "four_well", "color": "r", "linestyle": "-"} 
    protocol_harmonic_h1 = {"phi_1x": 0.1 * mu_factor_h0, "phi_2x": 0.0, "mu_12": 0.0665 * mu_factor_h0, "phi_1xdc": 1.1, "phi_2xdc": 0, "name": "h1", "color": "r", "linestyle": "-"} 
    protocol_harmonic_h2 = {"phi_1x": 0.1 * mu_factor_h0, "phi_2x": 0.0, "mu_12": 0.0665 * mu_factor_h0, "phi_1xdc": 2.0, "phi_2xdc": 0, "name": "h2", "color": "g", "linestyle": "-"} 
    protocol_harmonic_h3 = {"phi_1x": 0.1 * mu_factor_h1, "phi_2x": 0.0, "mu_12": 0.0665 * mu_factor_h1, "phi_1xdc": 1.1, "phi_2xdc": 0, "name": "h3", "color": "b", "linestyle": "-"} 
    # protocol_harmonic_h4 = {"phi_1x": 0. * mu_factor_h0, "phi_2x": 0.0, "mu_12": 0.00 * mu_factor_h0, "phi_1xdc": 1.0, "phi_2xdc": 0, "name": "harmonic", "color": "r", "linestyle": "-"} 
    cutline_value = 1.5

    time_ratio = 1
    duration_fast_set = [20, 15, 20]
    duration_slwo_set = [25, 15, 25]

    duration_set = duration_fast_set
    protocol_list_harmonic = [
    # createProtocol(duration_set[0] * time_ratio, protocol_harmonic_h2, name = "flip erasure"),
    # createProtocol(duration_set[1] * time_ratio, protocol_harmonic_h3, name = "raise barrier"),
    # createProtocol(duration_set[2] * time_ratio, protocol_harmonic_h0, name = "4 well"),

    createProtocol(20, protocol_harmonic_h2, name = "flip erasure"),
    createProtocol(4, protocol_harmonic_h2, name = "flip erasure"),
    # createProtocol(5, protocol_harmonic_h3, name = "raise barrier"),
    createProtocol(25, protocol_harmonic_h0, name = "4 well"),



    # NAND
    # createProtocol(10, protocol_harmonic_v1, name = "harmonic"),
    # createProtocol(12, protocol_harmonic_v2, name = "harmonic"),
    # createProtocol(14, protocol_harmonic_v3, name = "harmonic"),
    # createProtocol(10, protocol_harmonic_v0, name = "barrier raise"),
    ] 


if beta_1 == 2.3:
# for beta = 2.3
    protocol_harmonic_h0 = {"phi_1x": 0.0, "phi_2x": 0.0, "mu_12": 0.0, "phi_1xdc": 0, "phi_2xdc": 0, "name": "four_well", "color": "r", "linestyle": "-"} 
    protocol_harmonic_h1 = {"phi_1x": 0.1 * mu_factor_h1, "phi_2x": 0.0, "mu_12": 0.05 * mu_factor_h1, "phi_1xdc": 1.5, "phi_2xdc": 0, "name": "h1", "color": "r", "linestyle": "-"} 
    protocol_harmonic_h2 = {"phi_1x": 0.1 * mu_factor_h2, "phi_2x": 0.0, "mu_12": 0.05 * mu_factor_h2, "phi_1xdc": 2.30, "phi_2xdc": 0, "name": "h2", "color": "g", "linestyle": "-"} 
    protocol_harmonic_h3 = {"phi_1x": 0.1 * mu_factor_h3, "phi_2x": 0.0, "mu_12": 0.05 * mu_factor_h3, "phi_1xdc": 1.5, "phi_2xdc": 0, "name": "h3", "color": "b", "linestyle": "-"} 
    # protocol_harmonic_h4 = {"phi_1x": 0. * mu_factor_h0, "phi_2x": 0.0, "mu_12": 0.00 * mu_factor_h0, "phi_1xdc": 1.0, "phi_2xdc": 0, "name": "harmonic", "color": "r", "linestyle": "-"} 
    cutline_value = 2

    protocol_list_harmonic = [
        createProtocol(40, protocol_harmonic_h1, name = "pre flip"),
        createProtocol(10, protocol_harmonic_h2, name = "flip erasure"),
        createProtocol(5, protocol_harmonic_h3, name = "raise barrier"),
        createProtocol(10, protocol_harmonic_h0, name = "4 well final"),
        
    ] 

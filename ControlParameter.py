N = 10_000
dt = 1.0/1_00


# time_length_1 = 16
# time_length_2 = 16
# time_length_3 = 20
# time_length_4 = 65
# time_length_5 = 5
# time_length_6 = 20
# time_rest = 300


time_length_1 = 40
time_length_2 = 40
time_length_3 = 40
time_length_4 = 65
time_length_5 = 100
time_length_6 = 52
time_rest = 50

phi_1_dcx_off = 0
phi_1_dcx_on = 3.0
phi_2_dcx_off = 0
phi_2_dcx_on = 3.0
M_12_off = 0
M_12_on = -0.8

mapping_state_1_to_state_2_dict = {"00": ["00", "10"], 
                                   "01": ["00", "10"], 
                                   "10": ["01", "11"], 
                                   "11": ["01", "11"]}
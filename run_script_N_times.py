import os
import numpy as np

sweep_parameter = {
    "name": "t_1",
    "expt_number": 32,
    "description": "Want to set the ratio of change the protocol to the desired value and then take a rest of step 1",
    "parameter_list": np.linspace(0.01, 1, 20)
}




N = 100

for _ in range(0, N):
    os.system(f"python variance_analysis_TR_to_be_delete.py {sweep_parameter['name']} {52} \"{sweep_parameter['description']}\" {sweep_parameter['expt_number']}")
# # for _ in range(0, N):
# #     

# print( sweep_parameter["parameter_list"])
# for index, t in enumerate(sweep_parameter["parameter_list"]):
#     print('\x1b[6;30;42m' + f"{index}/{len(sweep_parameter['parameter_list'])} (expt {sweep_parameter['expt_number']})" + '\x1b[0m')
#     print("current parameter is: ", t)
#     os.system(f"python variance_analysis_TR_to_be_delete.py {sweep_parameter['name']} {t} \"{sweep_parameter['description']}\" {sweep_parameter['expt_number']}")
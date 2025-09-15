import os
import numpy as np

sweep_parameter = {
    "name": "protocolIndex",
    "expt_number": 2,
    "description": "I want to simulate the TR process for step 1.",
    "parameter_value": 1
    # "parameter_list": np.linspace(0.542, 0.544, 20)
}


N = 100

for _ in range(0, N):
    os.system(f"python TR_variance_analysis.py {sweep_parameter['name']} {sweep_parameter['parameter_value']} \"{sweep_parameter['description']}\" {sweep_parameter['expt_number']}")
# for _ in range(0, N):
#     


# print( sweep_parameter["parameter_list"])
# for index, t in enumerate(sweep_parameter["parameter_list"]):
#     print(f"{index}/{len(sweep_parameter['parameter_list'])} (expt {sweep_parameter['expt_number']})")
#     print("current parameter is: ", t)
#     os.system(f"python variance_analysis_TR_to_be_delete.py {sweep_parameter['name']} {t} \"{sweep_parameter['description']}\" {sweep_parameter['expt_number']}")
import os
import numpy as np

sweep_parameter = {
    "name": "t_5",
    "expt_number": 5,
    "description": "I want to the minimum work for the step 5 and run this for 100 times",
    "parameter_value": 60
    # "parameter_list": np.linspace(0.542, 0.544, 20)
}


N = 100

for _ in range(0, N):
    os.system(f"python WD_var_analysis_to_be_delete.py {sweep_parameter['name']} {sweep_parameter['parameter_value']} \"{sweep_parameter['description']}\" {sweep_parameter['expt_number']}")
# for _ in range(0, N):
#     


# print( sweep_parameter["parameter_list"])
# for index, t in enumerate(sweep_parameter["parameter_list"]):
#     print(f"{index}/{len(sweep_parameter['parameter_list'])} (expt {sweep_parameter['expt_number']})")
#     print("current parameter is: ", t)
#     os.system(f"python variance_analysis_TR_to_be_delete.py {sweep_parameter['name']} {t} \"{sweep_parameter['description']}\" {sweep_parameter['expt_number']}")
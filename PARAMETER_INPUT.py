import numpy as np

"""
parameter part
"""

pColor = {"00": "#061DF7", "01": "#FCEF51", "10": "#3FC7F2", "11": "#F187F4"}
mapping_state_1_to_state_2_dict_SWAP = {'00': ['01'], '01': ['00'], '10': ['10'], '11': ['11']}
mapping_state_1_to_state_2_dict_CE = {'00': ['00'], '01': ['01'], '10': ['00'], '11': ['11']}
mapping_state_1_to_state_2_dict_storage = {'00': ['00'], '01': ['01'], '10': ['10'], '11': ['11']}
mapping_state_1_to_state_2_erasure_flip = {'00': ['10'], '01': ['11'], '10': ['10'], '11': ['01']}



PHI_0 = 2.067833848 * 1e-15
k_B = 1.38e-23
T = 0.5
k_BT = k_B * T



I_p_1, I_p_2 = 2e-6 , 2e-6  # Amp
I_m_1, I_m_2 = 7e-9, 7e-9                                # Amp
R_1, R_2 = 371, 371                                # ohm
C_1, C_2 = 4e-9, 4e-9                              # F
L_1, L_2 = 1e-9, 1e-9                              # H

quick_doubler = lambda x1, x2: np.hstack([np.array([x1] * 2), np.array([x2]*2)])
I_p, I_m = quick_doubler(I_p_1, I_p_2), quick_doubler(I_m_1, I_m_2)

m_c = C_1
m_1 = C_1
m_2 = C_2
x_c = PHI_0 / (2 * np.pi)
time_scale_factor = 1
t_c = np.sqrt(L_1 * C_1)


U0_1 = m_c * x_c**2 / t_c**2
U0_2 = m_2 * x_c**2 / t_c**2
kappa_1, kappa_2, kappa_3, kappa_4 = k_BT/U0_1, k_BT/U0_1, k_BT/U0_1, k_BT/U0_1




lambda_1 = 2 * np.sqrt(L_1 * C_1) / (C_1 * R_1)
theta_1  = 1
eta_1    = np.sqrt(np.sqrt(L_1 * C_1)/ (R_1 * C_1)) * np.sqrt(2 * kappa_1 / 1**2)

lambda_2 = 2 * np.sqrt(L_1 * C_1) / (C_2 * R_2)
theta_2  = 1 / (C_2/C_1)
eta_2    = np.sqrt(np.sqrt(L_1 * C_1)/ (R_1 * C_1)) * np.sqrt(2 * kappa_2 * (R_1 * C_1**2) / (R_2 * C_2**2))

lambda_3 = 2 * np.sqrt(L_1 * C_1) / (C_1 * R_1)
theta_3  = 4
eta_3    = np.sqrt(np.sqrt(L_1 * C_1)/ (R_1 * C_1)) * np.sqrt(8 * kappa_3)

lambda_4 = 2 * np.sqrt(L_1 * C_1) / (C_2 * R_2)
theta_4  = 4 / (C_2/C_1)
eta_4    = np.sqrt(np.sqrt(L_1 * C_1)/ (R_1 * C_1)) * np.sqrt(8 * kappa_4 * (R_1 * C_1**2) / (R_2 * C_2**2))

gamma = 20


beta_1 = 2 * np.pi * L_1 * I_p_1 / PHI_0; 
beta_2 = 2 * np.pi * L_2 * I_p_2 / PHI_0;

d_beta_1 = 2 * np.pi * L_1 * I_m_1 / PHI_0; 
d_beta_2 = 2 * np.pi * L_2 * I_m_2 / PHI_0;


_lambda = np.array([lambda_1, lambda_2, lambda_3, lambda_4])
_theta  = np.array([theta_1, theta_2, theta_3, theta_4])
_eta  =   np.array([eta_1, eta_2, eta_3, eta_4])


protocol_key_dict = [
    (0, 'U0_1'), (1, 'U0_2'), 
    (2, 'gamma_1'), (3, 'gamma_2'), (4, 'beta_1'), (5, 'beta_2'), (6, 'd_beta_1'), (7, 'd_beta_2'), 
    (8, 'phi_1_x'), (9, 'phi_2_x'), (10, 'phi_1_dcx'), (11, 'phi_2_dcx'), (12, 'M_12'), (13, 'x_c')]

mapping_state_1_to_state_2_dict = {"00": ["00", "10"], "01": ["00", "10"], "10": ["01", "11"], "11": ["01", "11"]}

minimum_points_linear = [
  [2.68412197, 2.68412197], [ 2.68412197, -2.68412197], [-2.68412197,  2.68412197], [-2.68412197, -2.68412197],
    [2.68412197, -0.03718152], [-2.68412197, -0.03718152],
    [2.87338017, 2.10484416], [-2.87178059, -2.08655969],
    [2.94641907, 2.94641907], [-2.94641907, -2.94641907],
    [2.10484416, 2.87338017], [-2.08655969 -2.87178059],
    [-0.00302389,  2.68412197], [-0.00302389, -2.68412197]
]



minimum_points_non_linear = [
  [2.68412197, 2.68412197], [ 2.68412197, -2.68412197], [-2.68412197,  2.68412197], [-2.68412197, -2.68412197],
    [2.68412197, -0.03718152], [-2.68412197, -0.03718152], # mix in y
    [2.72231176, 1.89825524], [-2.72118417, -1.88970146], # conditional tilt in y
    [2.84468611, 2.84468611], [-2.84468611, -2.84468611], # raising barrier
    [1.89825524, 2.72231176], [-1.88970146, -2.72118417],  # conditional tilt in x
    [-0.00302389,  2.68412197], [-0.00302389, -2.68412197] # mix in x
]

min_U_at_each_non_linear_pot = np.array([-3.699849234102441, -1.4204442608247931, -1.9312984711892351, -6.595280780001745, -1.931298471189145, -1.4201112551718307, -3.699849234102441])








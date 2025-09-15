import edward_tools.Analysis_tool.minimum_value_of_potential as minimum_value_of_potential

import importlib
coupled_fq_potential_package = importlib.reload(coupled_fq_potential_package)
importlib.reload(minimum_value_of_potential)
time_array = np.linspace(0, protocol_time_array[-1], 100)


def potential_analysis(cfqr, t,  beta_1, beta_2, d_beta_1, d_beta_2, verbose = False)
    u1_1_array = []
    u2_1_array = []
    u3_1_array = []
    u4_1_array = []

    u1_2_array = []
    u2_2_array = []
    u3_2_array = []
    u4_2_array = []

    u5_array = []

    force_array = []
    minimum_value_data_points = []


    for t in time_array:
        minimum_value_data = minimum_value_of_potential.find_minimum_of_mix_in_Y(cfqr, t,  beta_1, beta_2, d_beta_1, d_beta_2, verbose = False)
        minimum_coordinate = minimum_value_data["coordinate"]
        minimum_value_data_points.append(minimum_value_data)
        phi_1, phi_2 = minimum_coordinate
        _params_at_t = cfqr.system.protocol.get_params(t)
        _phi_1dcx = _params_at_t[10]
        _phi_2dcx = _params_at_t[11]
        energy_break_down = coupled_flux_qubit_non_linear_approx_pot_break_down(phi_1, phi_2, _phi_1dcx, _phi_2dcx, _params_at_t)
        u1_1_array.append(energy_break_down["u1_1"])
        u2_1_array.append(energy_break_down["u2_1"])
        u3_1_array.append(energy_break_down["u3_1"])
        u4_1_array.append(energy_break_down["u4_1"])

        u1_2_array.append(energy_break_down["u1_2"])
        u2_2_array.append(energy_break_down["u2_2"])
        u3_2_array.append(energy_break_down["u3_2"])
        u4_2_array.append(energy_break_down["u4_2"])

        u5_array.append(energy_break_down["u5"])


        force_break_down = coupled_fq_potential_package.coupled_flux_qubit_non_linear_approx_force_break_down(phi_1, phi_2, _phi_1dcx, _phi_2dcx, _params_at_t)
        force_array.append(force_break_down)

    force_array = np.array(force_array)                                                 

    U_dp1_1,   U_dp1_2,   U_dp1_3,   U_dp1_4   = force_array[:, 0, 0], force_array[:, 0, 1], force_array[:, 0, 2], force_array[:, 0, 3]
    U_dp2_1,   U_dp2_2,   U_dp2_3,   U_dp2_4   = force_array[:, 1, 0], force_array[:, 1, 1], force_array[:, 1, 2], force_array[:, 1, 3]
    U_dp1dc_1,   U_dp1dc_2,   U_dp1dc_3,   U_dp1dc_4   = force_array[:, 2, 0], force_array[:, 2, 1], force_array[:, 2, 2], force_array[:, 2, 3]
    U_dp2dc_1,   U_dp2dc_2,   U_dp2dc_3,   U_dp2dc_4   = force_array[:, 3, 0], force_array[:, 3, 1], force_array[:, 3, 2], force_array[:, 3, 3]
    
    # draw force
    fig, ax = plt.subplots(2, 2, figsize = [10, 9])
    ax_flatten = ax.flatten()

    ax_flatten[0].plot(time_array, U_dp1_1, label = "U_dp1_1")
    ax_flatten[0].plot(time_array, U_dp1_2, label = "U_dp1_2")
    ax_flatten[0].plot(time_array, U_dp1_3, label = "U_dp1_3")
    ax_flatten[0].plot(time_array, U_dp1_4, label = "U_dp1_4")
    ax_flatten[0].legend()

    ax_flatten[1].plot(time_array, U_dp2_1, label = "U_dp2_1")
    ax_flatten[1].plot(time_array, U_dp2_2, label = "U_dp2_2")
    ax_flatten[1].plot(time_array, U_dp2_3, label = "U_dp2_3")
    ax_flatten[1].plot(time_array, U_dp2_4, label = "U_dp2_4")
    ax_flatten[1].legend()

    ax_flatten[2].plot(time_array, U_dp1dc_1, label = "U_dp1dc_1")
    ax_flatten[2].plot(time_array, U_dp1dc_2, label = "U_dp1dc_2")
    ax_flatten[2].plot(time_array, U_dp1dc_3, label = "U_dp1dc_3")
    ax_flatten[2].plot(time_array, U_dp1dc_4, label = "U_dp1dc_4")
    ax_flatten[2].legend()

    ax_flatten[3].plot(time_array, U_dp2dc_1, label = "U_dp2dc_1")
    ax_flatten[3].plot(time_array, U_dp2dc_2, label = "U_dp2dc_2")
    ax_flatten[3].plot(time_array, U_dp2dc_3, label = "U_dp2dc_3")
    ax_flatten[3].plot(time_array, U_dp2dc_4, label = "U_dp2dc_4")
    ax_flatten[3].legend()

    # draw potential
    fig, ax = plt.subplots(1, 3, figsize = [16, 4])
    ax[0].plot(time_array, np.array(u1_1_array), label = "u1_1")
    ax[0].plot(time_array, np.array(u2_1_array), label = "u2_1")
    ax[0].plot(time_array, np.array(u3_1_array), label = "u3_1")
    ax[0].plot(time_array, np.array(u4_1_array), label = "u4_1")
    ax[0].legend()

    ax[1].plot(time_array, np.array(u1_2_array), label = "u1_2")
    ax[1].plot(time_array, np.array(u2_2_array), label = "u2_2")
    ax[1].plot(time_array, np.array(u3_2_array), label = "u3_2")
    ax[1].plot(time_array, np.array(u4_2_array), label = "u4_2")
    ax[1].legend()

    ax[2].plot(time_array, np.array(u5_array), label = "u5")
    ax[2].legend()
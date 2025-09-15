
def comparing_potential_after_offset(energy_profile_array):
    fig, ax = plt.subplots(2, 3, figsize = [12, 8])
    ax_flatten = ax.flatten()

    current_step = 2
    ax_flatten[0].plot(energy_profile_array[current_step - 2][0], energy_profile_array[current_step - 2][1]-np.min(energy_profile_array[current_step - 2][1]), "--", label = r"4 well potential ($t_0$)")
    ax_flatten[0].plot(energy_profile_array[current_step][0], energy_profile_array[current_step][1]-np.min(energy_profile_array[current_step][1]), label = r"mix along y ($t_2$)")
    ax_flatten[0].legend()

    current_step = 3
    ax_flatten[1].plot(energy_profile_array[current_step - 1][0], energy_profile_array[current_step - 1][1]-np.min(energy_profile_array[current_step - 1][1]), "--", label = r"conditional tilt along $\varphi_2 (t_3)$")
    ax_flatten[1].plot(energy_profile_array[current_step][0], energy_profile_array[current_step][1]-np.min(energy_profile_array[current_step][1]), label = r"mix along y $(t_2)$")
    ax_flatten[1].legend()


    current_step = 4
    ax_flatten[2].plot(energy_profile_array[current_step - 1][0], energy_profile_array[current_step - 1][1]-np.min(energy_profile_array[current_step - 1][1]), "--", label = r"conditional tilt along $\varphi_2 (t_3)$")
    ax_flatten[2].plot(energy_profile_array[current_step][0], energy_profile_array[current_step][1]-np.min(energy_profile_array[current_step][1]), label = r"conditional tilt along $\varphi_1 (t_4)$")
    ax_flatten[2].legend()


    current_step = 5
    ax_flatten[3].plot(energy_profile_array[current_step - 1][0], energy_profile_array[current_step - 1][1]-np.min(energy_profile_array[current_step - 1][1]), "--", label = r"conditional tilt along $\varphi_1 (t_4)$")
    ax_flatten[3].plot(energy_profile_array[current_step][0], energy_profile_array[current_step][1]-np.min(energy_profile_array[current_step][1]), label = r"mix along $\varphi_1 (t_5)$")
    ax_flatten[3].legend()

    current_step = 7
    ax_flatten[4].plot(energy_profile_array[0][0], energy_profile_array[0][1]-np.min(energy_profile_array[0][1]), label = r"mix along $\varphi_1 (t_5)$")
    ax_flatten[4].plot(energy_profile_array[current_step - 2][0], energy_profile_array[current_step - 2][1]-np.min(energy_profile_array[current_step - 2][1]), "--", label = r"4 well potential $(t_7)$")
    ax_flatten[4].legend()
    
    
def drawPotentialTogether(cfqr, protocol_time_array, minimum_points, energy_profile_array = [])
    # 4 well
    contour_value = 40

    fig, ax = plt.subplots(1, 2, figsize=[9, 4])
    ax_flatten = ax.flatten()
    X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[0], cutlineDirection = "v")
    plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_points[0][0],\
                                contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = contour_value,
                                showGraph=True)
    energy_profile_array.append([plt_line_data_at_t['cutline_plot']['plotAxis'], plt_line_data_at_t['cutline_plot']['targetU']])
    
    
    fig, ax = plt.subplots(1, 2, figsize=[9, 4])
    ax_flatten = ax.flatten()
    X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[1], cutlineDirection = "v")
    plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_points[1][0],\
                                contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = contour_value, 
                                showGraph=True)
    energy_profile_array.append([plt_line_data_at_t['cutline_plot']['plotAxis'], plt_line_data_at_t['cutline_plot']['targetU']])

    
    
    fig, ax = plt.subplots(1, 2, figsize=[9, 4])
    ax_flatten = ax.flatten()
    X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[2], cutlineDirection = "v")
    plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_points[2][0],\
                                contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = contour_value,  
                                showGraph=True)
    energy_profile_array.append([plt_line_data_at_t['cutline_plot']['plotAxis'], plt_line_data_at_t['cutline_plot']['targetU']])
    print("min U: ", np.min(U))
    
    # find_minimum_of_raise_barrier
    fig, ax = plt.subplots(1, 2, figsize=[9, 4])
    ax_flatten = ax.flatten()
    X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[3], cutlineDirection = "v")
    plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_points[3][0],\
                                contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = contour_value,  
                                showGraph=True)
    energy_profile_array.append([plt_line_data_at_t['cutline_plot']['plotAxis'], plt_line_data_at_t['cutline_plot']['targetU']])
    print("min U: ", np.min(U))
    
    
    ### conditional tilt in x
    fig, ax = plt.subplots(1, 2, figsize=[9, 4])
    ax_flatten = ax.flatten()

    X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[4], cutlineDirection = "h")
    plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_points[4][1],\
                                 contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = contour_value,  
                                showGraph=True)
    energy_profile_array.append([plt_line_data_at_t['cutline_plot']['plotAxis'], plt_line_data_at_t['cutline_plot']['targetU']])
    print("min U: ", np.min(U))
    
    
    ### mix in x
    fig, ax = plt.subplots(1, 2, figsize=[9, 4])
    ax_flatten = ax.flatten()
    X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[5], cutlineDirection = "h")
    plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_points[5][1],\
                                 contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = contour_value,  
                                showGraph=True)
    energy_profile_array.append([plt_line_data_at_t['cutline_plot']['plotAxis'], plt_line_data_at_t['cutline_plot']['targetU']])
    print("min U: ", np.min(U))
    
    
def showPotentialChangePlots():
    """ This function shows how does a potential change from one form to another one with time using scatter plots"""
    
    # from t_1 to t_2
    min_U_along_cutline = []

    fig, ax = plt.subplots(1, 2, figsize=[15, 6])
    ax_flatten = ax.flatten()
    for _t in np.linspace(0, 40, 6):
        X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = _t, cutlineDirection = "v")
        plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = -2.54,\
                                    contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = 20,
                                    showGraph=True)
        ax_flatten[1].set_title("t = 0 and 240")
        min_U_along_cutline.append(min(plt_line_data_at_t['cutline_plot']['targetU']))
        
    
    # from t_2 to t_3
    min_U_along_cutline = []
    fig, ax = plt.subplots(1, 2, figsize=[15, 6])
    ax_flatten = ax.flatten()
    for _t in np.linspace(40, 80, 6):
        X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = _t, cutlineDirection = "v")
        U = U - np.min(U)
        plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = -2.54,\
                                    contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = 20,
                                    showGraph=True)
        ax_flatten[1].set_title("t = 0 and 240")
        min_U_along_cutline.append(min(plt_line_data_at_t['cutline_plot']['targetU']))

    # from t_3 to t_4
    min_U_along_cutline = []
    fig, ax = plt.subplots(1, 2, figsize=[15, 6])
    ax_flatten = ax.flatten()
    for _t in np.linspace(80, 120, 6):
        X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = _t, cutlineDirection = "v")
        plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = -2.54,\
                                    contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = 20,
                                     showGraph=True)
        ax_flatten[1].set_title("t = 0 and 240")
        min_U_along_cutline.append(min(plt_line_data_at_t['cutline_plot']['targetU']))
        
        
    # from t_4 to t_5
    min_U_along_cutline = []
    fig, ax = plt.subplots(1, 2, figsize=[15, 6])
    ax_flatten = ax.flatten()
    for _t in np.linspace(120, 160, 6):
        X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = _t, cutlineDirection = "v")
        plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = -2.54,\
                                    contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = 20,
                                     showGraph=True)
        ax_flatten[1].set_title("t = 0 and 240")
        min_U_along_cutline.append(min(plt_line_data_at_t['cutline_plot']['targetU']))
        

def formSymmetricalWorkOffsetGraph():
    """
    The purpose of this function is to form a graph is to trace the value of minimum potential as time, and graph it as energy vs time. I want this plot because it is hard to find out the minimum and then offset the value of potential through the main loop. It is easier to do afterward editing. 
    However, one problem was that this function is trapped by a saddle point and cannot leave the saddle point. I have no good solution for this problem now. This function is just a copy from the notebook.
    """
    for i, x in enumerate(range(0, len(work_mean))):
        trial_solution = find_minimum_for_all_potential(cfqr, x * params['dt'], beta_1, beta_2, d_beta_1, d_beta_2, 
                                   verbose = False, guess = [minimum_sol_set[i][0]])
        minimum_sol_set.append(trial_solution)
        
    min_energy = [d[1] for d in minimum_sol_set[1:]]
    left_half_min_energy = min_energy[:len(min_energy)//2]
    right_half_min_energy = left_half_min_energy[::-1]

    combined_min_energy = np.concatenate([left_half_min_energy, right_half_min_energy])
    
    np.cumsum((work_mean[1:] - work_mean[:-1]) - (combined_min_energy[1:] - combined_min_energy[:-1]))
    plt.plot(combined_min_energy)
    
    
    
def show_graph_of_no_offset_potential():
    """
    It is a function to check the potential shape of the system. It is used before the actual simulation happens.
    """
    min_U_along_cutline = []
    plt_line_data_at_t_array = []

    fig, ax = plt.subplots(3, 4, figsize=[15, 11])
    ax_flatten = ax.flatten()
    contour_value = 40

    # 4 well
    X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[0], cutlineDirection = "v")
    plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_points[0][0],\
                                contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = contour_value,
                                showGraph=True)
    min_U_along_cutline.append(min(plt_line_data_at_t['cutline_plot']['targetU']))
    plt_line_data_at_t_array.append(plt_line_data_at_t['cutline_plot'])

    # mix in y
    X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[1], cutlineDirection = "v")
    plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_points[1][0],\
                                contour_plt=ax_flatten[2], cutline_plt=ax_flatten[3], contours = contour_value, 
                                showGraph=True)
    min_U_along_cutline.append(min(plt_line_data_at_t['cutline_plot']['targetU']))
    plt_line_data_at_t_array.append(plt_line_data_at_t['cutline_plot'])

    # conditional tilt in y
    X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[2], cutlineDirection = "v")
    plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_points[2][0],\
                                contour_plt=ax_flatten[4], cutline_plt=ax_flatten[5], contours = contour_value,  
                                showGraph=True)
    min_U_along_cutline.append(min(plt_line_data_at_t['cutline_plot']['targetU']))
    plt_line_data_at_t_array.append(plt_line_data_at_t['cutline_plot'])

    # raise barrier
    # X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[3], cutlineDirection = "v")
    # plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = -2.62524709,\
    #                             contour_plt=ax_flatten[6], cutline_plt=ax_flatten[7], contours = contour_value,  
    #                             showGraph=True)
    # min_U_along_cutline.append(min(plt_line_data_at_t['cutline_plot']['targetU']))
    # plt_line_data_at_t_array.append(plt_line_data_at_t['cutline_plot'])

    # conditional tilt in x
    X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[3], cutlineDirection = "h")
    plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_points[3][1],\
                                 contour_plt=ax_flatten[8], cutline_plt=ax_flatten[9], contours = contour_value,  
                                showGraph=True)
    min_U_along_cutline.append(min(plt_line_data_at_t['cutline_plot']['targetU']))
    plt_line_data_at_t_array.append(plt_line_data_at_t['cutline_plot'])


    # mix in x
    X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[4], cutlineDirection = "h")
    plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_points[4][1],\
                                 contour_plt=ax_flatten[10], cutline_plt=ax_flatten[11], contours = contour_value,  
                                showGraph=True)
    min_U_along_cutline.append(min(plt_line_data_at_t['cutline_plot']['targetU']))
    plt_line_data_at_t_array.append(plt_line_data_at_t['cutline_plot'])

    # X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[6], cutlineDirection = "h")
    # plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = +2.68412197,\
                                 # contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = 20,
    #                             showGraph=True)
    # min_U_along_cutline.append(min(plt_line_data_at_t['cutline_plot']['targetU']))
    # plt_line_data_at_t_array.append(plt_line_data_at_t['cutline_plot'])

    
def findPotential(cfqr, _t, state):
    """to trace the potential of different categories of a particle"""
    _params_at_t = cfqr.system.protocol.get_params(_t)
    def cfq_potential(state):
        phi_1, phi_2, phi_1dc, phi_2dc = state
        return coupled_flux_qubit_non_linear_approx_pot_break_down(phi_1, phi_2, phi_1dc, phi_2dc, _params_at_t)
    return np.array(list(cfq_potential(state).values()))
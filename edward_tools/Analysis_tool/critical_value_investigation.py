def graphForCriticalValueOfPhi_xdc(parameterArray):
    XYU_array = []
    
    final_t = 40
    for p_array in parameterArray:   
        print(p_array)
        protocol_list = [
            {"duration":final_t, "phi_1xdcx": p_array[0], "phi_2xdc": p_array[1],  "mu_12": p_array[2],  "name":"(1) mix in y direction"},
        ]
        computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, protocol_list)
        storage_protocol, comp_protocol = create_system(computation_protocol_parameter_dict)
        cfqr = cfq_runner.coupledFluxQubitRunner(potential = coupled_fq_pot, params = params, storage_protocol= storage_protocol, computation_protocol= comp_protocol)
        cfqr.initialize_sim(); 
        cfqr.set_sim_attributes()
        init_state_saved = cfqr.init_state
        
        
        fig, ax = plt.subplots(1, 2, figsize = [12, 6])
        X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = final_t, cutlineDirection = "v")
        plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = -2.68412197,\
                                    contour_plt=ax[0], cutline_plt=ax[1], contours = 20,
                                    showGraph=True)

        XYU_array.insert(0, plt_line_data_at_t)
        
    return XYU_array

"""
Example usage
XYU_array = graphForCriticalValueOfPhi_xdc([0, 0, 0], [0, 1, 0])
combineCriticalValueGraphs(XYU_array)

"""
def combineCriticalValueGraphs(XYU_array):
    fig, ax = plt.subplots(1, 2, figsize = [14, 6])
    for index, item in enumerate(XYU_array):

    X, Y, U = item["contour_plot"]["X"], item["contour_plot"]["Y"], item["contour_plot"]["U"]
    contours, time = item["contour_plot"]["contours"], item["contour_plot"]["time"]
    _plotAxis = item["cutline_plot"]["plotAxis"]
    _targetU = item["cutline_plot"]["targetU"]
    ax[0].set_box_aspect(1)
    ax[1].set_box_aspect(1)

    ax[0].set_xlabel(r"$\varphi_{1}$", fontsize = 20)
    ax[0].set_ylabel(r"$\varphi_{2}$", fontsize = 20)
    ax[0].xaxis.set_tick_params(labelsize=15)
    ax[0].yaxis.set_tick_params(labelsize=15)

    
    ax[1].set_ylabel("potential energy", fontsize = 18)
    ax[1].set_xlabel(r"$\varphi_{2}$", fontsize = 20)
    ax[1].xaxis.set_tick_params(labelsize=15)
    ax[1].yaxis.set_tick_params(labelsize=15)

    
    ax[0].contourf(X, Y, U, contours)
    ax[0].vlines(x = -2.6, ymin = -5, ymax = 5, colors= "red")
    ax[1].plot(_plotAxis, _targetU, label = r"$\varphi_{2dcx}$ = " + f"{phi_1_dcx_on_array[index]}")
    # ax[1].legend(fontsize= 15)
#     ax[1].set_title(, fontsize = 15)
# "contour_plot": {"X": X, "Y": Y, "U": U, "contours": contours, "time": time},
#         "cutline_plot": {"plotAxis": plotAxis, "targetU": targetU, "time": time, "cutlineDirection": cutlineDirection, "cutlineValue": cutlineValue}
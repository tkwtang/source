def graphForCriticalValueOfPhi_xdc(parameterArray):
    """
    This is for graphing the cutline graph at some critical phi_xdc
    """
    XYU_array = []
    
    final_t = 40
    for p_array in parameterArray:   
        print(p_array)
        protocol_list = [
            {"duration":final_t, "phi_1_dcx": p_array[0], "phi_2_dcx": p_array[1],  "M_12": p_array[2],  "name":"(1) mix in y direction"},
        ]
        computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, protocol_list)
        storage_protocol, comp_protocol = create_system(computation_protocol_parameter_dict)
        cfqr = cfq_runner.coupledFluxQubitRunner(potential = coupled_fq_pot, params = params, storage_protocol= storage_protocol, computation_protocol= comp_protocol)
        cfqr.initialize_sim(); 
        cfqr.set_sim_attributes()
        init_state_saved = cfqr.init_state
        
        
        fig, ax = plt.subplots(1, 2, figsize = [12, 6])
        X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = final_t, cutlineDirection = "v")
        plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = -1.96436202,\
                                    contour_plt=ax[0], cutline_plt=ax[1], contours = 20,
                                    showGraph=True)

        XYU_array.insert(0, plt_line_data_at_t)
        
    return XYU_array
import importlib, os, datetime, sys
source_path = os.path.expanduser('~/Project/source/')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from IPython import display
from IPython.display import HTML
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from sus.protocol_designer.protocol import sequential_protocol

from quick_sim import setup_sim
import kyle_tools as kt


from edward_tools import coupled_fq_protocol_library, cfq_runner
import edward_tools.cfq_batch_sweep as cfq_batch_sweep
from edward_tools.couple_flux_qubit_metrics import fidelityEvaluation
from edward_tools.coupled_fq_potential import coupled_flux_qubit_non_linear_approx_pot, coupled_flux_qubit_non_linear_approx_force, coupled_flux_qubit_non_linear_approx_pot_break_down
import edward_tools.Analysis_tool.general_analysis_tools as general_analysis_tool
import edward_tools.Analysis_tool.minimum_value_of_potential as minimum_value_of_potential
from edward_tools.visualization import animate_sim_flux_qubit




coupled_fq_protocol_library = importlib.reload(coupled_fq_protocol_library)
create_system = coupled_fq_protocol_library.create_system
get_potential_shot_at_different_t = coupled_fq_protocol_library.get_potential_shot_at_different_t
get_potential_shot_at_different_t_1D = coupled_fq_protocol_library.get_potential_shot_at_different_t_1D
create_simple_protocol_parameter_dict = coupled_fq_protocol_library.create_simple_protocol_parameter_dict
coupled_fq_runner = importlib.reload(cfq_runner)
coupled_fq_protocol_library = importlib.reload(coupled_fq_protocol_library)
create_system = coupled_fq_protocol_library.create_system
get_potential_along_a_1D_cutline = coupled_fq_protocol_library.get_potential_along_a_1D_cutline
plotCutlines = coupled_fq_protocol_library.plotCutlines

contourData = {
    "vmax": 0, "vmin": 500,
    "manual_domain": [np.array([-5, -5]), np.array([5, 5])],
    "contour_range": [0, 400],
    "title": ""
}

cutlineInformation = {
    "cutlineList": [(-2, "v","red")],
    "cutlineGeneratingFunction": None,
    "cutlineXLimit": [-4,4],
    "cutlineYLimit": [50, 400]
}

particleInformation = {
    "showParticles": False,
    "project_item": ["00", "01"],
    "particle_opacity": 0.5,
    "pColor": {"00": "#061DF7", "01": "red", "10": "#3FC7F2", "11": "#F187F4"}
}

animation_setting = {
    "frame_skip": 10,
    "save_path": None, 
    "save_dict": None,
    "interval": 100,
    "blit": False
}

protocol_graph_setting = {
    "key": ["phi_1_x", "phi_2_x", "phi_1_dcx", "phi_2_dcx", "M_12"]

}

potential_setting = {
    "pot": coupled_flux_qubit_non_linear_approx_pot,
    "force": coupled_flux_qubit_non_linear_approx_force,
    "number_of_params": 14,
    "dimension": 4,
    "manual_domain": [np.array([-5, -5]), np.array([5, 5])]
}

class CFQR_MANAGER():
    def __init__(self, cfqr_data_list, potential_setting = potential_setting):
        self.cfqr_list = []
        for cfqr_data in cfqr_data_list:
            _cfqr = self.createCFQR(cfqr_data = cfqr_data, potential_setting = potential_setting)
            self.cfqr_list.append(_cfqr)

    def add_cfqr(self, cfqr_list = []):
        [self.cfqr_list.append(_cfqr) for _cfqr in cfqr_list]
        
        
    def createCFQR(self, cfqr_data, potential_setting = potential_setting, default_init_state = "default_init_state.npy", **kwargs):
        cfqr_params = cfqr_data['cfqr_params']
        params = cfqr_data["params"]
        protocol_list = cfqr_data["protocol_list"]
        initial_external_parameter_dict = cfqr_data['initial_external_parameter_dict']

        C_factor, L_factor, R_factor = cfqr_params['C_factor'], cfqr_params['L_factor'], cfqr_params['R_factor']
        I_m_factor = cfqr_params['I_m_factor']
        T = cfqr_params['T']
        I_p_1, I_p_2 = cfqr_params['I_p_1'], cfqr_params['I_p_2']
        I_m_1, I_m_2 = cfqr_params['I_m_1'], cfqr_params['I_m_2']
        R_1, R_2 = cfqr_params["R_1"] * R_factor, cfqr_params["R_2"] * R_factor
        C_1, C_2 = cfqr_params["C_1"] * C_factor, cfqr_params["C_2"] * C_factor
        L_1, L_2 = cfqr_params["L_1"] * L_factor, cfqr_params["L_2"] * L_factor 
        phi_1_x_on, phi_2_x_on = cfqr_params["phi_1_x_on"], cfqr_params["phi_2_x_on"] 
        phi_1_dcx_on, phi_2_dcx_on = cfqr_params["phi_1_dcx_on"], cfqr_params["phi_2_dcx_on"] 
        M_12_on =  cfqr_params["M_12_on"]
        gamma = cfqr_params["gamma"]

        
        has_velocity = True
        PHI_0 = 2.067833848 * 1e-15
        k_B = 1.38e-23
        k_BT = k_B * T
        freq = 1/np.sqrt(C_1 * L_1)
        characteristic_time = np.sqrt(C_1 * C_factor * L_1 * L_factor)

        m_c, m_1, m_2 = C_1, C_1, C_2
        x_c = PHI_0 / (2 * np.pi)
        time_scale_factor = 1
        t_c = np.sqrt(L_1 * C_1)
        v_c = x_c / t_c


        U0_1 = m_c * x_c**2 / t_c**2 / k_BT
        U0_2 = m_2 * x_c**2 / t_c**2 / k_BT
        kappa_1, kappa_2, kappa_3, kappa_4 = 1/U0_1, 1/U0_1, 1/U0_1, 1/U0_1

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

        gamma = 10


        beta_1 = 2 * np.pi * L_1 * I_p_1 / PHI_0; 
        beta_2 = 2 * np.pi * L_2 * I_p_2 / PHI_0;

        beta_1 = 2.3
        beta_2 = 2.3

        d_beta_1 = 2 * np.pi * L_1 * I_m_1 / PHI_0; 
        d_beta_2 = 2 * np.pi * L_2 * I_m_2 / PHI_0;


        _damping_factor = 1
        _lambda = np.array([lambda_1, lambda_2, lambda_3, lambda_4])
        _theta  = np.array([theta_1, theta_2, theta_3, theta_4])
        _eta  =   np.array([eta_1, eta_2, eta_3, eta_4])

        v_1 = np.random.normal(0, np.sqrt(k_BT/m_1)) / v_c
        v_2 = np.random.normal(0, np.sqrt(k_BT/m_2)) / v_c
        v_3 = np.random.normal(0, np.sqrt(k_BT/(m_1/4))) / v_c
        v_4 = np.random.normal(0, np.sqrt(k_BT/(m_2/4))) / v_c

        """
        # step 1: Define potential
        
        """
        
        coupled_fq_default_param = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x_c]
        [phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound] = np.array([4, 4, 4, 4])/time_scale_factor

        coupled_fq_domain = [[-phi_1_bound, -phi_2_bound, -phi_1dc_bound, -phi_2dc_bound], \
                             [phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound]]

        
        coupled_fq_pot = Potential(potential_setting['pot'], potential_setting['force'], potential_setting['number_of_params'], potential_setting['dimension'] , default_params = coupled_fq_default_param,  relevant_domain = coupled_fq_domain)

        """
        # step 2: Define initial condition and protocol
        """
        manual_domain=[np.array([-5, -5]), np.array([5, 5])]
        phi_1_dcx , phi_2_dcx = 0, 0
        phi_1_dc, phi_2_dc = phi_1_dcx, phi_2_dcx

        phi_1_x_off, phi_2_x_off = 0, 0
        phi_1_dcx_off, phi_2_dcx_off, M_12_off = 0, 0, 0

        zeroDissipation = False
        params['sim_params'] = [_lambda, _theta, _eta]

        if zeroDissipation:
            params['sim_params'] = [_lambda * 0, _theta, _eta * 0]

        params['circuit_parameter'] = {
            "C_factor":C_factor, "L_factor": L_factor, "I_m_factor": I_m_factor, "T": T, 
            "I_p_1": I_p_1, "I_p_2": I_p_2, "I_m_1": I_m_1, "I_m_2": I_m_2,
            "R_1": R_1, "R_2": R_2, "C_1": C_1, "C_2": C_2, "L_1": L_1, "L_2": L_2, 
            "phi_1_x_on": phi_1_x_on, "phi_2_x_on": phi_1_x_on,
            "phi_1_dcx_on": phi_1_dcx_on, "phi_2_dcx_on": phi_2_dcx_on, "M_12_on": M_12_on,
            "gamma": gamma
        }
        
        
        params['capacitance'] = [C_1, C_2, C_1/4, C_2/4]
        params['mass_special'] = [1, 1, 1/4, 1/4]
        params['v_c'] = x_c/t_c
        params['k_BT'] = k_BT
        params['U0'] = U0_1
    
    
        protocol_time_array = [item["duration"] for item in protocol_list]
        protocol_time_array.insert(0, 0)
        protocol_time_array = np.cumsum(protocol_time_array)
        protocol_time_index_array = protocol_time_array / params['dt'] - 1
        protocol_time_index_array[0] = 0
        protocol_time_index_array = protocol_time_index_array.astype(int)


        """
        # step 3: create the relevant storage protocol and computation protocol
        """
        
        initial_parameter_dict = {
            "U0_1": U0_1,     "U0_2": U0_2,     "gamma_1": gamma,  "gamma_2": gamma,
            "beta_1": beta_1,   "beta_2": beta_2,   "d_beta_1": d_beta_1 ,   "d_beta_2": d_beta_2, 'x_c': x_c,
            "phi_1_x": initial_external_parameter_dict["phi_1_x"], 
            "phi_2_x": initial_external_parameter_dict["phi_2_x"],  
            "phi_1_dcx": initial_external_parameter_dict["phi_1_dcx"],  
            "phi_2_dcx": initial_external_parameter_dict["phi_2_dcx"],
            "M_12": initial_external_parameter_dict["M_12"]
        }
        
        
        
        computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, \
                                                                            protocol_list)
        storage_protocol, comp_protocol = create_system(computation_protocol_parameter_dict)

        """
        # step 4: create the coupled_fq_runner
        """
        init_state_saved = np.load(default_init_state)
        cfqr = cfq_runner.coupledFluxQubitRunner(potential = coupled_fq_pot, params = params, protocol_list = protocol_list,\
                                                        storage_protocol= storage_protocol, \
                                                        computation_protocol= comp_protocol, measure_all_states=True)
        cfqr.initialize_sim()
        cfqr.set_sim_attributes(init_state=init_state_saved)
        return cfqr
    
    
    def show_time_evolution_of_protocol(self, time_array_step = 5, cfqr_index = 0, contourData = contourData, 
                                        cutlineInformation = cutlineInformation, fig_ax = None, offset_potential = False):
        if not fig_ax:
            fig, ax = plt.subplots(1, 2, figsize=[10, 5])

        ax_flatten = ax.flatten()
        
        _value, _direction, _color = cutlineInformation['cutlineList'][0]
        cfqr_chosen = self.cfqr_list[cfqr_index]
        time_array = np.linspace(cfqr_chosen.protocol_time_array[0], cfqr_chosen.protocol_time_array[-1], time_array_step)
        
        for _i, _t in enumerate(time_array):
            X_1, Y_1, U_1, cutlineDirection_1, _targetAxis_1, _plotAxis_1 = get_potential_along_a_1D_cutline(cfqr_chosen, t = _t, cutlineDirection = _direction)
            plt_line_data_at_t_1 = plotCutlines(X_1, Y_1, U_1, cutlineDirection_1, cutlineValue = _value, 
                                cutline_label = _t, contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], 
                                contours = 20, showGraph=True, cutlineColor = plt.cm.tab10.colors[_i], x_range = [-4, 4], 
                                cutlineYLimit = cutlineInformation['cutlineYLimit'], offset_potential = offset_potential, 
                                contour_range = contourData['contour_range'], time = _t)
        ax[1].legend()
    
    
    def compare_potential_at_a_moment(self, _t = 100, legend=True, 
        plot_axis = [0, 1], slice_values = None, fig_ax=None, 
        contourData = contourData, cutlineInformation = cutlineInformation, particleInformation = particleInformation,
        animation_setting = animation_setting, protocol_graph_setting = protocol_graph_setting, 
        ax0_title = None, offset_potential = False, **pot_kwargs):
        """
            Compare the potential of two protocol at the same time
        """
        if not fig_ax:
            fig, ax = plt.subplots(1, 2, figsize=[10, 5])
        ax_flatten = ax.flatten()
        
        _value, _direction, _color = cutlineInformation['cutlineList'][0]
        cfqr_list = self.cfqr_list
        cfqr_1 = cfqr_list[0]
        cfqr_2 = cfqr_list[1]
        
        
        X_1, Y_1, U_1, cutlineDirection_1, _targetAxis_1, _plotAxis_1 = get_potential_along_a_1D_cutline(cfqr_1, t = _t, cutlineDirection = _direction)
        X_2, Y_2, U_2, cutlineDirection_2, _targetAxis_2, _plotAxis_2 = get_potential_along_a_1D_cutline(cfqr_2, t = _t, cutlineDirection = _direction)
        
        plt_line_data_at_t_1 = plotCutlines(X_1, Y_1, U_1, cutlineDirection_1, cutlineValue = _value, cutline_label = "protocol_1",
                                contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = 20, 
                                showGraph=True, cutlineColor = "red", x_range = [-4, 4], 
                                cutlineYLimit = cutlineInformation['cutlineYLimit'], offset_potential = offset_potential, 
                                contour_range = contourData['contour_range'], time = _t)
        plt_line_data_at_t_2 = plotCutlines(X_2, Y_2, U_2, cutlineDirection_2, cutlineValue = _value, cutline_label = "protocol_2",
                                contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = 20, 
                                showGraph=True, cutlineColor = "blue", x_range = [-4, 4], 
                                cutlineYLimit = cutlineInformation['cutlineYLimit'], offset_potential = offset_potential, 
                                contour_range = contourData['contour_range'], time = _t)
    
    
    
    
    def compare_animation_of_potential_with_cutline_and_projection(self, legend=True, 
        plot_axis = [0, 1], slice_values = None, fig_ax=None, 
        contourData = contourData, cutlineInformation = cutlineInformation, particleInformation = particleInformation,
        animation_setting = animation_setting, protocol_graph_setting = protocol_graph_setting, 
        ax0_title = None, offset_potential = False, **pot_kwargs):
        """
        Animate the two protocol at the same time
        """

        names = [r"$\varphi_1$", r"$\varphi_2$", r"$\varphi_{1dc}$", r"$\varphi_{2dc}$"]
        plot_axis = [0, 1]

        if not fig_ax:
            fig, ax = plt.subplots(1, 2, figsize=[10, 5])
        else:
            fig, ax = fig_ax
        ax_flatten = ax.flatten()

        # things in common
        cfqr_list = self.cfqr_list
        cfqr_1 = cfqr_list[0]
        cfqr_2 = cfqr_list[1]
        index_skip_in_all_time_array = int(animation_setting['frame_skip']/cfqr_1.params['dt'])
        time_array = cfqr_list[0].protocol_all_time_array[::index_skip_in_all_time_array]

        # things in difference
        parms_at_init_1 = cfqr_1.protocol.get_params(0)
        parms_at_init_2 = cfqr_2.protocol.get_params(0)


        if particleInformation['showParticles']:
            all_state = cfqr.sim.output.all_state['states']
            N, nsteps, N_dim = np.shape(all_state)[0], np.shape(all_state)[1], np.shape(all_state)[2]
            state_lookup = separate_by_state(all_state[:, :, (0, 1), :])

            phi_1 = all_state[:,::animation_setting['frame_skip'], 0, 0]
            phi_2 = all_state[:,::animation_setting['frame_skip'], 1, 0]
            phi_1dc = all_state[:,::animation_setting['frame_skip'], 2, 0]
            phi_2dc = all_state[:,::animation_setting['frame_skip'], 3, 0]
            particle_potential = cfqr.system.potential.potential(phi_1, phi_2, phi_1dc, phi_2dc, parms_at_init)

            phi_1_lim = (np.min(phi_1), np.max(phi_1))
            phi_2_lim = (np.min(phi_2), np.max(phi_2))
            scat_kwargs = {'alpha': particleInformation["particle_opacity"], 'zorder':10}
            scat = [ax_flatten[0].scatter(phi_1[state_lookup[key], 0], phi_2[state_lookup[key], 0], c = particleInformation['pColor'][key], **scat_kwargs) for key in state_lookup]
        else:
            phi_1_lim = (-5, 5)
            phi_2_lim = (-5, 5)

        _value, _direction, _color = cutlineInformation['cutlineList'][0]



        X_1, Y_1, U_1, cutlineDirection_1, _targetAxis_1, _plotAxis_1 = get_potential_along_a_1D_cutline(cfqr_1, t = time_array[0], cutlineDirection = _direction)
        ax_flatten[0].set(xlim=phi_1_lim, ylim=phi_2_lim, xlabel=names[plot_axis[0]], ylabel=names[plot_axis[1]])
        ax_flatten[0].set_aspect(1)




        def animate(i):
            parms_at_i_1 = cfqr_1.protocol.get_params(time_array[i])
            parms_at_i_2 = cfqr_2.protocol.get_params(time_array[i])
            ax_flatten[1].clear()

            if particleInformation['showParticles']:
                x_i = phi_1[:, i]
                y_i = phi_2[:, i]
                particle_potential_i = cfqr.system.potential.potential(phi_1[:, i], phi_2[:, i], phi_1dc[:, i], phi_2dc[:, i], parms_at_i)

                scat_2 = []
                for key in state_lookup:
                    if key in particleInformation['project_item']:
                        _scatter = ax_flatten[1].scatter(phi_1[state_lookup[key], 0], particle_potential[state_lookup[key], 0], c = particleInformation['pColor'][key], **scat_kwargs)

                    else:
                        _scatter = None
                    scat_2.append(_scatter)

            for item in cutlineInformation['cutlineList']:
                _value, _direction, _color = item

                X_1, Y_1, U_1, cutlineDirection_1, _targetAxis_1, _plotAxis_1 = get_potential_along_a_1D_cutline(cfqr_1, t = time_array[i], cutlineDirection = _direction)
                X_2, Y_2, U_2, cutlineDirection_2, _targetAxis_2, _plotAxis_2 = get_potential_along_a_1D_cutline(cfqr_2, t = time_array[i], cutlineDirection = _direction)
                plt_line_data_at_t_1 = plotCutlines(X_1, Y_1, U_1, cutlineDirection_1, cutlineValue = _value, cutline_label = "protocol_1",\
                                        contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = 20, 
                                        showGraph=True, cutlineColor = _color, x_range = [-4, 4], 
                                        cutlineYLimit = cutlineInformation['cutlineYLimit'], offset_potential = offset_potential, 
                                        contour_range = contourData['contour_range'], time = time_array[i])
                plt_line_data_at_t_2 = plotCutlines(X_2, Y_2, U_2, cutlineDirection_2, cutlineValue = _value,\
                                        contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = 20, cutline_label = "protocol_2", 
                                        showGraph=True, cutlineColor = "blue", x_range = [-4, 4], 
                                        cutlineYLimit = cutlineInformation['cutlineYLimit'], offset_potential = offset_potential, 
                                        contour_range = contourData['contour_range'], time = time_array[i])




            params_at_t_i_1 = cfqr_1.protocol.get_params(time_array[i])

            if particleInformation['showParticles']:
                for _i, item in enumerate(state_lookup):
                    scat[_i].set_offsets(np.c_[x_i[state_lookup[item]], y_i[state_lookup[item]]])

                    if item in particleInformation['project_item'] and scat_2[_i]:
                        scat_2[_i].set_offsets(np.c_[y_i[state_lookup[item]], particle_potential_i[state_lookup[item]]])
        ax_flatten[1].legend()
        ani = FuncAnimation(fig, animate, interval=animation_setting['interval'], frames=len(time_array), blit=animation_setting['blit'])
        return ani, fig, ax

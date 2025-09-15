import sys
import os
# source_path = os.path.expanduser('~/Project/source/')
# sys.path.append(source_path)
# sys.path.append(os.path.expanduser('~/Project/source/simtools/'))

import numpy as np
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from kyle_tools.multisim import SimManager, FillSpace
from SimRunner import SaveParams, SaveSimOutput, SaveFinalWork
from infoenginessims.simprocedures import basic_simprocedures as sp
from infoenginessims.simprocedures import running_measurements as rp
from infoenginessims.simprocedures import trajectory_measurements as tp
from infoenginessims.simprocedures.basic_simprocedures import ReturnFinalState, MeasureWorkDone, MeasureStepValue
from scipy.optimize import fsolve
from IPython.display import Image
from edward_tools import coupled_fq_protocol_library
from quick_sim import setup_sim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .initial_state_sampling import self_defined_initial_state

from edward_tools import visualization
from IPython import display
from scipy import optimize

default_params_dict = {}

pColor = {"00": "#061DF7", "01": "#FCEF51", "10": "#3FC7F2", "11": "#F187F4"}

class coupledFluxQubitRunner(SimManager):
    def __init__(self, potential = None, name_func = [None, None], params = default_params_dict, potential_default_param = None, storage_protocol = None, computation_protocol = None , measure_all_states = False, protocol_list = None, has_velocity = True):
        """
        params: parameters for the simulation such as time, lambda, theta and eta
        override_potential_parameter: to override the default parameter for the potential
        """
        self.potential = potential
        self.params = params
        self.save_name = name_func
        self.has_velocity = has_velocity
        self.override_potential_parameter = potential_default_param
        self.storage_protocol = storage_protocol
        self.computation_protocol = computation_protocol
        self.save_procs =  [SaveParams(), SaveSimOutput(), SaveFinalWork()]
        self.sampleSize = round(self.params['N'] * params['percentage'])
        self.as_step = params['as_step']
        self.measure_all_state = measure_all_states
        self.protocol_list = protocol_list
        self.pColor = {"00": "#061DF7", "01": "#FCEF51", "10": "#3FC7F2", "11": "#F187F4"}
        self.protocol_key = ['U0_1', 'U0_2', 'gamma_1', 'gamma_2', 'beta_1', 'beta_2', 'd_beta_1', 'd_beta_2', 'phi_1x', 'phi_2x', 'phi_1xdc', 'phi_2xdc', 'mu_12', 'x_c']
        
        


    def verify_param(self, key, val):
        return True

    def initialize_sim(self):
        self.potential.default_params = self.override_potential_parameter
        self.eq_protocol = self.storage_protocol or  self.potential.trivial_protocol().copy()

        self.potential.default_params = np.array(self.override_potential_parameter)
        self.protocol = self.computation_protocol or self.potential.trivial_protocol().copy()
        # print(f"from fq_runner.py: system.protocol.t_i = {self.protocol.t_i}, system.protocol.t_f = {self.protocol.t_f}")

        self.eq_system = System(self.eq_protocol, self.potential)
        self.eq_system.axes_label = ["phi_1", "phi_2", "phi_1dc", "phi_2dc"]
        self.eq_system.has_velocity = self.has_velocity
        self.eq_system.capacitance = self.params['capacitance']
        self.eq_system.mass = self.params['mass']
        self.eq_system.v_c = self.params['v_c']
        self.eq_system.k_BT = self.params['k_BT']
        self.eq_system.U0 = self.params['U0']

        self.system = System(self.protocol, self.potential)
        self.system.axes_label = ["phi_1", "phi_2", "phi_1dc", "phi_2dc"]
        self.system.has_velocity = self.has_velocity
        self.system.capacitance = self.params['capacitance']
        self.system.mass = self.params['mass']
        self.system.v_c = self.params['v_c']
        self.system.k_BT = self.params['k_BT']
        self.system.U0 = self.params['U0']
        
        self.createProtocolTimeArray(self.protocol_list, self.params)
        
        
    def set_sim_attributes(self, init_state = None, manual_domain = None, axes = None, percentage = 1, as_step = np.s_[::], verbose = True, extra_constraint = None):
        if init_state is not None:
            print("use old initial_state")
            self.init_state = init_state
        else:
            print("generating new initial_state")
            if extra_constraint:
                self.init_state = self_defined_initial_state(self.eq_system, self.params['N'], extra_constraint = extra_constraint)
            else:
                self.init_state = self.eq_system.eq_state(self.params['N'], t=0, beta=self.params['beta'], manual_domain = manual_domain, axes = axes)

        print(f"as step value: {self.as_step}, sampleSize: {self.sampleSize}" )

        
        if self.params['measureWorkWithOffset'] == True:
            work_measurement_procedure = sp.MeasureWorkDoneWithOffset(rp.get_dW, trial_request=np.s_[::self.sampleSize], step_request=self.as_step, protocol_time_index_array = self.params['protocol_time_index_array'], measurement_params =  self.params)
        else: 
            work_measurement_procedure = sp.MeasureWorkDone(rp.get_dW, trial_request=np.s_[::self.sampleSize], step_request=self.as_step)
            
        self.procs = [work_measurement_procedure, sp.ReturnFinalState()]

        
        if self.measure_all_state:
            self.procs.append(sp.MeasureAllState(trial_request=np.s_[:self.sampleSize], step_request=self.as_step),)
            
        
        if verbose:
            print(f"from cfq_runner.py, The as_step is {as_step} and dt is {self.params['dt']}")

        # edward added this, to override the 200 states only in all states.
        # self.procs[1] = sp.MeasureAllState()

        sim_kwargs = {
                        'damping':self.params['lambda'],
                        'temp':1/self.params['beta'],
                        'dt':self.params['dt'],
                        'procedures':self.procs,
                        'sim_params': self.params['sim_params']
            }
        

        self.sim = setup_sim(self.system, self.init_state, verbose = verbose,**sim_kwargs)
        self.sim.reference_system = self.eq_system
        
        return

    
    def show_initial_state(self):
        """plot a graph of initial state and tell the information of initial KE"""
        init_state = self.init_state
        avg_init_KE = np.mean(self.system.get_kinetic_energy(init_state), axis = 0)
        
        plt.title(f"average initial KE = {avg_init_KE}")
        plt.scatter(init_state[:,0,0], init_state[:,1,0])

    def show_final_state(self):
        """plot a graph of initial state and tell the information of initial KE"""
        final_state = self.get_all_state()[:,-1,...]
        avg_init_KE = np.mean(self.system.get_kinetic_energy(final_state), axis = 0)
        
        plt.title(f"average initial KE = {avg_init_KE}")
        plt.scatter(init_state[:,0,0], init_state[:,1,0])
        
        
    def plotState(self, state):
        """plot a graph of a given state"""
        plt.scatter(state[:,0,0], state[:,1,0])
        
        
    def get_all_state(self):
        """get the all_state result from the output"""
        return self.sim.output.all_state['states']
    
    def get_pColor(self):
        """get the color of each particle"""
        return self.pColor

    
    def createProtocolTimeArray(self, protocol_list, params):
        """Return the following four arrays: protocol_time_array, protocol_time_index_array, protocol_all_time_array, protocol_all_time_index_array"""
        protocol_time_array = [item["duration"] for item in protocol_list]
        protocol_time_array.insert(0, 0)
        protocol_time_array = np.cumsum(protocol_time_array)
        
        protocol_time_index_array = protocol_time_array / params['dt'] - 1
        protocol_time_index_array[0] = 0
        protocol_time_index_array = protocol_time_index_array.astype(int)
        
        protocol_all_time_index_array = np.array(range(0, int(protocol_time_index_array[-1]) + 2))
        protocol_all_time_array = protocol_all_time_index_array * params['dt']
        
        params['protocol_time_array'] = protocol_time_array
        params['protocol_time_index_array'] = protocol_time_index_array
        self.protocol_time_array = protocol_time_array
        self.protocol_time_index_array = protocol_time_index_array
        self.protocol_all_time_array = protocol_all_time_array
        self.protocol_all_time_index_array = protocol_all_time_index_array
        
        return protocol_time_array, protocol_time_index_array, protocol_all_time_array, protocol_all_time_index_array
    
    
    def plotGraphWithContourGraph(self, t = 0, cutlineDirection = "v", cutlineValue = 0, contour_range = [0, 450], showGraph = None, saveGraph = None, graph_ax = None, graph_title = " ", x_range = None, y_range = None, mark_critical_points = None):
        if not graph_ax:
            fig, ax = plt.subplots(1, 2, figsize = [9, 4.5])
            ax[0].set_box_aspect(aspect=1)
        
        X, Y, U, cutlineDirection, _targetAxis, _plotAxis = coupled_fq_protocol_library.get_potential_along_a_1D_cutline(self, t = t, cutlineDirection =cutlineDirection, cutlineValue = cutlineValue, resolution = 100, contours = 5, manual_domain = [[-5, -5], [5, 5]])
            
        plotLineData = coupled_fq_protocol_library.plotCutlines(X, Y, U, cutlineDirection = cutlineDirection, cutlineValue = cutlineValue, x_range = x_range, y_range = y_range, contour_range = contour_range, cutlineColor = 'red' , contour_plt = graph_ax[0], cutline_plt = graph_ax[1], contours = 5, time = None, graph_title = graph_title, showGraph = showGraph, saveGraph = saveGraph)
        
        if mark_critical_points:
            pass
        return plotLineData
        
        
        
    def showAnimation(self, protocol_list, computation_protocol_parameter_dict, manual_domain = [[-5, -5], [5, 5]], has_velocity = True, pColor = ["blue", "orange", "green", "red"], frame_skip = 10, color_key = False, color_bar_range = None , showFormat = "mp4"):
        # animations
        vmin, vmax = 0, 100
        # phi_1_dc, phi_2_dc = phi_1_dcx, phi_2_dcx

        all_state = self.sim.output.all_state['states']
        self.system.protocol_list = protocol_list
        time_range = (computation_protocol_parameter_dict["t"][0], computation_protocol_parameter_dict["t"][-1])

        ani,_,_ = visualization.animate_sim_flux_qubit(all_state, system = self.system ,
                                           times = time_range, frame_skip=frame_skip, color_by_state=True, 
                                           pColor =pColor, vmin = vmin, vmax = vmax, color_key = color_key,
                                           color_bar_range = color_bar_range, manual_domain = manual_domain,
                                           has_velocity=has_velocity)

        # from IPython.display import HTML
        if showFormat == "gif":
            writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
            ani.save('scatter.gif', writer=writer)
            Image(filename="scatter.gif")
            plt.show()
        else:
            video = ani.to_html5_video()
            html = display.HTML(video)
            display.display(html)
    
    
    
    
    
    def getParameterTimeSeries(self):
        reducedTimeArray = self.getReducedTimeArray()
        reducedParams = np.array([self.system.protocol.get_params(_t) for _t in reducedTimeArray])
        return {
            "phi_1x": reducedParams[:, 8], "phi_2x": reducedParams[:, 9], "phi_1xdc": reducedParams[:, 10], "phi_2xdc": reducedParams[:, 11], "mu_12": reducedParams[:, 12]
        }


    def showPamameterTimeGraph(self, protocol_time_array, protocol_all_time_array):
        params_at_all_time = np.array([self.system.protocol.get_params(_t) for _t in protocol_all_time_array])
        phi_1x_at_all_time = params_at_all_time[:, 8]
        phi_2x_at_all_time = params_at_all_time[:, 9]
        phi_1xdc_at_all_time = params_at_all_time[:, 10]
        phi_2xdc_at_all_time = params_at_all_time[:, 11]
        phi_M_12_at_all_time = params_at_all_time[:, 12]
    
        fig, ax  = plt.subplots(1, 2, figsize = [12, 4])

        ax[0].plot(protocol_all_time_array, phi_1xdc_at_all_time, label = r"$\varphi_{1xdc}$")
        ax[0].plot(protocol_all_time_array, phi_2xdc_at_all_time, label = r"$\varphi_{2xdc}$")
        ax[0].plot(protocol_all_time_array, phi_M_12_at_all_time, label = r"$M_{12}$")
        ax[0].set_ylabel("value")
        ax[0].set_xlabel("time")
        ax[0].legend()

        ax[1].plot(protocol_all_time_array, phi_1x_at_all_time, label = r"$\varphi_{1x}$")
        ax[1].plot(protocol_all_time_array, phi_2x_at_all_time, label = r"$\varphi_{2x}$")
        ax[1].set_xlabel("time")
        ax[1].legend()

        for i, x in enumerate(protocol_time_array):
            t_text = f"$t_{i}$"
            ax[0].text(x-10, 3.61, t_text)
            ax[1].text(x-10, 0.0355, t_text)


            ax[0].vlines(x, ymin = -0.8, ymax = 3.3, linestyle = '--', color = 'grey', alpha = 0.5)
            ax[1].vlines(x, ymin = -0.2, ymax = 0.02, linestyle = '--', color = 'grey', alpha = 0.5)

        
    
    def separate_by_state(self):
        self.particle_index = visualization.separate_by_state(self.get_all_state()[:, :, (0, 1), :])
        
    
    def updateProtocol(self, new_protocol_list):
        computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, \
                                                                        new_protocol_list)
        new_storage_protocol, new_comp_protocol = coupled_fq_protocol_library.create_system(computation_protocol_parameter_dict, modifiedFunction = None)
        self.computation_protocol = new_comp_protocol
        self.initialize_sim()


    def create_params_dict_with_external_parameters(self, new_params_list, output_format = "dict"):
        params_list = self.system.protocol.get_params(0)
        for key in new_params_list.keys():
            keyIndex = self.protocol_key.index(key)
            params_list[keyIndex] = new_params_list[key]
            
        if output_format == "list":
            return params_list
        else:
            return dict(zip(self.protocol_key, params_list))

    def convert_params_list_into_dict(self, params_list):
        return dict(zip(self.protocol_key, params_list))
    
    def get_minimum_at_t_for_a_set_of_parameter(self, _t, verbose = True, guess = [(0, 0)], showPE = None):
        """To find the minimum point for a given time, overlapped with the find_all_minimum_points_for_all_potential function in minimum_value_of_potential.py in Analysis tool. """
        _params_at_t = self.system.protocol.get_params(_t)
        beta_1 = _params_at_t[4]
        beta_2 = _params_at_t[5]
        d_beta_1 = _params_at_t[6]
        d_beta_2 = _params_at_t[7]
        _phi_1x = _params_at_t[8]
        _phi_2x = _params_at_t[9]
        _phi_1xdc = _params_at_t[10]
        _phi_2xdc = _params_at_t[11]
        _M_12 = _params_at_t[12]
        _xi = 1 / (1 - _M_12**2)
        _phi_1dc = _phi_1xdc
        _phi_2dc = _phi_2xdc

        def Fcn(coord):
            _phi_1, _phi_2 = coord
            u1_1 = 1/2 * _xi * (_phi_1 - _phi_1x)**2
            u3_1 = beta_1 * np.cos(_phi_1) * np.cos(_phi_1dc/2)
            u4_1 = -d_beta_1 * np.sin(_phi_1) * np.sin(_phi_1dc/2)

            u1_2 = 1/2 * _xi * (_phi_2 - _phi_2x)**2        
            u3_2 = beta_2 * np.cos(_phi_2) * np.cos(_phi_2dc/2)
            u4_2 = -d_beta_2 * np.sin(_phi_2) * np.sin(_phi_2dc/2)
            u5 = _M_12 * _xi * (_phi_1 - _phi_1x) * (_phi_2 - _phi_2x)

            return u1_1 + u1_2 + u3_1 + u3_2 + u4_1 + u4_2 + u5

        solution_set = [optimize.fmin(Fcn, _g, disp=False) for _g in guess]
        energy_set = [self.system.potential.potential(sol[0], sol[1], _phi_1xdc, _phi_2xdc, _params_at_t) for sol in solution_set]
        
        if showPE:
            return [solution_set, energy_set]
        else:
            return solution_set

# [0: 'U0_1', 1: 'U0_2', 2: 'gamma_1', 3: 'gamma_2', 4: 'beta_1', 5: 'beta_2', 6: 'd_beta_1', 7: 'd_beta_2', 8: 'phi_1_x', 9: 'phi_2_x', 10: 'phi_1_dcx', 11: 'phi_2_dcx', 12: 'M_12', 13: 'x_c']
        
    
    def find_critical_point_for_all_potential(self, _t, verbose = True, guess = [(0, 0)], showPE = False):
        _params_at_t = self.system.protocol.get_params(_t)
        beta_1 = _params_at_t[4]
        beta_2 = _params_at_t[5]
        d_beta_1 = _params_at_t[6]
        d_beta_2 = _params_at_t[7]
        _phi_1x = _params_at_t[8]
        _phi_2x = _params_at_t[9]
        _phi_1xdc = _params_at_t[10]
        _phi_2xdc = _params_at_t[11]
        _M_12 = _params_at_t[12]
        _xi = 1 / (1 - _M_12**2)
        _phi_1dc = _phi_1xdc
        _phi_2dc = _phi_2xdc

        def Fcn(x):
            return [
                _xi * x[0] - beta_1 * np.sin(x[0]) * np.cos(_phi_1xdc/2) + d_beta_1 * np.cos(x[0]) * np.sin(_phi_1xdc/2) 
                        + _M_12 * _xi * x[1],
                _xi * x[1] - beta_2 * np.sin(x[1]) * np.cos(_phi_2xdc/2) + d_beta_2 * np.cos(x[1]) * np.sin(_phi_2xdc/2) 
                    + _M_12 * _xi * x[0]
            ]

        solution_set = [fsolve(Fcn, _g) for _g in guess]
        energy_set = [self.system.potential.potential(sol[0], sol[1], _phi_1xdc, _phi_2xdc, _params_at_t) for sol in solution_set]

        if showPE:
                return [solution_set, energy_set]
        else:
            return solution_set
        
    
    def getFidelityBarChart(self):
        metastable_state_array = ["00", "01", "10", "11"]
        fig, ax = plt.subplots(2, 2, figsize=[7, 7])
        fidelity_time_array = np.array(self.sim.fidelity_time_array).astype(float)
        
        # 00
        ax[1][0].bar(["00", "01", "10", "11"], fidelity_time_array[-1,:,0], color = pColor.values())
        ax[1][0].set_ylim(0, 1)
        ax[1][0].axhline(0.5, linestyle='--')

        # 01
        ax[0][0].bar(["00", "01", "10", "11"], fidelity_time_array[-1,:,1], color = pColor.values())

        ax[0][0].axhline(0.5, linestyle='--')
        ax[0][0].set_ylim(0, 1)

        # 10
        ax[1][1].bar(["00", "01", "10", "11"], fidelity_time_array[-1,:,2], color = pColor.values())
        ax[1][1].set_ylim(0, 1)
        ax[1][1].axhline(0.5, linestyle='--')

        # 11
        ax[0][1].bar(["00", "01", "10", "11"], fidelity_time_array[-1,:,3], color = pColor.values())
        ax[0][1].set_ylim(0, 1)
        ax[0][1].axhline(0.5, linestyle='--')
    
    
    def getFidelityTimeGraph(self):
        """
        trace the fidelity of the systems  at different time as 4 different plots
        """
        metastable_state_array = ["00", "01", "10", "11"]
        
        fidelity_time_array = np.array(self.sim.fidelity_time_array, dtype = float)

        fig, ax = plt.subplots(2, 2, figsize = [10, 8])
        ax_flatten = ax.flatten()

        for i, _ax in enumerate(ax_flatten):
            _ax.plot(self.sim.target_step_index, fidelity_time_array[:, 0, i], c = pColor["00"], label = "00")
            _ax.plot(self.sim.target_step_index, fidelity_time_array[:, 1, i], c = pColor["01"], label = "01")
            _ax.plot(self.sim.target_step_index, fidelity_time_array[:, 2, i], c = pColor["10"], label = "10")
            _ax.plot(self.sim.target_step_index, fidelity_time_array[:, 3, i], c = pColor["11"], label = "11")
            _ax.set_title(f"time evolution of fidelity inside {metastable_state_array[i]}")
            _ax.legend()
        plt.show()
    

    def getIndexOfParticles(self, state = None):
        if not type(state) == np.ndarray:
            state = self.init_state
        return visualization.separate_by_state_2(state[:, (0, 1), :])
    
    def get_state_by_particle_type(self, label = None):
        # label = 00, 01, 10 and 11
        all_states = self.sim.output.all_state['states']
        
        if label:
            _index = self.getIndexOfParticles()[label]
            return  all_states[_index]
        else:
            return self.getIndexOfParticles()
    

    def get_failure_particles(self, mapping_dict):
        wrong_particle_array = []
        for index_i, index_f in mapping_dict.items():
            initial_state_index = self.getIndexOfParticles()
            selected_final_state = self.sim.output.final_state[initial_state_index[index_i]]
            index_of_selected_final_state = self.getIndexOfParticles(selected_final_state)

            correct_well_query = [index_of_selected_final_state[item] for item in index_f]
            if len(correct_well_query) > 1:
                correct_well_query = np.logical_or(correct_well_query)
            else:
                correct_well_query = correct_well_query[0]

            wrong_particle_index = ~correct_well_query
            number_of_wrong_particle = np.sum(wrong_particle_index)

            if number_of_wrong_particle > 0:
                target_particles = self.get_all_state()[initial_state_index[index_i]]
                wrong_particle = target_particles[wrong_particle_index]
                wrong_particle_array.append(wrong_particle)
        if len(wrong_particle_array) == 0:
            return np.array([])
        else:
            return np.vstack(wrong_particle_array)


            
    
    def plot_distribution_with_cutline(self, t, time_array=None, params = None, legend=False, slice_values = None, fig_ax=None, vmin = None, vmax = None, cmap = "afmhot", pcolor = {"00": "blue", "01": "orange", "10": "green", "11": "red"}, cutlineInformation = [(-3, "v","red")], **pot_kwargs):
        names = [r"$\varphi_1$", r"$\varphi_2$"]
        fig, ax = plt.subplots(1, 2, figsize=[10, 5])
        ax_flatten = ax.flatten()
        manual_domain=[np.array([-5, -5]), np.array([5, 5])]

        all_state = self.sim.output.all_state['states']
        phi_1_dc = all_state[..., 2, 0]
        phi_2_dc = all_state[..., 3, 0]
        all_state = all_state[:, :, (0, 1), :]

        N, nsteps, N_dim = np.shape(all_state)[0], np.shape(all_state)[1], np.shape(all_state)[2]
        which_axes = [np.s_[..., i, 0] for i in range(N_dim)]
        x_array = [all_state[item] for item in which_axes]
        state_lookup = separate_by_state(all_state)
        opacity = min(1, 300/N)

        time_array = time_array[::frame_skip]
        x = x_array[0][:,::frame_skip]
        y = x_array[1][:,::frame_skip]
        x_lim = (np.min(x), np.max(x))
        y_lim = (np.min(y), np.max(y))
        scat_kwargs = {'alpha':opacity, 'zorder':10}
        scat = [ax_flatten[0].scatter(x[state_lookup[key], 0], y[state_lookup[key], 0], **scat_kwargs) for key in state_lookup]
        scat_1 = [ax_flatten[1].scatter(x[state_lookup[key], 0], y[state_lookup[key], 0], **scat_kwargs) for key in state_lookup]
        
        modified_manual_domain = [(manual_domain[0][1], manual_domain[0][0]), \
                                  (manual_domain[1][1], manual_domain[1][0])]
        resolution = 100
        U, X_mesh = self.system.lattice(t, resolution, axes=(0, 1),\
                                    manual_domain=modified_manual_domain, slice_values = slice_values)
        X, Y = X_mesh[0], X_mesh[1]

        # 5. find vmin, vmax, x_min, x_max, y_min and y_max
        vmin, vmax = np.min(U), np.max(U)
        x_min, x_max = np.min(X), np.max(X)
        y_min, y_max = np.min(Y), np.max(Y)
        cont = contour_plt.contourf(X, Y, U, 40, vmin = 0, vmax = 400, cmap = "afmhot")
        
        
        
        ##############
        
        min_U_along_cutline = []
        fig, ax = plt.subplots(5, 4, figsize=[15, 18])
        ax_flatten = ax.flatten()
        cutlineRange = [-3.5, 3.5]

        X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[0], cutlineDirection = "v")
        plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_point_data[0][0][0],\
                                contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = 20,
                                showGraph=True, time=protocol_time_array[0], cutlineRange = cutlineRange)
        min_U = np.min(U)
        scat = [ax_flatten[2 * 0].scatter(x[state_lookup[key], protocol_time_index_array[0]], y[state_lookup[key], protocol_time_index_array[0]], c = pColor[key],**scat_kwargs) for key in state_lookup]


        for _i in range(1, 5):
            X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[_i], cutlineDirection = "v")
            plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_point_data[_i][0][0],\
                                    contour_plt=ax_flatten[2*_i], cutline_plt=ax_flatten[2*_i + 1], contours = 20,
                                    showGraph=True, time=protocol_time_array[_i], cutlineRange = cutlineRange)
            min_U = np.min(U)
            _index = list(simResult['cfqr'].sim.target_step_index).index(protocol_time_index_array[_i]+1)
            scat = [ax_flatten[2 * _i].scatter(x[state_lookup[key], _index], y[state_lookup[key], _index], c = pColor[key], **scat_kwargs) for key in state_lookup]

        for _i in range(5, 9):
            X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[_i], cutlineDirection = "h")
            plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_point_data[_i][0][1],\
                                    contour_plt=ax_flatten[2*_i], cutline_plt=ax_flatten[2*_i + 1], contours = 20,
                                    showGraph=True, time=protocol_time_array[_i], cutlineRange = cutlineRange)
            min_U = np.min(U)
            _index = list(simResult['cfqr'].sim.target_step_index).index(protocol_time_index_array[_i]+1)
            scat = [ax_flatten[2 * _i].scatter(x[state_lookup[key], _index], y[state_lookup[key], _index], c = pColor[key], **scat_kwargs) for key in state_lookup]

            
    def getReducedTimeArray(self, protocol_all_time_array = None):
        as_step = self.params['as_step']
        if not protocol_all_time_array:
            protocol_all_time_array = self.protocol_all_time_array
            # simResult['cfqr'].protocol_all_time_array[simResult['cfqr'].sim.output.all_state['step_indices']]
        return protocol_all_time_array[as_step]
    

    def get_KE_of_all_particles(self, state, label):
        _state_phi_1 = state.copy()
        _state_phi_1[..., (1, 2, 3), 1] = 0

        _state_phi_2 = state.copy()
        _state_phi_2[..., (0, 2, 3), 1] = 0

        _state_phi_1dc = state.copy()
        _state_phi_1dc[..., (0, 1, 3), 1] = 0

        _state_phi_2dc = state.copy()
        _state_phi_2dc[..., (0, 1, 2), 1] = 0


        KE_phi_1, KE_phi_2, KE_phi_1dc, KE_phi_2dc = np.mean(self.system.get_kinetic_energy(_state_phi_1), axis = 0), \
                                                     np.mean(self.system.get_kinetic_energy(_state_phi_2), axis = 0), \
                                                     np.mean(self.system.get_kinetic_energy(_state_phi_1dc), axis = 0), \
                                                     np.mean(self.system.get_kinetic_energy(_state_phi_2dc), axis = 0)
        
        KE_phi_1_std, KE_phi_2_std, KE_phi_1dc_std, KE_phi_2dc_std = np.std(self.system.get_kinetic_energy(_state_phi_1), axis = 0) * 3, \
                                                                     np.std(self.system.get_kinetic_energy(_state_phi_2), axis = 0) * 3, \
                                                                     np.std(self.system.get_kinetic_energy(_state_phi_1dc), axis = 0)  * 3, \
                                                                     np.std(self.system.get_kinetic_energy(_state_phi_2dc), axis = 0) * 3
        
        KE_mean_array = [KE_phi_1, KE_phi_1dc, KE_phi_2, KE_phi_2dc]
        KE_std_array = [KE_phi_1_std, KE_phi_1dc_std, KE_phi_2_std, KE_phi_2dc_std ]

        return KE_mean_array, KE_std_array

    


    def KE_analysis_of_individual_particles(self):

        pIndex = self.getIndexOfParticles()
        """To investigate the average KE of each type of particles"""
        all_state = self.get_all_state()

        index_of_00 = pIndex["00"]
        index_of_01 = pIndex["01"]
        index_of_10 = pIndex["10"]
        index_of_11 = pIndex["11"]

        KE_mean_array_00, KE_std_array_00 = self.get_KE_of_all_particles(all_state[index_of_00], "00")
        KE_mean_array_01, KE_std_array_01 = self.get_KE_of_all_particles(all_state[index_of_01], "01")
        KE_mean_array_10, KE_std_array_10 = self.get_KE_of_all_particles(all_state[index_of_10], "10")
        KE_mean_array_11, KE_std_array_11 = self.get_KE_of_all_particles(all_state[index_of_11], "11")
        
        return np.array([
            [KE_mean_array_00[0], KE_mean_array_01[0], KE_mean_array_10[0], KE_mean_array_11[0]],
            [KE_std_array_00[0], KE_std_array_01[0], KE_std_array_10[0], KE_std_array_11[0]]
        ])
    




    def phi_analysis_of_individual_particles(self, target_axis = "phi_1"):
        
        mapping = {"phi_1": 0, "phi_2": 1, "phi_1dc": 2, "phi_2dc": 3}
        target_axis_index = mapping[target_axis]

        pIndex = self.getIndexOfParticles()
        """To investigate the average KE of each type of particles"""
        all_state = self.get_all_state()

        index_of_00 = pIndex["00"]
        index_of_01 = pIndex["01"]
        index_of_10 = pIndex["10"]
        index_of_11 = pIndex["11"]

        phi_00 = all_state[index_of_00, ..., target_axis_index, 0]
        phi_01 = all_state[index_of_01, ..., target_axis_index, 0]
        phi_10 = all_state[index_of_10, ..., target_axis_index, 0]
        phi_11 = all_state[index_of_11, ..., target_axis_index, 0]

        phi_mean_array_00, phi_std_array_00 = np.mean(phi_00, axis = 1), np.std(phi_00) * 3
        phi_mean_array_01, phi_std_array_01 = np.mean(phi_01, axis = 1), np.std(phi_01) * 3
        phi_mean_array_10, phi_std_array_10 = np.mean(phi_10, axis = 1), np.std(phi_10) * 3
        phi_mean_array_11, phi_std_array_11 = np.mean(phi_11, axis = 1), np.std(phi_11) * 3
        
        return np.array([
            [phi_mean_array_00, phi_mean_array_01, phi_mean_array_10, phi_mean_array_11],
            [phi_std_array_00, phi_std_array_01, phi_std_array_10, phi_std_array_11]
        ])
    

    def criticalPointMonitor(self, as_step = np.s_[::100]):

        min_00_array, min_00_PE_array = [], []
        min_01_array, min_01_PE_array  = [], []
        min_10_array, min_10_PE_array  = [], []
        min_11_array, min_11_PE_array  = [], []
        barrier_00_01_array, barrier_00_01_PE_array = [], []
        barrier_10_11_array, barrier_10_11_PE_array = [], []
        barrier_00_10_array, barrier_00_10_PE_array = [], []
        barrier_01_11_array, barrier_01_11_PE_array = [], []

        protocol_all_time_array = self.protocol_all_time_array
        t_array = protocol_all_time_array[as_step]
        for _t in t_array:

            min_location, min_PE = self.get_minimum_at_t_for_a_set_of_parameter(_t,  guess=[[-2, -2], [-2, 2], [2, -2], [2, 2]], showPE=True)

            min_00 = min_location[0]
            min_01 = min_location[1]
            min_10 = min_location[2]
            min_11 = min_location[3]
            min_00_PE = min_PE[0]
            min_01_PE = min_PE[1]
            min_10_PE = min_PE[2]
            min_11_PE = min_PE[3]

            min_00_array.append(min_00)
            min_01_array.append(min_01)
            min_10_array.append(min_10)
            min_11_array.append(min_11)
            min_00_PE_array.append(min_00_PE)
            min_01_PE_array.append(min_01_PE)
            min_10_PE_array.append(min_10_PE)
            min_11_PE_array.append(min_11_PE)


            barrier_location, barrier_PE = self.find_critical_point_for_all_potential(_t, guess=[[-2, 0], [2, 0], [0, -2], [0, 2]], showPE=True)
            barrier_00_01 = barrier_location[0]
            barrier_10_11 = barrier_location[1]
            barrier_00_10 = barrier_location[2]
            barrier_01_11 = barrier_location[3]
            barrier_00_01_PE = barrier_PE[0]
            barrier_10_11_PE = barrier_PE[1]
            barrier_00_10_PE = barrier_PE[2]
            barrier_01_11_PE = barrier_PE[3]


            barrier_00_01_array.append(barrier_00_01)
            barrier_10_11_array.append(barrier_10_11)
            barrier_00_10_array.append(barrier_00_10)
            barrier_01_11_array.append(barrier_01_11)
            barrier_00_01_PE_array.append(barrier_00_01_PE)
            barrier_10_11_PE_array.append(barrier_10_11_PE)
            barrier_00_10_PE_array.append(barrier_00_10_PE)
            barrier_01_11_PE_array.append(barrier_01_11_PE)

        return {
            "t_array": t_array,
            "00": {"location": min_00_array, "PE": np.array(min_00_PE_array)},
            "01": {"location": min_01_array, "PE": np.array(min_01_PE_array)},
            "10": {"location": min_10_array, "PE": np.array(min_10_PE_array)},
            "11": {"location": min_11_array, "PE": np.array(min_11_PE_array)},
            "00_01": {"location": barrier_00_01_array, "PE": np.array(barrier_00_01_PE_array)},
            "10_11": {"location": barrier_10_11_array, "PE": np.array(barrier_10_11_PE_array)},
            "00_10": {"location": barrier_00_10_array, "PE": np.array(barrier_00_10_PE_array)},
            "01_11": {"location": barrier_01_11_array, "PE": np.array(barrier_01_11_PE_array)}
        }
    
    
    def createNewParticle(self, particle_coordinates):
        self.particle_test = particle_coordinates
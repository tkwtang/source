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

class simRunner(SimManager):
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
        self.pColor = {"0": "#061DF7", "1": "#FCEF51"}
        
    def verify_param(self, key, val):
        return True

    def initialize_sim(self):
        self.potential.default_params = self.override_potential_parameter
        self.eq_protocol = self.storage_protocol
        
        self.potential.default_params = np.array(self.override_potential_parameter)
        self.protocol = self.computation_protocol

        self.eq_system = System(self.eq_protocol, self.potential)
        self.eq_system.axes_label = ["x"]
        self.eq_system.has_velocity = self.has_velocity
        self.eq_system.U0 = 1

        self.system = System(self.protocol, self.potential)
        self.system.axes_label = ["x"]
        self.system.has_velocity = self.has_velocity
        self.system.U0 = 1
        
        self.createProtocolTimeArray(self.protocol_list, self.params)
        
        
    def set_sim_attributes(self, init_state = None, manual_domain = None, axes = None, percentage = 1, as_step = np.s_[::], verbose = True, extra_constraint = None):
        if init_state is not None:
            print("use old initial_state")
            self.init_state = init_state
        else:
            print("generating new initial_state")
            self.init_state = self.eq_system.eq_state(self.params['N'], t=0, beta=self.params['beta'], manual_domain = manual_domain, axes = axes)

        print(f"as step value: {self.as_step}, sampleSize: {self.sampleSize}")

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
        # phi_1dc, phi_2dc = phi_1xdc, phi_2xdc

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
    
    
    
    
    def showPamameterTimeGraph(self, protocol_time_array, protocol_all_time_array):
        params_at_all_time = np.array([self.system.protocol.get_params(_t) for _t in protocol_all_time_array])
        phi_1x_at_all_time = params_at_all_time[:, 8]
        phi_2x_at_all_time = params_at_all_time[:, 9]
        phi_1xdc_at_all_time = params_at_all_time[:, 10]
        phi_2xdc_at_all_time = params_at_all_time[:, 11]
        phi_mu_12_at_all_time = params_at_all_time[:, 12]
    
        fig, ax  = plt.subplots(1, 2, figsize = [12, 4])

        ax[0].plot(protocol_all_time_array, phi_1xdc_at_all_time, label = r"$\varphi_{1xdc}$")
        ax[0].plot(protocol_all_time_array, phi_2xdc_at_all_time, label = r"$\varphi_{2xdc}$")
        ax[0].plot(protocol_all_time_array, phi_mu_12_at_all_time, label = r"$M_{12}$")
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
    
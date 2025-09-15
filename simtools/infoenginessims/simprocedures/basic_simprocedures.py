from numpy import empty, s_, histogramdd, mean, shape, array, average, sign, zeros
from scipy.stats import sem
import numpy as np
# import numpy as np


class SimProcedure:
    """Base class for simulation procedures.

    _Methods_
    do_initial_task: Called during the simulation's run method after all of its
        initialization steps and right before the main state evolution loop.
    do_intermediate_task: Called at the end of each step of the simulation's
        main state evolution loop.
    do_final_task: Called right before the end of the simulation's run method.
        Should return the output appropriate the procedure.  If the output of
        the simulation should have no contribution from this procedure, the
        output of this method should be None.
    """

    def do_initial_task(self, simulation):
        pass

    def do_intermediate_task(self):
        pass

    def do_final_task(self):
        return None


# --------- State Measurements --------- #

class ReturnFinalState(SimProcedure):
    """Measurement that returns the supposedly existent final next_states."""

    def do_initial_task(self, simulation, output_name='final_state'):

        self.simulation = simulation
        self.output_name = output_name

    def do_final_task(self):
        return self.simulation.next_state

class ReturnInitialState(SimProcedure):
    """Measurement that returns the supposedly existent final next_states."""

    def do_initial_task(self, simulation, output_name='initial_state'):

        self.simulation = simulation
        self.output_name = output_name

    def do_final_task(self):
        return self.simulation.initial_state

#
class MeasureAllState(SimProcedure):
    """Measurement that returns for a subset of trials.

    The trial_indices argument can take lists of indices (integer array
    indexing), slices, and numpy index expressions.
    """

    def __init__(self, step_request=s_[:], trial_request=s_[:],
                 output_name='all_state'):

        self.step_request = step_request
        self.trial_request = trial_request
        self.output_name = output_name

    def do_initial_task(self, simulation):

        self.simulation = simulation

        initial_state = simulation.initial_state

        state_shape = initial_state.shape[1:]
        # nstate_dims = initial_state.shape[1]

        trial_indices = range(self.simulation.ntrials)[self.trial_request]
        step_indices = range(self.simulation.nsteps + 1)[self.step_request]

        all_states_shape = [len(trial_indices), len(step_indices)]
        all_states_shape.extend(state_shape)

        states = empty(all_states_shape)

        self.all_state = {'step_indices': step_indices,
                          'trial_indices': trial_indices,
                          'states': states}

        try:

            step_index = step_indices.index(0)
            initial_state = self.simulation.initial_state

            states[:, step_index, ...] = initial_state[trial_indices, ...]

        except ValueError:
            pass

    def do_intermediate_task(self):
        next_step = self.simulation.current_step + 1

        try:

            step_indices = self.all_state['step_indices']
            step_index = step_indices.index(next_step)

            next_state = self.simulation.next_state
            trial_indices = self.all_state['trial_indices']
            states = self.all_state['states']

            states[:, step_index, ...] = next_state[trial_indices, ...]

        except ValueError:
            pass

    def do_final_task(self):
        step_indices = range(self.simulation.nsteps + 1)[self.step_request]
        self.all_state['step_indices'] = step_indices
        self.all_state['states'] = self.all_state['states'][:, :len(step_indices), ...]

        return self.all_state

class MeasureStepValue(SimProcedure):
    """Measurement that returns a value for a subset of trials.

    The trial_indices argument can take lists of indices (integer array
    indexing), slices, and numpy index expressions.
    """

    def __init__(self, get_value, output_name='all_value', step_request=s_[:], trial_request=s_[:]):
        self.get_val = get_value
        self.output_name = output_name
        self.step_request = step_request
        self.trial_request = trial_request



    def do_initial_task(self, simulation):

        self.simulation = simulation

        initial_val = self.get_val(self.simulation, self.trial_request)

        val_shape = shape(initial_val)

        step_indices = range(self.simulation.nsteps + 1)[self.step_request]


        all_val_shape = [len(step_indices), *val_shape]

        vals = empty(all_val_shape)

        vals[0, ...] = initial_val

        self.all_value = {'step_indices': step_indices, 'trial_indices': self.trial_request, 'values': vals}


    def do_intermediate_task(self):

        next_step = self.simulation.current_step + 1

        try:

            step_indices = self.all_value['step_indices']
            step_index = step_indices.index(next_step)

            try:
                next_value = self.get_val(self.simulation, self.trial_request)

                vals = self.all_value['values']

                vals[step_index, ...] = next_value

            except ValueError:
                print('shape fail')


        except ValueError:
            print("some error in the intermediate task of MeasureStepValue")
            pass

    def do_final_task(self):
        step_indices = range(self.simulation.nsteps + 1)[self.step_request]
        self.all_value['step_indices'] = step_indices
        self.all_value['values'] = self.all_value['values'][:len(step_indices)]

        return self.all_value

class MeasureMeanValue(MeasureStepValue):

    """Measurement that returns a value for a subset of trials.

    The trial_indices argument can take lists of indices (integer array
    indexing), slices, and numpy index expressions.
    """
    def __init__(self, get_value, output_name='all_value', step_request=s_[:], trial_request=s_[:], weights=None):
        self.get_val = lambda x,y: [average(get_value(x,y), axis=0, weights=weights), sem(get_value(x,y))]
        self.output_name = output_name
        self.step_request = step_request
        self.trial_request = trial_request

    def do_final_task(self):
        step_indices = range(self.simulation.nsteps + 1)[self.step_request]
        nvals = len(step_indices)
        self.all_value['step_indices'] = step_indices
        self.all_value['std_error'] = self.all_value['values'][:nvals,1,...]
        self.all_value['values'] = self.all_value['values'][:nvals,0,...]

        return self.all_value

class TerminateOnMean(MeasureMeanValue):
    def __init__(self, get_value, target=1, **kwargs):
        kw_args = {'output_name':'all_value', 'step_request':s_[:], 'trial_request':s_[:], 'weights':None}
        kw_args.update(kwargs)
        MeasureMeanValue.__init__(self, get_value, **kwargs)
        self.target=target

    def do_intermediate_task(self):
        try:
            next_step = self.simulation.current_step + 1
            step_indices = self.all_value['step_indices']
            step_index = step_indices.index(next_step)
            MeasureMeanValue.do_intermediate_task(self,)
            c_val = self.all_value['values'][step_index-1, 0, ...]
            next_val = self.all_value['values'][step_index, 0, ...]
            if sign(c_val-self.target) != sign(next_val-self.target):
                self.terminate = True
        except:
            pass


class MeasureAllStateDists(SimProcedure):
    """Records a running set of state histograms."""

    def __init__(self, bins, step_request=s_[:],
                 output_name='all_state_dists'):

        self.bins = bins
        self.step_request = step_request
        self.output_name = output_name

    def do_initial_task(self, simulation):

        self.simulation = simulation

        if self.bins is None:
            self.bins = simulation.initial_dist.bins

        step_indices = range(self.simulation.nsteps + 1)[self.step_request]
        hists = []

        self.all_dists = {'step_indices': step_indices, 'hists': hists}

        if 0 in step_indices:

            initial_state = simulation.initial_state
            bins = self.bins

            dist = histogramdd(initial_state, bins=bins)
            hists.append(dist)

    def do_intermediate_task(self):

        next_step = self.simulation.current_step + 1

        if next_step in self.all_dists['step_indices']:

            next_state = self.simulation.next_state
            bins = self.bins
            hists = self.all_dists['hists']

            dist = histogramdd(next_state, bins=bins)
            hists.append(dist)

    def do_final_task(self):

        return self.all_dists

class MeasureWorkDone(SimProcedure):
    """Written by Edward. Update using the method written by Kyle."""

    def __init__(self, get_dW, output_name='work_done', step_request=s_[:], trial_request = s_[:], system = None):

        # self.get_dvalue = get_dvalue
        self.output_name = output_name
        self.step_request = step_request
        self.trial_request = trial_request
        self.get_val = get_dW
        


    def do_initial_task(self, simulation):
        self.simulation = simulation
        # ntrial =  simResult["cfqr"].sim.initial_state.shape[0]
        # self.simulation.work_dist_time_array2 = zeros([simulation.ntrials, simulation.nsteps])
        self.simulation.work_dist_time_array = []
        self.simulation.work_statistic_array = empty([simulation.nsteps, 2])
        self.simulation.work_dist_array = zeros(simulation.ntrials)
        
        

    def do_intermediate_task(self):
        dW = self.get_val(self.simulation)        
        self.simulation.work_dist_array += dW
        current_step = self.simulation.current_step
        self.simulation.work_dist_time_array.append(np.copy(self.simulation.work_dist_array))
        self.simulation.work_statistic_array[current_step, :] = [np.mean(self.simulation.work_dist_array), np.std(self.simulation.work_dist_array)]
    
        # simulation.work_statistic_array[current_step] =

from scipy import optimize
import edward_tools.couple_flux_qubit_metrics as couple_flux_qubit_metrics

class MeasureWorkDoneWithOffset(SimProcedure):
    """Written by Edward. Update using the method written by Kyle."""

    def __init__(self, get_dW, output_name='work_done_offset', step_request=s_[:], trial_request = s_[:], protocol_time_index_array = None, system = None, measurement_params = None):
        # self.get_dvalue = get_dvalue
        self.output_name = output_name
        self.step_request = step_request
        self.trial_request = trial_request
        self.get_val = get_dW
        self.protocol_time_index_array = protocol_time_index_array
        self.monitor_work_dist_in_whole_process = measurement_params['monitor_work_dist_in_whole_process']
        self.applyOffset = measurement_params['applyOffset']

    def do_initial_task(self, simulation):
        print(self.step_request, self.trial_request)
        
        target_step_index = list(range(simulation.nsteps+1)[self.step_request])
        step_length = len(target_step_index)
         
        
        self.simulation = simulation
        self.mapping_state_1_to_state_2_dict = {"00": ["00", "10"], "01": ["00", "10"], "10": ["01", "11"], "11": ["01", "11"]}
        self.simulation.target_step_index = target_step_index
        self.simulation.work_statistic_array = empty([step_length, 2])
        self.simulation.work_dist_array = zeros(simulation.ntrials)
        self.simulation.work_dist_time_array = []
        # self.simulation.force_array = empty([len(self.protocol_time_index_array), simulation.ntrials, N_dim, 3])
        
        if self.monitor_work_dist_in_whole_process:
            self.simulation.work_dist_time_array_whole_process = zeros([simulation.ntrials, simulation.nsteps+1])
        
        
        
        """key step information"""
        N_dim = 4
        self.simulation.keyStep_all_states = empty([len(self.protocol_time_index_array), simulation.ntrials, N_dim, 2])
        self.simulation.keyStep_all_states[0] = self.simulation.initial_state
        
        self.simulation.keyStep_work_distribution = empty([len(self.protocol_time_index_array[1:]) + 1, simulation.ntrials ])
        self.simulation.keyStep_work_statistic = zeros([len(self.protocol_time_index_array[1:]) + 1, 2]) # this array is to hold the work done at the time step of the protocol list for calculation of work done in each key time step
        # self.simulation.minimum_point_information = empty([simulation.nsteps, 4, 3])
        
        self.simulation.fidelity_time_array = []
        self.simulation.minimum_point_energy = self.find_minimum_for_all_potential(0)
        

    def do_intermediate_task(self):
        current_step = self.simulation.current_step
        current_time = self.simulation.current_time
        current_minimum_energy = self.find_minimum_for_all_potential(current_time)
        
        dW = self.get_val(self.simulation)
        
        if self.applyOffset:
            self.simulation.work_dist_array +=  dW - (current_minimum_energy - self.simulation.minimum_point_energy)
        else:
            self.simulation.work_dist_array +=  dW
            
        self.simulation.minimum_point_energy = current_minimum_energy
        
        if self.monitor_work_dist_in_whole_process:
            self.simulation.work_dist_time_array_whole_process[:, current_step] = self.simulation.work_dist_array

        
        if current_step in self.protocol_time_index_array[1:]: # skip index = 0, the substage information
            _i = np.where(self.protocol_time_index_array[1:] == current_step)[0][0]
            self.simulation.keyStep_all_states [_i+1] = self.simulation.current_state
            self.simulation.keyStep_work_distribution[_i+1] = self.simulation.work_dist_array
            self.simulation.keyStep_work_statistic[_i+1, :] = [np.mean(self.simulation.work_dist_array), np.std(self.simulation.work_dist_array)]

        
        if current_step in self.simulation.target_step_index:
            _i = self.simulation.target_step_index.index(current_step)
            self.simulation.work_dist_time_array.append(np.copy(self.simulation.work_dist_array))
            self.simulation.work_statistic_array[_i, :] = [np.mean(self.simulation.work_dist_array), np.std(self.simulation.work_dist_array)]
            
            # measure the fidelity in the key target step
            initial_phi_1_phi_2 = self.simulation.initial_state[:, (0, 1), :]
            current_phi_1_phi_2   = self.simulation.current_state[:, (0, 1), :]
            fidelity = couple_flux_qubit_metrics.fidelityEvaluation(initial_phi_1_phi_2, current_phi_1_phi_2, self.mapping_state_1_to_state_2_dict)
            fidelity = [d['final_percentage'] for d in fidelity]
            self.simulation.fidelity_time_array.append(fidelity)
    
    def do_final_task(self):
        self.do_intermediate_task()
    
        
    def find_minimum_for_all_potential(self, _t, guess = [(-2, -2), (-2, 2), (2, -2), (2, 2)]):
        _params_at_t = self.simulation.system.protocol.get_params(_t)
        _phi_1x = 0
        _phi_2x = 0
        protocol_key = ['U0_1', 'U0_2', 'gamma_1', 'gamma_2', 'beta_1', 'beta_2', 'd_beta_1', 'd_beta_2', 
                'phi_1_x', 'phi_2_x', 'phi_1_dcx', 'phi_2_dcx', 'M_12', 'x_c']
        
        beta_1 = _params_at_t[4]
        beta_2 = _params_at_t[5]
        d_beta_1 = _params_at_t[6]
        d_beta_2 = _params_at_t[7]
        _phi_1x = _params_at_t[8]
        _phi_2x = _params_at_t[9]
        _phi_1dcx = _params_at_t[10]
        _phi_2dcx = _params_at_t[11]
        _M_12 = _params_at_t[12]
        
        _phi_1dc = _phi_1dcx
        _phi_2dc = _phi_2dcx
        _xi = 1 / (1 - _M_12**2)

        def Fcn(coord):
            _phi_1, _phi_2 = coord
            u1_1 = 1/2 * _xi * (_phi_1 - _phi_1x)**2
            u3_1 = beta_1 * np.cos(_phi_1) * np.cos(_phi_1dc/2)
            u4_1 = d_beta_1 * np.sin(_phi_1) * np.sin(_phi_1dc/2)

            u1_2 = 1/2 * _xi * (_phi_2 - _phi_2x)**2        
            u3_2 = beta_2 * np.cos(_phi_2) * np.cos(_phi_2dc/2)
            u4_2 = d_beta_2 * np.sin(_phi_2) * np.sin(_phi_2dc/2)

            u5 = _M_12 * _xi * (_phi_1 - _phi_1x) * (_phi_2 - _phi_2x)

            return u1_1 + u1_2 + u3_1 + u3_2 + u4_1 + u4_2 + u5
        
        result = np.empty([4, 3])
        
#         for _i, _g in enumerate(guess):
#             sol = optimize.fmin(Fcn, _g, disp=False) 
#             result[_i, (0, 1)] = sol
#             result[_i, 2] = self.simulation.system.potential.potential(sol[0], sol[1], _phi_1dcx, _phi_2dcx, _params_at_t)
        
        # sol = optimize.fmin(Fcn, guess[0], disp=False) 
        
        solution_set = [optimize.fmin(Fcn, _g, disp=False) for _g in guess]
        energy_set = [self.simulation.system.potential.potential(sol[0], sol[1], _phi_1dcx, _phi_2dcx, _params_at_t) for sol in solution_set]

        return np.min(energy_set)            
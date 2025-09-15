import sys, os

sys.path.append(os.path.expanduser('~/source'))
from quick_sim import setup_sim
import kyle_tools as kt
import numpy as np
from kyle_tools.multisim import SimManager, FillSpace
from sus.protocol_designer import *
from sus.library.potentials import even_1DW, even_1DW_lifed, fredkin_pot



sys.path.append(os.path.expanduser('~/source/simtools/'))
# from infoenginessims.api import *
from infoenginessims.simprocedures import basic_simprocedures as sp
from infoenginessims.simprocedures import running_measurements as rp
from infoenginessims.simprocedures import trajectory_measurements as tp
from infoenginessims.simprocedures.basic_simprocedures import ReturnFinalState


default_parameters = {'k':1, 'depth':5, 'location':1, 'beta':1., 'scale':1., 'dt':1/10_000, 'lambda':.01, 'N':10_000, 'target_work':None, 'k_rel':1}



class BitFlipRunner(SimManager):
    def __init__(self, name_func = [None, None], params=default_parameters):
        # self.potential = even_1DW
        self.potential = even_1DW_lifed
        self.params = params
        self.save_name = name_func
        self.has_velocity = True

        self.save_procs = [SaveParams(), SaveSimOutput(), SaveFinalWork()]

    def verify_param(self, key, val):
        keys = list(self.params.keys())
        objectives = ['{}>0'] * len(keys)
        obj_dict = {k:v for k,v in zip(keys, objectives)}
        obj_dict['hold'] = '0 <= {} < 1'


        return eval(obj_dict[key].format(val))


    def initialize_sim(self):
        key_list = ['depth', 'location', 'k', 'shifted']
        d, l, k, c = [self.params[k] for k in key_list]


        self.potential.default_params = [d/l**4, -4*d/l**2, c]
        self.eq_protocol = self.potential.trivial_protocol().copy()

        self.potential.default_params = [0, k, 0]
        self.protocol =  self.potential.trivial_protocol().copy()

        self.system = System(self.protocol, self.potential)
        self.system.has_velocity = self.has_velocity
        self.eq_system = System(self.eq_protocol, self.potential)
        self.eq_system.has_velocity= self.has_velocity
        self.system.protocol.normalize()
        self.system.protocol.time_stretch(np.pi/np.sqrt(k))
        print("from SimRunner.py, set_sim_attributes")
        self.set_sim_attributes()

        return

    def set_sim_attributes(self):
        self.init_state = self.eq_system.eq_state(self.params['N'], t=0, beta=self.params['beta'])

        #
        as_step = max(1, int((self.system.protocol.t_f/self.params['dt'])/500))



        self.procs = self.set_simprocs(as_step)

        print()
        print("The as_tep is", as_step)
        print("The dt is", self.params['dt'],)

        # edward added this, to override the 200 states only in all states.
        # self.procs[2] = sp.MeasurexAllState()


        # orignial
        if self.params['noise']:
            # old code. The last parameter in sim_params is equal to 1. The simulationis  under a noisy environment.
            sim_kwargs = {'damping':self.params['lambda'], 'temp':1/self.params['beta'], 'dt':self.params['dt'], 'procedures':self.procs
            }
        else:
            # new. The last parameter in sim_params is equal to 0 because this cna turn off the noise. This is for the case of with / without damping under a noiseless environment
            sim_kwargs = {'damping':self.params['lambda'], 'temp':1/self.params['beta'], 'dt':self.params['dt'],
            'procedures': self.procs,
            'sim_params': [1, 1, 0]
            }

        self.sim = setup_sim(self.system, self.init_state, **sim_kwargs)
        self.sim.reference_system = self.eq_system
        return


    def analyze_output(self):
        if not hasattr(self.sim.output, 'final_W'):
            final_state = self.sim.output.final_state
            init_state = self.sim.initial_state
            U0 = self.system.get_potential(init_state, 0) - self.eq_system.get_potential(init_state, 0)
            UF = self.eq_system.get_potential(final_state, 0) - self.system.get_potential(final_state, 0)
            final_W = U0 + UF
            setattr(self.sim.output, 'final_W', final_W)


    def set_simprocs(self, as_step):
        return [
            sp.ReturnFinalState(),
            # sp.TerminateOnMean(rp.get_time_constant_work, target=self.params['target_work'], step_request=np.s_[::as_step], output_name='all_W'),
            # sp.MeasureAllState(trial_request=np.s_[:200], step_request=np.s_[::as_step]),
            # tp.CountJumps(output_name='jump_trajectories', state_slice=np.s_[...,0]),
            ]

class FredkinRunner(BitFlipRunner):
    def __init__(self, name_func = [None, None], params=default_parameters):
        self.potential = fredkin_pot
        self.params = params
        self.save_name = name_func
        self.has_velocity = True

    def initialize_sim(self):
        key_list = ['depth', 'location', 'k']
        d, l, k = [self.params[k] for k in key_list]


        self.potential.default_params = [d/(l**4), -4*d/(l**2), 0, k]
        self.eq_protocol = self.potential.trivial_protocol().copy()

        self.potential.default_params[-1] = self.potential.default_params = [d/(l**4), -4*d/(l**2), 1, k]
        self.protocol =  self.potential.trivial_protocol().copy()

        self.system = System(self.protocol, self.potential)
        self.system.has_velocity = self.has_velocity
        self.eq_system = System(self.eq_protocol, self.potential)
        self.eq_system.has_velocity= self.has_velocity

        self.system.protocol.normalize()
        self.system.protocol.time_stretch(np.pi/np.sqrt(k))

        self.set_sim_attributes()
        return




class SaveSimLight():
    def run(self, SimManager):
        N = SimManager.params['N']
        sim_dict = {}
        sim_dict.update({'work_stats':SimManager.sim.output.work_stats})
        sim_dict.update({'all_W':SimManager.sim.output.all_W})
        final_W = SimManager.sim.output.final_W
        fluc_hist = ft_hist(final_W)
        counts, bins = np.histogram(final_W, bins=int(2*N**(1/3)))
        sim_dict.update({'ft_hist':fluc_hist,'work_hist':[bins,counts]})
        SimManager.save_dict.update({'sim_dict':sim_dict})


class SaveParams():
    def run(self, SimManager):
        SimManager.save_dict.update({'params':SimManager.params})

class SaveSimOutput():
    def run(self, SimManager):
        keys = ['final_state', 'all_state', 'all_W']
        vals = [getattr(SimManager.sim.output,item) for item in keys]
        sim_dict = { k:v for k,v in zip(keys, vals)}
        sim_dict.update({'init_state':SimManager.sim.initial_state, 'nsteps':SimManager.sim.nsteps})

        SimManager.save_dict.update({'sim_dict':sim_dict})

class SaveFinalWork():
    def run(self, SimManager):
        SimManager.save_dict.update({'final_W':SimManager.sim.output.final_W})

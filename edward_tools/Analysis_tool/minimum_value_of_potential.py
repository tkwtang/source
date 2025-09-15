import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import importlib, os, datetime
from edward_tools.coupled_fq_potential import coupled_flux_qubit_non_linear_approx_pot, coupled_flux_qubit_non_linear_approx_force
# def get_work_done_from_time_array(work_statistic_array, time_array):
# def find_minimum_of_mix_in_X(cfqr, _t, beta_1, beta_2, d_beta_1, d_beta_2, verbose = True):
# def find_minimum_for_all_potential(cfqr, _t, beta_1, beta_2, d_beta_1, d_beta_2, verbose = True, guess = [(0, 0)]):
# def find_minimum_for_all_potential_2(cfqr, _t, beta_1, beta_2, d_beta_1, d_beta_2, verbose = True, guess = [(0, 0)]):
# def traceWorkDoneForOneParticle():
# def plot_work_dist_with_time(work_dist_time_array):



def get_work_done_from_time_array(work_statistic_array, time_array):
    pass

["four_well", "four_well", "mix_in_y", "conditional_tilt_in_y", "conditional_tilt_in_x", "mix_in_x", "four_well", "four_well"]




def obtain_minimum_points(cfqr, potential_type_and_time_array, beta_1, beta_2, d_beta_1, d_beta_2, verbose = False):
    result = []
    for (t, potential_type) in potential_type_and_time_array:
        if potential_type == "four_well":
            result.append(find_minimum_of_four_well(cfqr, t, beta_1, beta_2, d_beta_1, d_beta_2, verbose))
        if potential_type == "mix_in_y":
            result.append(find_minimum_of_mix_in_Y(cfqr, t, beta_1, beta_2, d_beta_1, d_beta_2, verbose))
        if potential_type == "conditional_tilt_in_y":
            result.append(find_minimum_of_conditional_tilt_in_Y(cfqr, t, beta_1, beta_2, d_beta_1, d_beta_2, verbose))
        if potential_type == "raise_barrier":
            result.append(find_minimum_of_raise_barrier(cfqr, t, beta_1, beta_2, d_beta_1, d_beta_2, verbose))
        if potential_type == "conditional_tilt_in_x":
            result.append(find_minimum_of_conditional_tilt_in_X(cfqr, t, beta_1, beta_2, d_beta_1, d_beta_2, verbose))
        if potential_type == "mix_in_x":
            result.append(find_minimum_of_mix_in_X(cfqr, t, beta_1, beta_2, d_beta_1, d_beta_2, verbose))
    return result
    

def find_minimum_of_four_well(cfqr, _t, beta_1, beta_2, d_beta_1, d_beta_2, verbose = True):
    # 4 well potential
    _params_at_t = cfqr.system.protocol.get_params(_t)
    _phi_1dcx = _params_at_t[10]
    _phi_2dcx = _params_at_t[11]
    _mu_12 = _params_at_t[12]

    def Fcn_4_well(x):
        return [x[0] - beta_2 * np.sin(x[0]), 
                x[1] - beta_2 * np.sin(x[1])
        ]
    sol_1 = fsolve(Fcn_4_well, [2, 2])
    sol_2 = fsolve(Fcn_4_well, [2, -2])
    sol_3 = fsolve(Fcn_4_well, [-2, 2])
    sol_4 = fsolve(Fcn_4_well, [-2, -2])
    sol_5 = fsolve(Fcn_4_well, [0, 2])
    sol_6 = fsolve(Fcn_4_well, [0, -2])
    
    _params_at_t = cfqr.system.protocol.get_params(_t)
    _phi_1dcx = _params_at_t[10]
    _phi_2dcx = _params_at_t[11]
    _min_potential = cfqr.system.potential.potential(sol_3[0], sol_3[1], _phi_1dcx, _phi_2dcx, _params_at_t)

    
    
    if verbose:
        print(f"time: {_t}, _phi_1dcx: {_phi_1dcx}, _phi_2dcx: {_phi_2dcx}, _mu_12: {_mu_12}")
        print("4 wells locations: ", sol_1, sol_2, sol_3, sol_4)
        print("saddle points locations: ", sol_5, sol_6)
        print("The value of minimum potential is: ", cfqr.system.potential.potential(sol_1[0], sol_1[1], _phi_1dcx, _phi_2dcx, _params_at_t))
    return {"coordinate": sol_3, "min_E": _min_potential}
    


def find_minimum_of_mix_in_Y(cfqr, _t, beta_1, beta_2, d_beta_1, d_beta_2, verbose = True):
    _params_at_t = cfqr.system.protocol.get_params(_t)
    _phi_1dcx = _params_at_t[10]
    _phi_2dcx = _params_at_t[11]
    _mu_12 = _params_at_t[12]

    def Fcn_Mix_in_y(x):
        return [
              x[0] - beta_2 * np.sin(x[0]), 
              x[1] - beta_2 * np.sin(x[1]) * np.cos(_phi_2dcx/2) + d_beta_2 * np.cos(x[1]) * np.sin(_phi_2dcx/2)
        ]
    sol_1 = fsolve(Fcn_Mix_in_y, [2, 1.5])
    sol_2 = fsolve(Fcn_Mix_in_y, [-2, -1.5])
    _min_potential = cfqr.system.potential.potential(sol_2[0], sol_2[1], _phi_1dcx, _phi_2dcx, _params_at_t)
    
    if verbose:
        print(f"time: {_t}, _phi_1dcx: {_phi_1dcx}, _phi_2dcx: {_phi_2dcx}, _mu_12: {_mu_12}")
        print("minima locations: ", sol_1, sol_2)
        print("The value of minimum potential is: ", cfqr.system.potential.potential(sol_1[0], sol_1[1], _phi_1dcx, _phi_2dcx, _params_at_t))
    return {"coordinate": sol_2, "min_E": _min_potential}

def find_minimum_of_conditional_tilt_in_Y(cfqr, _t, beta_1, beta_2, d_beta_1, d_beta_2, verbose = True):
    _params_at_t = cfqr.system.protocol.get_params(_t)
    _phi_1dcx = _params_at_t[10]
    _phi_2dcx = _params_at_t[11]
    _mu_12 = _params_at_t[12]
    _xi = 1 / (1 - _mu_12**2)

    def Fcn_conditional_tilt_in_y(x):
        return [_xi * x[0] - beta_1 * np.sin(x[0]) + _mu_12 * _xi * x[1],
                _xi * x[1] - beta_2 * np.sin(x[1]) * np.cos(_phi_2dcx/2) + d_beta_2 * np.cos(x[1]) * np.sin(_phi_2dcx/2) 
                + _mu_12 * _xi * x[0]]

    sol_1 = fsolve(Fcn_conditional_tilt_in_y, [2.6, 2])
    sol_2 = fsolve(Fcn_conditional_tilt_in_y, [-2.6, -2])
    _min_potential = cfqr.system.potential.potential(sol_2[0], sol_2[1], _phi_1dcx, _phi_2dcx, _params_at_t)
    
    if verbose:
        print(f"time: {_t}, _phi_1dcx: {_phi_1dcx}, _phi_2dcx: {_phi_2dcx}, _mu_12: {_mu_12}")
        print("minima locations: ", sol_1, sol_2)
        print("The value of minimum potential is: ", cfqr.system.potential.potential(sol_1[0], sol_1[1], _phi_1dcx, _phi_2dcx, _params_at_t))

    return {"coordinate": sol_2, "min_E": _min_potential}

def find_minimum_of_raise_barrier(cfqr, _t, beta_1, beta_2, d_beta_1, d_beta_2, verbose = True):
    _params_at_t = cfqr.system.protocol.get_params(_t)
    _phi_1dcx = _params_at_t[10]
    _phi_2dcx = _params_at_t[11]
    _mu_12 = _params_at_t[12]
    _xi = 1 / (1 - _mu_12**2)
    print(_phi_1dcx, _phi_2dcx, _mu_12)

    def Fcn_raise_barrier(x):
        return [_xi * x[0] - beta_1 * np.sin(x[0]) + _mu_12 * _xi * x[1],
                _xi * x[1] - beta_2 * np.sin(x[1]) + _mu_12 * _xi * x[0]
               ]

    sol_1 = fsolve(Fcn_raise_barrier, [2.6, 2])
    sol_2 = fsolve(Fcn_raise_barrier, [-2.6, -2])
    _min_potential = cfqr.system.potential.potential(sol_2[0], sol_2[1], _phi_1dcx, _phi_2dcx, _params_at_t)
    
    if verbose:
        print(f"time: {_t}, _phi_1dcx: {_phi_1dcx}, _phi_2dcx: {_phi_2dcx}, _mu_12: {_mu_12}")
        print("minima locations: ", sol_1, sol_2)
        print("The value of minimum potential is: ", cfqr.system.potential.potential(sol_1[0], sol_1[1], _phi_1dcx, _phi_2dcx, _params_at_t))
    return {"coordinate": sol_2, "min_E": _min_potential}


def find_minimum_of_conditional_tilt_in_X(cfqr, _t, beta_1, beta_2, d_beta_1, d_beta_2, verbose = True):
    _params_at_t = cfqr.system.protocol.get_params(_t)
    _phi_1dcx = _params_at_t[10]
    _phi_2dcx = _params_at_t[11]
    _mu_12 = _params_at_t[12]
    _xi = 1 / (1 - _mu_12**2)

    def Fcn_conditional_tilt_in_x(x):
        return [
              _xi * x[0] - beta_1 * np.sin(x[0]) * np.cos(_phi_1dcx/2) + d_beta_1 * np.cos(x[0]) * np.sin(_phi_1dcx/2) 
                    + _mu_12 * _xi * x[1],
              _xi * x[1] - beta_1 * np.sin(x[1]) + _mu_12 * _xi* x[0],
        ]
    sol_1 = fsolve(Fcn_conditional_tilt_in_x, [2, 2])
    sol_2 = fsolve(Fcn_conditional_tilt_in_x, [-2, -2])
    _min_potential = cfqr.system.potential.potential(sol_1[0], sol_1[1], _phi_1dcx, _phi_2dcx, _params_at_t)

    if verbose:
        print(f"time: {_t}, _phi_1dcx: {_phi_1dcx}, _phi_2dcx: {_phi_2dcx}, _mu_12: {_mu_12}")
        print("minima locations: ", sol_1, sol_2)
        print("The value of minimum potential is: ", cfqr.system.potential.potential(sol_1[0], sol_1[1], _phi_1dcx, _phi_2dcx, _params_at_t))
    return {"coordinate": sol_1, "min_E": _min_potential}



# Mix in x
def find_minimum_of_mix_in_X(cfqr, _t, beta_1, beta_2, d_beta_1, d_beta_2, verbose = True):
    _params_at_t = cfqr.system.protocol.get_params(_t)
    _phi_1dcx = _params_at_t[10]
    _phi_2dcx = _params_at_t[11]
    _mu_12 = _params_at_t[12]
    _xi = 1 / (1 - _mu_12**2)

    def Fcn_Mix_in_x(x):
        return [
            _xi * x[0] - beta_1 * np.sin(x[0]) * np.cos(_phi_1dcx) + d_beta_1 * np.cos(x[0]) * np.sin(_phi_1dcx/2),
            _xi * x[1] - beta_2 * np.sin(x[1])
        ]
    sol_1 = fsolve(Fcn_Mix_in_x, [2, 2])
    sol_2 = fsolve(Fcn_Mix_in_x, [-2, 2])
    _min_potential = cfqr.system.potential.potential(sol_1[0], sol_1[1], _phi_1dcx, _phi_2dcx, _params_at_t)

    if verbose:
        print(f"time: {_t}, _phi_1dcx: {_phi_1dcx}, _phi_2dcx: {_phi_2dcx}, _mu_12: {_mu_12}")
        print("minimum locations:", sol_1, sol_2)
        print("The value of minimum potential is: ", cfqr.system.potential.potential(sol_1[0], sol_1[1], _phi_1dcx, _phi_2dcx, _params_at_t))
    return {"coordinate": sol_1, "min_E": _min_potential}


def find_minimum_for_all_potential(cfqr, _t, beta_1, beta_2, d_beta_1, d_beta_2, verbose = True, guess = [(0, 0)]):
    _params_at_t = cfqr.system.protocol.get_params(_t)
    _phi_1dcx = _params_at_t[10]
    _phi_2dcx = _params_at_t[11]
    _mu_12 = _params_at_t[12]
    _xi = 1 / (1 - _mu_12**2)
    
    def Fcn(x):
        return [
            _xi * x[0] - beta_1 * np.sin(x[0]) * np.cos(_phi_1dcx/2) + d_beta_1 * np.cos(x[0]) * np.sin(_phi_1dcx/2) 
                    + _mu_12 * _xi * x[1],
            _xi * x[1] - beta_2 * np.sin(x[1]) * np.cos(_phi_2dcx/2) + d_beta_2 * np.cos(x[1]) * np.sin(_phi_2dcx/2) 
                + _mu_12 * _xi * x[0]
        ]

    solution_set = [fsolve(Fcn, _g) for _g in guess]
    sol_1 = solution_set[0]
    
    if verbose:
        print(f"time: {_t}, _phi_1dcx: {_phi_1dcx}, _phi_2dcx: {_phi_2dcx}, _mu_12: {_mu_12}")
        print("critical point locations:", solution_set)
        print("The value of minimum potential is: ", cfqr.system.potential.potential(sol_1[0], sol_1[1], _phi_1dcx, _phi_2dcx, _params_at_t))
        


from scipy import optimize

def find_minimum_for_all_potential_2(cfqr, _t, guess = [(0, 0)]):
    _params_at_t = cfqr.system.protocol.get_params(_t)
    _phi_1x = 0
    _phi_2x = 0
    
    
    beta_1 = _params_at_t[4]
    beta_2 = _params_at_t[5]
    d_beta_1 = _params_at_t[6]
    d_beta_2 = _params_at_t[7]
    _phi_1x = _params_at_t[8]
    _phi_2x = _params_at_t[9]
    _phi_1dcx = _params_at_t[10]
    _phi_2dcx = _params_at_t[11]
    _mu_12 = _params_at_t[12]

    _phi_1dc = _phi_1dcx
    _phi_2dc = _phi_2dcx
    _xi = 1 / (1 - _mu_12**2)

    
    def Fcn(coord):
        _phi_1, _phi_2 = coord
        u1_1 = 1/2 * _xi * (_phi_1 - _phi_1x)**2
        u3_1 = beta_1 * np.cos(_phi_1) * np.cos(_phi_1dc/2)
        u4_1 = d_beta_1 * np.sin(_phi_1) * np.sin(_phi_1dc/2)
        
        u1_2 = 1/2 * _xi * (_phi_2 - _phi_2x)**2        
        u3_2 = beta_2 * np.cos(_phi_2) * np.cos(_phi_2dc/2)
        u4_2 = d_beta_2 * np.sin(_phi_2) * np.sin(_phi_2dc/2)
        
        u5 = _mu_12 * _xi * (_phi_1 - _phi_1x) * (_phi_2 - _phi_2x)
        
        return u1_1 + u1_2 + u3_1 + u3_2 + u4_1 + u4_2 + u5

    solution_set = [optimize.fmin(Fcn, _g, disp=False) for _g in guess]
    sol_1 = list(solution_set[0])
    _potential = coupled_flux_qubit_non_linear_approx_pot(sol_1[0], sol_1[1], _phi_1dcx, _phi_2dcx, _params_at_t)
    # _min_potential = cfqr.system.potential.potential(sol_1[0], sol_1[1], _phi_1dcx, _phi_2dcx, _params_at_t)
    return sol_1, _potential

def find_all_minimum_points_for_all_potential(cfqr, _t, guess = [(0, 0)]):
    _params_at_t = cfqr.system.protocol.get_params(_t)
    beta_1 = _params_at_t[4]
    beta_2 = _params_at_t[5]
    d_beta_1 = _params_at_t[6]
    d_beta_2 = _params_at_t[7]
    _phi_1x = _params_at_t[8]
    _phi_2x = _params_at_t[9]
    _phi_1dcx = _params_at_t[10]
    _phi_2dcx = _params_at_t[11]
    _mu_12 = _params_at_t[12]
    _xi = 1 / (1 - _mu_12**2)
    _phi_1dc = _phi_1dcx
    _phi_2dc = _phi_2dcx
    
    def Fcn(coord):
        _phi_1, _phi_2 = coord
        u1_1 = 1/2 * _xi * (_phi_1 - _phi_1x)**2
        u3_1 = beta_1 * np.cos(_phi_1) * np.cos(_phi_1dc/2)
        u4_1 = -d_beta_1 * np.sin(_phi_1) * np.sin(_phi_1dc/2)
        
        u1_2 = 1/2 * _xi * (_phi_2 - _phi_2x)**2        
        u3_2 = beta_2 * np.cos(_phi_2) * np.cos(_phi_2dc/2)
        u4_2 = -d_beta_2 * np.sin(_phi_2) * np.sin(_phi_2dc/2)
        u5 = _mu_12 * _xi * (_phi_1 - _phi_1x) * (_phi_2 - _phi_2x)
        
        return u1_1 + u1_2 + u3_1 + u3_2 + u4_1 + u4_2 + u5

    solution_set = [optimize.fmin(Fcn, _g, disp=False) for _g in guess]
    energy_set = [coupled_flux_qubit_non_linear_approx_pot(sol[0], sol[1], _phi_1dcx, _phi_2dcx, _params_at_t) for sol in solution_set]
    return {"solution_set": solution_set, "energy_set": energy_set}


def traceWorkDoneForOneParticle():
    selected_particles  = simResult["cfqr"].sim.output.all_state["states"][0, :, :, 0]
    N_step_number, _ = selected_particles.shape
    step_time_array = np.array(simResult["cfqr"].sim.output.all_state["step_indices"]) * params['dt']

    U_a = np.zeros( N_step_number-1) 
    U_b = np.zeros(N_step_number-1)


    def findPotential(cfqr, _t, state):
        # print(state)
        _params_at_t = cfqr.system.protocol.get_params(_t)
        def cfq_potential(state):
            phi_1, phi_2, phi_1dc, phi_2dc = state
            return coupled_flux_qubit_non_linear_approx_pot(phi_1, phi_2, phi_1dc, phi_2dc, _params_at_t)
        return cfq_potential(state)



    for i in range(1, len(step_time_array)):

        # U_a[i-1] = findPotential(simResult["cfqr"], step_time_array[i], selected_particles[:, i-1, :])
        U_a[i-1] = findPotential(simResult["cfqr"], step_time_array[i], selected_particles[i-1, :])
        U_b[i-1] = findPotential(simResult["cfqr"],  step_time_array[i-1], selected_particles[i-1, :])
        

def plot_work_dist_with_time(cfqr, skipNumber = 100):
    work_dist_time_array = cfqr.sim.work_dist_time_array
    step_time_array = np.array(cfqr.sim.output.all_state["step_indices"]) * params['dt']
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5,5)
    points = [(0.1, 0.5), (0.5, 0.5), (0.9, 0.5)]
    _skipped_work_dist_time_array = work_dist_time_array[::skipNumber]
    _skipped_step_time_array = step_time_array[::skipNumber]
    
    def animate(i):
        ax.clear()
        # Get the point from the points list at index i
        _dist = _skipped_work_dist_time_array[i]
        work_min = np.min(_dist) - 5
        work_max = np.max(_dist) + 5
        bins = np.linspace(work_min, work_max, 50)

        # Plot that point using the x and y coordinates
        ax.hist(_dist, bins = bins)
        ax.set_title(f"t = {_skipped_step_time_array[i]}")
        # Set the x and y axis to display a fixed range
    ani = FuncAnimation(fig, animate, frames=len(_skipped_work_dist_time_array),
                        interval=100, repeat=False)
    plt.close()
    return ani




def try_to_use_vectorization_function_to_find_work_done_for_N_particles():
    """
    try_to_use_vectorization_function_to_find_work_done_for_N_particles, but failed because it was too slow.
    """
    selected_particles  = simResult["cfqr"].sim.output.all_state["states"][::5, :, :, 0]
    N_particles, N_step_number, _ = selected_particles.shape
    step_time_array = np.array(simResult["cfqr"].sim.output.all_state["step_indices"]) * params['dt']
    U_a = np.zeros([N_particles, N_step_number-1]) 
    U_b = np.zeros([N_particles, N_step_number-1])
    def findPotential2(cfqr, _t, state):
        # print(state)
        _params_at_t = cfqr.system.protocol.get_params(_t)
        def cfq_potential(state):
            phi_1, phi_2, phi_1dc, phi_2dc = state
            return coupled_flux_qubit_non_linear_approx_pot(phi_1, phi_2, phi_1dc, phi_2dc, _params_at_t)
        vectorized_cfq_potential = np.vectorize(cfq_potential, signature='(4)->()')
        return cfq_potential(state)
    def findPotential2(cfqr, _t, state):
        # print(state)
        _params_at_t = cfqr.system.protocol.get_params(_t)

        def cfq_potential(state):
            phi_1, phi_2, phi_1dc, phi_2dc = state
            return coupled_flux_qubit_non_linear_approx_pot(phi_1, phi_2, phi_1dc, phi_2dc, _params_at_t)
        # vectorized_cfq_potential = np.vectorize(cfq_potential, signature='(4)->()')
        return np.apply_along_axis(cfq_potential, 1, state)
        # return cfq_potential(state)
    for i in range(1,len(step_time_array)):
        U_a[:, i-1] = findPotential2(simResult["cfqr"], step_time_array[i], selected_particles[:, i-1, :])
        U_b[:, i-1] = findPotential2(simResult["cfqr"],  step_time_array[i-1], selected_particles[:, i-1, :])
        
        
        
def draw_energy_break_down_curve():
    u1_1_array = []
    u2_1_array = []
    u3_1_array = []
    u4_1_array = []

    u1_2_array = []
    u2_2_array = []
    u3_2_array = []
    u4_2_array = []

    u5_array = []

    for t in time_array:
        minimum_value_data = find_minimum_of_mix_in_Y(cfqr, t,  beta_1, beta_2, d_beta_1, d_beta_2, verbose = False)
        minimum_coordinate = minimum_value_data["coordinate"]
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

    

def drawGraphForPotentialAtDifferentTime(cfqr, pot_function, protocol_time_array, ax_flatten = []):
    if len(ax_flatten) == 0:
        fig, ax = plt.subplots(len(protocol_time_array), 1, figsize = [8, 40])
        ax_flatten = ax.flatten()
        
        
    minimum_value_data_points_all = []
    manual_domain=[np.array([-5, -5]), np.array([5, 5])]
    modified_manual_domain = [(manual_domain[0][1], manual_domain[0][0]), (manual_domain[1][1], manual_domain[1][0])]
    resolution = 50
    contours = 20
    pick_value_for_minimum_points = [(0, 1), (0, 1), (0, 1), (0, 2), (0, 1), (0, 3), (0, 3), (0, 2)]
    pick_up_coordinate = [1, 1, 1, 1, 0, 0, 0, 0]
    protocol_key = ['U0_1', 'U0_2', 'gamma_1', 'gamma_2', 'beta_1', 'beta_2', 'd_beta_1', 'd_beta_2', 
                'phi_1x', 'phi_2x', 'phi_1xdc', 'phi_2xdc', 'mu_12', 'x_c']
    

    for _t_index, _t in enumerate(protocol_time_array):
        minimum_value_data = find_all_minimum_points_for_all_potential(cfqr, _t, guess=[(-2, -2), (-2, 2), (2, -2), (2, 2)])
        _params_at_t = cfqr.protocol.get_params(_t)
        beta_1 = _params_at_t[4]
        beta_2 = _params_at_t[5]
        d_beta_1 = _params_at_t[6]
        d_beta_2 = _params_at_t[7]
        _phi_1x = _params_at_t[8]
        _phi_2x = _params_at_t[9]
        _phi_1dcx = _params_at_t[10]
        _phi_2dcx = _params_at_t[11]
        _mu_12 = _params_at_t[12]
        _xi = 1 / (1 - _mu_12**2)
        slice_values = [0, 0, _phi_1dcx, _phi_2dcx]
        
        minimum_value_data['potential_value'] = [pot_function(x, y, _phi_1dcx, _phi_2dcx, _params_at_t) for x,y in minimum_value_data["solution_set"]]
        del minimum_value_data['energy_set']
        pick_value = pick_value_for_minimum_points[_t_index] # this is for getting the correct solution coordinate
        minimum_value_data['potential_value'] = [minimum_value_data['potential_value'][_i] for _i in pick_value]
        minimum_value_data['solution_set'] = [minimum_value_data['solution_set'][_i] for _i in pick_value]
        minimum_value_data_points_all.append(minimum_value_data)
            
        U, X_mesh = cfqr.system.lattice_with_self_defined_potential(_t, pot_function, resolution, axes=(0, 1),\
                                    manual_domain=modified_manual_domain, slice_values = slice_values)
        x_array, y_array = X_mesh
        
        _plt = ax_flatten[_t_index]
        _plt.contourf(x_array, y_array, U, contours)
        _plt.set_title(f"t = {protocol_time_array[_t_index]}")
        _plt.set_box_aspect(aspect=1)


        for sol, pot in zip(minimum_value_data["solution_set"], minimum_value_data["potential_value"]):
            if pick_up_coordinate[_t_index] == 1:
                _plt.hlines(sol[1], xmax=5, xmin=-5, colors='red')
                _plt.text(-7.5, sol[1]-0.35, f'U = {pot:.6g}', color = 'r')
            else:
                _plt.vlines(sol[0], ymax=5, ymin=-5, colors='red')
                _plt.text(sol[0]-0.15, 6.0, f'U = {pot:.6g}', color = 'r')

        # _plt.colorbar()


def drawScanGraphForPotentialAtDifferentTime(cfqr, pot_function, protocol_time_array):
    fig, ax = plt.subplots(len(protocol_time_array), 1, figsize = [8, 40])
    ax_flatten = ax.flatten()
        
        
    minimum_value_data_points_all = []
    manual_domain=[np.array([-5, -5]), np.array([5, 5])]
    modified_manual_domain = [(manual_domain[0][1], manual_domain[0][0]), (manual_domain[1][1], manual_domain[1][0])]
    resolution = 50
    contours = 20
    pick_value_for_minimum_points = [(0, 1), (0, 1), (0, 1), (0, 2), (0, 1), (0, 3), (0, 3), (0, 2)]
    pick_up_coordinate = [1, 1, 1, 1, 0, 0, 0, 0]
    protocol_key = ['U0_1', 'U0_2', 'gamma_1', 'gamma_2', 'beta_1', 'beta_2', 'd_beta_1', 'd_beta_2', 
                'phi_1x', 'phi_2x', 'phi_1xdc', 'phi_2xdc', 'mu_12', 'x_c']
    

    for _t_index, _t in enumerate(protocol_time_array):
        minimum_value_data = minimum_value_of_potential.find_all_minimum_points_for_all_potential(cfqr, _t, guess=[(-2, -2), (-2, 2), (2, -2), (2, 2)])
        _params_at_t = cfqr.protocol.get_params(_t)
        beta_1 = _params_at_t[4]
        beta_2 = _params_at_t[5]
        d_beta_1 = _params_at_t[6]
        d_beta_2 = _params_at_t[7]
        _phi_1x = _params_at_t[8]
        _phi_2x = _params_at_t[9]
        _phi_1dcx = _params_at_t[10]
        _phi_2dcx = _params_at_t[11]
        _mu_12 = _params_at_t[12]
        _xi = 1 / (1 - _mu_12**2)
        slice_values = [0, 0, _phi_1dcx, _phi_2dcx]
        
        minimum_value_data['potential_value'] = [pot_function(x, y, _phi_1dcx, _phi_2dcx, _params_at_t) for x,y in minimum_value_data["solution_set"]]
        del minimum_value_data['energy_set']
        pick_value = pick_value_for_minimum_points[_t_index] # this is for getting the correct solution coordinate
        minimum_value_data['potential_value'] = [minimum_value_data['potential_value'][_i] for _i in pick_value]
        minimum_value_data['solution_set'] = [minimum_value_data['solution_set'][_i] for _i in pick_value]
        minimum_value_data_points_all.append(minimum_value_data)
            
        U, X_mesh = cfqr.system.lattice_with_self_defined_potential(_t, pot_function, resolution, axes=(0, 1),\
                                    manual_domain=modified_manual_domain, slice_values = slice_values)
        x_array, y_array = X_mesh
        
        selected_U = U[np.all(np.logical_and(y_array > 0, y_array < 5), axis = 1)]
        selected_x = x_array[np.all(np.logical_and(y_array > 0, y_array < 5), axis = 1)]

        for _x, _U in zip(selected_x, selected_U):
            ax_flatten[_t_index].plot(_x, _U)

            
from edward_tools.visualization import separate_by_state
from matplotlib.animation import FuncAnimation, PillowWriter

def cutlineWithDistribution(cfqr):
    # from potential_investigation.ipynb
    all_state = simResult['cfqr'].sim.output.all_state['states']
    all_state = all_state[:, :, (0, 1), :]
    N, nsteps, N_dim = np.shape(all_state)[0], np.shape(all_state)[1], np.shape(all_state)[2]
    which_axes = [np.s_[..., i, 0] for i in range(N_dim)]
    x_array = [all_state[item] for item in which_axes]
    state_lookup = separate_by_state(all_state)
    opacity = min(1, 300/N)
    x, y = x_array[0], x_array[1]
    x_lim = (np.min(x), np.max(x))
    y_lim = (np.min(y), np.max(y))
    scat_kwargs = {'alpha':opacity, 'zorder':10}
    min_U_along_cutline = []
    fig, ax = plt.subplots(4, 4, figsize=[15, 15])
    ax_flatten = ax.flatten()

    for _i in range(0, 5):
        X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[_i], cutlineDirection = "v")
        plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_point_data[_i][0][0],\
                                contour_plt=ax_flatten[2*_i], cutline_plt=ax_flatten[2*_i + 1], contours = 20,
                                showGraph=True)
        min_U = np.min(U)
        scat = [ax_flatten[2 * _i].scatter(x[state_lookup[key], protocol_time_index_array[_i]], y[state_lookup[key], protocol_time_index_array[_i]], **scat_kwargs) for key in state_lookup]

    for _i in range(5, 8):
        X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[_i], cutlineDirection = "h")
        plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_point_data[_i][0][1],\
                                contour_plt=ax_flatten[2*_i], cutline_plt=ax_flatten[2*_i + 1], contours = 20,
                                showGraph=True)
        min_U = np.min(U)
        scat = [ax_flatten[2 * _i].scatter(x[state_lookup[key], protocol_time_index_array[_i]], y[state_lookup[key], protocol_time_index_array[_i]], **scat_kwargs) for key in state_lookup]


    
    
from scipy import optimize
def curriedModifiedFunction(t, result, cfqr, guess = [(-2, -2)]):
    beta_1 = result[4]
    beta_2 = result[5]
    d_beta_1 = result[6]
    d_beta_2 = result[7]
    _phi_1x = result[8]
    _phi_2x = result[9]
    _phi_1dcx = result[10]
    _phi_2dcx = result[11]
    _mu_12 = result[12]
    _xi = 1 / (1 - _mu_12**2)
    
    _phi_1dc = _phi_1dcx
    _phi_2dc = _phi_2dcx

    def Fcn(coord):
        _phi_1, _phi_2 = coord
        u1_1 = 1/2 * _xi * (_phi_1 - _phi_1x)**2
        u3_1 = beta_1 * np.cos(_phi_1) * np.cos(_phi_1dc/2)
        u4_1 = -d_beta_1 * np.sin(_phi_1) * np.sin(_phi_1dc/2)
        
        u1_2 = 1/2 * _xi * (_phi_2 - _phi_2x)**2        
        u3_2 = beta_2 * np.cos(_phi_2) * np.cos(_phi_2dc/2)
        u4_2 = -d_beta_2 * np.sin(_phi_2) * np.sin(_phi_2dc/2)
        
        u5 = _mu_12 * _xi * (_phi_1 - _phi_1x) * (_phi_2 - _phi_2x)
        
        return u1_1 + u1_2 + u3_1 + u3_2 + u4_1 + u4_2 + u5

    solution_set = [optimize.fmin(Fcn, _g, disp=False) for _g in guess]
    sol_1 = list(solution_set[0])
    _potential = coupled_flux_qubit_non_linear_approx_pot(sol_1[0], sol_1[1], _phi_1dcx, _phi_2dcx, result)
    
    
    # result[8] = - np.abs(d_beta_1 * np.sin(sol_1[1]) * np.sin(_phi_1dcx/2) / (_xi * sol_1[1]))# phi_1x
    # result[9] = - np.abs(d_beta_2 * np.sin(sol_1[0]) * np.sin(_phi_2dcx/2) / (_xi * sol_1[0]))# phi_2x
    result[8] = 0
    result[9] = 0
    return result

        
# for _p in cfqr.sim.system.protocol.protocol_array:
#     _p.modifiedFunction = curriedModifiedFunction



def tilt_phi_1(t, result):
    beta_1, beta_2       = result[4], result[5]
    d_beta_1, d_beta_2   = result[6], result[7]
    _phi_1x, _phi_2x     = result[8], result[9]
    _phi_1x, _phi_2x     = 0, 0
    _phi_1dcx, _phi_2dcx = result[10], result[11]
    _mu_12                = result[12]
    _phi_1dc, _phi_2dc   = _phi_1dcx, _phi_2dcx
    _xi = 1 / (1 - _mu_12**2)

    
    def Fcn(_phi_1):
        # _phi_1, _phi_2 = coord
        u1_1 = 1/2 * _xi * (_phi_1 - _phi_1x)**2
        u3_1 = beta_1 * np.cos(_phi_1) * np.cos(_phi_1dc/2)
        u4_1 = d_beta_1 * np.sin(_phi_1) * np.sin(_phi_1dc/2)
        
        # u1_2 = 1/2 * _xi * (_phi_2 - _phi_2x)**2        
        # u3_2 = beta_2 * np.cos(_phi_2) * np.cos(_phi_2dc/2)
        # u4_2 = d_beta_2 * np.sin(_phi_2) * np.sin(_phi_2dc/2)
        
        # u5 = _mu_12 * _xi * (_phi_1 - _phi_1x) * (_phi_2 - _phi_2x)
        return u1_1 + u3_1 + u4_1

        
    solution_set = [optimize.fmin(Fcn, _g, disp=False) for _g in [-2, 2]]
    solution_set = [sol[0] for sol in solution_set]
    phi_1_a, phi_1_b = abs(min(solution_set)), -abs(min(solution_set))
    
    correct_phi_1x = 1/2 * (phi_1_a + phi_1_b) - 1/(phi_1_a - phi_1_b) * \
                     (d_beta_1 * (np.sin(phi_1_a) - np.sin(phi_1_b)) * np.sin(_phi_1dc/2) -\
                     beta_1 * np.cos(_phi_1dc/2) * (np.cos(phi_1_a) - np.cos(phi_1_b)))
    result[8] = correct_phi_1x
    return result


def tilt_phi_2(t, result):
    beta_1, beta_2       = result[4], result[5]
    d_beta_1, d_beta_2   = result[6], result[7]
    _phi_1x, _phi_2x     = result[8], result[9]
    _phi_1x, _phi_2x     = 0, 0
    _phi_1dcx, _phi_2dcx = result[10], result[11]
    _mu_12                = result[12]
    _phi_1dc, _phi_2dc   = _phi_1dcx, _phi_2dcx
    _xi = 1 / (1 - _mu_12**2)

    
    def Fcn(_phi_2):
        # _phi_1, _phi_2 = coord
        # u1_1 = 1/2 * _xi * (_phi_1 - _phi_1x)**2
        # u3_1 = beta_1 * np.cos(_phi_1) * np.cos(_phi_1dc/2)
        # u4_1 = d_beta_1 * np.sin(_phi_1) * np.sin(_phi_1dc/2)
        
        u1_2 = 1/2 * _xi * (_phi_2 - _phi_2x)**2        
        u3_2 = beta_2 * np.cos(_phi_2) * np.cos(_phi_2dc/2)
        u4_2 = d_beta_2 * np.sin(_phi_2) * np.sin(_phi_2dc/2)
        
        # u5 = _mu_12 * _xi * (_phi_1 - _phi_1x) * (_phi_2 - _phi_2x)
        return u1_2 + u3_2 + u4_2

        
    solution_set = [optimize.fmin(Fcn, _g, disp=False) for _g in [-2, 2]]
    solution_set = [sol[0] for sol in solution_set]
    phi_2_a, phi_2_b = abs(min(solution_set)), -abs(min(solution_set))
    
    correct_phi_2x = 1/2 * (phi_2_a + phi_2_b) - 1/(phi_2_a - phi_2_b) * \
                     (d_beta_2 * (np.sin(phi_2_a) - np.sin(phi_2_b)) * np.sin(_phi_2dc/2) -\
                     beta_2 * np.cos(_phi_2dc/2) * (np.cos(phi_2_a) - np.cos(phi_2_b)))
    result[9] = correct_phi_2x
    return result
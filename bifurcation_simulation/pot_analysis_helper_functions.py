from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation, PillowWriter
circuit_params = {
    "beta": 2.3, "d_beta": 0,
    "phi_1x": 0,   "phi_2x": 0, "phi_1xdc": 0, "phi_2xdc": 0, "M_12": 0
}


def pot_function(circuit_params=circuit_params):
    beta_1 = circuit_params["beta"]
    beta_2 = circuit_params["beta"]
    d_beta_1 = circuit_params["d_beta"]
    d_beta_2 = circuit_params["d_beta"]
    _phi_1x = circuit_params["phi_1x"]
    _phi_2x = circuit_params["phi_2x"]
    _phi_1xdc = circuit_params["phi_1xdc"]
    _phi_2xdc = circuit_params["phi_2xdc"]
    _M_12 = circuit_params["M_12"]
    _xi = 1
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

    return Fcn



def get_XYU(circuit_params, domain, resolution):
    """
    To get the 2D data of the potential for contour plots
    """
    x_vec = np.linspace(domain[0], domain[1], resolution)
    X, Y = np.meshgrid(x_vec, x_vec)
    U = pot_function(circuit_params)([X, Y])
    return X, Y, U



def first_derivative_of_pot_function(circuit_params=circuit_params):
    beta_1 = circuit_params["beta"]
    beta_2 = circuit_params["beta"]
    d_beta_1 = circuit_params["d_beta"]
    d_beta_2 = circuit_params["d_beta"]
    _phi_1x = circuit_params["phi_1x"]
    _phi_2x = circuit_params["phi_2x"]
    _phi_1xdc = circuit_params["phi_1xdc"]
    _phi_2xdc = circuit_params["phi_2xdc"]
    _M_12 = circuit_params["M_12"]
    _xi = 1
    _phi_1dc = _phi_1xdc
    _phi_2dc = _phi_2xdc
    
    def Fcn(x):
        return [
            _xi * (x[0] - _phi_1x) - beta_1 * np.sin(x[0]) * np.cos(_phi_1xdc/2) + d_beta_1 * np.cos(x[0]) * np.sin(_phi_1xdc/2) 
                    + _M_12 * _xi * (x[1] - _phi_2x),
            _xi * (x[1] - _phi_2x) - beta_2 * np.sin(x[1]) * np.cos(_phi_2xdc/2) + d_beta_2 * np.cos(x[1]) * np.sin(_phi_2xdc/2) 
                    + _M_12 * _xi * (x[0] - _phi_1x)
        ]
    return Fcn



def find_all_minimum_points_for_all_potential(circuit_params, guess = [(0, 0)]):
    """"
    To find all the maximum points, of the potential function
    """
    solution_set = [optimize.fmin(pot_function(circuit_params), _g, disp=False) for _g in guess]
    # energy_set = [coupled_flux_qubit_non_linear_approx_pot(sol[0], sol[1], _phi_1dcx, _phi_2dcx, _params_at_t) for sol in solution_set]
    return {"coord": solution_set}



def find_all_critical_points_for_all_potential(circuit_params, guess = [(0, 0)]):
    """"
    To find all the critical points, including the saddle, minimum and maximum points, of the potential function
    """
    solution_set = [optimize.fsolve(first_derivative_of_pot_function(circuit_params), _g) for _g in guess]
    # energy_set = [coupled_flux_qubit_non_linear_approx_pot(sol[0], sol[1], _phi_1dcx, _phi_2dcx, _params_at_t) for sol in solution_set]
    return {"coord": solution_set}



def barrier_height_searching(circuit_params = circuit_params, sweep_setting = {"parameter": "M_12", "range": np.linspace(0, 10, 20)}):
    """
    return the barrier height
    """
    result_array = []
    for _p in sweep_setting['range']:
        circuit_params[sweep_setting['parameter']] = _p
        critical_dict = find_all_critical_points_for_all_potential(circuit_params, 
                                            guess = [(-2, -2), (-2, 2), (2, -2), (2, 2), (-2,0), (0, -2), (2, 0), (0, 2)])
        critical_points = list(critical_dict.values())[0]
        critical_potential = [pot_function(circuit_params)([x, y]) for x, y in critical_points]
        delta_U00 = critical_potential[1] - critical_potential[0]
        delta_U11 = critical_potential[6] - critical_potential[3]
        delta_U10 = critical_potential[6] - critical_potential[2]
        delta_U01 = critical_potential[5] - critical_potential[4]
    
        X, Y, U = get_XYU(circuit_params, [-4, 4], 50)
        
        result_array.append({
            "coord": critical_points, 
             "barrier_height": {
                 "delta_U00": delta_U00, "delta_U01": delta_U01, 
                 "delta_U10": delta_U10, "delta_U11": delta_U11
            },
            "contour_data": {"X": X, "Y": Y, "U": U}
        })
        
    return result_array


""""
Functions for cutline plot
"""
cutline_setting = {"value": 0, "direction": "v", "color": "red"}
def plotCutline(cutline_setting, ax_setting):
    """
    plot a cutline with a given ax object
    """
    ax, h_axis_label, v_axis_label, X, Y, U = ax_setting.values()
    
    if cutline_setting["direction"] == "v": # for the case of verticle cutline
        target = X # this variable indicate the column for which the value of potential and the h_a
        target_with_cutline_value_inserted = np.sort(np.append(target[0], cutline_setting["value"]))
        target_index = target_with_cutline_value_inserted.tolist().index(cutline_setting["value"])
        v_axis = U[target_index]
        h_axis = Y.T[target_index]
    else:
        target = Y.T # this variable indicate the column for which the value of potential and the h_a
        target_with_cutline_value_inserted = np.sort(np.append(target[0], cutline_setting["value"]))
        target_index = target_with_cutline_value_inserted.tolist().index(cutline_setting["value"])
        v_axis = U[target_index]
        h_axis = X[target_index]
        
    line = ax.plot(h_axis, v_axis, color = cutline_setting['color'])[0]
    ax.set_xlabel(h_axis_label)
    ax.set_ylabel(v_axis_label)
    return line



def easy_plot_function(circuit_params, cutline_data_array, savePath = None):
    """
    This function is exactly the same as the following example: 
        2024/10/1 Example: plot cutlines on contour plot and show the potential of the cutline
    """
    fig, ax = plt.subplots(1,2, figsize = [8, 4])
    ax[0].set_xlim(-3, 3)
    ax[0].set_ylim(-3, 3)
    ax[0].set_aspect(1)

    # get the XYU
    resolution = 50
    print(circuit_params)
    X, Y, U = get_XYU(circuit_params, domain=[-4, 4], resolution=50)


    # plot contour_plot
    contour_plot = ax[0].contourf(X, Y, U, 40, vmin = 1, vmax = 6.5)


    # plot the critical points
    critical_dict = find_all_critical_points_for_all_potential(circuit_params, guess = [(-2, -2), (-2, 2), (2, -2), (2, 2), (-2,0), (0, -2), (2, 0), (0, 2)])
    critical_points = list(critical_dict.values())[0]
    critical_potential = [pot_function(circuit_params)([x, y]) for x, y in critical_points]
    critical_points_color = ["green", "red", "yellow", "orange", "white", "white", "white", "white"]

    for (x, y), color in zip(critical_points, critical_points_color):
        ax[0].plot(x, y, marker = "d", markersize = 8, color = color)
        print(f"{color} = ({x:.3g}, {y:.3g})")
    # fig.colorbar(ScalarMappable(norm = contour_plot_2.norm, cmap=contour_plot_2.cmap))


    # plot cutlines
    ax_setting = {"ax": ax[1], "h_axis_label": r"$\varphi_2$", "v_axis_label": r"U", "X": X, "Y": Y, "U": U}

    for cutline_setting in cutline_data_array:
        # plot cutline on the right graph
        plotCutline(cutline_setting, ax_setting)

        # plot cutline on the contour graph
        if cutline_setting['direction'] == "v":
            ymin, ymax = ax[0].set_ylim()
            ax[0].vlines(cutline_setting['value'], ymin, ymax, color = cutline_setting['color'])
        else:
            xmin, xmax = ax[0].set_ylim()
            ax[0].hlines(cutline_setting['value'], xmin, xmax, color = cutline_setting['color'])
    
    if savePath:
        fig.savefig(savePath)

    # plot the cutline on the contour graph

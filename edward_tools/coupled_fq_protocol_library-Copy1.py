import numpy as np
import sys
import os
from .fq_potential import fq_pot, fq_default_param
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
import matplotlib.pyplot as plt
from edward_tools.visualization import separate_by_state
from matplotlib.animation import FuncAnimation, PillowWriter


protocol_key = ['U0_1', 'U0_2', 'gamma_1', 'gamma_2', 'beta_1', 'beta_2', 'd_beta_1', 'd_beta_2', 'phi_1_x', 'phi_2_x', 'phi_1_dcx', 'phi_2_dcx', 'M_12', 'x_c']

def create_simple_protocol_parameter_dict(protocol_array):
    """
    simple means that the protocol doesn't have substeps.
    """
    result_dict = {}
    for i, k in enumerate(protocol_key):
        result_dict[k] = [protocol_array[i], protocol_array[i]]
    result_dict["t"] = [0, 1]
    return result_dict

def create_system(protocol_parameter_dict, domain = None, modifiedFunction = None):
    """
    This function is used to produce the storage and computation protocol

    input:
    1. comp_protocol_parameter_dict:
    - a dictionary contains the an array of time, which represents the time point at which the protocol is changed
    - the key is the name of the parameter
    - for parameters, they are arrays containing the value of the parameter at the particular time point

    2. modifiedFunction:
    - To modify the simple linear parametrization of the value
    - You can use more complex method to get the values such as calculate the value of phi_1x based on the value of other parameters
    
    output:
    1. comp_prototocl: the protocol for the computation system
    2. storage_protocol: : the protocol for the equilibrium system
    """
    # storage protocol, just take the first and last element of each term in the protocol key from the dict to form the storage protocol
    storage_t = (protocol_parameter_dict["t"][0], protocol_parameter_dict["t"][-1])
    storage_protocol_parameter_time_series = [np.array([protocol_parameter_dict[key][0], protocol_parameter_dict[key][-1]]) for key in protocol_key]
    storage_protocol_parameter_time_series = np.array(storage_protocol_parameter_time_series)
    storage_protocol = Protocol(storage_t, storage_protocol_parameter_time_series)
    storage_protocol.modifiedFunction = None
    # computation protocol, this part form the time series of the comp_protocol and join them to form Compound_Protocol
    comp_protocol_array = []
    comp_t = protocol_parameter_dict["t"]
    comp_protocol_parameter_time_series = [protocol_parameter_dict[key] for key in protocol_key]
    comp_protocol_parameter_time_series = np.array(comp_protocol_parameter_time_series).T

    for i in range(len(comp_t)-1):
        n_th_comp_time_array = (comp_t[i], comp_t[i+1])
        n_th_comp_protocol_parameter_array = np.array([comp_protocol_parameter_time_series[i], comp_protocol_parameter_time_series[i+1]]).T # in the form of array of [(p_n_i, p_n_f)]
        _p = Protocol(n_th_comp_time_array, n_th_comp_protocol_parameter_array)
        
        _p.modifiedFunction = modifiedFunction
        comp_protocol_array.append(_p)
    comp_protocol = Compound_Protocol(comp_protocol_array)
    comp_protocol.protocol_array  = comp_protocol_array

    return storage_protocol, comp_protocol


def create_system_from_storage_and_computation_protocol(storage_protocol_parameter_dict = None, comp_protocol_parameter_dict = None, domain = None):
    """
    This function is used to produce the storage and computation protocol

    input:
    1. input_parameters_dict:
    - a dictionary contains the an array of time, which represents the time point at which the protocol is changed
    - the key is the name of the parameter
    - for parameters, they are arrays containing the value of the parameter at the particular time point

    output:
    1. storage_protocol:
    - the protocol for the equilibrium system

    2. comp_prototocl
    - the protocol for the computation system
    """
    if comp_protocol_parameter_dict == None:
        print("please give me comp_protocol")


    if storage_protocol_parameter_dict is not None:
        print("storage_protocol_parameter_dict is not None")

    # storage protocol
    storage_t = storage_protocol_parameter_dict["t"]
    storage_protocol_parameter_time_series = [storage_protocol_parameter_dict[key] for key in protocol_key]
    storage_protocol_parameter_time_series = np.array(storage_protocol_parameter_time_series)
    storage_protocol = Protocol(storage_t, storage_protocol_parameter_time_series)

    # computation protocol
    comp_protocol_array = []
    comp_t = comp_protocol_parameter_dict["t"]
    comp_protocol_parameter_time_series = [comp_protocol_parameter_dict[key] for key in protocol_key]
    comp_protocol_parameter_time_series = np.array(comp_protocol_parameter_time_series).T

    for i in range(len(comp_t)-1):
        n_th_comp_time_array = (comp_t[i], comp_t[i+1])
        n_th_comp_protocol_parameter_array = np.array([comp_protocol_parameter_time_series[i], comp_protocol_parameter_time_series[i+1]]).T # in the form of array of [(p_n_i, p_n_f)]
        _p = Protocol(n_th_comp_time_array, n_th_comp_protocol_parameter_array)
        comp_protocol_array.append(_p)
    comp_protocol = Compound_Protocol(comp_protocol_array)

    return storage_protocol, comp_protocol

def customizedProtocol(initial_values_dict, protocol_list, normalized = False):
    protocol_key_array = ['U0_1', 'U0_2', 'gamma_1', 'gamma_2', 'beta_1', 'beta_2', 'd_beta_1', \
                    'd_beta_2', 'phi_1_x', 'phi_2_x', 'phi_1_dcx', 'phi_2_dcx', 'M_12', 'x_c']

    protocol_parameter_dict = {key: [value] for key, value in initial_values_dict.items()}
    protocol_parameter_dict["t"] = [0.0]


    for item in protocol_list:
        # add the duration to the time entry of the protocol_parameter_dict
        # print("*" * 20)
        # print(protocol_list)
        # print("*" * 20)
        protocol_parameter_dict["t"].append(protocol_parameter_dict["t"][-1] + item["duration"])

        for key in protocol_key_array:
            if key in item.keys(): # to check which key is present in the protocol_list_item.
                protocol_parameter_dict[key].append(item[key])
            else:
                protocol_parameter_dict[key].append(protocol_parameter_dict[key][-1])

    if normalized:
        protocol_parameter_dict["t"] = np.array(protocol_parameter_dict["t"])/ np.max(protocol_parameter_dict["t"])

    return protocol_parameter_dict


# def sliceThroughPhi_dc(simRunner, time = None, axis1 = 0, axis2 = 1, contours=10, resolution = 200, manual_domain=None, slice_values = None, surface = False, cbar=False, numberOfColumns = 3, vmin = None, vmax = None, steps = None):
#



def get_potential_shot_at_different_t(simRunner, protocol_parameter_dict, timeStep = None, axis1 = 0, axis2 = 1, contours=10, resolution = 200, manual_domain=None, slice_values = None, surface = False, cbar=False, numberOfColumns = 3, vmin = None, vmax = None):
    # print(protocol_parameter_dict)
    # to figure out which parameter has changed, and which have not been changed.
    changing_parameter_key = [key for key, value in protocol_parameter_dict.items() \
                            if len(set(value)) != 1]

    # to create the title of each subplot

    if timeStep:
        timeSeries = np.arange(protocol_parameter_dict["t"][0], protocol_parameter_dict["t"][-1] + timeStep, timeStep)
        # timeSeries = np.arange(0, 1 + timeStep, timeStep)
        changing_parameter_dict = {}


        for key in changing_parameter_key:
            if key != "t":
                keyIndex = protocol_key.index(key)
                changing_parameter_dict[key] = [simRunner.system.protocol.get_params(t)[keyIndex] for t in timeSeries]

    else:
        timeSeries = protocol_parameter_dict["t"]
        changing_parameter_dict = {key: protocol_parameter_dict[key] for key in changing_parameter_key}

    # print(changing_parameter_dict)
    # create the subplot_title
    subplot_title_array = []
    for key, value in changing_parameter_dict.items():
        array = [f"{key}: {v:.3g}" for v in value]
        subplot_title_array.append(array)
    subplot_title_array = list(zip(*subplot_title_array))

    # to create the subgraph with correct number of rows
    numberOfPlots = len(timeSeries)
    offset = 0 if numberOfPlots % numberOfColumns == 0 else 1
    numberOfRows = numberOfPlots // numberOfColumns + offset

    fig, ax = plt.subplots(numberOfRows, numberOfColumns, figsize=(18,4 * numberOfRows))


    def drawParameterGraphs(fig, ax, vmin, vmax):
        # vmin, vmax = 0, 0
        modified_manual_domain = [(manual_domain[0][1], manual_domain[0][0]), (manual_domain[1][1], manual_domain[1][0])]
        plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

        for i, t in enumerate(timeSeries):
            row = i // numberOfColumns
            column = i % numberOfColumns
            phi_1_dcx_index = protocol_key.index('phi_1_dcx')
            phi_2_dcx_index = protocol_key.index('phi_2_dcx')
            phi_1_dc_i = simRunner.system.protocol.get_params(t)[phi_1_dcx_index]
            phi_2_dc_i = simRunner.system.protocol.get_params(t)[phi_2_dcx_index]
            slice_values = [0, 0, phi_1_dc_i, phi_2_dc_i]

            U, X_mesh = simRunner.system.lattice(t, resolution, axes=(0, 1), manual_domain=modified_manual_domain, slice_values = slice_values)

            if (i==0) and not vmin and not vmax:
                vmin = np.min(U)
                vmax = np.max(U)
            # U, X_mesh = simRunner.system.lattice(t, resolution, axes=(axis1, axis2), manual_domain=manual_domain, slice_values = slice_values)
            X = X_mesh[0]
            Y = X_mesh[1]
            x_min, x_max = np.min(X), np.max(X)
            y_min, y_max = np.min(Y), np.max(Y)

            if surface is False:
                # subplot = fig.add_subplot(row+1, numberOfColumns, column + 1)
                # out = subplot.contourf(X, Y, U, contours)
                if cbar:
                    plt.colorbar(out)

                # This part is to prevent index error
                if len(timeSeries) > numberOfColumns: # when the number of graph is more than one row
                    subplot = ax[row][column]
                elif len(timeSeries) <= numberOfColumns and len(timeSeries) > 1: # when the number of graph is just one row
                    subplot = ax[column]
                elif len(timeSeries) == 1: # when the number of graph is 1
                    subplot = ax
                subplot.set_aspect(1)

                if len(subplot_title_array) > 0:
                    # subplot.set_title(f"t = {t:.3g}, " + ", ".join(subplot_title_array[i]))
                    pass
                else:
                    pass
                    # subplot.set_title(f"t = {t:.3g}")
                subplot.tick_params(axis='x', labelsize=13 )

                subplot.set_xlabel(r"$\varphi_1$", fontsize=15)
                subplot.set_ylabel(r"$\varphi_2$", fontsize=15)
                out = subplot.contourf(X, Y, U, contours, vmin = vmin, vmax = vmax)

                # cfqr.system.protocol.get_params(0)
            if surface is True:

                ax = fig.add_subplot(row+1, numberOfColumns, column + 1, projection='3d')
                surf = ax.plot_surface(X, Y, U)
                # ax.set_title(", ".join(subplot_title_array[i]))

    drawParameterGraphs(fig, ax, vmin, vmax)
    plt.show()


def get_potential_shot_at_different_t_1D(simRunner, protocol_parameter_dict, timeStep = None, axis1 = 0, axis2 = 1, targetAxis = 0, cutlineDirection = "v", cutlineValue = 0, contours=10, resolution = 200, manual_domain=None, slice_values = None, surface = False, cbar=False, numberOfColumns = 3, vmin = None, vmax = None):
    """
    cutlineDirection: vertical or horizontal cutline
    cutlineValue: the position of the cutline
    for example, cutlineDirection = 'v' and cutlineValue = '3' mean the cutline is x = 3.

    resolution: resolution of the contour plot
    """

    # to figure out which parameter has changed, and which have not been changed.
    changing_parameter_key = [key for key, value in protocol_parameter_dict.items() \
                            if len(set(value)) != 1]

    # to create the title of each subplot
    plotResultArray = []

    if timeStep:
        timeSeries = np.arange(protocol_parameter_dict["t"][0], protocol_parameter_dict["t"][-1] + timeStep, timeStep)
        # timeSeries = np.arange(0, 1 + timeStep, timeStep)
        changing_parameter_dict = {}


        for key in changing_parameter_key:
            if key != "t":
                keyIndex = protocol_key.index(key)
                changing_parameter_dict[key] = [simRunner.system.protocol.get_params(t)[keyIndex] for t in timeSeries]

    else:
        timeSeries = protocol_parameter_dict["t"]
        changing_parameter_dict = {key: protocol_parameter_dict[key] for key in changing_parameter_key}

    # print(changing_parameter_dict)
    # create the subplot_title
    subplot_title_array = []
    for key, value in changing_parameter_dict.items():
        array = [f"{key}: {v:.3g}" for v in value]
        subplot_title_array.append(array)
    subplot_title_array = list(zip(*subplot_title_array))

    # to create the subgraph with correct number of rows
    numberOfPlots = len(timeSeries * 2)
    offset = 0 if numberOfPlots % numberOfColumns == 0 else 2
    numberOfRows = numberOfPlots // numberOfColumns + offset

    fig, ax = plt.subplots(numberOfRows, numberOfColumns, figsize=(18,4 * numberOfRows))


    def drawParameterGraphs(fig, ax, vmin, vmax):
        # vmin, vmax = 0, 0
        modified_manual_domain = [(manual_domain[0][1], manual_domain[0][0]), (manual_domain[1][1], manual_domain[1][0])]

        for i, t in enumerate(timeSeries):
            contour_row = 2 * (i // numberOfColumns)
            cutline_row = contour_row + 1
            column = i % numberOfColumns
            phi_1_dcx_index = protocol_key.index('phi_1_dcx')
            phi_2_dcx_index = protocol_key.index('phi_2_dcx')
            phi_1_dc_i = simRunner.system.protocol.get_params(t)[phi_1_dcx_index]
            phi_2_dc_i = simRunner.system.protocol.get_params(t)[phi_2_dcx_index]
            slice_values = [0, 0, phi_1_dc_i, phi_2_dc_i]

            U, X_mesh = simRunner.system.lattice(t, resolution, axes=(0, 1), manual_domain=modified_manual_domain, slice_values = slice_values)

            if (i==0) and not vmin and not vmax:
                vmin = np.min(U)
                vmax = np.max(U)
            # U, X_mesh = simRunner.system.lattice(t, resolution, axes=(axis1, axis2), manual_domain=manual_domain, slice_values = slice_values)
            X = X_mesh[0]
            Y = X_mesh[1]
            x_min, x_max = np.min(X), np.max(X)
            y_min, y_max = np.min(Y), np.max(Y)


            if surface is False:
                # This part is to prevent index error
                if len(timeSeries) > numberOfColumns: # when the number of graph is more than one row
                    contour_subplot = ax[contour_row][column]
                    cutline_subplot = ax[cutline_row][column]
                elif len(timeSeries) <= numberOfColumns and len(timeSeries) > 1: # when the number of graph is just one row
                    contour_subplot = ax[contour_row][column]
                    cutline_subplot = ax[cutline_row][column]

                if len(subplot_title_array) > 0:
                    contour_subplot.set_title(f"t = {t:.3g}" + ", ".join(subplot_title_array[i]))
                else:
                    contour_subplot.set_title(f"t = {t:.3g}")

                # subplot.set_aspect(1)
                plotResult = plotCutlines(X, Y, U, cutlineDirection = cutlineDirection, cutlineValue = cutlineValue, contour_plt = contour_subplot, cutline_plt = cutline_subplot, contours = contours, time = t)
                plotResult["parameters"] = subplot_title_array
                plotResultArray.append(plotResult)
                # subplot.show()
                # plt.scatter(plotAxis, targetU[0])
                # if cutlineDirection == "v":
                #     targetAxis = Y
                #     targetIndex = np.sum(np.mean(targetAxis, axis=1) < cutline) - 1
                #     plotAxis = X[targetIndex, :]
                # if cutlineDirection == "h":
                #     targetAxis = X
                #     targetIndex = np.sum(np.mean(targetAxis, axis=1) < cutline) - 1
                #     plotAxis = Y[:, targetIndex]
                # # print(X[targetIndex, :])
                # # out1 = subplot.contourf(X, Y, U, contours, vmin = vmin, vmax = vmax)
                #
                #
                # if cutlineDirection == "v":
                #     subplot.axvline(x=cutline, ymin=np.min(plotAxis), ymax=np.max(plotAxis))
                # if cutlineDirection == "h":
                #     subplot.axhline(y=cutline, xmin=np.min(plotAxis), xmax=np.max(plotAxis))
                #
                # # targetIndex = np.sum(np.mean(targetAxis, axis=1) < cutline) - 1
                # # subplot.set_aspect(0)
                # out2 = subplot.plot(plotAxis, U[targetIndex, :])

                # cfqr.system.protocol.get_params(0)

    drawParameterGraphs(fig, ax, vmin, vmax)
    plt.show()
    return plotResultArray


def get_potential_along_a_1D_cutline(simRunner, t = 0, cutlineDirection = "v", cutlineValue = 0, \
                                     resolution = 100, contours = 5, manual_domain = [[-5, -5], [5, 5]]):
    # This fuction return an object with the following structure
    # {
    #    "contourData": {"X_grid", "Y_grid", "U_grid"},
    #    "cutlineData": {"culineDirection", "targetAxis", "plotAxis"}
    # }
    # you can use this function to obtain the relevant data to plot the contour plot at time t and the
    # potential along the cutline

    phi_1_dcx_index = protocol_key.index('phi_1_dcx')
    phi_2_dcx_index = protocol_key.index('phi_2_dcx')
    phi_1_dc_i = simRunner.protocol.get_params(t)[phi_1_dcx_index]
    phi_2_dc_i = simRunner.protocol.get_params(t)[phi_2_dcx_index]
    slice_values = [0, 0, phi_1_dc_i, phi_2_dc_i]


    # Call the system lattice methods, this will return (1) the potential energy at each grid point,
    # (2) the grid in the format of X = [[row1], [row2], [row3], ...] and
    # Y = [[column1], [column2], [column3], ...]. The axis parameter tells which axes do we want to
    # use as the x and y .
    modified_manual_domain = [(manual_domain[0][1], manual_domain[0][0]), \
                              (manual_domain[1][1], manual_domain[1][0])]
    U, X_mesh = simRunner.system.lattice(t, resolution, axes=(0, 1),\
                                manual_domain=modified_manual_domain, slice_values = slice_values)
    X, Y = X_mesh[0], X_mesh[1]

    # 5. find vmin, vmax, x_min, x_max, y_min and y_max
    vmin, vmax = np.min(U), np.max(U)
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)

    # 6. call the plot cutline function
    # plotAxis = the axis that we want to be the variable , targetAxis means to fixed the value of that axis
    if cutlineDirection == "h":
        _plotAxis = X
        _targetAxis = Y
        _plotU = U

    if cutlineDirection == "v":
        _plotAxis = Y.T
        _targetAxis = X.T
        _plotU = U.T

    plotAxis = _plotAxis[0] # since all the rows are the same, it doesn't matter which one is chosen
    targetAxis = _targetAxis[:, 0] # to take out the first value of the variable axis.

    # to find out the resolution of the target axis
    targetRange = (_targetAxis[-1] - _targetAxis[-2])/2

    # to find out the index of the cutline that is the closest to the target value
    # e.g the resolution may be something like [-0.05, 0.00, 0.05, 0.010, ... ] and my target value is 0.04
    # I cannot find the index of 0.04 because it does not exist in the array
    # The best I can do is to find the cutline closest to my target value, which is 0.05 in this case
    targetIndex = np.where(np.abs(targetAxis - cutlineValue) <= targetRange)[0][0]

    targetU = _plotU[targetIndex]

    return X, Y, U, cutlineDirection, _targetAxis, _plotAxis

    # return  {
    #     "contourData": {"X_grid": X, "Y_grid": Y, "U_grid": U},
    #     "cutlineData": {"culineDirection": cutlineDirection, "targetAxis": _targetAxis, "plotAxis": _plotAxis}
    # }


def plotCutlines(X, Y, U, cutlineDirection = "v", cutlineValue = 0, cutlineColor = 'red', cutlineRange = None, contour_plt = plt, cutline_plt = plt, contours = 5, time = None, graph_title = None, showGraph = None, cmap = "afmhot", cutlineYLimit = None):
    """
    X = all the rows of the mesh
    Y = all the columns of the mesh

    cutlineDirection: vertical or horizontal cutline
    cutlineValue: the position of the cutline
    for example, cutlineDirection = 'v' and cutlineValue = '3' mean the cutline is x = 3.

    resolution: resolution of the contour plot
    """
    if cutlineDirection == "h":
        _plotAxis = X
        _targetAxis = Y
        _plotU = U

    if cutlineDirection == "v":
        _plotAxis = Y.T
        _targetAxis = X.T
        _plotU = U.T

    plotAxis = _plotAxis[0]

    targetAxis = np.mean(_targetAxis, axis = 1)
    # to find out the resolution of the target axis
    targetRange = (targetAxis[-1] - targetAxis[-2])/2
    targetIndex = np.where(np.abs(targetAxis - cutlineValue) <= targetRange)[0][0]
    targetU = _plotU[targetIndex]
    
    min_U = np.min(targetU)

    # print(targetAxis, targetIndex, targetU)
    if showGraph:
        # cont = contour_plt.contourf(X, Y, U,  contours)
        cont = contour_plt.contourf(X, Y, U, 40, vmin = 0, vmax = 400, cmap = "afmhot")

        if cutlineDirection == "h":
            contour_plt.hlines(y = _targetAxis[targetIndex], xmin = np.min(_plotAxis), xmax = np.max(_plotAxis), colors= cutlineColor)
        if cutlineDirection == "v":
            contour_plt.vlines(x = _targetAxis[targetIndex], ymin = np.min(_plotAxis), ymax = np.max(_plotAxis), colors= cutlineColor)
        # _plt.show()

        cutline_plt.plot(plotAxis, targetU, color = cutlineColor)
        if graph_title:
            cutline_plt.set_title(f"{graph_title}")
        else:
            cutline_plt.set_title(f"t = {time}, U_min: {min_U:.3g}")
            
        if cutlineRange:
            cutline_plt.set_xlim(cutlineRange)
        if cutlineYLimit:
            cutline_plt.set_ylim(cutlineYLimit[0], cutlineYLimit[1])
        
    else:
        plt.close()
    return {
        "contour_plot": {"X": X, "Y": Y, "U": U, "contours": contours, "time": time},
        "cutline_plot": {"plotAxis": plotAxis, "targetU": targetU, "time": time, "cutlineDirection": cutlineDirection, "cutlineValue": cutlineValue}
    }



def animate_sim_flux_qubit_with_cutline(cfqr, frame_skip=30, time_array=None, params = None, legend=True, slice_values = None, fig_ax=None,  
                                        save_path = None, save_dict = None, vmin = None, vmax = None, 
                                        cutlineInformation = [(-3, "v","red")], cutlineYLimit = None, pColor = None, opacity = None, **pot_kwargs):
    names = [r"$\varphi_1$", r"$\varphi_2$"]
    fig, ax = plt.subplots(1, 2, figsize=[10, 5])
    ax_flatten = ax.flatten()
    manual_domain=[np.array([-5, -5]), np.array([5, 5])]
    
    all_state = cfqr.sim.output.all_state['states']
    phi_1_dc = all_state[..., 2, 0]
    phi_2_dc = all_state[..., 3, 0]
    all_state = all_state[:, :, (0, 1), :]

    N, nsteps, N_dim = np.shape(all_state)[0], np.shape(all_state)[1], np.shape(all_state)[2]
    which_axes = [np.s_[..., i, 0] for i in range(N_dim)]
    x_array = [all_state[item] for item in which_axes]
    state_lookup = separate_by_state(all_state)
    
    if opacity == None:
        opacity = min(1, 300/N)
    
    time_array = time_array[::frame_skip]
    x = x_array[0][:,::frame_skip]
    y = x_array[1][:,::frame_skip]
    x_lim = (np.min(x), np.max(x))
    y_lim = (np.min(y), np.max(y))
    scat_kwargs = {'alpha':opacity, 'zorder':10}
    scat = [ax_flatten[0].scatter(x[state_lookup[key], 0], y[state_lookup[key], 0], c = pColor[key], **scat_kwargs) for key in state_lookup]
    # scat_1 = [ax_flatten[1].scatter(x[state_lookup[key], 0], y[state_lookup[key], 0], **scat_kwargs) for key in state_lookup]
    
    if manual_domain:
        x_lim = (manual_domain[0][0], manual_domain[1][0])
        y_lim = (manual_domain[0][1], manual_domain[1][1])
        ax_flatten[0].set(xlim=x_lim, ylim=y_lim, xlabel=names[0], ylabel=names[1])

    ax_flatten[0].set(xlim=x_lim, ylim=y_lim, xlabel=names[0], ylabel=names[1])
    ax_flatten[0].set_aspect(1)

    def animate(i):
        x_i = x[:, i]
        y_i = y[:, i]
        ax_flatten[1].clear()
        
        for item in cutlineInformation:
            _value, _direction, _color = item
            X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = time_array[i] * params['dt'], cutlineDirection = _direction)
        # U = U - np.min(U)
        
            plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = _value,\
                                    contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = 20,
                                    showGraph=True, cutlineColor = _color, cutlineRange = [-4, 4], cutlineYLimit = cutlineYLimit)
        
        
        
        ax_flatten[1].set_title("")
        ax_flatten[1].set_xlabel(r"$\varphi_2$")
        ax_flatten[1].set_ylabel(r"potential ($k_BT$)")
        for i, item in enumerate(state_lookup):
            scat[i].set_offsets(np.c_[x_i[state_lookup[item]], y_i[state_lookup[item]]])
            # ax_flatten[0].scatter(x_i[state_lookup[item]], y_i[state_lookup[item]])
            

    ani = FuncAnimation(fig, animate, interval=100, frames=len(time_array), blit=False)

    return ani, fig, ax
import numpy as np
import sys
import os
from .fq_potential import fq_pot, fq_default_param
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
import matplotlib.pyplot as plt
from edward_tools.visualization import separate_by_state
from matplotlib.animation import FuncAnimation, PillowWriter
import uuid

protocol_key = ['U0_1', 'U0_2', 'gamma_1', 'gamma_2', 'beta_1', 'beta_2', 'd_beta_1', 'd_beta_2', 'phi_1x', 'phi_2x', 'phi_1xdc', 'phi_2xdc', 'mu_12', 'x_c']

# [0: 'U0_1', 1: 'U0_2', 2: 'gamma_1', 3: 'gamma_2', 4: 'beta_1', 5: 'beta_2', 6: 'd_beta_1', 7: 'd_beta_2', 8: 'phi_1x', 9: 'phi_2x', 10: 'phi_1xdc', 11: 'phi_2xdc', 12: 'mu_12', 13: 'x_c']

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

    
    if modifiedFunction == None:
        modifiedFunction = [None for _ in range(len(comp_t)-1)]
    
    print(modifiedFunction)
    
    for i in range(len(comp_t)-1):
        n_th_comp_time_array = (comp_t[i], comp_t[i+1])
        n_th_comp_protocol_parameter_array = np.array([comp_protocol_parameter_time_series[i], comp_protocol_parameter_time_series[i+1]]).T # in the form of array of [(p_n_i, p_n_f)]
        _p = Protocol(n_th_comp_time_array, n_th_comp_protocol_parameter_array)
        
        _p.modifiedFunction = modifiedFunction[i]
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
                    'd_beta_2', 'phi_1x', 'phi_2x', 'phi_1xdc', 'phi_2xdc', 'mu_12', 'x_c']

    protocol_parameter_dict = {key: [value] for key, value in initial_values_dict.items()}
    protocol_parameter_dict["t"] = [0.0]


    for item in protocol_list:
        # add the duration to the time entry of the protocol_parameter_dict
        protocol_parameter_dict["t"].append(protocol_parameter_dict["t"][-1] + item["duration"])

        for key in protocol_key_array:
            if key in item.keys(): # to check which key is present in the protocol_list_item.
                protocol_parameter_dict[key].append(item[key])
            else:
                protocol_parameter_dict[key].append(protocol_parameter_dict[key][-1])

    if normalized:
        protocol_parameter_dict["t"] = np.array(protocol_parameter_dict["t"])/ np.max(protocol_parameter_dict["t"])

    return protocol_parameter_dict


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
            phi_1xdc_index = protocol_key.index('phi_1xdc')
            phi_2xdc_index = protocol_key.index('phi_2xdc')
            phi_1_dc_i = simRunner.system.protocol.get_params(t)[phi_1xdc_index]
            phi_2_dc_i = simRunner.system.protocol.get_params(t)[phi_2xdc_index]
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
            phi_1xdc_index = protocol_key.index('phi_1xdc')
            phi_2xdc_index = protocol_key.index('phi_2xdc')
            phi_1_dc_i = simRunner.system.protocol.get_params(t)[phi_1xdc_index]
            phi_2_dc_i = simRunner.system.protocol.get_params(t)[phi_2xdc_index]
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

    drawParameterGraphs(fig, ax, vmin, vmax)
    plt.show()
    return plotResultArray



def get_potential_along_a_1D_cutline_by_overriding_parameters(simRunner, parameter_data = None, resolution = 100, cutlineDirection = "v", cutlineValue = 0, manual_domain = [[-5, -5], [5, 5]]):
    if not parameter_data:
        parameter_data = {'U0_1': 373.743081504123, 'U0_2': 373.743081504123, 'gamma_1': 10, 'gamma_2': 10,
                          'beta_1': 2.3, 'beta_2': 2.3, 'd_beta_1': 0.0, 'd_beta_2': 0.0,
                          'phi_1x': 0, 'phi_2x': 0, 'phi_1xdc': 0, 'phi_2xdc': 0, 'mu_12': 0,
                          'x_c': 3.291059784019349e-16}
    
    
    slice_values = [0, 0, parameter_data["phi_1xdc"], parameter_data["phi_2xdc"]]
    
    modified_manual_domain = [(manual_domain[0][1], manual_domain[0][0]), \
                              (manual_domain[1][1], manual_domain[1][0])]
    
    U, X_mesh = simRunner.system.lattice(0, resolution, axes=(0, 1), manual_domain=modified_manual_domain, slice_values = slice_values, override_params = parameter_data.values())
    X, Y = X_mesh[0], X_mesh[1]
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

    return {"cutlineDirection": cutlineDirection, "targetU": targetU, "plotAxis": plotAxis }





def plotCutlines(X, Y, U, cutlineDirection = "v", cutlineValue = 0, cutlineColor = 'red', cutline_label = None, x_range = None, y_range = None, contour_range = [0, 400], contour_plt = plt, cutline_plt = plt, contours = 5, time = None, graph_title = None, cmap = "afmhot", cutlineYLimit = None, offset_potential = False, dynamicCutline = False, showGraph = None, saveGraph = None):
    """
    This function is used to plot the contour plot and the cutline after the function get_potential_along_a_1D_cutline is called.
    
    X = all the rows of the mesh
    Y = all the columns of the mesh

    cutlineDirection: vertical or horizontal cutline
    cutlineValue: the position of the cutline
    for example, cutlineDirection = 'v' and cutlineValue = '3' mean the cutline is x = 3.

    dynamicCutline: to set the color of the cutline = None so that they will not show up in the graph, we use another line to represent this line.
    
    resolution: resolution of the contour plot
    
    _plotAxis: the x axis in the potential vs coord graph
    _plotU: the y axis in the potential vs coord graph
    
    _targetAxis: the target value of the graph
    """
    if cutlineDirection == "h":
        _plotAxis = X
        _plotU = U
        
        _targetAxis = Y
        

    if cutlineDirection == "v":
        _plotAxis = Y.T
        _plotU = U.T
        
        _targetAxis = X.T
        
        

    plotAxis = _plotAxis[0]

    targetAxis = np.mean(_targetAxis, axis = 1)
    # to find out the resolution of the target axis
    targetRange = (targetAxis[-1] - targetAxis[-2])/2
    targetIndex = np.where(np.abs(targetAxis - cutlineValue) <= targetRange)[0][0]
    targetU = _plotU[targetIndex]
    
    min_U = np.min(targetU)
    if offset_potential:
        targetU = targetU - min_U
    # print(targetAxis, targetIndex, targetU)


    cutline_plt.plot(plotAxis, targetU, color = cutlineColor, label = cutline_label)

    if graph_title:
        cutline_plt.set_title(f"{graph_title}")
    else:
        cutline_plt.set_title(f"t = {time}, U_min: {min_U:.3g}")

    if x_range:
        cutline_plt.set_xlim(x_range)
    
    if y_range:
        cutline_plt.set_ylim(y_range)
        
    if cutlineYLimit:
        cutline_plt.set_ylim(cutlineYLimit[0], cutlineYLimit[1])
        
        
        # cont = contour_plt.contourf(X, Y, U,  contours)
    cont = contour_plt.contourf(X, Y, U, 40, vmin = contour_range[0], vmax = contour_range[1], cmap = "afmhot")

    cutlineColor = cutlineColor if not dynamicCutline else "None" 
    
    if cutlineDirection == "h":
        _line = contour_plt.plot([np.min(_plotAxis), np.max(_plotAxis)], [_targetAxis[targetIndex], _targetAxis[targetIndex]],  color= cutlineColor,  linewidth = 1)
        
    if cutlineDirection == "v":
        _line = contour_plt.plot([_targetAxis[targetIndex], _targetAxis[targetIndex]], [np.min(_plotAxis), np.max(_plotAxis)],  color= cutlineColor,  linewidth = 1)
    
    if saveGraph:
        plt.savefig(f"coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{uuid.uuid4().hex}_plotline_graph.png")
        
    if not showGraph:
        plt.close()

    return {
        "contour_plot": {"X": X, "Y": Y, "U": U, "contours": contours, "time": time},
        "cutline_plot": {"plotAxis": plotAxis, "targetU": targetU, "time": time, "cutlineDirection": cutlineDirection, "cutlineValue": cutlineValue, "meaning": "This is the potential along the cutline in the contour plot"},
        # "contour_cutline_data": {"cutlineDirection" = cutlineDirection, , "meaning": "this is the x/y coordinate of the cutline in the contour plot"}
        "_line":  _line,
        "ax0": contour_plt,
        "ax1": cutline_plt
    }

def get_contour_and_cutline_data_with_given_params(simRunner, params_dict, cutlineData = {"cutlineValue": 0, "cutlineDirection": "v", "color": "red"}, contourData = { "vmin": 0, "vmax": 10, "resolution": 100}, manual_domain = [[-5, -5], [5, 5]]):
    phi_1xdc_index = protocol_key.index('phi_1xdc')
    phi_2xdc_index = protocol_key.index('phi_2xdc')
    phi_1_dc_i = params_dict["phi_1xdc"]
    phi_2_dc_i = params_dict["phi_2xdc"]
    slice_values = [0, 0, phi_1_dc_i, phi_2_dc_i]
    
     # Call the system lattice methods, this will return (1) the potential energy at each grid point,
    # (2) the grid in the format of X = [[row1], [row2], [row3], ...] and
    # Y = [[column1], [column2], [column3], ...]. The axis parameter tells which axes do we want to
    # use as the x and y .
    modified_manual_domain = [(manual_domain[0][1], manual_domain[0][0]), \
                              (manual_domain[1][1], manual_domain[1][0])]
    
    resolution = contourData["resolution"]
    U, X_mesh = simRunner.system.lattice(0, resolution, axes=(0, 1),\
                                manual_domain=modified_manual_domain, slice_values = slice_values, override_params = params_dict.values())
    X, Y = X_mesh[0], X_mesh[1]

    # 5. find vmin, vmax, x_min, x_max, y_min and y_max
    vmin, vmax = np.min(U), np.max(U)
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    
     # 6. call the plot cutline function
    # plotAxis = the axis that we want to be the variable , targetAxis means to fixed the value of that axis
    if cutlineData["cutlineDirection"] == "h":
        cutline_x_axis = X
        targetAxis = Y
        cutline_y_axis = U

    if cutlineData["cutlineDirection"] == "v":
        cutline_x_axis = Y.T
        targetAxis = X.T
        cutline_y_axis = U.T

    cutline_x_axis = cutline_x_axis[0] # since all the rows are the same, it doesn't matter which one is chosen
    targetAxis = targetAxis[:, 0] # to take out the first value of the variable axis.

    # to find out the resolution of the target axis
    targetRange = (targetAxis[-1] - targetAxis[-2])/2

    # to find out the index of the cutline that is the closest to the target value
    # e.g the resolution may be something like [-0.05, 0.00, 0.05, 0.010, ... ] and my target value is 0.04
    # I cannot find the index of 0.04 because it does not exist in the array
    # The best I can do is to find the cutline closest to my target value, which is 0.05 in this case
    targetIndex = np.where(np.abs(targetAxis - cutlineData["cutlineValue"]) <= targetRange)[0][0]
    
    return {
        "contourPlot": {
            "X": X, "Y": Y, "U": U
        },
        "cutlinePlot": {
            "x_axis": cutline_x_axis, "y_axis": cutline_y_axis[targetIndex], 
            "cutlineDirection": cutlineData["cutlineDirection"], "cutlineValue": cutlineData["cutlineValue"], 
            "cutlineColor": cutlineData["color"]
        }
    }



def params_list_to_dict(params_list):
    return dict(zip(protocol_key, params_list))


def get_potential_along_a_1D_cutline(simRunner, t = 0, cutlineDirection = "v", cutlineValue = 0, \
                                     resolution = 100, contours = 5, manual_domain = [[-5, -5], [5, 5]]):
    # This fuction return an object with the following structure
    # {
    #    "contourData": {"X_grid", "Y_grid", "U_grid"},
    #    "cutlineData": {"culineDirection", "targetAxis", "plotAxis"}
    # }
    # plotAxis = the axis that we want to be the variable , targetAxis means to fixed the value of that axis
    # you can use this function to obtain the relevant data to plot the contour plot at time t and the
    # potential along the cutline

    phi_1xdc_index = protocol_key.index('phi_1xdc')
    phi_2xdc_index = protocol_key.index('phi_2xdc')
    phi_1_dc_i = simRunner.protocol.get_params(t)[phi_1xdc_index]
    phi_2_dc_i = simRunner.protocol.get_params(t)[phi_2xdc_index]
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


    
    


    

contourData = {
    "vmax": None, "vmin": None,
    "manual_domain": [np.array([-5, -5]), np.array([5, 5])],
    "contour_range": [0, 400],
    "title": None
}

cutlineInformation = {
    "cutlineList": [(-2, "v","red")],
    "cutlineGeneratingFunction": None,
    "cutlineXLimit": [-4,4],
    "cutlineYLimit": [50, 400]
}

particleInformation = {
    "showParticles": True,
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
    "key": ["phi_1x", "phi_2x", "phi_1xdc", "phi_2xdc", "mu_12"],
    "showGraph": False
}



def animate_sim_flux_qubit_with_cutline_and_projection(cfqr, time_array=None, state_array = None, params = None, legend=True, 
    plot_axis = [0, 1], slice_values = None, fig_ax=None, 
    contourData = contourData, cutlineInformation = cutlineInformation, particleInformation = particleInformation,
    animation_setting = animation_setting, protocol_graph_setting = protocol_graph_setting, 
    ax0_title = None, offset_potential = False, **pot_kwargs):
    """
    This is very similar to the function animate_sim_flux_qubit_with_cutline, but it can project particles onto the cutlines
    """
    
    names = [r"$\varphi_1$", r"$\varphi_2$", r"$\varphi_{1dc}$", r"$\varphi_{2dc}$"]
    
    if not fig_ax:
        fig, ax = plt.subplots(1, 3, figsize=[15, 5])
    else:
        fig, ax = fig_ax
    ax_flatten = ax.flatten()
    

    parms_at_init = cfqr.protocol.get_params(0)
    plot_axis = [0, 1]
    print("new animation function")
    
    
    # params_at_all_time = np.array([cfqr.protocol.get_params(_t) for _t in  time_array]).T
    # params_at_all_time_dict = dict(zip(cfqr.protocol_key, params_at_all_time))
    # for key, value in params_at_all_time_dict.items():
    #     if key in protocol_graph_setting['key']:
    #         ax_flatten[2].plot(cfqr.protocol_all_time_array, value, label = key)
    # ax_flatten[2].legend()
    
    if particleInformation['showParticles']:
        N, nsteps, N_dim = np.shape(state_array)[0], np.shape(state_array)[1], np.shape(state_array)[2]
        state_lookup = separate_by_state(state_array[:, :, (0, 1), :])

        phi_1 = state_array[..., 0, 0]
        phi_2 = state_array[..., 1, 0]
        phi_1dc = state_array[..., 2, 0]
        phi_2dc = state_array[..., 3, 0]
        particle_potential = cfqr.system.potential.potential(phi_1, phi_2, phi_1dc, phi_2dc, parms_at_init)
        scat_kwargs = {'alpha': particleInformation["particle_opacity"], 'zorder':10}
        scat = [ax_flatten[0].scatter(phi_1[state_lookup[key], 0], phi_2[state_lookup[key], 0], c = particleInformation['pColor'][key], **scat_kwargs) for key in state_lookup]

    if contourData['domain_limit']:
        phi_1_lim = (contourData['domain_limit'][0], contourData['domain_limit'][1])
        phi_2_lim = phi_1_lim
    else:
        phi_1_lim = (-4, 4)
        phi_2_lim = (-4, 4)

    _value_1, _direction_1, _color_1 = cutlineInformation['cutlineList'][0]
    _value_2, _direction_2, _color_2 = cutlineInformation['cutlineList'][1]
    
    X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = time_array[0], cutlineDirection = _direction_1)

    
    ax_flatten[0].set(xlim=phi_1_lim, ylim=phi_2_lim, xlabel=names[plot_axis[0]], ylabel=names[plot_axis[1]])
    ax_flatten[0].set_aspect(1)
    
    projection_mapping = {"phi_1": phi_1, "phi_2": phi_2, "phi_1dc": phi_1dc, "phi_2dc": phi_2dc}
    particle_project_axis = particleInformation['project_axis']
    phi_project = projection_mapping[particle_project_axis]
            
    def animate(i):
        parms_at_i = cfqr.protocol.get_params(time_array[i])
        ax_flatten[1].clear()
        ax_flatten[2].clear()
        
        if particleInformation['showParticles']:
            x_i = phi_1[:, i]
            y_i = phi_2[:, i]
            particle_potential_i = cfqr.system.potential.potential(phi_1[:, i], phi_2[:, i], phi_1dc[:, i], phi_2dc[:, i], parms_at_i)
        
            scat_2 = [] # hold the scatter plots of each type of particles
            scat_3 = []
            
            for key in ["00", "10"]:
                _scatter_1 = ax_flatten[1].scatter( phi_2[state_lookup[key], 0], particle_potential[state_lookup[key], 0], c = particleInformation['pColor'][key], **scat_kwargs) # scatter plot of a particular type of particles
                scat_2.append(_scatter_1)
            for key in ["01", "11"]:
                _scatter_2 = ax_flatten[2].scatter( phi_2[state_lookup[key], 0], particle_potential[state_lookup[key], 0], c = particleInformation['pColor'][key], **scat_kwargs) # scatter plot of a particular type of particles
                scat_3.append(_scatter_2)

        for _j, item in enumerate(cutlineInformation['cutlineList']):
            _value, _direction, _color = item
            
            X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = time_array[i], cutlineDirection = _direction)

            plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = _value, contour_plt=ax_flatten[0], cutline_plt=ax_flatten[_j+1], contours = 20,
                                    showGraph=True, cutlineColor = _color, x_range = [-4, 4], cutlineYLimit = cutlineInformation['cutlineYLimit'][_j], offset_potential = offset_potential, contour_range = contourData['contour_range'], time = time_array[i])
        
            
        particle_group_1 = ["00", "10"]
        pGroup = state_lookup.keys()
        particle_group_2 = ["01", "11"]
        params_at_t_i = cfqr.protocol.get_params(time_array[i])

        if particleInformation['showParticles']:
            for _i, pType in enumerate(pGroup):
                scat[_i].set_offsets(np.c_[x_i[state_lookup[pType]], y_i[state_lookup[pType]]])

        if scat_2:
            for _k, pType in enumerate(particle_group_1):
                scat_2[_k].set_offsets(np.c_[phi_project[:, i][state_lookup[pType]], particle_potential_i[state_lookup[pType]]])

        if scat_3:
            for _k, pType in enumerate(particle_group_2):
                scat_3[_k].set_offsets(np.c_[phi_project[:, i][state_lookup[pType]], particle_potential_i[state_lookup[pType]]])
                
    ani = FuncAnimation(fig, animate, interval=animation_setting['interval'], frames=len(time_array), blit=animation_setting['blit'])
    return ani, fig, ax
    


# def animate_sim_flux_qubit_with_cutline_and_projection(cfqr, time_array=None, params = None, legend=True, 
#     plot_axis = [0, 1], slice_values = None, fig_ax=None, 
#     contourData = contourData, cutlineInformation = cutlineInformation, particleInformation = particleInformation,
#     animation_setting = animation_setting, protocol_graph_setting = protocol_graph_setting, 
#     ax0_title = None, offset_potential = False, **pot_kwargs):
#     """
#     This is very similar to the function animate_sim_flux_qubit_with_cutline, but it can project particles onto the cutlines
#     """
    
#     names = [r"$\varphi_1$", r"$\varphi_2$", r"$\varphi_{1dc}$", r"$\varphi_{2dc}$"]
    
#     if not fig_ax:
#         fig, ax = plt.subplots(1, 2, figsize=[10, 5])
#     else:
#         fig, ax = fig_ax
#     ax_flatten = ax.flatten()
    


#     index_skip_in_all_time_array = int(animation_setting['frame_skip']/params['dt'])
#     time_array = cfqr.protocol_all_time_array[::index_skip_in_all_time_array]
#     parms_at_init = cfqr.protocol.get_params(0)
#     plot_axis = [0, 1]
    

#     params_at_all_time = np.array([cfqr.protocol.get_params(_t) for _t in  cfqr.protocol_all_time_array]).T
#     params_at_all_time_dict = dict(zip(cfqr.protocol_key, params_at_all_time))
#     # for key, value in params_at_all_time_dict.items():
#     #     if key in protocol_graph_setting['key']:
#     #         ax_flatten[2].plot(cfqr.protocol_all_time_array, value, label = key)
#     # ax_flatten[2].legend()
    
#     if particleInformation['showParticles']:
#         all_state = cfqr.sim.output.all_state['states']
#         N, nsteps, N_dim = np.shape(all_state)[0], np.shape(all_state)[1], np.shape(all_state)[2]
#         state_lookup = separate_by_state(all_state[:, :, (0, 1), :])

#         phi_1 = all_state[:,::animation_setting['frame_skip'], 0, 0]
#         phi_2 = all_state[:,::animation_setting['frame_skip'], 1, 0]
#         phi_1dc = all_state[:,::animation_setting['frame_skip'], 2, 0]
#         phi_2dc = all_state[:,::animation_setting['frame_skip'], 3, 0]
#         particle_potential = cfqr.system.potential.potential(phi_1, phi_2, phi_1dc, phi_2dc, parms_at_init)
#         scat_kwargs = {'alpha': particleInformation["particle_opacity"], 'zorder':10}
#         scat = [ax_flatten[0].scatter(phi_1[state_lookup[key], 0], phi_2[state_lookup[key], 0], c = particleInformation['pColor'][key], **scat_kwargs) for key in state_lookup]

#     if contourData['domain_limit']:
#         phi_1_lim = (contourData['domain_limit'][0], contourData['domain_limit'][1])
#         phi_2_lim = phi_1_lim
#     else:
#         phi_1_lim = (-4, 4)
#         phi_2_lim = (-4, 4)

#     _value, _direction, _color = cutlineInformation['cutlineList'][0]
#     X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = time_array[0], cutlineDirection = _direction)

    
#     ax_flatten[0].set(xlim=phi_1_lim, ylim=phi_2_lim, xlabel=names[plot_axis[0]], ylabel=names[plot_axis[1]])
#     ax_flatten[0].set_aspect(1)
    
    


#     def animate(i):
#         parms_at_i = cfqr.protocol.get_params(time_array[i])
#         ax_flatten[1].clear()
        
#         if particleInformation['showParticles']:
#             x_i = phi_1[:, i]
#             y_i = phi_2[:, i]
#             particle_potential_i = cfqr.system.potential.potential(phi_1[:, i], phi_2[:, i], phi_1dc[:, i], phi_2dc[:, i], parms_at_i)
        
#             scat_2 = [] # hold the scatter plots of each type of particles
#             for key in state_lookup:
#                 if key in particleInformation['project_item']:
#                     _scatter = ax_flatten[1].scatter(phi_1[state_lookup[key], 0], particle_potential[state_lookup[key], 0], c = particleInformation['pColor'][key], **scat_kwargs) # scatter plot of a particular type of particles
#                 else:
#                     _scatter = None
#                 scat_2.append(_scatter)
            
#         for item in cutlineInformation['cutlineList']:
#             _value, _direction, _color = item

#             X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = time_array[i], cutlineDirection = _direction)

#             plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = _value,\
#                                     contour_plt=ax_flatten[0], cutline_plt=ax_flatten[1], contours = 20,
#                                     showGraph=True, cutlineColor = _color, x_range = [-4, 4], cutlineYLimit = cutlineInformation['cutlineYLimit'], offset_potential = offset_potential, contour_range = contourData['contour_range'], time = time_array[i])
        
            
            
#         params_at_t_i = cfqr.protocol.get_params(time_array[i])

#         if particleInformation['showParticles']:
#             for _i, item in enumerate(state_lookup):
#                 scat[_i].set_offsets(np.c_[x_i[state_lookup[item]], y_i[state_lookup[item]]])

#                 if item in particleInformation['project_item'] and scat_2[_i]:
#                     scat_2[_i].set_offsets(np.c_[y_i[state_lookup[item]], particle_potential_i[state_lookup[item]]])
#     print(animation_setting['interval'])
#     ani = FuncAnimation(fig, animate, interval=animation_setting['interval'], frames=len(time_array), blit=animation_setting['blit'])
#     return ani, fig, ax
    
    

def generate_state_at_different_time():
    
    min_U_along_cutline = []
    fig, ax = plt.subplots(5, 4, figsize=[15, 18])
    ax_flatten = ax.flatten()
    cutlineRange = [-3.5, 3.5]
    pColor = {"00": "#061DF7", "01": "#FCEF51", "10": "#3FC7F2", "11": "#F187F4"}
    cutline_xrange = [-3.5, 3.5]
    cutline_yrange = [0, 200]

    X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[0], cutlineDirection = "v")
    plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_point_data[0][0][0],\
                            contour_plt=ax_flatten[2*0], cutline_plt=ax_flatten[2*0 + 1], contours = 20,
                            showGraph=True, time=protocol_time_array[0], offset_potential = True, x_range = cutline_xrange, y_range = cutline_yrange, graph_title = " ")
    min_U = np.min(U)
    scat = [ax_flatten[2 * 0].scatter(x[state_lookup[key], protocol_time_index_array[0]], y[state_lookup[key], protocol_time_index_array[0]], c = pColor[key],**scat_kwargs) for key in state_lookup]
    ax_flatten[2 * 0].set_xticks([-4, 0, 4])
    ax_flatten[2 * 0].set_yticks([-4, 0, 4])

    for _i in range(1, 5):
        X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[_i], cutlineDirection = "v")

        plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_point_data[_i][0][0],\
                                contour_plt=ax_flatten[2*_i], cutline_plt=ax_flatten[2*_i + 1], contours = 20,
                                showGraph=True, time=protocol_time_array[_i], offset_potential = True, x_range = cutline_xrange, y_range = cutline_yrange, graph_title = " ")

        _index = list(simResult['cfqr'].sim.target_step_index).index(protocol_time_index_array[_i]+1)
        scat = [ax_flatten[2 * _i].scatter(x[state_lookup[key], _index], y[state_lookup[key], _index], c = pColor[key], **scat_kwargs) for key in state_lookup]
        ax_flatten[2 * _i].set_xticks([-4, 0, 4])
        ax_flatten[2 * _i].set_yticks([-4, 0, 4])

    for _i in range(5, 9):
        X, Y, U, cutlineDirection, _targetAxis, _plotAxis = get_potential_along_a_1D_cutline(cfqr, t = protocol_time_array[_i], cutlineDirection = "h")
        plt_line_data_at_t = plotCutlines(X, Y, U, cutlineDirection, cutlineValue = minimum_point_data[_i][0][1],\
                                contour_plt=ax_flatten[2*_i], cutline_plt=ax_flatten[2*_i + 1], contours = 20,
                                showGraph=True, time=protocol_time_array[_i], offset_potential = True, x_range = cutline_xrange, y_range = cutline_yrange, graph_title = " ")
        min_U = np.min(U)
        _index = list(simResult['cfqr'].sim.target_step_index).index(protocol_time_index_array[_i]+1)
        scat = [ax_flatten[2 * _i].scatter(x[state_lookup[key], _index], y[state_lookup[key], _index], c = pColor[key], **scat_kwargs) for key in state_lookup]
        ax_flatten[2 * _i].set_xticks([-4, 0, 4])
        ax_flatten[2 * _i].set_yticks([-4, 0, 4])
        # ax_flatten[2 * _i + 2].plot(plt_line_data_at_t['cutline_plot']['plotAxis'], plt_line_data_at_t['cutline_plot']['targetU'])
        
        

        
plt_figure_setting = {
    "horizontal_lim": [-4, 4], "vertical_lim": [0, 500], "ax": plt
}
def plot_potential_along_a_cutline(cfqr, t = 0, cutlineDirection = "v", cutlineValue = 0, \
                        resolution = 100, contours = 5, manual_domain = [[-5, -5], [5, 5]], plt_figure_setting = plt_figure_setting):
    phi_1xdc_index = protocol_key.index('phi_1xdc')
    phi_2xdc_index = protocol_key.index('phi_2xdc')
    phi_1_dc_i = cfqr.protocol.get_params(t)[phi_1xdc_index]
    phi_2_dc_i = cfqr.protocol.get_params(t)[phi_2xdc_index]
    slice_values = [0, 0, phi_1_dc_i, phi_2_dc_i]


    modified_manual_domain = [(manual_domain[0][1], manual_domain[0][0]), \
                              (manual_domain[1][1], manual_domain[1][0])]
    U, X_mesh = cfqr.system.lattice(t, resolution, axes=(0, 1),\
                                manual_domain=modified_manual_domain, slice_values = slice_values)
    X, Y = X_mesh[0], X_mesh[1]

    # 5. find vmin, vmax, x_min, x_max, y_min and y_max
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
    
    
    plt_figure_setting['ax'].plot(plotAxis, targetU, color = plt_figure_setting['color'], label = plt_figure_setting['label'])
    plt_figure_setting['ax'].set_xlim(plt_figure_setting['horizontal_lim'])
    plt_figure_setting['ax'].set_ylim(plt_figure_setting['vertical_lim'])
    plt_figure_setting['ax'].legend()


    
plt_figure_setting = {
    "horizontal_lim": [-4, 4], "vertical_lim": [-6, 6], "ax": plt
}
def plot_slope_along_a_cutline(cfqr, t = 0, cutlineDirection = "v", cutlineValue = 0, \
                        resolution = 100, contours = 5, manual_domain = [[-5, -5], [5, 5]], plt_figure_setting = plt_figure_setting):

    
    phi_1xdc_index = protocol_key.index('phi_1xdc')
    phi_2xdc_index = protocol_key.index('phi_2xdc')
    phi_1_dc_i = cfqr.protocol.get_params(t)[phi_1xdc_index]
    phi_2_dc_i = cfqr.protocol.get_params(t)[phi_2xdc_index]
    slice_values = [0, 0, phi_1_dc_i, phi_2_dc_i]


    modified_manual_domain = [(manual_domain[0][1], manual_domain[0][0]), \
                              (manual_domain[1][1], manual_domain[1][0])]
    F, X_mesh = cfqr.system.lattice(t, resolution, axes=(0, 1),\
                                manual_domain=modified_manual_domain, slice_values = slice_values, return_force=True)
    X, Y = X_mesh[0], X_mesh[1]

    # 5. find vmin, vmax, x_min, x_max, y_min and y_max
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)

    # 6. call the plot cutline function
    # plotAxis = the axis that we want to be the variable , targetAxis means to fixed the value of that axis
    if cutlineDirection == "h":
        _plotAxis = X
        _targetAxis = Y
        _plotSlope = -F

    if cutlineDirection == "v":
        _plotAxis = Y.T
        _targetAxis = X.T
        _plotSlope = -F.T


    plotAxis = _plotAxis[0] # since all the rows are the same, it doesn't matter which one is chosen
    targetAxis = _targetAxis[:, 0] # to take out the first value of the variable axis.

    # to find out the resolution of the target axis
    targetRange = (_targetAxis[-1] - _targetAxis[-2])/2

    # to find out the index of the cutline that is the closest to the target value
    # e.g the resolution may be something like [-0.05, 0.00, 0.05, 0.010, ... ] and my target value is 0.04
    # I cannot find the index of 0.04 because it does not exist in the array
    # The best I can do is to find the cutline closest to my target value, which is 0.05 in this case
    targetIndex = np.where(np.abs(targetAxis - cutlineValue) <= targetRange)[0][0]

    targetSlope = _plotSlope[targetIndex]
    # print(plotAxis.shape, targetSlope.shape)
    
    plt_figure_setting['ax'].plot(plotAxis, targetSlope[:, 1], color = plt_figure_setting['color'], label = plt_figure_setting['label'])
    plt_figure_setting['ax'].set_xlim(plt_figure_setting['horizontal_lim'])
    plt_figure_setting['ax'].set_ylim(plt_figure_setting['vertical_lim'])
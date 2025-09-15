import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from numba import njit
import datetime, itertools, json, operator

protocol_key = ['U0_1', 'U0_2', 'gamma_1', 'gamma_2', 'beta_1', 'beta_2', 'd_beta_1', 'd_beta_2', 'phi_1_x', 'phi_2_x', 'phi_1_dcx', 'phi_2_dcx', 'M_12']

@njit
def binary_partition(positions, boundary=0):
    '''
    takes a set of position coordinates and sets each value to either 0 or 1 depending on if it is below or above the boundary
    '''
    return (np.sign(positions-boundary)+1)/2


def separate_by_state_in_full_form(state, **kwargs):
    # the separate_by_state_2 is not good because it has to reduce the dimension, which is not obvious. It is better to reduce the dimension within this function, instead of doing that before hand.
    
    bool_array_00 = binary_partition(state[:, 0:2, 0]) == np.array([0., 0.])
    # print(bool_array_00)
    index_of_00 = np.all(bool_array_00, axis=1)

    bool_array_01 = binary_partition(state[:, 0:2, 0]) == np.array([0., 1.])
    # print(bool_array_01)
    index_of_01 = np.all(bool_array_01, axis=1)

    bool_array_10 = binary_partition(state[:, 0:2, 0]) == np.array([1., 0.])
    # print(bool_array_10)
    index_of_10 = np.all(bool_array_10, axis=1)

    bool_array_11 = binary_partition(state[:, 0:2, 0]) == np.array([1., 1.])
    # print(bool_array_11)
    index_of_11 = np.all(bool_array_11, axis=1)

    return {
        "00": index_of_00,
        "01": index_of_01,
        "10": index_of_10,
        "11": index_of_11,
    }

def separate_by_state_2(state, **kwargs):

    bool_array_00 = binary_partition(state[:, :, 0]) == np.array([0., 0.])
    # print(bool_array_00)
    index_of_00 = np.all(bool_array_00, axis=1)

    bool_array_01 = binary_partition(state[:, :, 0]) == np.array([0., 1.])
    # print(bool_array_01)
    index_of_01 = np.all(bool_array_01, axis=1)

    bool_array_10 = binary_partition(state[:, :, 0]) == np.array([1., 0.])
    # print(bool_array_10)
    index_of_10 = np.all(bool_array_10, axis=1)

    bool_array_11 = binary_partition(state[:, :, 0]) == np.array([1., 1.])
    # print(bool_array_11)
    index_of_11 = np.all(bool_array_11, axis=1)

    return {
        "00": index_of_00,
        "01": index_of_01,
        "10": index_of_10,
        "11": index_of_11,
    }

def separate_by_state(state, has_velocity = True, **kwargs):
    if has_velocity:
        initial_state = state[:, 0, ...]

        bool_array_00 = binary_partition(initial_state[:, :, 0]) == np.array([0., 0.])
        # print(bool_array_00)
        index_of_00 = np.all(bool_array_00, axis=1)

        bool_array_01 = binary_partition(initial_state[:, :, 0]) == np.array([0., 1.])
        # print(bool_array_01)
        index_of_01 = np.all(bool_array_01, axis=1)

        bool_array_10 = binary_partition(initial_state[:, :, 0]) == np.array([1., 0.])
        # print(bool_array_10)
        index_of_10 = np.all(bool_array_10, axis=1)

        bool_array_11 = binary_partition(initial_state[:, :, 0]) == np.array([1., 1.])
        # print(bool_array_11)
        index_of_11 = np.all(bool_array_11, axis=1)
        
    else: 
        initial_state = state[:, (0, 1)]

        bool_array_00 = binary_partition(initial_state) == np.array([0., 0.])
        # print(bool_array_00)
        index_of_00 = np.all(bool_array_00, axis=1)

        bool_array_01 = binary_partition(initial_state) == np.array([0., 1.])
        # print(bool_array_01)
        index_of_01 = np.all(bool_array_01, axis=1)

        bool_array_10 = binary_partition(initial_state) == np.array([1., 0.])
        # print(bool_array_10)
        index_of_10 = np.all(bool_array_10, axis=1)

        bool_array_11 = binary_partition(initial_state) == np.array([1., 1.])
        # print(bool_array_11)
        index_of_11 = np.all(bool_array_11, axis=1)

    return {
        "00": index_of_00,
        "01": index_of_01,
        "10": index_of_10,
        "11": index_of_11,
    }

def animate_sim_flux_qubit(all_state, times=[0,1], system=None, frame_skip=30, which_axes=None, axes_names=None,  color_by_state=None, key_state=None, color_key=None, color_bar_range = None, pColor = None, legend=True, alpha=None, slice_values = None, fig_ax=None,  save_path = None, save_dict = None, manual_domain = None, vmin = None, vmax = None, has_velocity = True,  **pot_kwargs):
    # to assign color to particles based on thier initial position
    if has_velocity:
        phi_1_dc = all_state[..., 2, 0]
        phi_2_dc = all_state[..., 3, 0]
        all_state = all_state[:, :, (0, 1), :]
    else:
        phi_1_dc = all_state[..., 2]
        phi_2_dc = all_state[..., 3]
        all_state = all_state[:, :, (0, 1)]
    

    N, nsteps, N_dim = np.shape(all_state)[0], np.shape(all_state)[1], np.shape(all_state)[2]
    
    if has_velocity:
        which_axes = [np.s_[..., i, 0] for i in range(N_dim)]
    else: 
        which_axes = [np.s_[..., i] for i in range(N_dim)]
        
    x_array = [all_state[item] for item in which_axes]
    
    if has_velocity:
        state_lookup = separate_by_state(all_state, has_velocity = has_velocity)
    else:
        initial_state = all_state[:, 0]
        state_lookup = separate_by_state(initial_state, has_velocity = has_velocity)

        
    if fig_ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig, ax = fig_ax
    plt.close()
    samples = np.linspace(0, nsteps-1, nsteps)[::frame_skip]
    time = np.linspace(times[0], times[1], nsteps + 1)

    opacity = min(1, 300/N)

    
    x = x_array[0]
    y = x_array[1]
    names = (r'$\varphi_1$', r'$\varphi_2$')

    x_lim = (np.min(x), np.max(x))
    y_lim = (np.min(y), np.max(y))
    txt = fig.suptitle('t={:.2f}'.format(times[0]))

    scat_kwargs = {'alpha':opacity, 'zorder':10}


    scat = [ax.scatter(x[state_lookup[key], 0], y[state_lookup[key], 0], c = pColor[key], **scat_kwargs) for key in state_lookup]
    
    fig.legend(state_lookup)
    #
    if manual_domain:
        x_lim = (manual_domain[0][0], manual_domain[1][0])
        y_lim = (manual_domain[0][1], manual_domain[1][1])
        ax.set(xlim=x_lim, ylim=y_lim, xlabel=names[0], ylabel=names[1])
    else:
        manual_domain = np.array([[x_lim[0], y_lim[0]], [x_lim[1], y_lim[1]]]) * 2

    ax.set(xlim=x_lim, ylim=y_lim, xlabel=names[0], ylabel=names[1])
    ax.set_aspect(1)


    # if system is not None:
    #     pot, pout = system.show_potential(times[0], ax=ax, cbar=False, surface=False, manual_domain = manual_domain, contours = 10, **pot_kwargs)
    #     pot.title.set_visible(False)
    #     print(pot, pout)

    def animate(i):
#         index = int(samples[i])
#         t_c = time[index]
        
#         x_i = x[:, index]
#         y_i = y[:, index]
        t_c = time[i]
        x_i = x[:, i]
        y_i = y[:, i]
        phi_1_dcx_index = protocol_key.index('phi_1_dcx')
        phi_2_dcx_index = protocol_key.index('phi_2_dcx')
        phi_1_dc_i = system.protocol.get_params(t_c)[phi_1_dcx_index]
        phi_2_dc_i = system.protocol.get_params(t_c)[phi_2_dcx_index]

        slice_values = [0, 0, phi_1_dc_i, phi_2_dc_i]

        # to check if the two numbers are close to each other or not.
        # phi_1_dc_avg = np.mean(phi_1_dc[:, index])
        # phi_2_dc_avg = np.mean(phi_2_dc[:, index])
        # print("from visualization.py, phi_1_dc_avg, phi_2_dc_avg, phi_1_dc_i, phi_2_dc_i", phi_1_dc_avg, phi_2_dc_avg, phi_1_dc_i, phi_2_dc_i)

        for j, item in enumerate(state_lookup):
            scat[j].set_offsets(np.c_[x_i[state_lookup[item]], y_i[state_lookup[item]]])
        txt.set_text(f't={t_c:.2g} {getProtocolSubstepName(system.protocol_list, t_c)}')

        if system:
            system.state_lookup  = state_lookup
            modified_manual_domain = [(manual_domain[0][1], manual_domain[0][0]), (manual_domain[1][1], manual_domain[1][0])]
            # print(modified_manual_domain)
            U, X_mesh = system.lattice(t_c, 80, axes=(0, 1), manual_domain=modified_manual_domain, slice_values = slice_values)
            X = X_mesh[0]
            Y = X_mesh[1]
            x_min, x_max = np.min(X), np.max(X)
            y_min, y_max = np.min(Y), np.max(Y)

            # cont = ax.contourf(X, Y, U, 20, vmin = vmin, vmax = vmax)
            cont = ax.contourf(X, Y, U, 40, vmin = 0, vmax = 400, cmap = "afmhot")
            
        if i == 0:
            colorbar = fig.colorbar(cont, ax = ax)
            
            if color_bar_range:
                colorbar.ax.set_ylim(color_bar_range[0], color_bar_range[1])

            
            # ax.set(xlim=x_lim, ylim=y_lim, xlabel=names[0], ylabel=names[1])
            # pot, pout =system.show_potential(t_c, ax=ax, cbar=False, surface=False, manual_domain = manual_domain, **pot_kwargs)
            # pot.figure.canvas.draw()
            # pot.title.set_visible(False)
    #
    ani = FuncAnimation(fig, animate, interval=100, frames=len(samples), blit=False)
    #
    # if save_path and save_dict:
    #     fps = 20
    #     writergif = PillowWriter(fps=30)
    #     ani.save(save_path, writer=writergif)
    #
    #     with open("coupled_flux_qubit_protocol/data.json") as f:
    #         data = json.load(f)
    #         save_dict["gif_save_path"] = save_path
    #         data.append(save_dict)
    #         with open("coupled_flux_qubit_protocol/data.json", "w+") as f_w:
    #             json.dump(data, f_w)
    #
    # plt.close()
    return ani, fig, ax



def getProtocolSubstepName(protocol_list, t):
    time_array = [item["duration"] for item in protocol_list]
    name_array = [item["name"] for item in protocol_list]
    cumulative_time_array = list(itertools.accumulate(time_array, operator.add))

    targetIndex = 0

    for i, x in enumerate(cumulative_time_array):
        if i == len(cumulative_time_array) - 1:
            targetIndex = i
            break
        elif i == 0 and t < cumulative_time_array[i]:
            targetIndex = i
            break
        else:
            if t >= cumulative_time_array[i] and t <= cumulative_time_array[i+1]:
                targetIndex = i + 1
                break

    fraction = ""
    if targetIndex == 0:
        fraction += f"{t:.3g}/{time_array[targetIndex]}"
    else:
        fraction += f"{t - cumulative_time_array[targetIndex-1]:.3g}/{time_array[targetIndex]}"

    return f"(step {targetIndex}: {name_array[targetIndex]}, {fraction})"
    # print(time_array, cumulative_time_array, name_array[targetIndex])

    
def plotFidelityBarChart(fidelity_time_array, plot_index = [-1], ax_flatten = None):
    fidelity_time_array = np.array(fidelity_time_array, dtype = 'float')
    for x in plot_index:
        _f = fidelity_time_array[x]
        
        if not ax_flatten:
            fig, ax = plt.subplots(2, 2, figsize=[3, 3])
            ax_flatten = ax.flatten()
        
        pColor = {"00": "#061DF7", "01": "#FCEF51", "10": "#3FC7F2", "11": "#F7B6D2"}

        # 00
        
        ax_flatten[2].bar(["00", "01", "10", "11"], _f[:,0], color = pColor.values())
        ax_flatten[2].set_ylim(0, 1)
        ax_flatten[2].axhline(0.5, linestyle='--')

        # 01
        ax_flatten[0].bar(["00", "01", "10", "11"], _f[:,1], color = pColor.values())
        ax_flatten[0].set_ylim(0, 1)
        ax_flatten[0].axhline(0.5, linestyle='--')
        ax_flatten[0].set_xticks([])

        # 10
        ax_flatten[3].bar(["00", "01", "10", "11"], _f[:,2], color = pColor.values())
        ax_flatten[3].set_ylim(0, 1)
        ax_flatten[3].axhline(0.5, linestyle='--')
        ax_flatten[3].set_yticks([])


        # 11
        ax_flatten[1].bar(["00", "01", "10", "11"], _f[:,3], color = pColor.values())
        ax_flatten[1].set_ylim(0, 1)
        ax_flatten[1].axhline(0.5, linestyle='--')
        ax_flatten[1].set_yticks([])
        ax_flatten[1].set_xticks([])
        
        
def playAnimationInHtmlVideo(ani):
    display.HTML(ani.to_html5_video())

def plotTwoState(cfqr, state_1, state_2, p_index = None):
    if not type(p_index) == np.ndarray:
        particle_index = cfqr.getIndexOfParticles(state_1)
    pColor = {"00": "#061DF7", "01": "#FCEF51", "10": "#3FC7F2", "11": "#F187F4"}
    fig, ax = plt.subplots(1, 2, figsize = [10, 5])
    for key in pColor.keys():
        ax[0].scatter(state_1[particle_index[key], 0, 0], state_1[particle_index[key], 1, 0], c = pColor[key])
        ax[1].scatter(state_2[particle_index[key], 0, 0], state_2[particle_index[key], 1, 0], c = pColor[key])
    ax[0].set_xlim(-4, 4)
    ax[0].set_ylim(-4, 4)
    ax[0].set_title("init state")
    ax[0].set_xlabel("phi_1")
    ax[0].set_ylabel("phi_2")
    ax[0].set_aspect(1)

    ax[1].set_xlim(-4, 4)
    ax[1].set_ylim(-4, 4)
    ax[1].set_title("final state")
    ax[1].set_xlabel("phi_1")
    ax[1].set_ylabel("phi_2")
    ax[1].set_aspect(1)
import numpy as np
def extra_constraint_00_and_11_only(state):
    # This extra constraint is for creating an initial state such that all the state are at 00 and 11 only
    # used in the function set_sim_attributes
    # to check if the particle is at the position of bit 11
    tf_array_11 = np.logical_and(state[:,0, 0] > 0, state[:,1, 0] > 0)

    # to check if the particle is at the position of bit 00
    tf_array_00 = np.logical_and(state[:,0, 0] < 0, state[:,1, 0] < 0)

    tf_array_00_or_11 = np.logical_or(tf_array_00, tf_array_11).astype(int)

    return tf_array_00_or_11

def self_defined_initial_state(cfqr, Nsample, t=None, resolution=500, beta=1, manual_domain=None, axes=None, slice_vals=None, verbose=True, extra_constraint = None):
        if cfqr.potential.N_dim >= 4 and (axes is None or len(axes) > 4):
            if resolution > 100:
                if cfqr.potential.N_dim >= 4:
                    resolution = 50
                print('new resolution is {}'.format(resolution))

        NT = Nsample
        state = np.zeros((max(100, int(2*NT)), cfqr.potential.N_dim, 2))

        def get_prob(cfqr, state):
            if cfqr.has_velocity is False:
                state = state[:, :, 0]
            E_curr = cfqr.get_energy(state, t)
            Delta_U = E_curr-U0

            if extra_constraint:
                return np.exp(-beta * Delta_U) * extra_constraint(state)

            return np.exp(-beta * Delta_U)

        if t is None:
            t = cfqr.protocol.t_i

        U, X = cfqr.lattice(t, resolution, axes=axes, slice_values=slice_vals, manual_domain=manual_domain)
        mins = []; maxes = []
        for item in X:
            mins.append(np.min(item))
            maxes.append(np.max(item))

        U0 = np.min(U); i = 0

        if axes is None:
            axes = [_ for _ in range(1, cfqr.potential.N_dim + 1)]
        axes = np.array(axes) - 1

        while i < Nsample:
            test_coords = np.zeros((NT, cfqr.potential.N_dim, 2))
            if slice_vals is not None:
                test_state[:, :, 0] = slice_vals

            # this is the state generated to test
            test_coords[:, axes, 0] = np.random.uniform(mins, maxes, (NT, len(axes)))

            p = get_prob(cfqr, test_coords)

            decide = np.random.uniform(0, 1, NT)
            n_sucesses = np.sum(p > decide)
            if i == 0:
                ratio = max(n_sucesses/NT, .05)

            state[i:i+n_sucesses, :, :] = test_coords[p > decide, :, :]
            i = i + n_sucesses
            if verbose:
                print(f"\r found {i} samples out of {Nsample}", end="")

            NT = max(int((Nsample-i)/ratio), 100)

        print("\nfrom system: finish the while loop in sampling.")
        state = state[0:Nsample, :, :]
        # print("the state is", state)

        if cfqr.has_velocity:
            state[:, :, 1] = np.random.normal(0, np.sqrt(1/(cfqr.mass*beta)), (Nsample, cfqr.potential.N_dim))
        else:
            return state[:, :, 0]

        return state

def plot_initial_state(state):
    plt.plot(state[:, 0, 0], state[:, 1, 0])

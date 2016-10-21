import numpy as np

def plot_glm(data,
             weights,
             adjacency,
             firingrates,
             std_firingrates=None,
             fig=None,
             axs=None,
             handles=None,
             title=None,
             figsize=(6, 3),
             W_lim=3,
             pltslice=slice(0, 500),
             data_index=0,
             N_to_plot=2):
    """
    Plot the parameters of the model
    :return:
    """
    Y = data
    W, A = weights, adjacency
    N = W.shape[0]

    # Do the imports here so that plotting stuff isn't loaded
    # unless it is necessary
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if handles is None:
        # If handles are not given, create a new plot
        handles = []

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(N_to_plot, 3)
        W_ax = fig.add_subplot(gs[:, 0])
        A_ax = fig.add_subplot(gs[:, 1])
        lam_axs = [fig.add_subplot(gs[i, 2]) for i in range(N_to_plot)]
        axs = (W_ax, A_ax, lam_axs)

        # Weight matrix
        h_W = W_ax.imshow(W[:, :, 0], vmin=-W_lim, vmax=W_lim, cmap="RdBu", interpolation="nearest")
        W_ax.set_xlabel("pre")
        W_ax.set_ylabel("post")
        W_ax.set_xticks(np.arange(N))
        W_ax.set_xticklabels(np.arange(N) + 1)
        W_ax.set_yticks(np.arange(N))
        W_ax.set_yticklabels(np.arange(N) + 1)
        W_ax.set_title("Weights")

        # Colorbar
        divider = make_axes_locatable(W_ax)
        cbax = divider.new_horizontal(size="5%", pad=0.05)
        fig.add_axes(cbax)
        plt.colorbar(h_W, cax=cbax)
        handles.append(h_W)

        # Adjacency matrix
        h_A = A_ax.imshow(A, vmin=0, vmax=1, cmap="Greys", interpolation="nearest")
        A_ax.set_xlabel("pre")
        A_ax.set_ylabel("post")
        A_ax.set_title("Adjacency")
        A_ax.set_xticks(np.arange(N))
        A_ax.set_xticklabels(np.arange(N) + 1)
        A_ax.set_yticks(np.arange(N))
        A_ax.set_yticklabels(np.arange(N) + 1)

        # Colorbar
        divider = make_axes_locatable(A_ax)
        cbax = divider.new_horizontal(size="5%", pad=0.05)
        fig.add_axes(cbax)
        plt.colorbar(h_A, cax=cbax)
        handles.append(h_A)

        # Plot the true and inferred rates
        for n in range(min(N, N_to_plot)):
            tn = np.where(Y[pltslice, n])[0]
            lam_axs[n].plot(tn, np.ones_like(tn), 'ko', markersize=4)

            # If given, plot the mean+-std of the firing rates
            if std_firingrates is not None:
                sausage_plot(np.arange(pltslice.start, pltslice.stop),
                             firingrates[pltslice, n],
                             std_firingrates[pltslice,n],
                             sgax=lam_axs[n],
                             alpha=0.5)

            h_fr = lam_axs[n].plot(firingrates[pltslice, n], label="True")[0]
            lam_axs[n].set_ylim(-0.05, 1.1)
            lam_axs[n].set_ylabel("$\lambda_{}(t)$".format(n + 1))

            if n == 0:
                lam_axs[n].set_title("Firing Rates")

            if n == min(N, N_to_plot) - 1:
                lam_axs[n].set_xlabel("Time")
            handles.append(h_fr)

        if title is not None:
            handles.append(fig.suptitle(title))

        plt.tight_layout()

    else:
        # If we are given handles, update the data
        handles[0].set_data(W[:, :, 0])
        handles[1].set_data(A)
        for n in range(min(N, N_to_plot)):
            handles[2 + n].set_data(np.arange(pltslice.start, pltslice.stop), firingrates[pltslice, n])

        if title is not None:
            handles[-1].set_text(title)
        plt.pause(0.001)

    return fig, axs, handles


def sausage_plot(x, y, yerr, sgax=None, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    T = x.size
    assert x.shape == y.shape == yerr.shape == (T,)

    # Get axis
    if sgax is None:
        sgax = plt.gca()

    # Compute envelope
    env = np.zeros((T*2,2))
    env[:,0] = np.concatenate((x, x[::-1]))
    env[:,1] = np.concatenate((y + yerr, y[::-1] - yerr[::-1]))

    # Add the patch
    sgax.add_patch(Polygon(env, **kwargs))
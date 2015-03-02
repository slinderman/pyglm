"""
Analyze the inferred locations under the eigenmodel
"""
import numpy as np
import os
import gzip
import cPickle
import matplotlib.pyplot as plt

from hips.plotting.colormaps import harvard_colors, gradient_cmap
from hips.plotting.layout import create_figure

from pyglm.utils.experiment_helper import load_data, load_results


def plot_connectivity(dataset, run, algs,):
    # Load the data and results
    train, test, true_model = load_data(dataset)
    res_dir = os.path.join("results", dataset, "run%03d" % run)
    results = load_results(dataset, run, algs)

    # samples = results["gibbs"][0]

    ###########################################################
    # Get the average connectivity
    ###########################################################
    if "bfgs" in algs:
        W_mean = results["bfgs"].W.sum(2)
    elif "gibbs" in algs:
        W_samples = [smpl.weight_model.W_effective
                         for smpl in results["gibbs"][0]]

        offset = len(W_samples) // 2
        W_samples = np.array(W_samples[offset:])
        W_mean = W_samples.mean(0).sum(2)
    else:
        raise Exception("Unsupported algorithm")

    W_lim = np.amax(abs(W_mean))

    ###########################################################
    # Plot the inferred connectivity
    ###########################################################
    # Create a crimson-blue colormap
    colors = harvard_colors()
    cmap = gradient_cmap([colors[1],
                          np.array([1.0, 1.0, 1.0]),
                          colors[0]])

    # Plot inferred connectivity
    fig = create_figure((4,3))
    ax = fig.add_subplot(111)

    im = ax.imshow(np.kron(W_mean, np.ones((15,15))),
                   vmin=-W_lim, vmax=W_lim,
                  interpolation="none", cmap=cmap,
                  extent=(1,27,27,1))

    # Plot separators
    ax.plot([16.5, 16.5], [1,27], 'k:')
    ax.plot([1,27], [16.5, 16.5], 'k:')
    ax.set_xlim([1,27])
    ax.set_ylim([27,1])

    # cbar_ticks = np.array([-0.01, 0.0, 0.01])
    # cbar = fig.colorbar(im,
    #                     values=np.linspace(-W_lim, W_lim, 500),
    #                     boundaries=np.linspace(-W_lim, W_lim, 500),
    #                     ticks=cbar_ticks)
    # cbar.set_ticklabels(['-0.01', '0', '+0.01'])

    cbar = fig.colorbar(im,
                        values=np.linspace(-W_lim, W_lim, 500),
                        boundaries=np.linspace(-W_lim, W_lim, 500))

    ax.set_xlabel("$n$")
    ax.set_ylabel("$n'$")
    ax.set_title("Inferred $W_{n' \\to n}$")

    plt.subplots_adjust(left=0.2, bottom=0.2)

    # Save the figure
    fig_path = os.path.join(res_dir, "gibbs_conn.pdf")
    fig.savefig(fig_path)


def approximate_rgc_locs():
    """
    Approximate the RGC locations given Figure 2 of Pillow 2008
    :return:
    """
    on_locs = np.zeros((11,2))
    on_locs[0,:]  = [0.5, 2.0]
    on_locs[1,:]  = [0.9, 0.9]
    on_locs[2,:]  = [1.5, 0.2]
    on_locs[3,:]  = [1.5, 1.5]
    on_locs[4,:]  = [1.5, 2.5]
    on_locs[5,:]  = [1.0, 3.5]
    on_locs[6,:]  = [2.0, 3.5]
    on_locs[7,:]  = [2.5, 2.0]
    on_locs[8,:]  = [2.5, 1.0]
    on_locs[9,:]  = [3.5, 1.5]
    on_locs[10,:] = [3.3, 3.0]

    # Center the ON locs
    on_locs -= np.array([[2,2]])

    # Manually approximate the OFF locations
    off_locs = np.zeros((16,2))
    off_locs[0,:]  = [0.5, 2.5]
    off_locs[1,:]  = [0.5, 3.5]
    off_locs[2,:]  = [0.6, 1.5]
    off_locs[3,:]  = [1.5, 2.2]
    off_locs[4,:]  = [1.5, 3.0]
    off_locs[5,:]  = [1.5, 3.5]
    off_locs[6,:]  = [2.5, 3.5]
    off_locs[7,:]  = [2.5, 3.0]
    off_locs[8,:]  = [1.7, 1.3]
    off_locs[9,:]  = [1.0, 0.8]
    off_locs[10,:] = [2.5, 2.5]
    off_locs[11,:] = [1.7, 0.5]
    off_locs[12,:] = [2.5, 1.5]
    off_locs[13,:] = [2.8, 0.5]
    off_locs[14,:] = [3.5, 2.5]
    off_locs[15,:] = [3.5, 1.3]

    # Center the OFF locs
    off_locs -= np.array([[2,2]])

    # Stack up the locations
    L = np.vstack((off_locs, on_locs))
    return L

# plot_connectivity("rgc_bern_eigen_60T", run=1,
#                   algs=("gibbs",))


plot_connectivity("rgc_nb_eigen_300T", run=1,
                  algs=("bfgs",))

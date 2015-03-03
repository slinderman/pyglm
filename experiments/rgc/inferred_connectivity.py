"""
Analyze the inferred locations under the eigenmodel
"""
import numpy as np
import os
import gzip
import cPickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from hips.plotting.colormaps import harvard_colors, gradient_cmap
from hips.plotting.layout import create_figure
colors = harvard_colors()

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

    cbar_ticks = np.array([-1.0, 0.0, 1.])
    cbar = fig.colorbar(im,
                        values=np.linspace(-W_lim, W_lim, 500),
                        boundaries=np.linspace(-W_lim, W_lim, 500),
                        ticks=cbar_ticks)
    cbar.set_ticklabels(['-1.0', '0', '+1.0'])
    #
    # cbar = fig.colorbar(im,
    #                     values=np.linspace(-W_lim, W_lim, 500),
    #                     boundaries=np.linspace(-W_lim, W_lim, 500))

    ax.set_xlabel("$n$")
    ax.set_ylabel("$n'$")
    ax.set_title("Inferred $W_{n' \\to n}$")

    plt.subplots_adjust(left=0.2, bottom=0.2)

    # Save the figure
    fig_path = os.path.join(res_dir, "gibbs_conn.pdf")
    fig.savefig(fig_path)


def plot_impulse_responses(dataset, run):
        # Load the data and results
    from pyglm.models import BernoulliEigenmodelPopulation
    model = BernoulliEigenmodelPopulation(
        N=27, dt=1.0, dt_max=10.0, B=5)
    res_dir = os.path.join("results", dataset, "run%03d" % run)
    results = load_results(dataset, run, ["gibbs"])
    samples = results["gibbs"][0]

    ###########################################################
    # Get the average connectivity
    ###########################################################
    W_samples = [smpl.weight_model.W_effective for smpl in samples]
    offset = len(W_samples) // 2
    W_samples = np.array(W_samples[offset:])
    W_lim = np.amax(abs(W_samples))
    imp_lim = W_lim * np.amax(abs(model.basis.basis))

    syn_sort = np.argsort(W_samples.mean(0).sum(2).ravel())
    i_synapses = zip(*np.unravel_index(syn_sort[:3], (27,27)))
    e_synapses = zip(*np.unravel_index(syn_sort[-3:], (27,27)))
    synapses = i_synapses + e_synapses

    # synapses = [(1,1)]

    for i,(n_pre, n_post) in enumerate(synapses):
        print "Plotting syn: ", n_pre, "->", n_post
        W_mean = W_samples[:,n_pre, n_post,:].mean(0)
        W_cov = np.cov(W_samples[:,n_pre, n_post, :].T)

        basis = model.basis.basis
        L = basis.shape[0]
        dt = 0.01
        tt = dt * np.arange(L)

        imp_mean = basis.dot(W_mean)
        imp_std  = np.sqrt(np.diag(basis.dot(W_cov).dot(basis.T)))

        # Plot inferred impulse response
        fig = create_figure((3,2))
        ax = fig.add_subplot(111)

        # Plot the axes
        ax.plot([dt, L*dt], [0,0], ":k", lw=0.5)

        # Plot the inferred impulse response
        ax.plot(dt * np.arange(L), imp_mean, '-k',
                lw=2.0)

        # Add error bars
        from matplotlib.patches import Polygon
        verts = list(zip(tt, imp_mean + 3*imp_std)) + \
                list(zip(tt[::-1], imp_mean[::-1] - 3*imp_std[::-1]))

        col = int(i<3)
        poly = Polygon(verts,
                       facecolor=colors[col],
                       edgecolor=colors[col], alpha=0.75)
        ax.add_patch(poly)

        ax.set_xlabel("$\Delta t [ms]$")
        xscale = 0.001
        xticks = ticker.FuncFormatter(lambda x, pos: '{0:.0f}'.format(x/xscale))
        ax.xaxis.set_major_formatter(xticks)

        ax.set_xlim([dt, (L-1)*dt])
        ax.set_ylabel("$\\mathbf{B} \\mathbf{w}_{%d \\to %d}$" % (n_pre, n_post))
        ax.set_ylim([-imp_lim, imp_lim])

        plt.subplots_adjust(left=0.25, bottom=0.25)

        # Save the figure
        fig_path = os.path.join(res_dir, "synapse_%d_%d.pdf" % (n_pre, n_post))
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


# plot_connectivity("rgc_nb_eigen_300T", run=1,
#                   algs=("gibbs",))

plot_impulse_responses("rgc_nb_eigen_300T", run=1)
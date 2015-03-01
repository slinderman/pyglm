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


def load_data(dataset=""):
    base_dir = os.path.join("data", dataset)
    assert os.path.exists(base_dir), \
        "Could not find data directory: " + base_dir

    model_path = os.path.join(base_dir, "model.pkl.gz")
    model = None
    if os.path.exists(model_path):
        with gzip.open(model_path, "r") as f:
            model = cPickle.load(f)

    train_path = os.path.join(base_dir, "train.pkl.gz")
    with gzip.open(train_path, "r") as f:
        train = cPickle.load(f)

    test_path = os.path.join(base_dir, "test.pkl.gz")
    with gzip.open(test_path, "r") as f:
        test = cPickle.load(f)

    return train, test, model

def load_results(dataset="", run=0,
                 algorithms=("bfgs", "gibbs","vb")):

    base_dir = os.path.join("results", dataset, "run%03d" % run)
    assert os.path.exists(base_dir), \
        "Could not find results directory: " + base_dir

    results = {}
    for alg in algorithms:
        res_path = os.path.join(base_dir, alg + ".pkl.gz")
        if os.path.exists(res_path):
            print "\tLoading ", alg, " results..."
            with gzip.open(res_path, "r") as f:
                results[alg] = cPickle.load(f)

    return results

def plot_connectivity(dataset, run, algs,):
    # Load the data and results
    train, test, true_model = load_data(dataset)
    res_dir = os.path.join("results", dataset, "run%03d" % run)
    results = load_results(dataset, run, algs)

    # samples = results["gibbs"][0]

    ###########################################################
    # Get the average connectivity
    ###########################################################
    W_samples = [smpl.weight_model.W_effective
                     for smpl in results["gibbs"][0]]

    offset = len(W_samples) // 2
    W_samples = np.array(W_samples[offset:])
    W_mean = W_samples.mean(0).sum(2)
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

    # fig.colorbar(im)
    # cbar_ticks = np.array([-0.12, -0.06, 0.0, 0.06, 0.12])
    cbar_ticks = np.array([-0.01, 0.0, 0.01])
    cbar = fig.colorbar(im,
                        values=np.linspace(-W_lim, W_lim, 500),
                        boundaries=np.linspace(-W_lim, W_lim, 500),
                        ticks=cbar_ticks)
    # import pdb; pdb.set_trace()
    # cbar.set_ticks(cbar_ticks)
    # cbar.set_ticklabels(['-0.12', '-0.06', '0', '+0.06', '+0.12'])
    # cbar.set_ticklabels(['0', '+0.12'])
    cbar.set_ticklabels(['-0.01', '0', '+0.01'])

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

plot_connectivity("rgc_bern_eigen_60T", run=1,
                  algs=("gibbs",))

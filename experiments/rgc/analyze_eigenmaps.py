"""
Analyze the inferred locations under the eigenmodel
"""
import numpy as np
import os
import gzip
import cPickle
import itertools as it
import matplotlib.pyplot as plt

from hips.plotting.colormaps import harvard_colors
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

def analyze_eigenmaps(dataset, run, algs,):
    # Load the data and results
    train, test, true_model = load_data(dataset)
    res_dir = os.path.join("results", dataset, "run%03d" % run)
    results = load_results(dataset, run, algs)

    samples = results["gibbs"][0]

    ###########################################################
    # Get the latent location samples
    ###########################################################
    N_samples = len(samples)

    # Rotate the samples since they are invariant to rotation
    print "Aligning samples"
    # F_star = samples[-1].network.F
    # lmbda_star = samples[-1].network.lmbda
    F_star = approximate_rgc_locs()
    lmbda_star = 0.1 * np.ones(2)
    F_samples = []
    for s,smpl in enumerate(samples):
        R = smpl.network.adjacency_dist.\
            compute_optimal_rotation(F_star, lmbda_star)
        F_samples.append(smpl.network.F.dot(R))

        # print "rho: %.3f" % (smpl.weight_model.A.sum() / 27.0**2)

    F_samples = np.array(F_samples)

    ###########################################################
    # Scatter plot some of the locations
    ###########################################################
    print "Plotting..."
    offset = N_samples // 2
    subsmpl = 10

    # OFF Cells
    fig = create_figure((4,3))
    ax = fig.add_subplot(111)
    # col = harvard_colors()
    col_itr = it.cycle(harvard_colors()[:6])
    # symbols = ["*", "o", "s"]
    sym_itr = it.cycle(["*", "o", "s", "^", "p"])
    for n in xrange(16):
        # color = colors[ n % len(colors)]
        color = col_itr.next()
        symbol = sym_itr.next()
        ax.plot(F_samples[offset::subsmpl,n,0],
                 F_samples[offset::subsmpl,n,1],
                 linestyle="none",
                 color=color,
                 marker=symbol,
                 markersize=7.5,
                 markerfacecolor=color,
                 markeredgecolor="none",
                 alpha=0.5)

        ax.plot(2*F_star[n,0], 2*F_star[n,1],
                color=color,
                marker=symbol,
                markersize=10,
                markerfacecolor=color,
                markeredgecolor="k",
                alpha=1.0)

        # Fix limits, plot axes
        lim=5
        ax.plot([-lim, lim], [0, 0], ':k', lw=0.5)
        ax.plot([0, 0], [-lim, lim], ':k', lw=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)


        ax.set_title("Inferred OFF Cell Embedding")

    # Save the figure
    fig_path = os.path.join(res_dir, "eigenmap_off.pdf")
    fig.savefig(fig_path)

    # OFF Cells
    fig = create_figure((4,3))
    ax = fig.add_subplot(111)
    # col = harvard_colors()
    col_itr = it.cycle(harvard_colors()[:6])
    # symbols = ["*", "o", "s"]
    sym_itr = it.cycle(["*", "o", "s", "^", "p"])
    for n in xrange(16,27):
        # color = colors[ n % len(colors)]
        color = col_itr.next()
        symbol = sym_itr.next()
        ax.plot(F_samples[offset::subsmpl,n,0],
                 F_samples[offset::subsmpl,n,1],
                 linestyle="none",
                 color=color,
                 marker=symbol,
                 markersize=7.5,
                 markerfacecolor=color,
                 markeredgecolor="none",
                 alpha=0.5)

        ax.plot(2*F_star[n,0], 2*F_star[n,1],
                color=color,
                marker=symbol,
                markersize=10,
                markerfacecolor=color,
                markeredgecolor="k",
                alpha=1.0)

        # Fix limits, plot axes
        lim=5
        ax.plot([-lim, lim], [0, 0], ':k', lw=0.5)
        ax.plot([0, 0], [-lim, lim], ':k', lw=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)


        ax.set_title("Inferred ON Cell Embedding")

    # Save the figure
    fig_path = os.path.join(res_dir, "eigenmap_on.pdf")
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

analyze_eigenmaps("rgc_bern_eigen_60T", run=1,
                 algs=("gibbs",))

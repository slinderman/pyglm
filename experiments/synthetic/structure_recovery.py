import os
import gzip
import cPickle
import numpy as np

from sklearn.metrics import roc_curve, precision_recall_curve

import matplotlib.pyplot as plt

from hips.plotting.layout import create_figure
from hips.plotting.colormaps import harvard_colors

from pyglm.utils.experiment_helper import load_data, load_results


def latent_structure_score(F_true, lmbda_true, inf_eigenmodel):
    # Compute optimal rotation
    R = inf_eigenmodel.compute_optimal_rotation(F_true, lmbda_true)
    F_rot = inf_eigenmodel.F.dot(R)

    sqerr = ((F_true - F_rot)**2).sum()
    return sqerr

def mean_latent_structure_score(F_true, lmbda_true, samples):
        sqerrs = []
        for s in samples:
            net = s.network.adjacency_dist
            sqerrs.append(latent_structure_score(F_true, lmbda_true, net))

        mse = np.array(sqerrs)
        return mse

def plot_latent_structure(ax, A_true, F_true, lmbda_true, samples):
    N = A_true.shape[0]

    ax.plot(F_true[:,0], F_true[:,1], 'ks', ls="none")
    for n1 in xrange(N):
        for n2 in xrange(N):
            if A_true[n1,n2]:
                ax.plot([F_true[n1,0], F_true[n2,0]],
                        [F_true[n1,1], F_true[n2,1]], 'k')


    h = ax.plot(F_true[:,0], F_true[:,1], 'ro', ls="none")[0]
    for i,s in enumerate(samples):
        print "Sample ", i
        net = s.network.adjacency_dist
        R = net.compute_optimal_rotation(F_true, lmbda_true)
        F_rot = net.F.dot(R)

        h.set_data(F_rot[:,0], F_rot[:,1])
        plt.pause(0.1)




def run_structure_recovery(dataset, run, algs):

    # Load the data and results
    train, test, true_model = load_data(dataset)
    res_dir = os.path.join("results", dataset, "run%03d" % run)
    results = load_results(dataset, run, algs)

    # Plot predictive likelihood vs wall clock time
    # fig = create_figure((4,3))
    # ax = fig.add_subplot(111, aspect="equal")
    # ax.plot([0,1], [0,1], ':k', lw=0.5)
    # col = harvard_colors()
    #
    # assert "bfgs" in results

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Helpers to parse results tuple
    samples    = lambda x: x[0]
    vlbs       = lambda x: np.array(x[1])
    lps        = lambda x: np.array(x[1])
    plls       = lambda x: np.array(x[2])
    timestamps = lambda x: np.array(x[3])

    # Get the true adjacency matrix
    A_true = true_model.weight_model.A
    F_true = true_model.network.adjacency_dist.F
    lmbda_true = true_model.network.adjacency_dist.lmbda

    # if 'svi' in results:
    #     svi_mse = mean_latent_structure_score(F_true, samples(results["svi"]))
    #     print "SVI:\t", svi_mse
    #
    # if 'vb' in results:
    #     vb_mse = mean_latent_structure_score(F_true, samples(results["vb"]))
    #     print "VB:\t", vb_mse
    #
    # if 'gibbs' in results:
    #     gibbs_sqerr = mean_latent_structure_score(F_true, lmbda_true, samples(results["gibbs"]))
    #     print "Gibbs:\t", gibbs_sqerr.mean()
    #     plt.plot(gibbs_sqerr)
    #
    # if "bfgs" in results:
    #     pass

    plt.ion()
    plot_latent_structure(ax, A_true, F_true, lmbda_true, samples(results["vb"]))

    plt.show()

    # Put a legend above
    # plt.legend(loc=4, prop={'size':9})

    # ax.set_xlabel('FPR')
    # ax.set_ylabel('TPR')
    #
    # ax.set_title("ROC Curve")
    # plt.tight_layout()

    # Save the figure
    # fig_path = os.path.join(res_dir, "latent_structure.pdf")
    # fig.savefig(fig_path)

run_structure_recovery("synth_nb_eigen_K50_T10000", run=1,
                     algs=("bfgs", "gibbs", "vb", "svi"))

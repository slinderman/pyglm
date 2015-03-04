import os
import sys
import numpy as np

import matplotlib.pyplot as plt

from hips.plotting.layout import create_figure
from hips.plotting.colormaps import harvard_colors, gradient_cmap

from pyglm.utils.experiment_helper import load_data, load_results

#
# def latent_structure_score(F_true, lmbda_true, inf_eigenmodel):
#     # Compute optimal rotation
#     R = inf_eigenmodel.compute_optimal_rotation(F_true, lmbda_true)
#     F_rot = inf_eigenmodel.F.dot(R)
#
#     sqerr = ((F_true - F_rot)**2).sum()
#     return sqerr
#
# def mean_latent_structure_score(F_true, lmbda_true, samples):
#         sqerrs = []
#         for s in samples:
#             net = s.network.adjacency_dist
#             sqerrs.append(latent_structure_score(F_true, lmbda_true, net))
#
#         mse = np.array(sqerrs)
#         return mse
#
# def plot_latent_structure(ax, A_true, F_true, lmbda_true, samples):
#     N = A_true.shape[0]
#
#     ax.plot(F_true[:,0], F_true[:,1], 'ks', ls="none")
#     for n1 in xrange(N):
#         for n2 in xrange(N):
#             if A_true[n1,n2]:
#                 ax.plot([F_true[n1,0], F_true[n2,0]],
#                         [F_true[n1,1], F_true[n2,1]], 'k')
#
#
#     h = ax.plot(F_true[:,0], F_true[:,1], 'ro', ls="none")[0]
#     for i,s in enumerate(samples):
#         print "Sample ", i
#         net = s.network.adjacency_dist
#         R = net.compute_optimal_rotation(F_true, lmbda_true)
#         F_rot = net.F.dot(R)
#
#         h.set_data(F_rot[:,0], F_rot[:,1])
#         plt.pause(0.1)


def plot_latent_embedding(samples=None, vb=None, F_true=None,
                          ax=None, color=[0,0,0]):

    if ax is None: ax = plt.gca()


    if samples is not None:
        F_samples = np.array([s.network.adjacency_dist.F
                              for s in samples])

        F_2D = F_samples.reshape((-1,2))
        lim = 3 * np.std(F_2D[:])

        cmap = gradient_cmap([np.ones(3), color])
        ax.hist2d(F_2D[:,0], F_2D[:,1], bins=100, cmap=cmap)

        # for F in F_samples:
        #     ax.plot(F[:,0], F[:,1], 'o', ls="none", color=color)


    elif vb is not None:
        eigenmodel = vb.network.adjacency_dist
        eigenmodel.resample_from_mf()
        F_2D = vb.network.adjacency_dist.mf_mu_F.copy()

        # Rotate the sample
        # R = vb.network.adjacency_dist.compute_optimal_rotation(F_true,
        #                                                        np.ones(2))
        R = np.eye(2)
        F_2D = F_2D.dot(R)

        ax.plot(F_2D[:,0], F_2D[:,1], 'o', color=color, ls="none")
        lim = np.amax(abs(F_2D)) + 0.2

    else:
        raise Exception("Must give me either gibbs samples or vb model")


    if F_true is not None:
        ax.plot(F_true[:,0], F_true[:,1], 'ks', ls="none")

    # Plot axes
    ax.plot([0,0], [-lim, lim], 'k:', lw=0.5)
    ax.plot([-lim, lim], [0,0], 'k:', lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title('Latent embedding ')

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


    if 'gibbs' in results:
        # gibbs_sqerr = mean_latent_structure_score(F_true, lmbda_true, samples(results["gibbs"]))
        # print "Gibbs:\t", gibbs_sqerr.mean()
        # plt.plot(gibbs_sqerr)
        plot_latent_embedding(samples=samples(results["gibbs"])[-1:],
                              F_true=F_true, ax=ax,
                              color=harvard_colors()[2])

    # if 'svi' in results:
    #     plot_latent_embedding(vb=samples(results["svi"])[-1],
    #                           F_true=F_true, ax=ax,
    #                           color=harvard_colors()[0])
    #
    #
    # if 'vb' in results:
    #     plot_latent_embedding(vb=samples(results["vb"])[-1],
    #                           F_true=F_true, ax=ax,
    #                           color=harvard_colors()[1])
    #
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

args = sys.argv
assert len(args) == 3
dataset = args[1]
run = int(args[2])

print "Dataset: ", dataset
print "Run:     ", run


run_structure_recovery(dataset=dataset, run=1,
                       algs=("bfgs", "gibbs", "vb", "svi"))

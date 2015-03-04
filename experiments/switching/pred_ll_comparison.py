import os
import sys
import gzip
import cPickle
import numpy as np
from scipy.misc import logsumexp

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from hips.plotting.layout import create_figure
from hips.plotting.colormaps import harvard_colors

from pyglm.models import HomogeneousPoissonModel

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

def plot_pred_ll_vs_time(dataset, run, algs, Z=1.0, nbins=4):

    # Load the data and results
    train, test, true_model = load_data(dataset)
    res_dir = os.path.join("results", dataset, "run%03d" % run)
    results = load_results(dataset, run, algs)

    # Plot predictive likelihood vs wall clock time
    fig = create_figure((6.5,3))
    ax = fig.add_subplot(111)
    col = harvard_colors()
    plt.grid()

    # Compute the max and min time in seconds
    N = train.shape[1]
    homog_model = HomogeneousPoissonModel(N)
    homog_model.add_data(train)
    homog_model.fit()
    homog_pll = homog_model.heldout_log_likelihood(test)

    # Normalize PLL by number of time bins
    Z = float(test.shape[0])

    # # DEBUG
    # true_pll = true_model.heldout_log_likelihood(train)
    # import pdb; pdb.set_trace()

    # assert "bfgs" in results
    # t_bfgs = timestamps["bfgs"]
    t_bfgs = 1.0
    t_start = 1.0
    t_stop = 2.0

    # Helpers to parse results tuple
    samples    = lambda x: x[0]
    vlbs       = lambda x: np.array(x[1])
    lps        = lambda x: np.array(x[1])
    plls       = lambda x: np.array(x[2])
    timestamps = lambda x: np.array(x[3])

    for i, alg in enumerate(algs):
        if alg == "bfgs": continue
        if alg in results:
            t_gibbs = timestamps(results[alg])
            t_gibbs = t_bfgs + t_gibbs
            t_stop = max(t_stop, t_gibbs[-1])
            ax.semilogx(t_gibbs, (plls(results[alg]) - homog_pll)/Z,
                        color=col[i], label=alg.upper(), lw=1.5)

    if "bfgs" in results:
        bfgs_model = results["bfgs"]
        bfgs_pll = bfgs_model.heldout_log_likelihood(test)
        ax.semilogx([t_start, t_stop],
                    [(bfgs_pll - homog_pll)/Z,
                     (bfgs_pll - homog_pll)/Z],
                    color=col[len(algs)-1], lw=1.5, label="MAP" )

    if true_model is not None:
        true_ll = true_model.heldout_log_likelihood(test)
        ax.semilogx([t_start, t_stop],
                    [(true_ll - homog_pll)/Z,
                     (true_ll - homog_pll)/Z],
                    color='k', lw=1.5, label="True" )

    # Put a legend above
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=5, mode="expand", borderaxespad=0.,
               prop={'size':9})

    ax.set_xlim(t_start, t_stop)

    # Format the ticks
    # plt.locator_params(nbins=nbins)
    logxscale = 3
    xticks = ticker.FuncFormatter(lambda x, pos: '{0:.1f}'.format(x/10.**logxscale))
    ax.xaxis.set_major_formatter(xticks)
    ax.set_xlabel('Time ($10^{%d}$ s)' % logxscale)

    # logyscale = 4
    # yticks = ticker.FuncFormatter(lambda y, pos: '{0:.3f}'.format(y/10.**logyscale))
    # ax.yaxis.set_major_formatter(yticks)
    # ax.set_ylabel('Pred. LL ($ \\times 10^{%d}$)' % logyscale)
    ax.set_ylabel('Pred. LL (bps)')
    # ax.set_ylim(0, 0.18)

    # ylim = ax.get_ylim()
    # ax.plot([t_bfgs, t_bfgs], ylim, '--k')
    # ax.set_ylim(ylim)

    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.2)
    # plt.title("Predictive Log Likelihood ($T=%d$)" % T_train)
    plt.show()

    # Save the figure
    fig_path = os.path.join(res_dir, "pred_ll_vs_time.pdf")
    fig.savefig(fig_path)



def plot_pred_ll_bar(dataset, run, algs, Z=1.0, nbins=4):

    # Load the data and results
    train, test, true_model = load_data(dataset)
    res_dir = os.path.join("results", dataset, "run%03d" % run)
    results = load_results(dataset, run, algs)

    # Plot predictive likelihood vs wall clock time
    fig = create_figure((6.5,3))
    ax = fig.add_subplot(111)
    col = harvard_colors()
    plt.grid()

    # Compute the max and min time in seconds
    N = train.shape[1]
    homog_model = HomogeneousPoissonModel(N)
    homog_model.add_data(train)
    homog_model.fit()
    homog_pll = homog_model.heldout_log_likelihood(test)

    # Normalize PLL by number of time bins
    Z = float(test.shape[0])


    # Helpers to parse results tuple
    samples    = lambda x: x[0]
    vlbs       = lambda x: np.array(x[1])
    lps        = lambda x: np.array(x[1])
    plls       = lambda x: np.array(x[2])
    timestamps = lambda x: np.array(x[3])

    N_avg = 100

    for i, alg in enumerate(algs):
        if alg == "bfgs": continue
        if alg in results:
            alg_pll = plls(results[alg])
            mean_pll = -np.log(N_avg) + logsumexp(alg_pll[-N_avg:])
            ax.bar(i, (mean_pll - homog_pll)/Z,
                        color=col[i], label=alg.upper(), lw=1.5)

    if "bfgs" in results:
        bfgs_model = results["bfgs"]
        bfgs_pll = bfgs_model.heldout_log_likelihood(test)
        ax.bar(len(algs)-1, (bfgs_pll - homog_pll)/Z,
               color=col[len(algs)-1], label="MAP", lw=1.5)

    if true_model is not None:
        true_ll = true_model.heldout_log_likelihood(test)
        ax.bar(len(algs), (true_ll - homog_pll) / Z,
                    color='k', lw=1.5, label="True" )

    # Put a legend above
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=5, mode="expand", borderaxespad=0.,
               prop={'size':9})


    # Format the ticks
    ax.set_ylabel('Pred. LL (bps)')
    ax.set_ylim(0.4,0.8)

    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.2)
    # plt.title("Predictive Log Likelihood ($T=%d$)" % T_train)
    plt.show()

    # Save the figure
    fig_path = os.path.join(res_dir, "pred_ll_bar.pdf")
    fig.savefig(fig_path)


args = sys.argv
assert len(args) == 3
dataset = args[1]
run = int(args[2])

print "Dataset: ", dataset
print "Run:     ", run


# plot_pred_ll_vs_time(dataset=dataset, run=1,
#                      algs=("empty-hsmm",
#                            "er-hsmm",
#                            "er-hmm",
#                            "bfgs",
#                             ))


plot_pred_ll_bar(dataset=dataset, run=1,
                     algs=("empty-hsmm",
                           "er-hsmm",
                           "er-hmm",
                           "bfgs",
                            ))



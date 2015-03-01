import os
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
    fig = create_figure((4,3))
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

    assert "bfgs" in results
    # t_bfgs = timestamps["bfgs"]
    t_bfgs = 1.0
    t_start = 1.0
    t_stop = 0.0

    # Helpers to parse results tuple
    samples    = lambda x: x[0]
    vlbs       = lambda x: np.array(x[1])
    lps        = lambda x: np.array(x[1])
    plls       = lambda x: np.array(x[2])
    timestamps = lambda x: np.array(x[3])

    if 'svi' in results:
        isreal = ~np.isnan(plls(results['svi']))
        svis = plls(results['svi'])[isreal]
        t_svi = timestamps(results['svi'])[isreal]
        t_svi = t_bfgs + t_svi - t_svi[0]
        t_stop = max(t_stop, t_svi[-1])
        ax.semilogx(t_svi, (svis - homog_pll)/Z,
                    color=col[0], label="SVI", lw=1.5)

    if 'vb' in results:
        t_vb = timestamps(results['vb'])
        t_vb = t_bfgs + t_vb
        t_stop = max(t_stop, t_vb[-1])
        ax.semilogx(t_vb, (plls(results['vb']) - homog_pll)/Z,
                    color=col[1], label="VB", lw=1.5)

    if 'gibbs' in results:
        t_gibbs = timestamps(results['gibbs'])
        t_gibbs = t_bfgs + t_gibbs
        t_stop = max(t_stop, t_gibbs[-1])
        ax.semilogx(t_gibbs, (plls(results['gibbs']) - homog_pll)/Z,
                    color=col[2], label="Gibbs", lw=1.5)


    # Extend lines to t_st
    # if 'svi' in plls and 'svi' in timestamps:
    #     final_svi_pll = -np.log(4) + logsumexp(svis[-4:])
    #     ax.semilogx([t_svi[-1], t_stop],
    #                 [(final_svi_pll - plls['homog'])/Z,
    #                  (final_svi_pll - plls['homog'])/Z],
    #                 '--',
    #                 color=col[0], lw=1.5)

    # if 'vb' in plls and 'vb' in timestamps:
    #     ax.semilogx([t_vb[-1], t_stop],
    #                 [(plls['vb'][-1] - plls['homog'])/Z,
    #                  (plls['vb'][-1] - plls['homog'])/Z],
    #                 '--',
    #                 color=col[1], lw=1.5)
    #
    if "bfgs" in results:
        bfgs_model = results["bfgs"]
        bfgs_pll = bfgs_model.heldout_log_likelihood(test)
        ax.semilogx([t_start, t_stop],
                    [(bfgs_pll - homog_pll)/Z,
                     (bfgs_pll - homog_pll)/Z],
                    color=col[3], lw=1.5, label="MAP" )

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

plot_pred_ll_vs_time("synth_nb_eigen_K50_T10000", run=1,
                     algs=("bfgs", "gibbs", "vb", "svi"))

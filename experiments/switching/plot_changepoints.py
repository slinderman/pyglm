import os
import sys
import numpy as np

import matplotlib.pyplot as plt

from hips.plotting.layout import create_figure
from hips.plotting.colormaps import harvard_colors

from pyglm.models import HomogeneousPoissonModel
from pyglm.utils.experiment_helper import load_results, load_data




def plot_changepoint_pr(dataset, run, algs):

    # Load the data and results
    train, test, true_model = load_data(dataset)
    res_dir = os.path.join("results", dataset, "run%03d" % run)
    results = load_results(dataset, run, algs)

    # Plot predictive likelihood vs wall clock time
    fig = create_figure((6.5,3))
    ax = fig.add_subplot(111)
    col = harvard_colors()

    # Helpers to parse results tuple
    samples    = lambda x: x[0]
    vlbs       = lambda x: np.array(x[1])
    lps        = lambda x: np.array(x[1])
    plls       = lambda x: np.array(x[2])
    timestamps = lambda x: np.array(x[3])
    stateseqs     = lambda x: np.array(x[4])

    N_avg = 100
    plot_slice = slice(0,1000)

    assert "bfgs" not in algs

    if true_model is not None:
        true_states = true_model.hidden_state_sequence[0][plot_slice]
        true_changepoints = np.diff(true_states) != 0
        ax.plot(true_changepoints, color='k', label='True', lw=2.0)

    D = len(algs)
    assert D==3
    col_inds = [0,1,3]

    for i, alg in enumerate(algs):
        if alg in results:
            # ax = fig.add_subplot(D,1,i+1)

            # Plot the true changepoints
            # if true_model is not None:
            #     true_states = true_model.hidden_state_sequence[0][plot_slice]
            #     true_changepoints = np.diff(true_states) != 0
            #     ax.plot(true_changepoints, color='k', label='True', lw=3)

            # Plot the inferred changepoints
            states = stateseqs(results[alg])[:,plot_slice]
            changepoints = np.diff(states, axis=1) != 0
            changepoint_pr = changepoints[-N_avg:,:].sum(0) / float(N_avg)
            ax.plot(changepoint_pr, color=col[col_inds[i]], label=alg.upper(),
                    lw=6-2*i)

            # ax.set_ylim(0,1)
            # ax.set_yticks([0,0.5, 1])
            # ax.set_ylabel("Changepoint Prob.")
            # ax.set_ylabel(alg.upper())
            #
            # if i == 0:
            #     ax.set_title("Changepoint Probability")
            #
            # if i < D-1:
            #     ax.set_xticklabels([])
            #     ax.set_xlabel("Time")


    ax.set_ylim(0,1)
    ax.set_ylabel("Changepoint Prob.")
    ax.set_xlabel("Time")



    # Put a legend above
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=5, mode="expand", borderaxespad=0.,
               prop={'size':9})

    #
    # # Format the ticks
    # ax.set_ylabel('Changepoint Prob.')
    # ax.set_ylim(0,1)

    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    # plt.title("Predictive Log Likelihood ($T=%d$)" % T_train)
    plt.show()

    # Save the figure
    fig_path = os.path.join(res_dir, "changepoint_pr.pdf")
    fig.savefig(fig_path)


args = sys.argv
assert len(args) == 3
dataset = args[1]
run = int(args[2])

print "Dataset: ", dataset
print "Run:     ", run


plot_changepoint_pr(dataset=dataset, run=1,
                     algs=("empty-hsmm",
                           "er-hsmm",
                           "er-hmm"
                            ))



import os
import gzip
import cPickle
import numpy as np

from sklearn.metrics import roc_curve, precision_recall_curve

import matplotlib.pyplot as plt

from hips.plotting.layout import create_figure
from hips.plotting.colormaps import harvard_colors

from pyglm.utils.experiment_helper import load_data, load_results

def compute_pred_lls(samples, test):
    F_test = samples[0].augment_data(test)["F"]
    plls = np.zeros(len(samples))
    for s,smpl in enumerate(samples):
            plls[s] = smpl.heldout_log_likelihood(test, F=F_test)

    return plls



def run_link_prediction(dataset, run, algs):

    # Load the data and results
    train, test, true_model = load_data(dataset)
    res_dir = os.path.join("results", dataset, "run%03d" % run)
    results = load_results(dataset, run, algs)

    # Plot predictive likelihood vs wall clock time
    fig = create_figure((4,3))
    ax = fig.add_subplot(111, aspect="equal")
    ax.plot([0,1], [0,1], ':k', lw=0.5)
    col = harvard_colors()

    assert "bfgs" in results

    # Helpers to parse results tuple
    samples    = lambda x: x[0]
    vlbs       = lambda x: np.array(x[1])
    lps        = lambda x: np.array(x[1])
    plls       = lambda x: np.array(x[2])
    timestamps = lambda x: np.array(x[3])

    # Get the true adjacency matrix
    A_true = true_model.weight_model.A.ravel()

    if 'svi' in results:
        pass

    if 'vb' in results:
        vb_model = samples(results["vb"])[-1]
        W_vb = vb_model.weight_model.mf_expected_W().sum(2)
        score = abs(W_vb).ravel()
        fpr, tpr, _ = roc_curve(A_true, score)
        ax.plot(fpr, tpr, color=col[1], lw=1.5, label="VB")

    if 'gibbs' in results:
        W_samples = [smpl.weight_model.W_effective.sum(2)
                     for smpl in samples(results["gibbs"])]

        offset = len(W_samples) // 2
        W_samples = np.array(W_samples[offset:])
        score = abs(W_samples.mean(0)).ravel()
        fpr, tpr, _ = roc_curve(A_true, score)
        ax.plot(fpr, tpr, color=col[2], lw=1.5, label="Gibbs")

    if "bfgs" in results:
        score = abs(results["bfgs"].W).ravel()
        fpr, tpr, _ = roc_curve(A_true, score)
        ax.plot(fpr, tpr, color=col[3], lw=1.5, label="MAP")

    # Put a legend above
    plt.legend(loc=4, prop={'size':9})

    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')

    ax.set_title("ROC Curve")
    plt.tight_layout()

    # Save the figure
    fig_path = os.path.join(res_dir, "roc.pdf")
    fig.savefig(fig_path)

    ##########################################################
    # Do the same for link prediction
    ##########################################################
    fig = create_figure((4,3))
    ax = fig.add_subplot(111, aspect="equal")

    if 'svi' in results:
        pass

    if 'vb' in results:
        vb_model = samples(results["vb"])[-1]
        W_vb = vb_model.weight_model.mf_expected_W().sum(2)
        score = abs(W_vb).ravel()
        prec, recall, _ = precision_recall_curve(A_true, score)
        ax.plot(recall, prec, color=col[1], lw=1.5, label="VB")
    if 'gibbs' in results:
        W_samples = [smpl.weight_model.W_effective.sum(2)
                     for smpl in samples(results["gibbs"])]

        offset = len(W_samples) // 2
        W_samples = np.array(W_samples[offset:])
        score = abs(W_samples.mean(0)).ravel()
        prec, recall, _ = precision_recall_curve(A_true, score)
        ax.plot(recall, prec, color=col[2], lw=1.5, label="Gibbs")

    if "bfgs" in results:
        score = abs(results["bfgs"].W).ravel()
        prec, recall, _ = precision_recall_curve(A_true, score)
        ax.plot(recall, prec, color=col[3], lw=1.5, label="MAP")

    # Put a legend above
    plt.legend(loc=1, prop={'size':9})

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')

    ax.set_title("Precision-Recall Curve")
    plt.tight_layout()

    # Save the figure
    fig_path = os.path.join(res_dir, "precision_recall.pdf")
    fig.savefig(fig_path)

run_link_prediction("synth_nb_eigen_K50_T10000", run=1,
                     algs=("bfgs", "gibbs", "vb"))

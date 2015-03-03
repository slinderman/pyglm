import cPickle
import os
import gzip
import numpy as np

if not os.environ.has_key("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

from hips.plotting.colormaps import harvard_colors, gradient_cmap
cm.register_cmap("harvard", ListedColormap(harvard_colors()))

from pyglm.switching_models import NegativeBinomialHDPHSMM

def generate_synthetic_data(seed=None):
    """
    Create a population model and generate a spike train from it.
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    ###########################################################
    # Create a population model
    ###########################################################
    N = 20          # Number of neurons
    M = 4           # Number of states
    T = 1000        # Number of time bins
    dt = 1.0        # Time bin width
    dt_max = 100.0  # Max time of synaptic influence
    B = 1           # Number of basis functions for the weights

    #   Bias hyperparameters
    bias_hypers = {"mu_0": -4.0, "sigma_0": 0.25}

    ###########################################################
    #   Network hyperparameters
    ###########################################################
    network_hypers = {"p": 0.05, "mu_0": np.zeros(B), "Sigma_0": 0.5**2*np.eye(B)}

    hdp_hmm_hypers = {"alpha": 6., "gamma": 6.,
                      "init_state_concentration": 6.}

    ###########################################################
    # Create the model with these parameters
    ###########################################################
    true_model = \
        NegativeBinomialHDPHSMM(
            N=N, M=M, dt=dt, dt_max=dt_max, B=B,
            bias_hypers=bias_hypers,
            network_hypers=network_hypers,
            hdp_hmm_hypers=hdp_hmm_hypers)

    ###########################################################
    # Sample from the true model
    ###########################################################
    S, stateseq = true_model.generate(T=T, keep=True)
    print "Number of generated spikes:\t", S.sum(0)

    ###########################################################
    #  Plot the network, the spike train and mean rate
    ###########################################################
    # plt.figure()
    # for m in xrange(M):
    #     population = true_model.populations[m]
    #     plt.subplot(1,M,m+1)
    #     plt.imshow(population.weight_model.W_effective.sum(2),
    #                vmin=-1.0, vmax=1.0,
    #                interpolation="none", cmap="RdGy")
    # plt.show()

    # plt.figure()
    # R = true_model.compute_rate()
    # N_plot = 3
    # for n in xrange(N_plot):
    #     plt.subplot(N_plot,1,n+1)
    #     t = np.arange(T) * dt
    #     plt.plot(t, R[:,n], '-r')
    #     spks = np.where(S[:,n])[0]
    #     plt.plot(t[spks], S[spks,n], 'ko', markerfacecolor="k")
    # plt.show()

    true_model.plot()
    plt.set_cmap("harvard")
    plt.show()




    #
    # ###########################################################
    # # Pickle and save the data
    # ###########################################################
    # out_dir  = os.path.join('data', "synthetic")
    # # out_name = 'synthetic_sbm_K%d_C%d_T%d.pkl' % (N,C,T)
    # out_name = 'synthetic_er_K%d_T%d.pkl' % (N,T)
    # out_path = os.path.join(out_dir, out_name)
    # with open(out_path, 'w') as f:
    #     print "Saving output to ", out_path
    #     cPickle.dump((S, true_model.copy_sample()), f, protocol=-1)
    #
    # with gzip.open(out_path + ".gz", 'w') as f:
    #     cPickle.dump((S, true_model.copy_sample()), f, protocol=-1)
    #
    # # Sample test data
    # T_test = 1000
    # S_test = true_model.generate(T=T_test, keep=False)

    # Pickle and save the data
    # out_dir  = os.path.join('data', "synthetic")
    # out_name = 'synthetic_er_K%d_T%d_test.pkl.gz' % (N,T)
    # out_path = os.path.join(out_dir, out_name)
    # with gzip.open(out_path, 'w') as f:
    #     print "Saving output to ", out_path
    #     cPickle.dump((S_test, true_model.copy_sample()), f, protocol=-1)

generate_synthetic_data()
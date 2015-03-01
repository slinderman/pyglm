import cPickle
import os
import gzip
import numpy as np
# np.seterr(all="raise")

if not os.environ.has_key("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pyglm.models import NegativeBinomialEigenmodelPopulation

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
    N = 50                                                  # Number of neurons
    T = 100000                                               # Number of time bins
    dt = 1.0                                                # Time bin width
    dt_max = 100.0                                          # Max time of synaptic influence
    B = 1                                                   # Number of basis functions for the weights

    #   Bias hyperparameters
    bias_hypers = {"mu_0": -4.0, "sigma_0": 0.25}

    ###########################################################
    #   Network hyperparameters
    ###########################################################
    network_hypers = {"p": 0.05, "mu_0": np.zeros(B), "Sigma_0": 1.0**2*np.eye(B),
                      "sigma_F": 1.0}

    ###########################################################
    # Create the model with these parameters
    ###########################################################
    true_model = NegativeBinomialEigenmodelPopulation(N=N, dt=dt, dt_max=dt_max, B=B,
                                            bias_hypers=bias_hypers,
                                            network_hypers=network_hypers)

    # Override the mean weight
    true_model.network.weight_dist.mu = np.zeros(B)
    true_model.network.weight_dist.sigma = 0.5**2 * np.eye(B)
    true_model.weight_model.resample()

    plt.figure()
    W_lim = np.amax(abs(true_model.weight_model.W_effective.sum(2)))
    plt.imshow(true_model.weight_model.W_effective.sum(2),
               vmin=-W_lim, vmax=W_lim,
               interpolation="none", cmap="RdGy")
    plt.colorbar()
    plt.show()

    true_model.network.plot(true_model.weight_model.A)

    ###########################################################
    # Sample from the true model
    ###########################################################
    S, Psi_hat = true_model.generate(T=T, keep=True, return_Psi=True)
    R_hat = true_model.observation_model.xi * np.exp(Psi_hat)
    print "Number of generated spikes:\t", S.sum(0)

    ###########################################################
    #  Plot the network, the spike train and mean rate
    ###########################################################
    plt.figure()
    R = true_model.compute_rate(true_model.data_list[0])

    # Basic sanity checks
    print "E_N:  ", R.sum(axis=0)
    print "N:    ", S.sum(axis=0)

    for n in xrange(N):
        plt.subplot(N,1,n+1)
        t = np.arange(T) * dt
        plt.plot(t, R[:,n], '-r')
        plt.plot(t, R_hat[:T,n], '-b')
        spks = np.where(S[:,n])[0]
        plt.plot(t[spks], S[spks,n], 'ko', markerfacecolor="k")
    plt.show()

    ###########################################################
    # Pickle and save the data
    ###########################################################
    out_dir  = os.path.join('data', "synthetic")
    out_name = 'synthetic_nb_er_K%d_T%d.pkl' % (N,T)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, 'w') as f:
        print "Saving output to ", out_path
        cPickle.dump((S, true_model.copy_sample()), f, protocol=-1)

    with gzip.open(out_path + ".gz", 'w') as f:
        cPickle.dump((S, true_model.copy_sample()), f, protocol=-1)

    # Sample test data
    T_test = 1000                                           # Number of time bins for test data set
    S_test = true_model.generate(T=T_test, keep=False)

    # Pickle and save the data
    out_dir  = os.path.join('data', "synthetic")
    # out_name = 'synthetic_sbm_K%d_C%d_T%d_test.pkl.gz' % (N,C,T)
    out_name = 'synthetic_nb_er_K%d_T%d_test.pkl.gz' % (N,T)
    out_path = os.path.join(out_dir, out_name)
    with gzip.open(out_path, 'w') as f:
        print "Saving output to ", out_path
        cPickle.dump((S_test, true_model.copy_sample()), f, protocol=-1)

generate_synthetic_data(1964982120)
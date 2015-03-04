import cPickle
import os
import gzip
import numpy as np
np.seterr(all="raise")

if not os.environ.has_key("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pyglm.models import NegativeBinomialEigenmodelPopulation

def generate_synthetic_data(dataset, seed=None):
    """
    Create a population model and generate a spike train from it.
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    res_dir = os.path.join("data", dataset)

    ###########################################################
    # Create a population model
    ###########################################################
    N = 50                                                  # Number of neurons
    T = 10000                                              # Number of time bins
    dt = 1.0                                                # Time bin width
    dt_max = 10.0                                          # Max time of synaptic influence
    B = 1                                                   # Number of basis functions for the weights

    #   Bias hyperparameters
    bias_hypers = {"mu_0": -4.0, "sigma_0": 0.25}

    ###########################################################
    #   Network hyperparameters
    ###########################################################
    network_hypers = {"p": 0.01, "mu_0": 0.*np.ones(B), "Sigma_0": 1**2*np.eye(B),
                      "sigma_F": 1.0}

    ###########################################################
    # Create the model with these parameters
    ###########################################################
    true_model = NegativeBinomialEigenmodelPopulation(N=N, dt=dt, dt_max=dt_max, B=B,
                                            bias_hypers=bias_hypers,
                                            network_hypers=network_hypers)

    ###########################################################
    #   Override the sample with some serious structure
    ###########################################################
    eigenmodel = true_model.network.adjacency_dist
    M = 4
    th = np.linspace(0,2*np.pi, M, endpoint=False)
    centers = np.hstack((np.cos(th)[:,None], np.sin(th)[:,None]))
    # centers = [[1,1], [1,-1], [-1,-1], [-1,1]]
    for m, center in enumerate(centers):
        start = m*N//M
        end = min((m+1)*N//M, N)
        eigenmodel.F[start:end, :] = \
            center + 0.1 * np.random.randn(end-start,2)

    # Override the mean weight
    true_model.network.weight_dist.mu = 0.25 * np.ones(B)
    true_model.network.weight_dist.sigma = 0.5**2 * np.eye(B)
    true_model.weight_model.resample()

    # Force self inhibition
    for n in xrange(N):
        true_model.weight_model.W[n,n,:] = -1
        true_model.weight_model.A[n,n] = 1

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

    for n in xrange(5):
        plt.subplot(5,1,n+1)
        t = np.arange(1000) * dt
        plt.plot(t, R[:1000,n], '-r')
        plt.plot(t, R_hat[:1000,n], '-b')
        spks = np.where(S[:1000,n])[0]
        plt.plot(t[spks], S[spks,n], 'ko', markerfacecolor="k")
    plt.show()

    # Sample test data
    T_test = 1000                                           # Number of time bins for test data set
    S_test = true_model.generate(T=T_test, keep=False)


    ###########################################################
    # Pickle and save the data
    ###########################################################
    with gzip.open(os.path.join(res_dir, "train.pkl.gz"), 'w') as f:
        # print "Saving output to ", out_path
        cPickle.dump(S, f, protocol=-1)

    with gzip.open(os.path.join(res_dir, "model.pkl.gz"), 'w') as f:
        print "Saving output to ", os.path.join(res_dir, "model.pkl.gz")
        cPickle.dump(true_model.copy_sample(), f, protocol=-1)

    with gzip.open(os.path.join(res_dir, "test.pkl.gz"), 'w') as f:
        cPickle.dump(S_test, f, protocol=-1)


generate_synthetic_data("synth_nb_eigen_long", 2979744453)
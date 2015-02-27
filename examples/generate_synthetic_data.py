import cPickle
import os
import gzip
import numpy as np

if not os.environ.has_key("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pyglm.models import Population

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
    N = 5                                                  # Number of neurons
    # C = 1                                                   # Number of clusters
    T = 10000                                               # Number of time bins
    dt = 1.0                                                # Time bin width
    dt_max = 100.0                                          # Max time of synaptic influence
    B = 1                                                   # Number of basis functions for the weights

    #   Bias hyperparameters
    bias_hypers = {"mu_0": -1.0, "sigma_0": 0.25}

    ###########################################################
    #   Network hyperparameters
    ###########################################################
    # c = np.arange(C).repeat((N // C))                       # Neuron to cluster assignments
    # p = 0.5 * np.ones((C,C))                                      # Probability of connection for each pair of clusters
    # # p = 0.9 * np.eye(C) + 0.05 * (1-np.eye(C))              # Probability of connection for each pair of clusters
    # mu = np.zeros((C,C,B))                                  # Mean weight for each pair of clusters
    # Sigma = np.tile( 3**2 * np.eye(B)[None,None,:,:], (C,C,1,1))    # Covariance of weight for each pair of clusters
    # network_hypers = {'C': C, 'c': c, 'p': p, 'mu': mu, 'Sigma': Sigma}

    ###########################################################
    # Create the model with these parameters
    ###########################################################
    network_hypers = {"Sigma_0": 10*np.eye(B)}
    true_model = Population(N=N, dt=dt, dt_max=dt_max, B=B,
                            bias_hypers=bias_hypers,
                            network_hypers=network_hypers)

    ###########################################################
    # Sample from the true model
    ###########################################################
    S, Psi_hat = true_model.generate(T=T, keep=True, return_Psi=True)
    print "Number of generated spikes:\t", S.sum(0)

    ###########################################################
    #  Plot the network, the spike train and mean rate
    ###########################################################
    plt.figure()
    plt.imshow(true_model.weight_model.W_effective.sum(2), vmin=-1.0, vmax=1.0, interpolation="none", cmap="RdGy")

    plt.figure()
    Psi = true_model.activation_model.compute_psi(true_model.data_list[0])
    R = true_model.compute_rate(true_model.data_list[0])

    # Basic sanity checks
    assert np.allclose(Psi, Psi_hat[:T,:])
    print "E_N:  ", R.sum(axis=0)
    print "N:    ", S.sum(axis=0)

    for n in xrange(N):
        plt.subplot(N,1,n+1)
        t = np.arange(T) * dt
        plt.plot(t, Psi[:,n], '-r')
        plt.plot(t, Psi_hat[:T,n], '-b')
        spks = np.where(S[:,n])[0]
        plt.plot(t[spks], S[spks,n], 'ko', markerfacecolor="k")
    plt.show()

    ###########################################################
    # Pickle and save the data
    ###########################################################
    out_dir  = os.path.join('data', "synthetic")
    # out_name = 'synthetic_sbm_K%d_C%d_T%d.pkl' % (N,C,T)
    out_name = 'synthetic_er_K%d_T%d.pkl' % (N,T)
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
    out_name = 'synthetic_er_K%d_T%d_test.pkl.gz' % (N,T)
    out_path = os.path.join(out_dir, out_name)
    with gzip.open(out_path, 'w') as f:
        print "Saving output to ", out_path
        cPickle.dump((S_test, true_model.copy_sample()), f, protocol=-1)

generate_synthetic_data()
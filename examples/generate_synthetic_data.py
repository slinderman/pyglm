import cPickle
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

from pyglmdos.models import _GibbsPopulation

def generate_synthetic_data(seed=None):
    """
    Create a population model and generate a spike train from it.
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    # Create a population model
    N = 2                                                   # Number of neurons
    C = 1                                                   # Number of clusters
    T = 10000                                               # Number of time bins
    dt = 1.0                                                # Time bin width
    dt_max = 1000.0                                          # Max time of synaptic influence
    B = 3                                                   # Number of basis functions for the weights

    ##
    #   Bias hyperparameters
    ##
    bias_hypers = {"mu_0": -3.0, "sigma_0": 0.25}

    ##
    #   Network hyperparameters
    ##
    c = np.arange(C).repeat((N // C))                       # Neuron to cluster assignments
    p = np.ones((C,C))                                      # Probability of connection for each pair of clusters
    # p = 0.9 * np.eye(C) + 0.05 * (1-np.eye(C))              # Probability of connection for each pair of clusters
    mu = np.zeros((C,C,B))                                  # Mean weight for each pair of clusters
    Sigma = np.tile(np.eye(B)[None,None,:,:], (C,C,1,1))    # Covariance of weight for each pair of clusters
    T_test = 1000                                           # Number of time bins for test data set

    ##
    # Create the model with these parameters
    ##
    network_hypers = {'C': C, 'c': c, 'p': p, 'mu': mu, 'Sigma': Sigma}
    true_model = _GibbsPopulation(N=N, dt=dt, B=B,
                                  bias_hypers=bias_hypers,
                                  network_hypers=network_hypers)

    # Plot the true network
    plt.ion()
    plt.imshow(true_model.weight_model.W_effective.sum(2), vmin=-1.0, vmax=1.0, interpolation="none", cmap="RdGy")
    plt.pause(0.001)

    ##
    # Sample from the true model
    ##
    S = true_model.generate(T=T, keep=False)
    print "Number of generated spikes:\t", S.sum(0)

    ##
    # Pickle and save the data
    ##
    out_dir  = os.path.join('data', "synthetic")
    out_name = 'synthetic_K%d_C%d_T%d.pkl' % (N,C,T)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, 'w') as f:
        print "Saving output to ", out_path
        cPickle.dump((S, true_model), f, protocol=-1)

    with gzip.open(out_path + ".gz", 'w') as f:
        cPickle.dump((S, true_model), f, protocol=-1)

    # Sample test data
    S_test = true_model.generate(T=T_test, keep=False)

    # Pickle and save the data
    out_dir  = os.path.join('data', "synthetic")
    out_name = 'synthetic_test_K%d_C%d_T%d.pkl' % (N,C,T_test)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, 'w') as f:
        print "Saving output to ", out_path
        cPickle.dump((S_test, true_model), f, protocol=-1)

generate_synthetic_data()
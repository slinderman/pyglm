import cPickle
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

from pyglmdos.models import _GibbsPopulation

def generate_synthetic_data(seed=None):
    """
    Create a discrete time Hawkes model and generate from it.

    :return:
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    # Create a true model
    T_test=1000

    # Small network:
    # Seed: 1957629166
    C = 1
    K = 2
    T = 10000
    dt = 1.0
    B = 3
    c = np.arange(C).repeat((K // C))
    p = 0.9 * np.eye(C) + 0.05 * (1-np.eye(C))
    mu = np.zeros((C,C,B))
    Sigma = np.tile(np.eye(B)[None,None,:,:], (C,C,1,1))

    # Create the model with these parameters
    network_hypers = {'C': C, 'c': c, 'p': p, 'mu': mu, 'Sigma': Sigma}
    true_model = _GibbsPopulation(N=K, dt=dt, B=B,
                                  network_hypers=network_hypers)

    # Plot the true network
    plt.ion()
    plt.imshow(true_model.weight_model.W_effective, vmin=-1.0, vmax=1.0, interpolation="none", cmap="RdGy")
    plt.pause(0.001)

    # Sample from the true model
    S = true_model.generate(T=T, keep=False)
    print "Number of generated spikes:\t", S.sum(0)

    # Pickle and save the data
    out_dir  = os.path.join('data', "synthetic")
    out_name = 'synthetic_K%d_C%d_T%d.pkl' % (K,C,T)
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
    out_name = 'synthetic_test_K%d_C%d_T%d.pkl' % (K,C,T_test)
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, 'w') as f:
        print "Saving output to ", out_path
        cPickle.dump((S_test, true_model), f, protocol=-1)

generate_synthetic_data()
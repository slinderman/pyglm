"""
Fit the RGC data with a simple logistic regression model
"""
import numpy as np
import os
import cPickle
import gzip
# np.seterr(all='raise')

import matplotlib.pyplot as plt

from pyglm.models import StandardBernoulliPopulation

def demo(seed=None):
    """
    Fit a weakly sparse
    :return:
    """
    if seed is None:
        seed = np.random.randint(2**32)

    print "Setting seed to ", seed
    np.random.seed(seed)

    ###########################################################
    # Load the RGC data
    ###########################################################
    base_path = os.path.join("data", "rgc", "rgc_60T")
    data_path = base_path + ".pkl"
    with open(data_path, 'r') as f:
        data = cPickle.load(f)

    S      = data["S"].astype(np.int32)
    T      = data["T"]
    N      = data["N"]
    dt     = data["dt"]

    # Set model parameters
    B      = 3
    dt_max = 0.100

    ###########################################################
    # Create a test spike-and-slab model
    ###########################################################

    # Copy the network hypers.
    model = StandardBernoulliPopulation(N=N, dt=dt, dt_max=dt_max, B=B)
    model.add_data(S)

    ###########################################################
    # Fit the test model with L1-regularized logistic regression
    ###########################################################
    model.fit(L1=True)

    ###########################################################
    # Plot the true and inferred network
    ###########################################################
    plt.figure()
    W_eff = model.W.sum(2)
    lim = np.amax(abs(W_eff))
    plt.imshow(W_eff,
               vmax=lim, vmin=-lim,
               interpolation="none", cmap="RdGy")
    plt.suptitle("Inferred network")

    #
    # Plot the true and inferred rates
    #
    plt.figure()
    R = model.compute_rate(model.data_list[0])
    for n in xrange(2):
        plt.subplot(N,1,n+1)
        plt.plot(np.arange(T), R[:,n], '-r', lw=1)
        plt.ylim([0,1])
    plt.show()

    ###########################################################
    # Save the fit model
    ###########################################################
    results_path = base_path + ".standard_fit.l1.pkl.gz"
    with gzip.open(results_path, 'w') as f:
        cPickle.dump(model, f, protocol=-1)


demo(1234)

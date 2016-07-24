import numpy as np
import os
import cPickle
import gzip
# np.seterr(all='raise')

if not os.environ.has_key("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

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
    # Load some example data.
    # See data/synthetic/generate.py to create more.
    ###########################################################
    base_path = os.path.join("data", "synthetic", "synthetic_K20_C1_T10000")
    data_path = base_path + ".pkl.gz"
    with gzip.open(data_path, 'r') as f:
        S, true_model = cPickle.load(f)
        true_model.add_data(S)

    T      = S.shape[0]
    N      = true_model.N
    B      = true_model.B
    dt     = true_model.dt
    dt_max = true_model.dt_max

    ###########################################################
    # Create a test spike-and-slab model
    ###########################################################

    # Copy the network hypers.
    test_model = StandardBernoulliPopulation(N=N, dt=dt, dt_max=dt_max, B=B,
                                             basis_hypers=true_model.basis_hypers)
    test_model.add_data(S)
    # F_test = test_model.basis.convolve_with_basis(S_test)

    ###########################################################
    # Fit the test model with L1-regularized logistic regression
    ###########################################################
    test_model.fit(L1=True)

    ###########################################################
    # Plot the true and inferred network
    ###########################################################
    plt.figure()
    plt.subplot(121)
    plt.imshow(true_model.weight_model.W_effective.sum(2),
               vmax=1.0, vmin=-1.0,
               interpolation="none", cmap="RdGy")
    plt.suptitle("True network")

    # Plot the inferred network
    plt.subplot(122)
    plt.imshow(test_model.W.sum(2),
               vmax=1.0, vmin=-1.0,
               interpolation="none", cmap="RdGy")
    plt.suptitle("Inferred network")

    #
    # Plot the true and inferred rates
    #
    plt.figure()
    R_true = true_model.compute_rate(true_model.data_list[0])
    R_test = test_model.compute_rate(test_model.data_list[0])
    for n in xrange(N):
        plt.subplot(N,1,n+1)
        plt.plot(np.arange(T), R_true[:,n], '-k', lw=2)
        plt.plot(np.arange(T), R_test[:,n], '-r', lw=1)
        plt.ylim([0,1])
    plt.show()

    ###########################################################
    # Save the fit model
    ###########################################################
    results_path = base_path + ".standard_fit.pkl.gz"
    with gzip.open(results_path, 'w') as f:
        cPickle.dump(test_model, f, protocol=-1)


demo(1234)

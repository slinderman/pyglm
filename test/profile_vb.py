import numpy as np
import os
import cPickle
import gzip
import time

from pyglm.models import NegativeBinomialEigenmodelPopulation
from pyglm.utils.profiling import show_line_stats

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
    base_path = os.path.join("data", "synthetic", "synthetic_nb_eigen_K50_T10000")
    data_path = base_path + ".pkl.gz"
    init_path = base_path + ".standard_fit.pkl.gz"
    test_path = os.path.join("data", "synthetic", "synthetic_nb_eigen_K50_T100000_test.pkl.gz")
    with gzip.open(data_path, 'r') as f:
        S, true_model = cPickle.load(f)

    # Load the test data
    with gzip.open(test_path, 'r') as f:
        S_test, _ = cPickle.load(f)

    T      = S.shape[0]
    N      = true_model.N
    B      = true_model.B
    dt     = true_model.dt
    dt_max = true_model.dt_max

    ###########################################################
    # Create and fit a standard model for initialization
    ###########################################################
    with gzip.open(init_path, 'r') as f:
        init_model = cPickle.load(f)

    ###########################################################
    # Create a test spike-and-slab model
    ###########################################################

    # Copy the network hypers.
    test_model = NegativeBinomialEigenmodelPopulation(N=N, dt=dt, dt_max=dt_max, B=B,
                            basis_hypers=true_model.basis_hypers,
                            observation_hypers=true_model.observation_hypers,
                            activation_hypers=true_model.activation_hypers,
                            weight_hypers=true_model.weight_hypers,
                            bias_hypers=true_model.bias_hypers,
                            network_hypers=true_model.network_hypers)
    test_model.add_data(S)

    # Initialize with the standard model
    test_model.initialize_with_standard_model(init_model)

    # Convolve the test data for fast heldout likelihood calculations
    F_test = test_model.basis.convolve_with_basis(S_test)

    ###########################################################
    # Fit the test model with Gibbs sampling
    ###########################################################
    N_samples = 2
    plot_interval = np.inf
    samples = [test_model.copy_sample()]
    vlbs = [test_model.get_vlb()]
    plls = [test_model.heldout_log_likelihood(S_test, F=F_test)]
    timestamps = [0]
    start = time.clock()
    for itr in xrange(N_samples):
        print ""
        print "VB iteration ", itr
        print "VLB: ", vlbs[-1]

        test_model.meanfield_coordinate_descent_step()
        vlbs.append(test_model.get_vlb())

        # Resample from MF
        test_model.resample_from_mf()
        plls.append(test_model.heldout_log_likelihood(S_test, F=F_test))
        samples.append(test_model.copy_sample())
        timestamps.append(time.clock()-start)


    show_line_stats()

demo(11223344)

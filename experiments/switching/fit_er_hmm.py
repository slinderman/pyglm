import numpy as np
import os
import sys
import cPickle
import gzip
import time
if not os.environ.has_key("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pyglm.switching_models import NegativeBinomialHDPHMM
from pyglm.utils.experiment_helper import load_data, load_results

def fit_with_gibbs(dataset, run, seed=None):
    """
    Fit the dataset using Gibbs sampling
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
    train, test, true_model = load_data(dataset)
    res_dir = os.path.join("results", dataset, "run%03d" % run)
    assert os.path.exists(res_dir), "Results directory does not exist: " + res_dir
    standard_results = load_results(dataset, run=run,
                                    algorithms=["bfgs"])

    T      = train.shape[0]
    N      = true_model.N
    M      = true_model.M
    B      = true_model.B
    dt     = true_model.dt
    dt_max = true_model.dt_max

    # Create and fit a standard model for initialization
    init_model = standard_results["bfgs"]

    ###########################################################
    # Create a test spike-and-slab model
    ###########################################################
    # Copy the network hypers.
    test_model = NegativeBinomialHDPHMM(N=N, M=M, dt=dt, dt_max=dt_max, B=B,
                            basis_hypers=true_model.basis_hypers,
                            observation_hypers=true_model.observation_hypers,
                            activation_hypers=true_model.activation_hypers,
                            weight_hypers=true_model.weight_hypers,
                            bias_hypers=true_model.bias_hypers,
                            network_hypers=true_model.network_hypers,
                            hdp_hmm_hypers=true_model.hdp_hmm_hypers)

    test_model.add_data(train)

    # Initialize with the standard model
    # test_model.initialize_with_standard_model(init_model)

    # Convolve the test data for fast heldout likelihood calculations
    packed_test = test_model.add_data(test)
    test_model.pop_data()

    ###########################################################
    # Fit the test model with Gibbs sampling
    ###########################################################
    # raw_input("Press any key to continue...\n")
    N_samples = 1000
    samples = [test_model.copy_sample()]
    lps = [test_model.log_likelihood()]
    plls = [test_model.heldout_log_likelihood(test, packed_data=packed_test)]
    timestamps = [0]
    start = time.clock()
    for itr in xrange(N_samples):
        lps.append(test_model.log_likelihood())
        plls.append(test_model.heldout_log_likelihood(test, packed_data=packed_test))
        samples.append(test_model.copy_sample())
        timestamps.append(time.clock()-start)

        print ""
        print "Gibbs iteration ", itr
        print "LP: ", lps[-1]
        print "PLL: ", plls[-1]

        test_model.resample_model()

        # Save intermediate sample
        with gzip.open(
                os.path.join(res_dir,
                             "gibbs.er.hdphmm.itr%04d.pkl.gz" % itr),
                "w") as f:
            cPickle.dump((test_model.copy_sample(), timestamps[-1]), f, protocol=-1)


    plt.ioff()

    ###########################################################
    # Save the results
    ###########################################################
    results_path = os.path.join(res_dir, "gibbs.er.hdphmm.pkl.gz")
    with gzip.open(results_path, 'w') as f:
        cPickle.dump((samples, lps, plls, timestamps), f, protocol=-1)



args = sys.argv
assert len(args) == 3
dataset = args[1]
run = int(args[2])

print "Dataset: ", dataset
print "Run:     ", run
fit_with_gibbs(dataset, run, seed=11223344)

import numpy as np
import os
import cPickle
import gzip

# Turn on line profiling with `export PROFILING=True`
from pyglm.utils.profiling import show_line_stats

from pyglm.models import Population

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
    data_path = os.path.join("data", "synthetic", "synthetic_K20_C1_T10000.pkl.gz")
    with gzip.open(data_path, 'r') as f:
        S, true_model = cPickle.load(f)

    T      = S.shape[0]
    N      = true_model.N
    B      = true_model.B
    dt     = true_model.dt
    dt_max = true_model.dt_max

    ###########################################################
    # Create a test spike-and-slab model
    ###########################################################

    # Copy the network hypers.
    test_model = Population(N=N, dt=dt, dt_max=dt_max, B=B,
                            basis_hypers=true_model.basis_hypers,
                            observation_hypers=true_model.observation_hypers,
                            activation_hypers=true_model.activation_hypers,
                            weight_hypers=true_model.weight_hypers,
                            network_hypers=true_model.network_hypers)
    test_model.add_data(S)

    ###########################################################
    # Fit the test model with Gibbs sampling
    ###########################################################
    N_samples = 10
    samples = []
    lps = []
    for itr in xrange(N_samples):
        lps.append(test_model.log_probability())
        samples.append(test_model.copy_sample())

        print ""
        print "Gibbs iteration ", itr
        print "LP: ", lps[-1]

        test_model.resample_model()

    with open("gibbs_profile.txt", "w") as f:
        show_line_stats(f)



demo(11223344)

import numpy as np

from graphistician.adjacency import BernoulliAdjacencyDistribution
from graphistician.weights import FixedGaussianWeightDistribution
from graphistician.networks import FactorizedNetworkDistribution


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


    N = 27          # Number of neurons
    T = 60000       # Number of time bins
    dt = 1.0        # Time bin width
    dt_max = 10.0   # Max time of synaptic influence
    B = 2           # Number of basis functions for the weights

    #   Bias hyperparameters
    bias_hypers = {"mu_0": -1.0, "sigma_0": 0.25}

    p = 0.5                 # Probability of connection for each pair of clusters
    mu = np.zeros((B,))     # Mean weight for each pair of clusters
    sigma = 1.0 * np.eye(B) # Covariance of weight for each pair of clusters

    # Define the true network model for the GLM
    true_network = FactorizedNetworkDistribution(
        N,
        BernoulliAdjacencyDistribution, {"p": p},
        FixedGaussianWeightDistribution, {"B": B, "mu": mu, "sigma": sigma})

    true_model = Population(N=N, dt=dt, dt_max=dt_max, B=B,
                       bias_hypers=bias_hypers,
                       network=true_network)

    ###########################################################
    # Create a test spike-and-slab model
    ###########################################################
    # Create another copy of the model with the true network model
    test_network = FactorizedNetworkDistribution(
        N,
        BernoulliAdjacencyDistribution, {"p": p},
        FixedGaussianWeightDistribution, {"B": B, "mu": mu, "sigma": sigma})

    test_model = Population(N=N, dt=dt, dt_max=dt_max, B=B,
                            bias_hypers=bias_hypers,
                            network=test_network)


    # Sample some synthetic data from the true model
    S = true_model.generate(T=T, keep=True, verbose=False)

    # Add training data in chunks
    chunksz = 1024
    for offset in xrange(0, T, chunksz):
        test_model.add_data(S[offset:min(offset+chunksz,T)])

    ###########################################################
    # Fit the test model with Gibbs sampling
    ###########################################################
    N_samples = 3
    samples = []
    lps = []
    for itr in xrange(N_samples):
        lps.append(test_model.log_probability())
        samples.append(test_model.copy_sample())

        print ""
        print "Gibbs iteration ", itr
        print "LP: ", lps[-1]

        test_model.collapsed_resample_model()

    with open("gibbs_profile.txt", "w") as f:
        show_line_stats(f)



demo(11223344)

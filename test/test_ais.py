# Run as script using 'python -m test.synth'
import sys

import numpy as np
np.random.seed(0)

from scipy.misc import logsumexp
import matplotlib.pyplot as plt

from graphistician.adjacency import BernoulliAdjacencyDistribution, BetaBernoulliAdjacencyDistribution
from graphistician.weights import FixedGaussianWeightDistribution

from graphistician.networks import FactorizedNetworkDistribution

from pyglm.models import Population

def ais(model, N_samples=10, B=1000, steps_per_B=1):
    """
    Use AIS to approximate the marginal likelihood of a GLM under various
    network models
    """
    betas = np.linspace(0, 1, B)

    # Sample m points
    lw = np.zeros(N_samples)
    for m in range(N_samples):
        # Initialize the model with a draw from the prior
        # This is equivalent to sampling with temperature=0
        model.collapsed_resample_model(temperature=0.0)

        # Keep track of the log of the m-th weight
        # It starts at zero because the prior is assumed to be normalized
        lw[m] = 0.0

        # Sample the intermediate distributions
        for b in xrange(1,B):
        # for (b,beta) in zip(range(1,B), betas[1:]):
            # print "M: %d\tBeta: %.3f" % (m,beta)
            sys.stdout.write("M: %d\tBeta: %.3f \r" % (m,betas[b]))
            sys.stdout.flush()

            # Compute the ratio of this sample under this distribution
            # and the previous distribution. The difference is added
            # to the log weight
            curr_lp = model.log_probability(temperature=betas[b])
            prev_lp = model.log_probability(temperature=betas[b-1])
            lw[m] += curr_lp - prev_lp

            # Sample the model at temperature betas[b]
            # Take some number of steps per beta in hopes that
            # the Markov chain will reach equilibrium.
            for s in range(steps_per_B):
                model.collapsed_resample_model(temperature=betas[b])

        print ""
        print "W: %f" % lw[m]

    # Compute the mean of the weights to get an estimate of the normalization constant
    log_Z = -np.log(N_samples) + logsumexp(lw)

    # TODO: Compute 95% confidence interval with bootstrap

    return log_Z


###########################################################
N = 10                                                  # Number of neurons
T = 10000                                                # Number of time bins
dt = 1.0                                                # Time bin width
dt_max = 10.0                                           # Max time of synaptic influence
B = 2                                                   # Number of basis functions for the weights

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

# Create another copy of the model with the true network model
test_network = FactorizedNetworkDistribution(
    N,
    BernoulliAdjacencyDistribution, {"p": p},
    FixedGaussianWeightDistribution, {"B": B, "mu": mu, "sigma": sigma})

test_model = Population(N=N, dt=dt, dt_max=dt_max, B=B,
                        bias_hypers=bias_hypers,
                        network=test_network)

# Define an alternative model for comparison
alt_network = FactorizedNetworkDistribution(
    N,
    BernoulliAdjacencyDistribution, {"p": p / 2.0},
    FixedGaussianWeightDistribution, {"B": B, "mu": mu, "sigma": sigma})

alt_model = Population(N=N, dt=dt, dt_max=dt_max, B=B,
                       bias_hypers=bias_hypers,
                       network=alt_network)

# Sample some synthetic data from the true model
S = true_model.generate(T=T, keep=True, verbose=False)

# Add training data in chunks
chunksz = 1024
for offset in xrange(0, T, chunksz):
    test_model.add_data(S[offset:min(offset+chunksz,T)])
    alt_model.add_data(S[offset:min(offset+chunksz,T)])

# Estimate the marginal likelihood under the true and alternative models
test_ml = ais(test_model, B=1000)
alt_ml  = ais(alt_model, B=1000)


# Plot the marginal likelihood results
plt.figure()
plt.bar(np.arange(2), [test_ml, alt_ml], width=0.8)
plt.xticks(np.arange(2)+0.4)
plt.gca().set_xticklabels(['True', 'Alt'], rotation='vertical')
plt.show(block=True)

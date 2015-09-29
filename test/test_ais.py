import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt

from graphistician.adjacency import BernoulliAdjacencyDistribution, BetaBernoulliAdjacencyDistribution
from graphistician.weights import FixedGaussianWeightDistribution

from graphistician.networks import FactorizedNetworkDistribution

from pyglm.models import Population


N = 10           # Number of neurons
T = 1000         # Number of time bins
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

# Plot estimated marginal likelihood as a function of number of steps, B
# N_samples = 20
# Bs = [100]
# mls = []
# ml_stds = []
# lws = []
# for B in Bs:
#     print "B =", B
#     ml, ml_std, lw = test_model.ais(N_samples=N_samples, B=B, verbose=False, full_output=True)
#     mls.append(ml)
#     ml_stds.append(ml_std)
#     lws.append(lw)

# Plot the marginal likelihood results
# plt.figure()
# plt.subplot(121)
# for B, ml, ml_std in zip(Bs, mls, ml_stds):
#     plt.plot([B, B], [ml-2*ml_std, ml+2*ml_std], 'k-s')
# plt.plot(Bs, mls, '+', linestyle="")
#
# plt.subplot(122)
# plt.boxplot(np.array(lws).T, positions=Bs, widths=5)
# plt.xlim(-5, Bs[-1]+10)
#
# plt.show(block=True)


# Define an alternative model for comparison
alt_network = FactorizedNetworkDistribution(
    N,
    BetaBernoulliAdjacencyDistribution, {},
    FixedGaussianWeightDistribution, {"B": B, "mu": mu, "sigma": sigma})

alt_model = Population(N=N, dt=dt, dt_max=dt_max, B=B,
                       bias_hypers=bias_hypers,
                       network=alt_network)

# Add training data in chunks
chunksz = 1024
for offset in xrange(0, T, chunksz):
    alt_model.add_data(S[offset:min(offset+chunksz,T)])

# Estimate the marginal likelihood under the true and alternative models
test_ml, test_ml_std = test_model.ais(N_samples=20, B=1000, verbose=True)
alt_ml, alt_ml_std  = alt_model.ais(N_samples=20, B=1000, verbose=True)


# Plot the marginal likelihood results
plt.figure()
plt.bar(np.arange(2), [test_ml, alt_ml], width=0.8)
plt.xticks(np.arange(2)+0.4)
plt.gca().set_xticklabels(['True', 'Alt'], rotation='vertical')
plt.show(block=True)

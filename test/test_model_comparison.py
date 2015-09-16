"""
Demo of a model comparison test where we synthesize a network
and then try to fit it with a variety of network models.
We compare models with heldout predictive likelihood.
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt

from pybasicbayes.util.text import progprint_xrange

from graphistician.adjacency import \
    LatentDistanceAdjacencyDistribution, \
    SBMAdjacencyDistribution, \
    BetaBernoulliAdjacencyDistribution

from graphistician.weights import \
    NIWGaussianWeightDistribution, \
    SBMGaussianWeightDistribution

from graphistician.networks import FactorizedNetworkDistribution

from pyglm.models import Population

seed = 1234
# seed = np.random.randint(2**32)

# Create an latent distance model with N nodes and D-dimensional locations
N = 30      # Number of training neurons
T = 10000   # Number of training time bins
B = 1       # Dimensionality of the weights
D = 2       # Dimensionality of the feature space

# Define the true network model for the GLM
true_net_model = FactorizedNetworkDistribution(
    N+1,
    LatentDistanceAdjacencyDistribution, {},
    NIWGaussianWeightDistribution, {})

true_model = Population(N+1, B=B, network=true_net_model)

# Generate synthetic data from the model
Sfull = true_model.generate(keep=False, T=T)
Strain = Sfull[:,:N]
Stest = Sfull[:,N:].ravel()

# Define a list of network models to use for comparison
adj_models = [
    LatentDistanceAdjacencyDistribution,
    SBMAdjacencyDistribution,
    BetaBernoulliAdjacencyDistribution,
]

weight_models = [
    NIWGaussianWeightDistribution,
    SBMGaussianWeightDistribution
]

# Iterate over network models
results = []
N_samples = 100
for adj_model, weight_model in itertools.product(adj_models, weight_models):
    # Create a GLM with the specified network model
    test_net_model = FactorizedNetworkDistribution(N, adj_model, {}, weight_model, {})
    test_model = Population(N, B=B, network=test_net_model)
    test_model.add_data(Strain)

    # Initialize outputs
    lps  = [test_model.log_probability()]
    plls = [test_model.heldout_neuron_log_likelihood(Strain, Stest)]

    print "A: ", adj_model.__name__
    print "W: ", weight_model.__name__

    # Fit the model with Gibbs sampling
    for smpl in progprint_xrange(N_samples):
        # Resample the model parameters and compute its log probability
        test_model.collapsed_resample_model()

        # Compute log prob and pred ll
        lps.append(test_model.log_probability())
        plls.append(test_model.heldout_neuron_log_likelihood(Strain, Stest))

    results.append((adj_model.__name__ + "-" + weight_model.__name__, lps, plls))


colors = ['b', 'r', 'g', 'y', 'm', 'k']
plt.figure()
for col, (cls, lps, plls) in zip(colors, results):
    plt.subplot(121)
    plt.plot(lps, color=col)
    plt.xlabel("Iteration")
    plt.ylabel("LP")

    plt.subplot(122)
    plt.plot(plls, color=col)
    plt.xlabel("Iteration")
    plt.ylabel("PLL")

plt.show()


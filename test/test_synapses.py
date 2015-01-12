"""
Unit tests for the synapse models
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot, invgamma

from pyglm.populations import *
from pyglm.deps.pybasicbayes.distributions import DiagonalGaussian


seed = np.random.randint(2**16)
print "Setting random seed to ", seed
np.random.seed(seed)

def create_simple_population(N=1, dt=0.001, T=100,
                             alpha_0=10.0, beta_0=10.0,
                             mu_bias=-3.0, sigma_bias=0.5**2,
                             mu_w=-0.5, sigma_w=0.5**2,
                             rho=0.5):
    # Set the model parameters
    B = 1       # Number of basis functions
    neuron_hypers = {'alpha_0' : alpha_0,
                     'beta_0' : beta_0}

    global_bias_hypers= {'mu' : mu_bias,
                         'sigmasq' : sigma_bias}

    network_hypers = {'rho' : rho,
                      'weight_prior_class' : DiagonalGaussian,
                      'weight_prior_hypers' :
                          {
                              'mu' : mu_w * np.ones((B,)),
                              'sigmas' : sigma_w * np.ones(B)
                          },

                      'refractory_rho' : rho,
                      'refractory_prior_class' : DiagonalGaussian,
                      'refractory_prior_hypers' :
                          {
                              'mu' : mu_w * np.ones((B,)),
                              'sigmas' : sigma_w * np.ones(B)
                          },
                     }

    population = ErdosRenyiBernoulliPopulation(
            N, B=B, dt=dt,
            global_bias_hypers=global_bias_hypers,
            neuron_hypers=neuron_hypers,
            network_hypers=network_hypers,
            )

    population.generate(size=T, keep=True)
    return population

def unit_test_resample():
    """
    Test the conditional distribution over weights of a neuron
    """
    population = create_simple_population()
    synapse = population.neuron_models[0].synapse_models[0]

    # Choose random covariates x,y
    T = 100
    x = np.random.randn(T,synapse.D_in)
    y = np.random.randn(T,1)
    xy = np.hstack((x,y))

    # Choose two random weights
    w1 = np.random.randn(synapse.D_in, 1)
    w2 = np.random.randn(synapse.D_in, 1)

    # Make sure the ratio of posterior distributions p(w1 | xy) / p(w2 | xy)
    # is equal to the ratio of joints p(w1, xy) / p(w2, xy)
    # These calculations execute different code paths and make sure
    # that we are computing sufficient statistics correctly
    synapse.set_weights(w1)
    cond1 = synapse.cond_w(xy).log_likelihood(w1)
    joint1 = synapse.log_likelihood(xy) + synapse.weights_prior.log_likelihood(w1)

    synapse.set_weights(w2)
    cond2 = synapse.cond_w(xy).log_likelihood(w2)
    joint2 = synapse.log_likelihood(xy) + synapse.weights_prior.log_likelihood(w2)

    print "log cond ratio: ", cond1 - cond2
    print "log joint ratio: ", joint1 - joint2

    assert np.allclose(cond1 - cond2,
                       joint1 - joint2), \
           "ERROR: Ratio of conditionals does not match ratio of joints!"

def unit_test_log_marginal_likelkhood():
    """
    Unit test the log marginal likelihood calculation
    """
    from pyglm.deps.pybasicbayes.distributions import GaussianFixed
    population = create_simple_population()
    synapse = population.neuron_models[0].synapse_models[0]

    # Choose random covariates x,y
    T = 100
    D = synapse.D_in
    X = np.random.randn(T,D)
    y = np.random.randn(T,1)
    data = np.hstack((X,y))

    mll = synapse.log_marginal_likelihood(data)

    # Debug the marginal likelihood calculation
    mu_marg = X.dot(synapse.mu_w.T).reshape((T,))
    # Covariances add
    Sig_marg = np.asscalar(synapse.sigma) * np.eye(T) + X.dot(synapse.Sigma_w.dot(X.T))
    # Compute the marginal log likelihood
    mll_true = GaussianFixed(mu_marg, Sig_marg).log_likelihood(y)

    print "MLL: ", mll
    print "MLL true: ", mll_true

    assert np.allclose(mll, mll_true), "ERROR: Marginal log likelihood calculations do not match!"

unit_test_resample()
unit_test_log_marginal_likelkhood()


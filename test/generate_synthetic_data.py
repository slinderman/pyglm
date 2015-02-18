"""
Generate synthetic datasets for testing. We try to tune the model
parameters such that the number of synthetic spikes is realistic.
"""
import numpy as np
import matplotlib.pyplot as plt

from oldpyglm.populations import ErdosRenyiBernoulliPopulation
from oldpyglm.deps.pybasicbayes.distributions import DiagonalGaussian
from oldpyglm.utils.utils import logit


def create_simple_population(N=1, B=3, dt=0.001,
                             mu_bias=-3.0, sigma_bias=0.5**2,
                             mu_w=0.0, sigma_w=1.0**2,
                             rho=0.25):
    # Set the model parameters
    neuron_hypers = {'alpha_0' : 3.0, 'beta_0' : 0.1}

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

    return population

def get_er_bernoulli_prior(N, mu_w, sigma_w, dt,
                           refractory_mu_w=None,
                           desired_rate=25):
    """
    Compute stable parameters for an Erdos Renyi population with
    N neurons and Gaussian weight distribution. This is purely
    heuristic, there is not theory of stability for this model.

    :param N:               Number of neurons
    :param mu_w:            Mean weight
    :param sigma_w:         Variance of weight
    :param dt:              Time bin size
    :param refractory_mu_w: The mean self weight, if different from mu_w
    :param desired_rate:    Desired firing rate (spikes/sec)
    :return:                Set of prior parameters for the population
    """

    # First set the mean bias to achieve the desired average rate
    mu_bias = logit(desired_rate * dt)

    # Set the variance of the bias such that three s.d. is ~10% of mean
    sigma_bias = ((mu_bias-logit(desired_rate*dt*0.9))/3.0)**2

    # Set the sparsity level such that the eigenvalues of the effective
    # weight matrix are less than one. This is an intuition based on theoretical
    # results for the stability of Poisson neurons.
    tol = 0.3
    maxeig = 1.0-tol

    # With a refractory weight we can afford slightly stronger weights
    if refractory_mu_w is not None:
        maxeig -= refractory_mu_w

    # TODO: Factor in mu_w. For now make sure it's ~= 0
    assert np.allclose(mu_w, 0.0), "ERROR: Stability calculations are based on mu_w ~= 0!"

    rho = maxeig**2/N/sigma_w
    rho = min(rho, 1.0)

    return mu_bias, sigma_bias, rho

def generate_data(N, T, dt):
    """
    Draw a random population from the prior. Set the prior parameters
    such that the population will yield reasonable spike trains.

    :param N:
    :param T:
    :param dt:
    :return:
    """
    mu_w = 0.0
    sigma_w = 1.0
    # Get prior parameters that will hopefully be stable
    mu_bias, sigma_bias, rho = get_er_bernoulli_prior(N, mu_w=mu_w,
                                                      )

    population = create_simple_population(N)
    print "A: ", population.A

    # Sample data from the first population
    S, _ = population.generate(size=T, keep=False)
    print "N spikes:\n", ["\t%d: %d\n" % (i,s) for i,s in enumerate(S.sum(0))]


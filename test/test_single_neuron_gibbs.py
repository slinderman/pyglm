"""
Unit tests for the synapse models
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot, invgamma

from oldpyglm.populations import *
from oldpyglm.deps.pybasicbayes.distributions import DiagonalGaussian


seed = np.random.randint(2**16)
print "Setting random seed to ", seed
np.random.seed(seed)

def create_simple_population(N=1, dt=0.001, T=1000,
                             alpha_0=100.0, beta_0=10.0,
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

def test_single_neuron_gibbs():
    """
    Test the mean field updates for synapses
    """
    population = create_simple_population(N=2, T=10000)
    neuron = population.neuron_models[0]
    synapse = neuron.synapse_models[0]
    data = neuron.data_list[0]

    plt.ion()
    plt.figure()
    plt.plot(data.psi, '-b')
    plt.plot(np.nonzero(data.counts)[0], data.counts[data.counts>0], 'ko')
    psi = plt.plot(data.psi, '-r')[0]
    plt.show()


    print "A_true: ", neuron.An
    print "W_true: ", neuron.weights
    print "b_true: ", neuron.bias

    print "--" * 20
    print "A:      ", neuron.An
    print "W:      ", neuron.weights
    print "bias:   ", neuron.bias
    print "--" * 20


    raw_input("Press enter to continue...")

    N_iter = 10000
    vlbs   = np.zeros(N_iter)
    for itr in xrange(N_iter):
        vlbs[itr] = neuron.resample_model()

        print "Iteration: ", itr
        print "A:      ", neuron.An
        print "W:      ", neuron.weights
        print "bias:   ", neuron.bias
        print "--" * 20

        psi.set_data(np.arange(data.T), data.psi)

        plt.pause(0.001)

test_single_neuron_gibbs()


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

def create_simple_population(N=10, dt=0.001, T=1000,
                             alpha_0=1.0, beta_0=1.0,
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

def test_meanfield_update_synapses():
    """
    Test the mean field updates for synapses
    """
    population = create_simple_population(N=10, T=10000)
    neuron = population.neuron_models[0]
    synapse = neuron.synapse_models[0]
    data = neuron.data_list[0]

    plt.ion()
    plt.figure()
    plt.plot(data.psi, '-b')
    plt.plot(np.nonzero(data.counts)[0], data.counts[data.counts>0], 'ko')
    mf_psi = plt.plot(data.mf_expected_psi(), '-r')
    ln_sigma_psi1 = plt.plot(data.mf_expected_psi() + 2*np.sqrt(data.mf_marginal_variance_psi()), ':r')
    ln_sigma_psi2 = plt.plot(data.mf_expected_psi() - 2*np.sqrt(data.mf_marginal_variance_psi()), ':r')
    plt.show()


    print "A_true: ", neuron.An
    print "W_true: ", neuron.weights
    print "b_true: ", neuron.bias

    print "--" * 20

    print "mf_rho: ", neuron.mf_rho
    print "mf_mu:  ", neuron.mf_mu_w
    print "mf_sig: ", neuron.mf_Sigma_w
    print "mf_mu_b: ", neuron.bias_model.mf_mu_bias
    print "mf_sigma_b: ", neuron.bias_model.mf_sigma_bias

    print "--" * 20

    raw_input("Press enter to continue...")

    N_iter = 100
    vlbs   = np.zeros(N_iter)
    for itr in xrange(N_iter):
        vlbs[itr] = neuron.meanfield_coordinate_descent_step()

        print "Iteration: ", itr, "\tVLB: ", vlbs[itr]
        print "mf_rho: ", neuron.mf_rho
        # print "mf_mu:  ", neuron.mf_mu_w
        # print "mf_sig: ", neuron.mf_Sigma_w
        # print "mf_mu_b: ", neuron.bias_model.mf_mu_bias
        # print "mf_sigma_b: ", neuron.bias_model.mf_sigma_bias

        print "--" * 20

        mu_psi = data.mf_expected_psi()
        sig_psi = np.sqrt(data.mf_marginal_variance_psi())
        mf_psi[0].set_data(np.arange(data.T), mu_psi)
        ln_sigma_psi1[0].set_data(np.arange(data.T), mu_psi + 2*sig_psi)
        ln_sigma_psi2[0].set_data(np.arange(data.T), mu_psi - 2*sig_psi)

        plt.pause(0.001)

    plt.ioff()
    plt.figure()
    plt.plot(vlbs)
    plt.xlabel("Iteration")
    plt.ylabel("VLB")
    plt.show()

test_meanfield_update_synapses()


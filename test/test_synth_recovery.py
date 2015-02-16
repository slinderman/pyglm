from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from pyglm.populations import ErdosRenyiBernoulliPopulation
from pyglm.deps.pybasicbayes.distributions import DiagonalGaussian


# seed = np.random.randint(2**16)
seed = 1234
print "Setting random seed to ", seed
np.random.seed(seed)

def create_simple_population(N=1, B=1,
                             mu_bias=-3.0, sigma_bias=0.5**2,
                             mu_w=0.0, sigma_w=1.0**2,
                             rho=0.25):
    dt = 0.001

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

def test_synth_recovery(N=2, T=10000, N_samples=1000):
    # Create two populations
    true_population = create_simple_population(N=N)
    test_population = create_simple_population(N=N)
    test_population.initialize_to_empty()

    print "true A: ", true_population.A

    # Sample data from the first population
    S, _ = true_population.generate(size=T, keep=True)
    print "N spikes: ", S.sum(0)

    # Fit the second population to the synthetic data
    test_population.add_data(S)

    bias_samples = []
    sigmas_samples = []
    A_samples = []
    w_samples = []

    for s in xrange(N_samples):
        print "Iteration: ", s
        test_population.resample_model(do_resample_network=False,
                                       do_resample_bias_prior=False,
                                       do_resample_latent=False)

        # Collect samples
        bias_samples.append(test_population.biases.copy())
        sigmas_samples.append(test_population.etas)
        A_samples.append(test_population.A.copy())
        w_samples.append(test_population.weights.copy())

        print "Sigma: ", test_population.etas

    # Convert samples to arrays
    offset = N_samples // 2
    bias_samples = np.array(bias_samples)[offset:,...]
    sigmas_samples = np.array(bias_samples)
    w_samples = np.array(w_samples)[offset:,...]
    A_samples = np.array(A_samples)[offset:,...]

    # Compute means and standard deviations
    bias_mean = bias_samples.mean(0)
    bias_std = bias_samples.std(0)
    print "True bias: \n", true_population.biases
    print "Mean bias: \n", bias_mean, " +- ", bias_std

    w_mean = w_samples.mean(0)
    w_std = w_samples.std(0)
    print "True w: \n", true_population.weights * true_population.A[:,:,None]
    print "Mean w: \n", w_mean, " +- ", w_std

    A_mean = A_samples.mean(0)
    print "True A: \n", true_population.A
    print "Mean A: \n", A_mean

    plt.plot(sigmas_samples[:,0,0])

test_synth_recovery()


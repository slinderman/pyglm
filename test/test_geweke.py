from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot

from pyglm.populations import *
from pyglm.deps.pybasicbayes.distributions import DiagonalGaussian, GaussianFixed
from pyglm.utils.basis import  Basis


seed = np.random.randint(2**16)
# seed = 1234
print "Setting random seed to ", seed
np.random.seed(seed)

################
#  parameters  #
################
N = 1
dt = 0.001
T = 100
N_samples = 10000

# Basis parameters
B = 2       # Number of basis functions
dt_max = 0.1      # Number of time bins over which the basis extends
basis_parameters = {'type' : 'cosine',
                    'n_eye' : 0,
                    'n_bas' : B,
                    'a' : 1.0/120,
                    'b' : 0.5,
                    'L' : 100,
                    'orth' : False,
                    'norm' : False
                    }
basis = Basis(B, dt, dt_max, basis_parameters)

#############################
#  generate synthetic data  #
#############################
spike_train_hypers = {}

# global_bias_class = GaussianFixed
mu_bias = -3
sigma_bias = 0.001
global_bias_hypers= {'mu' : mu_bias,
                     'sigmasq' : sigma_bias}
# global_bias_hypers = {
#                      'mu_0' : -3.0,
#                      'kappa_0' : 1.0,
#                      'sigmasq_0' : 1.0,
#                      'nu_0' : 1.0
#                     }

mu_w = 0
sigma_w = 1.0
network_hypers = {
                  # 'rho' : 1.0,
                  'weight_prior_class' : DiagonalGaussian,
                  'weight_prior_hypers' :
                      {
                          'mu_0' : 0.0 * np.ones((basis.B,)),
                          'nus_0' : 1.0/N**2,
                          'alphas_0' : 10.0,
                          'betas_0' : 10.0
                      },

                  # 'refractory_rho' : 0.5,
                  'refractory_prior_class' : DiagonalGaussian,
                  # 'refractory_prior_hypers' :
                  #     {
                  #         'mu_0' : 0.0 * np.ones((basis.B,)),
                  #         'nus_0' : 1.0/N,
                  #         'alphas_0' : 1.,
                  #         'betas_0' : 1.
                  #     },
                  'refractory_prior_hypers' :
                      {
                          'mu' : mu_w * np.ones((basis.B,)),
                          'sigmas' : sigma_w * np.ones(basis.B)
                      },
                 }

population = CompleteBernoulliPopulation(
        N, basis,
        global_bias_hypers=global_bias_hypers,
        neuron_hypers=spike_train_hypers,
        network_hypers=network_hypers,
        )

S, Xs = population.generate(size=T, keep=False)
Xs = [X[:T,:] for X in Xs]
data = np.hstack(Xs + [S])

##############
# plotting  #
#############
# t_lim = [0,1]
# axs, true_lns = population.plot_mean_spike_counts(Xs, dt=dt, S=S, color='k', t_lim=t_lim)
# plt.ion()
# plt.show()
# plt.pause(0.01)


#############
#  sample!  #
#############
# Initialize the parameters with an empty network
population.add_data(data)
population.initialize_to_empty()


ll_samples = []
A_samples = []
w_samples = []
bias_samples = []
sigma_samples = []

for s in range(N_samples):
    print "Iteration ", s
    ll = population.heldout_log_likelihood(data)
    print "LL: ", ll
    population.resample_model(do_resample_bias=False,
                              do_resample_bias_prior=False,
                              do_resample_latent=False,
                              do_resample_network=False,
                              do_resample_sigma=False,
                              do_resample_synapses=True)

    # Remove old data
    population.pop_data()

    # Generate new data (Geweke step)
    S, Xs = population.generate(size=T, keep=True)
    Xs = [X[:T,:] for X in Xs]
    data = np.hstack(Xs + [S])

    # Plot this sample
    # population.plot_mean_spike_counts(Xs, dt=dt, lns=true_lns)

    # Collect samples
    ll_samples.append(ll)
    A_samples.append(population.A.copy())
    bias_samples.append(population.biases.copy())
    sigma_samples.append(population.sigmas.copy())
    w_samples.append(population.weights.copy())

# Convert samples to arrays
A_samples = np.array(A_samples)
w_samples = np.array(w_samples)
bias_samples = np.array(bias_samples)
sigma_samples = np.array(sigma_samples)

# Make Q-Q plots
fig = plt.figure()
A_mean = A_samples.mean(0)
print "Mean A: \n", A_mean

w_mean = w_samples.mean(0)
print "Mean w: \n", w_mean
w_ax = fig.add_subplot(111)
w_dist = norm(0, sigma_w)
probplot(w_samples[:,0,0,0], dist=w_dist, plot=w_ax)

bias_mean = bias_samples.mean(0)
print "Mean bias: ", bias_mean

sigma_mean = sigma_samples.mean(0)
print "Mean sigma: ", sigma_mean

plt.show()




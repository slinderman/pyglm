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
thin = 100

# Basis parameters
B = 1       # Number of basis functions
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
mu_bias = -3.0
sigma_bias = 0.5**2
global_bias_hypers= {'mu' : mu_bias,
                     'sigmasq' : sigma_bias}
# global_bias_hypers = {
#                      'mu_0' : -3.0,
#                      'kappa_0' : 1.0,
#                      'sigmasq_0' : 1.0,
#                      'nu_0' : 1.0
#                     }

mu_w = -0.5
sigma_w = 0.5**2
network_hypers = {
                  # 'rho' : 1.0,
                  'weight_prior_class' : DiagonalGaussian,
                  'weight_prior_hypers' :
                      # {
                      #     'mu_0' : 0.0 * np.ones((basis.B,)),
                      #     'nus_0' : 1.0/N**2,
                      #     'alphas_0' : 10.0,
                      #     'betas_0' : 10.0
                      # },
                      {
                          'mu' : mu_w * np.ones((basis.B,)),
                          'sigmas' : sigma_w * np.ones(basis.B)
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

ll_samples = []
A_samples = []
w_samples = []
bias_samples = []
sigma_samples = []

for s in xrange(N_samples):
    print "Iteration ", s
    ll = population.heldout_log_likelihood(data)
    print "LL: ", ll

    # Resampling is trickier because of the augmentation.
    # Here we're trying to integrate out omega by thinning.
    for _ in xrange(thin):
        population.resample_model(do_resample_bias=False,
                                  do_resample_bias_prior=False,
                                  do_resample_latent=False,
                                  do_resample_network=False,
                                  do_resample_sigma=False,
                                  do_resample_synapses=True,
                                  do_resample_counts=False)

    # # Remove old data, filter it, and add it back
    # for n in population.neuron_models:
    #     S = n.data_list[0].counts[:,None]
    #     fS = population.filter_spike_train(S)
    #     n.data_list[0].X = fS

    # Remove old data
    population.pop_data()

    # Generate new data (Geweke step)
    S, Xs = population.generate(size=T, keep=True)
    Xs = [X[:T,:] for X in Xs]
    data = np.hstack(Xs + [S])
    print "N spikes: ", S.sum()

    # Plot this sample
    # population.plot_mean_spike_counts(Xs, dt=dt, lns=true_lns)

    # Collect samples
    ll_samples.append(ll)
    A_samples.append(population.A.copy())
    bias_samples.append(population.biases.copy())
    sigma_samples.append(population.sigmas.copy())
    w_samples.append(population.weights.copy())
    print "W: ", population.weights

# Convert samples to arrays
A_samples = np.array(A_samples)
w_samples = np.array(w_samples)
bias_samples = np.array(bias_samples)
sigma_samples = np.array(sigma_samples)

A_mean = A_samples.mean(0)
print "Mean A: \n", A_mean

w_mean = w_samples.mean(0)
w_std = w_samples.std(0)
print "Mean w: \n", w_mean, " +- ", w_std

# Make Q-Q plots
fig = plt.figure()
w_ax = fig.add_subplot(121)
w_dist = norm(mu_w, np.sqrt(sigma_w))
probplot(w_samples[:,0,0,0], dist=w_dist, plot=w_ax)

fig.add_subplot(122)
_, bins, _ = plt.hist(w_samples[:,0,0,0], 20, normed=True, alpha=0.2)
bincenters = 0.5*(bins[1:]+bins[:-1])
plt.plot(bincenters, w_dist.pdf(bincenters), 'r--', linewidth=1)
plt.show()

bias_mean = bias_samples.mean(0)
bias_std = bias_samples.std(0)
bias_dist = norm(mu_bias, np.sqrt(sigma_bias))
print "Mean bias: ", bias_mean, " +- ", bias_std
# Make Q-Q plots
fig = plt.figure()
bias_ax = fig.add_subplot(121)
probplot(bias_samples[:,0,0], dist=bias_dist, plot=bias_ax)

fig.add_subplot(122)
_, bins, _ = plt.hist(bias_samples[:,0,0], 20, normed=True, alpha=0.2)
bincenters = 0.5*(bins[1:]+bins[:-1])
plt.plot(bincenters, bias_dist.pdf(bincenters), 'r--', linewidth=1)
plt.show()


sigma_mean = sigma_samples.mean(0)
print "Mean sigma: ", sigma_mean

plt.show()




"""
Placeholder for making the ER figure
"""
import os
import cPickle
import gzip

import numpy as np
from hips.plotting.layout import create_axis_at_location, create_figure
from hips.plotting.colormaps import gradient_cmap

from deps.pybasicbayes.distributions import DiagonalGaussian
from pyglm.utils.datahelper import load_data
from utils.basis import  Basis
from pyglm.populations import ErdosRenyiNegativeBinomialPopulation





# Load the data
# res_dir = os.path.join('results', 'rgc_nb_er', 'runs', '2')
# itr = 480
# with gzip.open(os.path.join(res_dir, 'iter%03d' % itr), 'r') as infile:
#     params = cPickle.load(infile)
#     A = params[0]
#     weights = params[1]
#     biases = params[2]


A_smpls = []
W_smpls = []
bias_smpls = []
res_dir = os.path.join('results', 'rgc_nb_er', 'runs', '2')
for itr in np.arange(300,480,10):
    with gzip.open(os.path.join(res_dir, 'iter%03d' % itr), 'r') as infile:
        params = cPickle.load(infile)
        A = params[0]
        weights = params[1]
        biases = params[2]

        A_smpls.append(A)
        W_smpls.append(weights)
        bias_smpls.append(biases)

A_smpls = np.array(A_smpls)
W_smpls = np.array(W_smpls)
bias_smpls = np.array(bias_smpls)

Weff_smpls = A_smpls[:,:,:,None] * W_smpls

# TODO: Fix this hack
#
#  Load data
#
dat_file = os.path.join('data', 'rgc.mat')
data = load_data(dat_file)

# Bin with larger bins
dt_orig = 0.001
dt = 0.01
S_orig = data['S']
T,N = S_orig.shape
S = S_orig.reshape((-1, int(dt/dt_orig), N)).sum(1)
T,N = S.shape


# Recreate the basis
dt = 0.01
B = 3              # Number of impulse response basis functions
dt_max = 0.05      # Duration of impulse response
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

# TODO: Fix this hack
spike_train_hypers = {'xi' : 10}
global_bias_hypers= {'mu' : -4,
                     'sigmasq' : 1.0,
                     'mu_0' : -5.0,
                     'kappa_0' : 1.0,
                     'sigmasq_0' : 0.1,
                     'nu_0' : 10.0
                    }
network_hypers = {'rho' : 0.1,
                  'weight_prior_class' : DiagonalGaussian,
                  'weight_prior_hypers' :
                      {
                          'mu_0' : 0.0 * np.ones((basis.B,)),
                          'nus_0' : 1.0/N**2,
                          'alphas_0' : 1.0,
                          'betas_0' : 1.0
                      },
                  'refractory_rho' : 0.9,
                  'refractory_prior_class' : DiagonalGaussian,
                  'refractory_prior_hypers' :
                      {
                          'mu_0' : 0.0 * np.ones((basis.B,)),
                          'nus_0' : 1.0/N,
                          'alphas_0' : 1.,
                          'betas_0' : 1.
                      },
                 }

population = ErdosRenyiNegativeBinomialPopulation(
    N, basis,
    global_bias_hypers=global_bias_hypers,
    neuron_hypers=spike_train_hypers,
    network_hypers=network_hypers,
    )

population.A = A
population.weights = weights
population.biases = biases

# Filter the spike train
fS = population.filter_spike_train(S)
full_data = np.hstack((np.hstack(fS), S))

# Add the data to the population
# NOTE that we are treating the population like a model in that
# we are assuming this data is static
population.add_data(full_data)

#  ############
#  Begin plotting
#  Plot the effective weight matrix
#  ###############
fig = create_figure((2.5,2.5))
cmap = gradient_cmap([[1,0,0], [1,1,1],[0,0,0]])
ax = create_axis_at_location(fig, .5, .5, 1.5, 1.625)
population.plot_weighted_network(ax=ax, cmap=cmap)
ax.plot([0,N+1],[0,N+1], ':k')
ax.plot([0,N+1],[16,16], ':k')
ax.plot([16,16],[0,N+1], ':k')
ax.set_xlabel('Post')
ax.set_xlim(0,27)
ax.set_ylabel('Pre')
ax.set_ylim(27,0)
ax.set_title('Interaction weights')
# plt.tight_layout()
# Plot the firing rates

# Add a colorbar
cbar_ax = create_axis_at_location(fig, 2.1, .5, .1, 1.625)
# cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])

# Rather than using the colorbar method, directly
# instantiate a colorbar
wmax = np.amax(abs(population.W_effective))
print wmax
from matplotlib.colorbar import ColorbarBase
cbar_ticks = np.array([-1, 0.0, 1.0])
cbar = ColorbarBase(cbar_ax, cmap=cmap,
                    values=np.linspace(-1, 1, 500),
                    boundaries=np.linspace(-1, 1, 500),
                    ticks=cbar_ticks)
cbar.set_ticklabels(['-1', '0', '+1'])
#
#
fig.savefig('cosyne/figure1a.pdf')

# Make a figure of firing rates
# neurons_to_plot = [10, 19]
# t_lim = [58,60]
# fig2 = create_figure((2.5,2.5))
# ax1 = fig2.add_subplot(211)
# ax2 = fig2.add_subplot(212)
# axs = [ax1, ax2]
#
# population.plot_mean_spike_counts(np.hstack(fS), dt=dt,
#                                  S=S, color='k',
#                                  inds=neurons_to_plot,
#                                  t_lim=t_lim,
#                                  axs=axs)
#
# ax2.set_xlabel('Time [s]')
# ax1.set_ylabel('Neuron %d' % neurons_to_plot[0])
# ax2.set_ylabel('Neuron %d' % neurons_to_plot[1])
# ax1.set_ylim([0,200])
# ax2.set_ylim([0,200])
#
# ax1.set_title('Firing Rates')
# plt.tight_layout()
# fig2.savefig('cosyne/figure1b.pdf')

# Make a figure of a few impulse responses
# conns_to_plot = [[9, 19]]
# conns_to_plot = [ [19,19], [9, 19]]

# import pdb; pdb.set_trace()

# fig3 = create_figure((1.5,2.5))
# for i,conn in enumerate(conns_to_plot):
#     # ax = fig3.add_subplot(111)
#     if i == 0:
#         ax = create_axis_at_location(fig3, 0.4, 0.625,1.0,0.5)
#     else:
#         ax = create_axis_at_location(fig3, 0.4, 1.5,1.0,0.5)
#
#     w_mu = Weff_smpls[:,conn[0],conn[1],:].mean(0)
#     w_std = Weff_smpls[:,conn[0],conn[1],:].std(0)
#     imp = basis.basis.dot(w_mu)
#     imp_std = basis.basis.dot(w_std)
#     t = dt * np.arange(imp.size)
#
#     color = 'k' if i==0 else 'r'
#     sausage_plot(ax, t, imp, imp_std, color=color)
#
#     # impm1 = basis.basis.dot(w_mu - w_std)
#     # l1 = ax.plot(t, imp, '-', c='r')
#     # l2 = ax.plot(t, impp1, '--', c='r')
#     # l3 = ax.plot(t, impm1, '--', c='r')
#     ax.plot(t, np.zeros_like(imp), ':k')
#
#
#     ax.set_xlim(0,t[-1])
#     ax.set_xticks(np.linspace(0, t[-1], 5))
#     ax.set_xticklabels((np.linspace(0, t[-1], 5)*1000).astype(int))
#
#     if i == 0:
#         ax.set_xlabel('$\Delta t \mathrm{[ms]} $')
#
#
#     # ax.set_ylabel('$w_{%d\\to%d}(\Delta t)$' % (conn[0],conn[1]))
#     ax.set_ylim(-0.75, 0.75)
#     ax.set_yticks(     [-0.75, -0.5, -0.25, 0, 0.25, 0.5,  0.75])
#     ax.set_yticklabels([-0.75, '', '',  0, '', '', 0.75])
#     # if i == 0:
#     #     ax.set_title('Impulse response')
#     ax.set_title('$w_{%d\\to%d}(\Delta t)$' % (conn[0],conn[1]))


# plt.tight_layout()
# fig3.savefig('cosyne/figure1c.pdf' )

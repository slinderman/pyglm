from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoClip

from pybasicbayes.distributions import Regression, Gaussian
from pybasicbayes.util.text import progprint_xrange
from autoregressive.distributions import AutoRegression

from pyglm.utils.utils import logistic

import pyglm.dynamic_models
reload(pyglm.dynamic_models)
from pyglm.dynamic_models import PoissonLDS, ApproxPoissonLDS, PGEmissions

from hips.plotting.colormaps import harvard_colors
from hips.plotting.sausage import sausage_plot
colors = harvard_colors()

npr.seed(1234)


#########################
#  set some parameters  #
#########################

mu_init = np.array([0.,1.])
sigma_init = 0.01*np.eye(2)

N = 4
D = 2
A = 0.99*np.array([[np.cos(np.pi/24), -np.sin(np.pi/24)],
                   [np.sin(np.pi/24),  np.cos(np.pi/24)]])
sigma_states = 0.01*np.eye(2)

#C = np.array([[2.,0.]])
#b = np.array([[-2.]])
C = np.random.randn(N, D)
b = -2 * np.ones((N,1))

###################
#  generate data  #
###################
truemodel = PoissonLDS(
    init_dynamics_distn=Gaussian(mu_init, sigma_init),
    dynamics_distn=AutoRegression(A=A,sigma=sigma_states),
    emission_distn=PGEmissions(N, D, C=C, b=b))

T = 2000
data, z_true = truemodel.generate(T)
psi_true = z_true.dot(C.T) + truemodel.emission_distn.b.T
lmbda_true = np.exp(psi_true)


###############
#  fit model  #
###############
model = ApproxPoissonLDS(
    init_dynamics_distn=Gaussian(mu_0=np.zeros(D), sigma_0=np.eye(D),
                                 kappa_0=1.0, nu_0=D+1),
    dynamics_distn=AutoRegression(A = np.zeros((D,D)), sigma=1*np.eye(D),
                                  nu_0=D+1, S_0=D*np.eye(D),
                                  M_0=np.zeros((D, D)), K_0=D*np.eye(D)),
    emission_distn=PGEmissions(N, D, sigmasq_C=1.0, mu_b=-2))

# Mask off neuron 0
mask = np.ones((T,N), dtype=np.bool)
mask[T//2:,0] = False

model.add_data(data.X, mask=mask)
states = model.states_list[0]

###############
#  movie      #
###############
xlim = (0,300)
xticks = np.arange(xlim[0], xlim[1]+1, 50)
ylim = (-0.1, data.X[slice(*xlim),:].max() + .1)
yticks = np.arange(0, ylim[1])
# plt.ion()
fig = plt.figure(figsize=(8,4))
fig.patch.set_color("w")
hs = []
for n in xrange(N):
    plt.subplot(N,1,n+1)
    tn = np.where(data.X[:,n])[0]
    sn = data.X[tn,n]
    plt.plot(tn, sn, 'ko', markersize=6, markerfacecolor='k', ls="none")
    plt.plot(lmbda_true[:,n], color=colors[0], lw=2)
    hs.append(plt.plot(states.rate[:,n], color=colors[1], lw=2)[0])

    plt.ylabel("neuron $%d$" % (n+1))
    if n == N-1:
        plt.xlabel("time")
        plt.xticks(xticks)
    else:
        plt.xticks(xticks, [])

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.yticks([0,1,2,3])
plt.pause(0.001)

def update_plot(dopause=True):
    lmbda_inf = states.rate
    for n in xrange(N):
        hs[n].set_data(np.arange(T), lmbda_inf[:,n])

    if dopause:
        plt.pause(0.001)

def make_frame(t):
    model.resample_model()
    update_plot(dopause=False)
    return mplfig_to_npimage(fig)

def step():
    model.resample_model()
    # update_plot()
    return model.log_likelihood(), model.heldout_log_likelihood(), states.rate, states.omega


# Use moviemaker to make the spine movie
# movie_file = "bayesian_plds.mp4"
# movie = VideoClip(make_frame, duration=20)
# movie.write_videofile(movie_file, fps=30)

# Inference
N_samples = 500
results = [step() for _ in progprint_xrange(N_samples)]

# Plot average results
lls = np.array([r[0] for r in results])
hlls = np.array([r[1] for r in results])
lmbdas = np.array([r[2] for r in results[N_samples//2:]])
omegas = np.array([r[3] for r in results[N_samples//2:]])

lmbda_mean = lmbdas.mean(0)
lmbda_std = lmbdas.std(0)

omega_mean = omegas.mean(0)
omega_std = omegas.std(0)

for n in xrange(N):
    hs[n].set_data([],[])
    plt.subplot(N,1,n+1)
    sausage_plot(np.arange(T), lmbda_mean[:,n], lmbda_std[:,n],
                 color=colors[1], alpha=0.5)

    plt.plot(np.arange(T), lmbda_mean[:,n], color=colors[1])
plt.pause(0.001)

# plt.ion()
fig = plt.figure(figsize=(8,4))
for n in xrange(N):
    plt.subplot(N,1,n+1)
    sausage_plot(np.arange(T), omega_mean[:,n], omega_std[:,n],
                 color=colors[2], alpha=0.5)

    plt.plot(np.arange(T), omega_mean[:,n], color=colors[2])
plt.show()

plt.figure()
plt.plot(lls)
plt.xlabel("Iteration")
plt.ylabel("LL")

plt.figure()
plt.plot(hlls)
plt.xlabel("Iteration")
plt.ylabel("Heldout LL")


plt.show()


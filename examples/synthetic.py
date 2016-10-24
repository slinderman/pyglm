import importlib
import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")
plt.ion()

from pybasicbayes.util.text import progprint_xrange

from pyglm.utils.basis import cosine_basis
from pyglm.plotting import plot_glm
from pyglm.models import SparseBernoulliGLM

T = 10000   # Number of time bins to generate
N = 4       # Number of neurons
B = 1       # Number of "basis functions"
L = 100     # Autoregressive window of influence

# Create a cosine basis to model smooth influence of
# spikes on one neuron on the later spikes of others.
basis = cosine_basis(B=B, L=L) / L

# Generate some data from a model with self inhibition
true_model = \
    SparseBernoulliGLM(N, basis=basis,
                       regression_kwargs=dict(S_w=10.0, mu_b=-2.))
for n in range(N):
    true_model.regressions[n].a[n] = True
    true_model.regressions[n].W[n,:] = -2.0
_, Y = true_model.generate(T=T, keep=True)

# Plot the true model
fig, axs, handles = true_model.plot()
plt.pause(0.1)

# Create a test model for fitting
test_model = \
    SparseBernoulliGLM(N, basis=basis,
                       regression_kwargs=dict(S_w=10.0, mu_b=-2.))

test_model.add_data(Y)

# Plot the test model
fig, axs, handles = test_model.plot(title="Sample 0")
plt.pause(0.1)

# Fit with Gibbs sampling
def _collect(m):
    return m.log_likelihood(), m.weights, m.adjacency, m.biases, m.means[0]

def _update(m, itr):
    m.resample_model()
    test_model.plot(handles=handles,
                    pltslice=slice(0, 500),
                    title="Sample {}".format(itr+1))
    return _collect(m)

N_samples = 100
samples = []
for itr in progprint_xrange(N_samples):
    samples.append(_update(test_model, itr))

# Unpack the samples
samples = zip(*samples)
lps, W_smpls, A_smpls, b_smpls, fr_smpls = tuple(map(np.array, samples))

# Plot the log likelihood per iteration
fig = plt.figure(figsize=(4,4))
plt.plot(lps)
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.tight_layout()

# Plot the posterior mean and variance
W_mean = W_smpls[N_samples//2:].mean(0)
A_mean = A_smpls[N_samples//2:].mean(0)
fr_mean = fr_smpls[N_samples//2:].mean(0)
fr_std = fr_smpls[N_samples//2:].std(0)

fig, _, _ = plot_glm(Y, W_mean, A_mean, fr_mean,
                     std_firingrates=3*fr_std, title="Posterior Mean")

plt.show()
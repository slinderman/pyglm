import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

from pybasicbayes.util.text import progprint_xrange

from pyglm.utils.basis import cosine_basis
from pyglm.models import SparseBernoulliGLM

T = 10000   # Number of time bins to generate
N = 4       # Number of neurons
B = 1       # Number of "basis functions"
L = 100     # Autoregressive window of influence

# Create a cosine basis to model smooth influence of
# spikes on one neuron on the later spikes of others.
basis = cosine_basis(B=B, L=L) / L

# Generate some data from a model with self inhibition
true_model = SparseBernoulliGLM(N, basis=basis, S_w=10.0, mu_b=-2.)
for n in range(N):
    true_model.regressions[n].a[n] = True
    true_model.regressions[n].W[0,n,:] = -2.0
_, Y = true_model.generate(T=T, keep=True)

# Create a test model for fitting
test_model = SparseBernoulliGLM(N, basis=basis, S_w=2.0, mu_b=-2.)
test_model.add_data(Y)

# Fit with Gibbs sampling
def _collect(m):
    return m.log_likelihood(), m.weights, m.adjacency, m.biases

def _update(m):
    m.resample_model()
    return _collect(m)

N_samples = 100
samples = []
for _ in progprint_xrange(N_samples):
    samples.append(_update(test_model))

# Unpack the samples
samples = zip(*samples)
lps, W_smpls, A_smpls, b_smpls = tuple(map(np.array, samples))

# Plot the log likelihood per iteration
plt.figure()
plt.plot(lps)
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")

# Plot the true and inferred weight matrix
W_true = true_model.weights
W_mean = W_smpls[N_samples//2:].mean(0)
W_lim = max(abs(W_true).max(), abs(W_mean).max())

plt.figure()
plt.subplot(121)
plt.imshow(W_true[:,:,0], vmin=-W_lim, vmax=W_lim, cmap="RdBu", interpolation="nearest")
plt.xlabel("pre")
plt.ylabel("post")
plt.title("True Weights")
plt.colorbar()

plt.subplot(122)
plt.imshow(W_mean[:,:,0], vmin=-W_lim, vmax=W_lim, cmap="RdBu", interpolation="nearest")
plt.xlabel("pre")
plt.ylabel("post")
plt.title("Mean Posterior Weights")
plt.colorbar()

# Plot the true and inferred adjacency matrix
A_true = true_model.adjacency
A_mean = A_smpls[N_samples//2:].mean(0)

plt.figure()
plt.subplot(121)
plt.imshow(A_true, vmin=0, vmax=1, cmap="Greys", interpolation="nearest")
plt.xlabel("pre")
plt.ylabel("post")
plt.title("True Adjacency Matrix")

plt.subplot(122)
plt.imshow(A_mean, vmin=0, vmax=1, cmap="Greys", interpolation="nearest")
plt.xlabel("pre")
plt.ylabel("post")
plt.title("Mean Posterior Adjacency Matrix")

# Plot the true and inferred rates
plt.figure()
pltslice = slice(0, min(1000,T))
for n in range(N):
    plt.subplot(N,1,n+1)
    tn = np.where(Y[pltslice,n])[0]
    plt.plot(tn, np.ones_like(tn), 'ko', markersize=4)
    plt.plot(true_model.means[0][pltslice,n], label="True")
    plt.plot(test_model.means[0][pltslice,n], label="Test")
    plt.ylim(-0.05, 1.1)
    plt.ylabel("Probability")

    if n == 0:
        plt.title("True and Inferred Rates")
        plt.legend()

    if n == N-1:
        plt.xlabel("Time")

plt.show()
"""
Test the Bernoulli regression models.
"""
import numpy as np
# np.random.seed(1)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

from pybasicbayes.util.text import progprint_xrange
from pyglm.regression import SparseBernoulliRegression

N = 2
B = 1
T = 1000

# Make a regression model and simulate data
true_reg = SparseBernoulliRegression(N, B)
X = np.random.randn(T, N*B)
y = true_reg.rvs(X=X)

# Make a test regression and fit it
test_reg = SparseBernoulliRegression(N, B)
test_reg.a = np.bitwise_not(true_reg.a)

def _collect(r):
    return r.a.copy(), r.W.copy(), r.log_likelihood((X, y)).sum()

def _update(r):
    r.resample([(X,y)])
    return _collect(r)

smpls = [_collect(test_reg)]
for _ in progprint_xrange(100):
    smpls.append(_update(test_reg))

smpls = zip(*smpls)
As, Ws, lps = tuple(map(np.array, smpls))

# Plot the regression results
plt.figure()
lim = (-3, 3)
npts = 50
x1, x2 = np.meshgrid(np.linspace(*lim, npts), np.linspace(*lim, npts))

plt.subplot(121)
mu = true_reg.mean(np.column_stack((x1.ravel(), x2.ravel())))
plt.imshow(mu.reshape((npts, npts)),
           cmap="Greys", vmin=-0, vmax=1,
           alpha=0.8,
           extent=lim + tuple(reversed(lim)))
plt.scatter(X[:,0], X[:,1], c=y, vmin=0, vmax=1)
plt.xlim(lim)
plt.ylim(lim)
plt.colorbar()

plt.subplot(122)
mu = test_reg.mean(np.column_stack((x1.ravel(), x2.ravel())))
plt.imshow(mu.reshape((npts, npts)),
           cmap="Greys", vmin=0, vmax=1,
           alpha=0.8,
           extent=lim + tuple(reversed(lim)))
plt.scatter(X[:,0], X[:,1], c=y, vmin=0, vmax=1)
plt.xlim(lim)
plt.ylim(lim)
plt.colorbar()

print("True A: {}".format(true_reg.a))
print("Mean A: {}".format(As.mean(0)))

plt.show()


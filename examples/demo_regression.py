import numpy as np

from nb.models import ScalarRegressionFixedCov

# Make a model
D = 2
xi = 10
b = -1.0
A = np.ones((D+1,))
A[-1] = b

# A_true = np.ones((D,))
A_true = np.array([0,1])
mu_A = np.zeros((D,))
eta = 1.0
Sigma_A = eta * np.eye(D)
sigma =  0.1
true_model = ScalarRegressionFixedCov(w=A_true, sigma=sigma)

# Make synthetic data
T = 100
X = np.random.normal(size=(T,D))
# X = np.hstack((X, b*np.ones((T,1))))
xy = true_model.rvs(X)
y = xy[:,-1]

print "Max y:\t", np.amax(y)

# Scatter the data
import matplotlib.pyplot as plt
plt.figure()
plt.gca().set_aspect('equal')
plt.scatter(X[:,0], X[:,1], c=y, cmap='hot')
plt.colorbar()

# Plot A
l_true = plt.plot([0, A_true[0]], [0, A_true[1]], ':k')

# Fit the model with a matrix normal prior on A and sigma
inf_model = ScalarRegressionFixedCov(mu_w=mu_A, Sigma_w=Sigma_A, sigma=sigma)

# Plot the initial sample
l_inf = plt.plot([0, inf_model.w[0]], [0, inf_model.w[1]], '-k')

# Begin interactive plotting
plt.ion()
plt.show()

raw_input("Press any key to begin sampling...\n")

# MCMC
for i in range(100):
    print "ll:\t", inf_model.log_likelihood(xy)
    print "A:\t", inf_model.w
    inf_model.resample(xy)
    l_inf[0].set_data([0, inf_model.w[0]], [0, inf_model.w[1]])
    plt.pause(0.01)

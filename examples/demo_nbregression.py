import numpy as np
from nb.models import NegativeBinomialRegression


def test_nbregression():
    # Make a model
    D = 2
    xi = 10
    b = -1.0
    A = np.ones((1,D+1))
    A[0,-1] = b

    sigma =  0.001 * np.ones((1,1))
    true_model = NegativeBinomialRegression(A=A, sigma=sigma, xi=xi)

    # Make synthetic data
    T = 10000
    X = np.random.normal(size=(T,D))
    X = np.hstack((X, b*np.ones((T,1))))
    y = true_model.rvs(X, return_xy=False).reshape((T,))

    psi = X.dot(A.T)
    print "Max Psi:\t", np.amax(psi)
    print "Max p:\t", np.amax(np.exp(psi)/(1.0 + np.exp(psi)))
    print "Max y:\t", np.amax(y)

    # Scatter the data
    import matplotlib.pyplot as plt
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.scatter(X[:,0], X[:,1], c=y, cmap='hot')
    plt.colorbar()
    plt.show()

    # Fit the model with a matrix normal prior on A and sigma
    nu = D+1
    M = np.zeros((1,D+1))
    S = np.eye(1)
    K = np.eye(D+1)
    inf_model = NegativeBinomialRegression(nu_0=nu, S_0=S, M_0=M, K_0=K, xi=xi)

    # Add data
    inf_model.add_data(X, y)

    # MCMC
    for i in range(100):
        inf_model.resample()
        print "ll:\t", inf_model.log_likelihood(X, y)
        print "A:\t", inf_model.A
        print "sig:\t", inf_model.sigma



test_nbregression()

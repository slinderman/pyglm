import numpy as np

from pyglm.neuron import BernoulliSparseNeuron
from deps.pybasicbayes.distributions import Gaussian, GaussianFixedCov, GaussianFixedMean


def demo_ss_nbregression(T=1000, N=20, B=5, N_samples=100, do_plot=False, burnin=0.5):
    """
    A simple demo of spike and slab negative binomial regression

    :param T: Number of time bins
    :param N: Number of neurons
    :param B: Dimensionality of weights for each pair of neurons
    :param do_plot: whether or not to plot the
    :return:
    """
    # Make a model

    # NB dispersion
    xi = 10

    # Additive noise in psi
    sigma_psi = np.array(0.1)

    # Gaussian distributed term
    b = np.array([-3.0])
    sigma_b = np.array(1.0)

    # Multivariate Gaussian weights
    w_prior = Gaussian(np.zeros(B), 1.0/B * np.eye(B))
    w = w_prior.rvs(N)

    # Spike and slab indicator variables
    rho = min(0.8, 3./N)
    A = np.random.rand(N) < rho
    print "A: ", A

    # Make regression models for each dimension
    true_bias_model = GaussianFixedCov(mu=b, sigma=sigma_b)
    true_noise_model  = GaussianFixedMean(mu=np.zeros(1,), sigma=sigma_psi)
    true_regression_models = [ScalarRegressionFixedCov(w=w[d,:], sigma=sigma_psi) for d in range(N)]
    # true_model = SpikeAndSlabNegativeBinomialRegression(true_bias_model, true_regression_models, true_noise_model, As=A, xi=xi)
    true_model = SpikeAndSlabLogisticRegression(true_bias_model, true_regression_models, true_noise_model, As=A)

    # Make synthetic data
    datasets = 1
    Xss = []
    ys = []
    for i in range(datasets):
        X = np.random.normal(size=(T,N*B))
        Xs = [X[:,(B*d):(B*(d+1))].reshape((T,B)) for d in range(N)]
        y = true_model.rvs(Xs, return_xy=False)
        print "Max y:\t", np.amax(y)

        Xss.append(Xs)
        ys.append(y)

    if np.amax(y) > 100:
        raw_input("Max y > 100. This may crash. Are you sure you want to continue? Press any key to continue\n")

    # Fit with the same model
    inf_noise_model  = GaussianFixedMean(mu=np.zeros(1,), nu_0=1, lmbda_0=1*np.eye(1))
    inf_bias_model = GaussianFixedCov(mu_0=np.zeros((1,)), sigma_0=np.ones((1,1)), sigma=inf_noise_model.sigma)
    inf_regression_models = [ScalarRegressionFixedCov(weights_prior=w_prior,
                                                      sigma=inf_noise_model.sigma)
                             for _ in range(N)]

    # inf_model = SpikeAndSlabNegativeBinomialRegression(inf_bias_model, inf_regression_models, inf_noise_model,
    #                                                    rho_s=rho*np.ones(N), xi=xi)
    inf_model = SpikeAndSlabLogisticRegression(inf_bias_model, inf_regression_models, inf_noise_model,
                                               rho_s=rho*np.ones(N))

    # Add data
    for Xs,y in zip(Xss, ys):
        inf_model.add_data(Xs, y)

    # Prepare samples
    A_samples = []
    w_samples = []
    bias_samples = []
    sigma_samples = []
    ll_samples = []

    if do_plot:
        assert B == 1, "Can only plot 1D regressors"
        # Scatter the data
        import matplotlib.pyplot as plt
        plt.figure()
        plt.gca().set_aspect('equal')

        inds = np.where(ys[0])[0]
        plt.scatter(Xss[0][0][inds], Xss[0][1][inds], c=ys[0][inds], cmap='hot')
        plt.title('%d / %d Datapoints' % (len(np.unique(inds)), T))
        plt.colorbar(label='Count')

        # Plot A
        l_true = plt.plot([0, A[0] * w[0]], [0, A[1] * w[1]], ':k')

        # Plot the initial sample
        l_inf = plt.plot([0, inf_model.As[0] * inf_regression_models[0].w[0]],
                         [0, inf_model.As[1] * inf_regression_models[1].w[0]], '-k')

        plt.ion()
        plt.show()

        # MCMC
        raw_input("Press any key to continue...\n")
        for i in range(N_samples):
            print "Iteration ", i
            inf_model.resample()

            A_samples.append(inf_model.As.copy())
            w_samples.append(inf_model.weights.copy())
            bias_samples.append(inf_model.bias.copy())
            sigma_samples.append(inf_model.eta.copy())
            ll_samples.append(inf_model.log_likelihood(Xs,y))


            l_inf[0].set_data([0, inf_model.As[0] * inf_regression_models[0].w[0]],
                              [0, inf_model.As[1] * inf_regression_models[1].w[0]])
            plt.pause(0.001)
    else:
        # Profile instead of plot
        import cProfile, StringIO, pstats
        pr = cProfile.Profile()
        pr.enable()

        # MCMC
        for i in range(N_samples):
            print "Iteration ", i
            inf_model.resample()

            A_samples.append(inf_model.As.copy())
            w_samples.append(inf_model.weights.copy())
            bias_samples.append(inf_model.bias.copy())
            sigma_samples.append(inf_model.eta.copy())
            ll_samples.append(inf_model.log_likelihood(Xs,y))

        # END Profiling
        pr.disable()

        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()

    # Print posterior mean of samples
    offset = int(burnin*N_samples)
    print ""
    print "Posterior mean of samples %d:%d" % (offset, N_samples)
    print ""

    A_mean = np.array(A_samples)[offset:,:].mean(axis=0)
    print "True A: ", A.astype(np.float)
    print "Inf A:  ", A_mean
    print ""

    w_mean = np.array(w_samples)[offset:,:].mean(axis=0)
    weff_mean = (np.array(w_samples) * np.array(A_samples)[:,:,None]).mean(axis=0)
    print "True W: ", A[:,None] * w.astype(np.float)
    print "Inf W:  ", weff_mean
    print ""

    bias_mean = np.array(bias_samples)[offset:].mean(axis=0)
    print "True bias: ", true_model.bias
    print "Inf bias:  ", bias_mean
    print ""

    sigma_mean = np.array(sigma_samples)[offset:].mean(axis=0)
    print "True sigma: ", true_model.eta
    print "Inf sigma:  ", sigma_mean
    print ""

# Demo with a simple example
demo_ss_nbregression(T=1000, N=2, B=1, N_samples=500, do_plot=True)

# Profile the code on a bigger dataset
# demo_ss_nbregression(T=60000, N=27, B=3, N_samples=1, do_plot=False)
import abc
import copy
import sys

import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize

from pybasicbayes.abstractions import Model

from pyglm.utils.basis import CosineBasis
from pyglm.utils.utils import logistic, dlogistic_dx, logit

class HomogeneousPoissonModel(Model):

    def __init__(self, N):
        self.N = N
        # Initialize biases (rates)
        self.b = np.ones(N)

        self.data_list = []

    @property
    def bias(self):
        return self.b

    def add_data(self, S):
        assert S.ndim == 2 and S.shape[1] == self.N and \
            S.dtype==np.int32 and np.amin(S) >= 0

        self.data_list.append(S)

    def log_likelihood(self, S):
        assert S.ndim == 2 and S.shape[1] == self.N and \
            S.dtype==np.int32 and np.amin(S) >= 0

        log_Z = -gammaln(S+1)
        ll = (log_Z +
                S * np.log(self.b)[None, :]
                - self.b[None, :])

        # Return the log likelihood per time bin
        return ll.sum(1)

    def generate(self):
        raise NotImplementedError()

    def heldout_log_likelihood(self, S):
        return self.log_likelihood(S).sum()

    def fit(self):
        """
        Maximum likelihood fit
        :return:
        """
        # Compute the average rate
        T_tot = 0
        M_tot = np.zeros(self.N)    # Number of spikes

        for data in self.data_list:
            T_tot += data.shape[0]
            M_tot += data.sum(0)

        self.b = M_tot / float(T_tot)

class StandardBernoulliPopulation(Model):

    # Define the model components and their default hyperparameters
    _basis_class                = CosineBasis
    _default_basis_hypers       = {'norm': True, 'allow_instantaneous': False}


    def __init__(self, N, dt=1.0, dt_max=10.0, B=5,
                 basis=None, basis_hypers={},
                 allow_self_connections=True):
        """
        Initialize a discrete time network Hawkes model with K processes.

        :param N:  Number of processes
        """
        self.N      = N
        self.dt     = dt
        self.dt_max = dt_max
        self.B      = B
        self.allow_self_connections = allow_self_connections

        # Initialize the basis
        if basis is not None:
            # assert basis.B == B
            self.basis = basis
            self.B     = basis.B
        else:
            # Use the given basis hyperparameters
            self.basis_hypers = copy.deepcopy(self._default_basis_hypers)
            self.basis_hypers.update(basis_hypers)
            self.basis = self._basis_class(self.B, self.dt, self.dt_max,
                                           **self.basis_hypers)

        # Initialize the weights of the standard model.
        # We have a weight for the background
        self.b = 1e-2 * np.ones(self.N)
        # And a weight for each basis function of each presynaptic neuron.
        self.weights = 1e-4 * np.ones((self.N, self.N*self.B))
        if not self.allow_self_connections:
            self._remove_self_weights()

        # Initialize the data list to empty
        self.data_list = []

    @property
    def W(self):
        WB = self.weights.reshape((self.N,self.N, self.B))

        # DEBUG
        assert WB[0,0,self.B-1] == self.weights[0,self.B-1]
        assert WB[0,self.N-1,0] == self.weights[0,(self.N-1)*self.B]

        if self.B > 2:
            assert WB[self.N-1,self.N-1,self.B-2] == self.weights[self.N-1,-2]

        # Weight matrix is summed over impulse response functions
        W = np.transpose(WB, axes=[1,0,2])

        return W

    @property
    def bias(self):
        return self.b

    def _remove_self_weights(self):
        for n in xrange(self.N):
                self.weights[n,(n*self.B):(n+1)*self.B] = 1e-32


    def augment_data(self, S):
        assert isinstance(S, np.ndarray) and S.ndim == 2 and S.shape[1] == self.N \
               and np.amin(S) >= 0 and S.dtype == np.int32, \
               "Data must be a TxN array of event counts"

        T = S.shape[0]

        # Filter the data into a TxKxB array
        Ftens = self.basis.convolve_with_basis(S)

        # Flatten this into a T x (KxB) matrix
        # [F00, F01, F02, F10, F11, ... F(K-1)0, F(K-1)(B-1)]
        F = Ftens.reshape((T, self.N * self.B))
        assert np.allclose(F[:,0], Ftens[:,0,0])
        if self.B > 1:
            assert np.allclose(F[:,1], Ftens[:,0,1])
        if self.N > 1:
            assert np.allclose(F[:,self.B], Ftens[:,1,0])

        augmented_data = {"T": T, "S": S, "F": F}
        return augmented_data


    def add_data(self, S, F=None, minibatchsize=None):
        """
        Add a data set to the list of observations.
        First, filter the data with the impulse response basis,
        then instantiate a set of parents for this data set.

        :param S: a TxN matrix of of event counts for each time bin
                  and each neuron.
        """
        assert isinstance(S, np.ndarray) and S.ndim == 2 and S.shape[1] == self.N \
               and np.amin(S) >= 0 and S.dtype == np.int32, \
               "Data must be a TxN array of event counts"

        T = S.shape[0]

        if minibatchsize is None:
            minibatchsize = T

        for offset in np.arange(T, step=minibatchsize):
            end = min(offset+minibatchsize, T)
            S_mb = S[offset:end,:]

            augmented_data = self.augment_data(S_mb)

            # Add minibatch to the data list
            self.data_list.append(augmented_data)

    def copy_sample(self):
        """
        Return a copy of the parameters of the model
        :return: The parameters of the model (A,W,\lambda_0, \beta)
        """
        # return copy.deepcopy(self.get_parameters())

        # Shallow copy the data
        data_list = copy.copy(self.data_list)
        self.data_list = []

        # Make a deep copy without the data
        model_copy = copy.deepcopy(self)

        # Reset the data and return the data-less copy
        self.data_list = data_list
        return model_copy

    def generate(self,keep=True,**kwargs):
        raise NotImplementedError()

    def compute_rate(self, augmented_data):
        """
        Compute the rate of the augmented data

        :param index:   Which dataset to compute the rate of
        :param ns:      Which neurons to compute the rate of
        :return:
        """
        F = augmented_data["F"]
        R = np.zeros((augmented_data["T"], self.N))
        for n in xrange(self.N):
            Xn = F.dot(self.weights[n,:])
            Xn += self.bias[n]
            R[:,n] = logistic(Xn)

        return R

    def log_likelihood(self, augmented_data=None):
        """
        Compute the log likelihood of the augmented data
        :return:
        """
        ll = 0

        if augmented_data is None:
            datas = self.data_list
        else:
            datas = [augmented_data]

        ll = 0
        for data in datas:
            S = data["S"]
            R = self.compute_rate(data)
            R   = np.clip(R, 1e-32, 1-1e-32)
            ll += (S * np.log(R) + (1-S) * np.log(1-R)).sum()

        return ll

    def log_probability(self):
        lp = 0
        for data in self.data_list:
            lp += self.log_likelihood(data)
        return lp

    def heldout_log_likelihood(self, S=None, augmented_data=None):
        if S is not None and augmented_data is None:
            augmented_data = self.augment_data(S)
        elif S is None and augmented_data is None:
            raise Exception("Either S or augmented data must be given")

        return self.log_likelihood(augmented_data)


    def fit(self, L1=True):
        """
        Use scikit-learn's LogisticRegression model to fit the data

        :param L1:  If True, use L1 penalty on the coefficients
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import l1_min_c

        F = np.vstack([d["F"] for d in self.data_list])
        S = np.vstack([d["S"] for d in self.data_list])

        if L1:
            # Hold out some data for cross validation
            offset = int(0.75 * S.shape[0])
            T_xv = S.shape[0] - offset
            F_xv = F[offset:, ...]
            S_xv = S[offset:, ...]
            augmented_xv_data = {"T": T_xv, "S": S_xv, "F": F_xv}

            F    = F[:offset, ...]
            S    = S[:offset, ...]

            for n_post in xrange(self.N):
                # Get a L1 regularization path for inverse penalty C
                cs = l1_min_c(F, S[:,n_post], loss='log') * np.logspace(1, 4., 10)
                # The intercept is also subject to penalization, even though
                # we don't really want to penalize it. To counteract this effect,
                # we scale the intercept by a large value
                intercept_scaling = 10**6


                print "Computing regularization path for neuron %d ..." % n_post
                ints      = []
                coeffs    = []
                xv_scores = []
                lr = LogisticRegression(C=1.0, penalty='l1',
                                        fit_intercept=True, intercept_scaling=intercept_scaling,
                                        tol=1e-6)
                for c in cs:
                    print "Fitting for C=%.5f" % c
                    lr.set_params(C=c)
                    lr.fit(F, S[:,n_post])
                    ints.append(lr.intercept_.copy())
                    coeffs.append(lr.coef_.ravel().copy())
                    # xv_scores.append(lr.score(F_xv, S_xv[:,n_post]).copy())

                    # Temporarily set the weights and bias
                    self.b[n_post] = lr.intercept_
                    self.weights[n_post, :] = lr.coef_
                    xv_scores.append(self.heldout_log_likelihood(augmented_data=augmented_xv_data))

                # Choose the regularization penalty with cross validation
                print "XV Scores: "
                for c,score  in zip(cs, xv_scores):
                    print "\tc: %.5f\tscore: %.1f" % (c,score)
                best = np.argmax(xv_scores)
                print "Best c: ", cs[best]

                # Save the best weights
                self.b[n_post]          = ints[best]
                self.weights[n_post, :] = coeffs[best]

        else:
            # Just use standard L2 regularization
            for n_post in xrange(self.N):
                sys.stdout.write('.')
                sys.stdout.flush()

                lr = LogisticRegression(fit_intercept=True)
                lr.fit(F,S[:,n_post])
                self.b[n_post] = lr.intercept_
                self.weights[n_post,:] = lr.coef_

        print ""


class StandardNegativeBinomialPopulation(StandardBernoulliPopulation):
    """
    Overload the Bernoulli population with negative binomial likelihood
    """
    def __init__(self, N, xi=10, dt=1.0, dt_max=10.0, B=5,
                 basis=None, basis_hypers={},
                 allow_self_connections=True):
        super(StandardNegativeBinomialPopulation, self).\
            __init__(N, dt=dt, dt_max=dt_max, B=B,
                     basis=basis, basis_hypers=basis_hypers,
                     allow_self_connections=allow_self_connections)

        self.xi = xi

        # Set L1-regularization penalty
        self.lmbda = 0

    @property
    def T(self):
        """
        Total number of time bins
        :return:
        """
        return float(np.sum([d["T"] for d in self.data_list]))

    def compute_activation(self, augmented_data, n=None):
        """
        Compute the rate of the augmented data

        :param index:   Which dataset to compute the rate of
        :param ns:      Which neurons to compute the rate of
        :return:
        """
        F = augmented_data["F"]

        if n is None:
            X = np.zeros((augmented_data["T"], self.N))
            for n in xrange(self.N):
                w = self.weights[n,:]
                if not self.allow_self_connections:
                    offset = 1 + n * self.B
                    w[offset:offset+self.B] = 0
                X[:,n] = F.dot(w)
                X[:,n] += self.bias[n]
        else:
            w = self.weights[n,:]
            if not self.allow_self_connections:
                offset = 1 + n * self.B
                w[offset:offset+self.B] = 0
            X = F.dot(w)
            X += self.bias[n]

        return X

    def compute_rate(self, augmented_data, n=None):
        """
        Compute the rate of the augmented data

        :param index:   Which dataset to compute the rate of
        :param ns:      Which neurons to compute the rate of
        :return:
        """
        X = self.compute_activation(augmented_data, n=n)
        R = self.xi * np.exp(X) / self.dt
        return R

    def log_normalizer(self, S, n=None):
        if n is not None:
            S = S[:,n]

        return gammaln(S+self.xi) - gammaln(self.xi) - gammaln(S+1)

    def log_prior(self, n=None):
        lp = 0
        if n is None:
            for n in xrange(self.N):
                w = self.weights[n,:]
                if not self.allow_self_connections:
                    offset = 1 + n * self.B
                    w[offset:offset+self.B] = 0

            lp += -self.lmbda * np.abs(w).sum()
        else:
            w = self.weights[n,:]
            if not self.allow_self_connections:
                offset = 1 + n * self.B
                w[offset:offset+self.B] = 0
            lp += -self.lmbda * np.abs(w).sum()

        return lp

    def log_likelihood(self, augmented_data=None, n=None):
        """
        Compute the log likelihood of the augmented data
        :return:
        """
        if augmented_data is None:
            datas = self.data_list
        else:
            datas = [augmented_data]

        ll = 0
        for data in datas:
            S = data["S"][:,n] if n is not None else data["S"]
            Z = self.log_normalizer(S)
            X = self.compute_activation(data, n=n)
            P = logistic(X)
            P = np.clip(P, 1e-32, 1-1e-32)
            ll += (Z + S * np.log(P) + self.xi * np.log(1-P)).sum()

        return ll

    def log_probability(self):
        lp = self.log_prior()
        for data in self.data_list:
            lp += self.log_likelihood(data)
        return lp

    def heldout_log_likelihood(self, S=None, augmented_data=None):
        if S is not None and augmented_data is None:
            augmented_data = self.augment_data(S)
        elif S is None and augmented_data is None:
            raise Exception("Either S or augmented data must be given")

        return self.log_likelihood(augmented_data)

    def _neg_log_posterior(self, x, n):
        """
        Helper function to compute the negative log likelihood
        """
        assert x.shape == (1+self.N * self.B,)
        self.b[n] = x[0]
        self.weights[n,:] = x[1:]

        nlp =  -self.log_likelihood(n=n)
        nlp += -self.log_prior(n=n)

        return nlp / self.T

    def _grad_neg_log_posterior(self, x, n):
        """
        Helper function to compute the negative log likelihood
        """
        assert x.shape == (1+self.N * self.B,)
        self.b[n] = x[0]
        self.weights[n,:] = x[1:]

        # Compute the gradient
        d_ll_d_x = np.zeros_like(x)

        for data in self.data_list:
            S = data["S"][:,n]
            F = data["F"]
            X = self.compute_activation(data, n=n)
            P = logistic(X)
            P = np.clip(P, 1e-32, 1-1e-32)

            # Compute each term in the gradient
            d_ll_d_p  = S / P - self.xi / (1-P)     # 1xT
            d_p_d_psi = dlogistic_dx(P)             # technically TxT diagonal
            d_psi_d_b = 1.0                         # technically Tx1
            d_psi_d_w = F                           # TxNB

            # Multiply em up!
            d_ll_d_x[0]  += (d_ll_d_p * d_p_d_psi * d_psi_d_b).sum()
            d_ll_d_x[1:] += (d_ll_d_p * d_p_d_psi).dot(d_psi_d_w)

        # Compute gradient of the log prior
        d_lp_d_x = np.zeros_like(x)
        d_lp_d_x[1:] += -self.lmbda * np.sign(self.weights[n,:])

        d_lpost_d_x = d_ll_d_x + d_lp_d_x

        # Normalize by T
        d_lpost_d_x /= self.T

        # If self connections are disallowed, remove their gradient
        if not self.allow_self_connections:
            offset = 1 + n * self.B
            d_lpost_d_x[offset:offset+self.B] = 0

        return -d_lpost_d_x

    def _initialize_bias_to_mean(self):
        # Initialize the bias at the mean
        # and zero out the weights

        # To compute the mean bias:
        # E[s]  = 1/T \sum_t s_t  = \xi p / (1-p)
        # p/1-p = 1/(\xi T) \sum_t s_t = X
        #     p = X / (1+X)
        #     p = \sum_t s_t / (\xi T + \sum_t s_t)
        T_tot = 0
        N_tot = np.zeros(self.N)
        for data in self.data_list:
            T_tot += data["T"]
            N_tot += data["S"].sum(0)

        # Compute the mean of p
        pmean = N_tot / (self.xi * T_tot + N_tot)
        pmean = np.clip(pmean, 1e-8, 1-1e-8)

        self.b = logit(pmean)
        self.weights.fill(1e-3)

        if not self.allow_self_connections:
            self._remove_self_weights()

    def fit(self, L1=True, lmbdas=None):
        """
        Fit the negative binomial model using maximum likelihood
        """

        # Print progress
        itr = [0]
        def callback(x):
            if itr[0] % 10 == 0:
                print "Iteration: %03d\t LP: %.1f" % (itr[0], self.log_probability())
            itr[0] = itr[0] + 1


        if L1:
            temp_data_list = copy.copy(self.data_list)

            # Concatenate all the data
            F = np.vstack([d["F"] for d in self.data_list])
            S = np.vstack([d["S"] for d in self.data_list])

            # Hold out some data for cross validation
            offset = int(0.75 * S.shape[0])
            T_xv = S.shape[0] - offset
            F_xv = F[offset:, ...]
            S_xv = S[offset:, ...]
            augmented_xv_data = {"T": T_xv, "S": S_xv, "F": F_xv}

            # Train on the first part of the data
            F    = F[:offset, ...]
            S    = S[:offset, ...]

            self.data_list = []
            self.add_data(S, F)

            # Select the L1 regularization parameter using cross validation
            if lmbdas is None:
                lmbdas = np.logspace(-1,3,10)

            # Initialize to the mean
            self._initialize_bias_to_mean()

            # Fit neurons one at a time
            for n in xrange(self.N):
                print "Optimizing neuron ", n
                bs      = []
                weights    = []
                xv_scores = []

                x0 = np.concatenate(([self.b[n]], self.weights[n,:]))

                for lmbda in lmbdas:
                    print "Lambda: ", lmbda
                    self.lmbda = lmbda
                    itr[0] = 0
                    res = minimize(self._neg_log_posterior,
                                   x0,
                                   jac=self._grad_neg_log_posterior,
                                   args=(n,),
                                   callback=callback)

                    xf = res.x
                    bs.append(xf[0])
                    weights.append(xf[1:])
                    xv_scores.append(self.log_likelihood(augmented_xv_data, n=n))

                # Choose the regularization penalty with cross validation
                print "XV Scores: "
                for lmbda,score  in zip(lmbdas, xv_scores):
                    print "\tlmbda: %.3f\tscore: %.1f" % (lmbda,score)
                best = np.argmax(xv_scores)
                print "Best lmbda: ", lmbdas[best]

                self.b[n]         = bs[best]
                self.weights[n,:] = weights[best]

            # Restore the data list
            self.data_list = temp_data_list

        else:
            # Fit without any regularization
            self._initialize_bias_to_mean()

            # Fit neurons one at a time
            for n in xrange(self.N):
                print "Optimizing process ", n
                self.lmbda = 0
                itr[0] = 0
                x0 = np.concatenate(([self.b[n]], self.weights[n,:]))
                res = minimize(self._neg_log_posterior,
                               x0,
                               jac=self._grad_neg_log_posterior,
                               args=(n,),
                               callback=callback)

                xf = res.x
                self.b[n]         = xf[0]
                self.weights[n,:] = xf[1:]


class _PPCABase(Model):
    """
    PPCA corresponds to the generative model:
        A ~ MN(.,.)
        b ~ N(.,.)
        z_n ~ N(0, I)
        psi_{t,n} = A.dot(z_n) + b
        s_{t,n} ~ p(s | \sigma(psi_{t,n})
    """
    def __init__(self, S, D):
        self.S = S
        self.T, self.N = S.shape
        self.D =  D

        self.Z = np.zeros((self.N,self.D))
        self.omega = np.zeros((self.T, self.N))

        # Initialize regression model
        # from pybasicbayes.distributions.regression import Regression
        # S_0 = np.eye(self.T)
        # K_0 = np.eye(self.D+1)
        # M_0 = np.zeros((self.T, self.D+1))
        # nu_0 = self.T+2
        # self.regression = Regression(nu_0, S_0, M_0, K_0, affine=True)
        self.A = np.zeros((self.T, self.D))
        self.bias = np.zeros((self.T,))

        import pypolyagamma as ppg
        num_threads = ppg.get_omp_num_threads()
        seeds = np.random.randint(2**16, size=num_threads)
        self.ppgs = [ppg.PyPolyaGamma(seed) for seed in seeds]

    @abc.abstractproperty
    def a(self):
        raise NotImplementedError

    @abc.abstractproperty
    def b(self):
        raise NotImplementedError

    @property
    def kappa(self):
        return self.a - self.b/2.0

    # @property
    # def A(self):
    #     return self.regression.A[:,:-1]
    #
    # @property
    # def bias(self):
    #     return self.regression.A[:,-1]

    @property
    def psi(self):
        return self.A.dot(self.Z.T) + self.bias[:,None]

    def add_data(self,data):
        raise Exception

    def log_likelihood(self):
        return np.sum(self.log_likelihood_given_activation(self.S, self.psi))

    @abc.abstractmethod
    def log_likelihood_given_activation(self, S, psi):
        raise NotImplementedError

    def heldout_neuron_log_likelihood(self, Stest, M=100):
        from scipy.misc import logsumexp
        assert Stest.shape == (self.T,)

        # Sample a bunch of Zs
        Ztest = np.random.randn(self.D, M)
        psi_test = self.A.dot(Ztest) + self.bias[:,None]

        plls = np.array([np.sum(
                self.log_likelihood_given_activation(Stest, psi))
                for psi in psi_test.T])

        # Take the average of the predictive log likelihoods
        mean_pll = -np.log(M) + logsumexp(plls)

        # Use bootstrap to compute standard error
        subsamples = np.random.choice(plls, size=(100, M), replace=True)
        pll_subsamples = logsumexp(subsamples, axis=1) - np.log(M)
        std_pll = pll_subsamples.std()

        return mean_pll, std_pll


    def generate(self):
        raise NotImplementedError()

    def resample_model(self):
        # import ipdb ; ipdb.set_trace()
        self._resample_auxiliary_vars()
        self._resample_regression()
        self._resample_Z()

    def _resample_Z(self):
        """
        p(_n | A, kappa, omega) = p(z_n) * p(Az_n + b| kappa/omega, omega^-1)
        = p(Z) p(Z | A.T (kappa/omega-b), A.T omega A)
        """
        from pybasicbayes.util.stats import sample_gaussian
        kappa, omega = self.kappa, self.omega
        A,b = self.A, self.bias

        for n in range(self.N):
            h = np.zeros(self.D)
            J = np.eye(self.D)

            h += A.T.dot((kappa[:,n]/omega[:,n] - b) * omega[:,n])
            J += (A * omega[:,n][:,None]).T.dot(A)

            self.Z[n] = sample_gaussian(J=J, h=h)

    def _resample_regression(self):
        """
        p(A | Z, kappa, omega) = p(A) * p(AZ + b | kappa/omega, omega)
        = \prod_t N(Ab_t, 0, I) * N(Z'.T Ab_t | kappa_t/omega_t, omega_t)
        """
        # xy = np.hstack((self.Z, self.S.T))
        # self.regression.resample(xy)

        from pybasicbayes.util.stats import sample_gaussian
        kappa, omega = self.kappa, self.omega
        Z = np.hstack((self.Z, np.ones((self.N,1))))

        for t in xrange(self.T):
            h = np.zeros(self.D+1)
            J = np.eye(self.D+1)

            h += Z.T.dot(kappa[t])
            J += (Z * omega[t][:,None]).T.dot(Z)

            Abt = sample_gaussian(J=J, h=h)
            self.A[t], self.bias[t] = Abt[:-1], Abt[-1]

        # import ipdb; ipdb.set_trace()
        # print np.mean(self.A)
        # print np.mean(self.bias)

    def _resample_auxiliary_vars(self):
        import pypolyagamma as ppg
        b, psi = self.b, self.psi
        ppg.pgdrawvpar(self.ppgs, b.ravel(), psi.ravel(),
                       self.omega.ravel())


class _MixtureModelBase(Model):
    """
    Mixture of neurons model
    """
    def __init__(self, S, C):
        self.S, self.C = S, C
        self.T, self.N = S.shape
        self.c = np.random.randint(0,C, size=self.N)
        self.psis = np.zeros((self.T, self.C))

        from pybasicbayes.distributions.gaussian import ScalarGaussianNIX
        self.gaussian = ScalarGaussianNIX(mu_0=0, kappa_0=1, sigmasq_0=1.0, nu_0=2.0)


        import pypolyagamma as ppg
        num_threads = ppg.get_omp_num_threads()
        seeds = np.random.randint(2**16, size=num_threads)
        self.ppgs = [ppg.PyPolyaGamma(seed) for seed in seeds]

        self.omega = np.zeros((self.T, self.N))

    @abc.abstractproperty
    def a(self):
        raise NotImplementedError

    @abc.abstractproperty
    def b(self):
        raise NotImplementedError

    @property
    def kappa(self):
        return self.a - self.b/2.0

    @property
    def psi(self):
        return self.psis[:, self.c]

    def add_data(self,data):
        raise Exception

    def log_likelihood(self):
        return np.sum(self.log_likelihood_given_activation(self.S, self.psi))

    @abc.abstractmethod
    def log_likelihood_given_activation(self, S, psi):
        raise NotImplementedError

    def generate(self):
        raise NotImplementedError()

    def heldout_log_likelihood(self, S):
        return self.log_likelihood(S).sum()

    def resample_model(self):
        self._resample_auxiliary_vars()
        self._resample_c()
        self._resample_psi()

    def _resample_c(self):
        from pybasicbayes.util.stats import sample_discrete_from_log
        kappa, omega = self.kappa, self.omega
        mu = kappa / omega
        sigmasq = 1./omega
        for n in xrange(self.N):
            lps = np.zeros(self.C)
            for i in xrange(self.C):
                lps[i] += np.sum(-0.5 * (self.psis[:,i] - mu[:,n])**2 / sigmasq[:,n])

            self.c[n] = sample_discrete_from_log(lps)

    def _resample_psi(self):
        mu, sigmasq = self.gaussian.mu, self.gaussian.sigmasq
        kappa, omega = self.kappa, self.omega

        for i in range(self.C):
            h = mu / sigmasq * np.ones((self.T))
            J_diag = 1. / sigmasq * np.ones((self.T))

            for n in range(self.N):
                if self.c[n] == i:
                    h += kappa[:,n]
                    J_diag += omega[:,n]

            # Sample
            mu_i = h / J_diag
            sig_i = np.sqrt(1./J_diag)
            self.psis[:,i] = mu_i + sig_i * np.random.randn(self.T)

        # Resample global prior
        data = []
        for i in xrange(self.C):
            if np.sum(self.c==i) > 0:
                data.append(self.psis[:,i])
        data = np.concatenate(data)
        self.gaussian.resample(data)

    def _resample_auxiliary_vars(self):
        import pypolyagamma as ppg
        b, psi = self.b, self.psi
        ppg.pgdrawvpar(self.ppgs, b.ravel(), psi.ravel(),
                       self.omega.ravel())



class _BernoulliMixin(object):

    @property
    def a(self):
        if not hasattr(self, "_a"):
            self._a = self.S.copy()
        return self._a

    @property
    def b(self):
        if not hasattr(self, "_b"):
            self._b = np.ones(self.S.shape)
        return self._b

    def log_likelihood_given_activation(self, S, psi):
        p = logistic(psi)
        p = np.clip(p, 1e-32, 1-1e-32)
        ll = (S * np.log(p) + (1-S) * np.log(1-p))
        return ll


class _NegativeBinomialMixin(object):

    def __init__(self, *args):
        super(_NegativeBinomialMixin, self).__init__(*args)
        self.xi = 10

    @property
    def a(self):
        if not hasattr(self, "_a"):
            self._a = self.S.copy()
        return self._a

    @property
    def b(self):
        if not hasattr(self, "_b"):
            self._b = (self.S + self.xi).astype(np.float)
        return self._b

    def log_likelihood_given_activation(self, S, psi):
        p = logistic(psi)
        p = np.clip(p, 1e-32, 1-1e-32)
        xi = self.xi

        return self.log_normalizer(S, xi) \
               + S * np.log(p) \
               + xi * np.log(1-p)

    @staticmethod
    def log_normalizer(S, xi):
        return gammaln(S+xi) - gammaln(xi) - gammaln(S+1)


class BernoulliPPCA(_BernoulliMixin, _PPCABase):
    pass

class NegativeBinomialPPCA(_NegativeBinomialMixin, _PPCABase):
    pass

class BernoulliMixtureModel(_BernoulliMixin, _MixtureModelBase):
    pass

class NegativeBinomialMixtureModel(_NegativeBinomialMixin, _MixtureModelBase):
    pass
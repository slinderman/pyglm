"""
Background models for neural spike train activations.

For example:
    - Linear Dynamical System (LDS) model for background rates

"""
import numpy as np
from pyglm.abstractions import Component

class NoBackground(Component):
    """
    Null background model.
    """
    def __init__(self, population):
        self.population = population
        self.N = self.population.N

    def generate(self, T):
        return np.zeros((T,self.N))

    def mean_background_activation(self, augmented_data):
        return np.zeros(self.N)

    def resample(self, augmented_data):
        pass

    def meanfieldupdate(self, augmented_data):
        pass

    def get_vlb(self, augmented_data):
        return 0

    def resample_from_mf(self, augmented_data):
        pass

    def svi_step(self, augmented_data, minibatchfrac, stepsize):
        pass


class LinearDynamicalSystemBackground(Component):
    """
    Linear Dynamical System model for the background activation.
    Since the potentials for the activation are of a Gaussian form,
    we can perform conjugate Gibbs sampling or variational inference
    for a Gaussian LDS model.
    """
    def __init__(self, population, D=2,
                 A=None, C=None,
                 sigma_states=None,
                 sigma_C=1.0):

        self.population = population
        self.activation = population.activation_model
        self.N = self.population.N
        self.D = D

        from pybasicbayes.distributions import Gaussian
        self.init_dynamics_distn = Gaussian(mu_0=np.ones(D),
                                            kappa_0=1.0,
                                            sigma_0=0.000001 * np.eye(D),
                                            nu_0=3.0)

        from autoregressive.distributions import AutoRegression
        self.dynamics_distn = AutoRegression(A=A, sigma=sigma_states,
                                             nu_0=D+1.0, S_0=0.5 * np.eye(D),
                                             M_0=np.zeros((D,D)), K_0=0.5 * np.eye(D))

        # Initialize the emission matrix
        if C is None:
            self.C = sigma_C * np.random.randn(self.N, self.D)
        else:
            assert C.shape == (self.N, self.D)
            self.C = C

        self.sigma_C = sigma_C

    def augment_data(self, augmented_data):
        # Add a latent state sequence
        augmented_data["states"] = self.generate_states(augmented_data["T"])

    def log_likelihood(self, augmented_data):
        raise NotImplementedError

    def generate(self,T):
        states = self.generate_states(T)
        return states.dot(self.C.T)

    def generate_states(self, T):
        stateseq = np.empty((T,self.D))
        stateseq[0] = self.init_dynamics_distn.rvs()

        chol = np.linalg.cholesky(self.dynamics_distn.sigma)
        randseq = np.random.randn(T-1,self.D)

        for t in xrange(1,T):
            stateseq[t] = \
                self.dynamics_distn.A.dot(stateseq[t-1]) \
                + chol.dot(randseq[t-1])

        return stateseq

    def mean_background_activation(self, augmented_data):
        return augmented_data["states"].dot(self.C.T)

    def resample(self, augmented_data_list):
        self.resample_states(augmented_data_list)
        self.resample_parameters(augmented_data_list)

    def resample_states(self, augmented_data_list):
        from pylds.lds_messages import filter_and_sample

        for data in augmented_data_list:
            # Compute the residual activation from other components
            psi = self.activation.compute_psi(data)
            psi_residual = psi - self.mean_background_activation(data)

            # Get the observed mean and variance
            mu_obs = self.activation.new_mean(data)
            prec_obs = self.activation.new_precision(data)

            # Subtract off the activation from other components
            mu_obs -= psi_residual

            # Convert prec_obs into an array of diagonal covariance matrices
            sigma_obs = np.empty((data["T"], self.N, self.N), order="C")
            for t in xrange(data["T"]):
                sigma_obs[t,:,:] = np.diag(1./prec_obs[t,:])

            data["states"] = filter_and_sample(
                self.init_dynamics_distn.mu,
                self.init_dynamics_distn.sigma,
                self.dynamics_distn.A,
                self.dynamics_distn.sigma,
                self.C,
                sigma_obs,
                mu_obs)

    def resample_parameters(self, augmented_data_list):
        self.resample_init_dynamics_distn(augmented_data_list)
        self.resample_dynamics_distn(augmented_data_list)
        self.resample_emission_distn(augmented_data_list)

    def resample_init_dynamics_distn(self, augmented_data_list):
        states_list = [ad["states"][0] for ad in augmented_data_list]
        self.init_dynamics_distn.resample(states_list)

    def resample_dynamics_distn(self, augmented_data_list):
        from pyhsmm.util.general import AR_striding
        states_list = [ad["states"] for ad in augmented_data_list]
        strided_states_list = [AR_striding(s,1) for s in states_list]
        self.dynamics_distn.resample(strided_states_list)

    def resample_emission_distn(self, augmented_data_list):
        """
        Resample the observation vectors. Since the emission noise is diagonal,
        we can resample the columns of C independently
        :return:
        """
        # Get the prior
        prior_precision = 1./self.sigma_C * np.eye(self.D)
        prior_mean = np.zeros(self.D)
        prior_mean_dot_precision = prior_mean.dot(prior_precision)

        # Get the sufficient statistics from the likelihood
        lkhd_precision = np.zeros((self.N, self.D, self.D))
        lkhd_mean_dot_precision = np.zeros((self.N, self.D))

        for data in augmented_data_list:
            # Compute the residual activation from other components
            psi = self.activation.compute_psi(data)
            psi_residual = psi - self.mean_background_activation(data)

            # Get the observed mean and variance
            mu_obs = self.activation.new_mean(data)
            prec_obs = self.activation.new_precision(data)

            # Subtract off the residual
            mu_obs -= psi_residual

            # Update the sufficient statistics for each neuron
            for n in xrange(self.N):
                lkhd_precision[n,:,:] += (data["states"] * prec_obs[:,n][:,None]).T.dot(data["states"])
                lkhd_mean_dot_precision[n,:] += \
                    (mu_obs[:,n] * prec_obs[:,n]).T.dot(data["states"])

        # Sample each column of C
        for n in xrange(self.N):
            post_prec = prior_precision + lkhd_precision[n,:,:]
            post_cov  = np.linalg.inv(post_prec)
            post_mu   =  (prior_mean_dot_precision +
                          lkhd_mean_dot_precision[n,:]).dot(post_cov)
            post_mu   = post_mu.ravel()

            self.C[n,:] = np.random.multivariate_normal(post_mu, post_cov)

    ### Variational inference
    def meanfieldupdate(self, augmented_data): raise NotImplementedError
    def get_vlb(self, augmented_data): raise NotImplementedError
    def resample_from_mf(self, augmented_data): raise NotImplementedError
    def svi_step(self, augmented_data, minibatchfrac, stepsize): raise NotImplementedError
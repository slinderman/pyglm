import abc
import numpy as np

# from hips.distributions.polya_gamma import polya_gamma
from pypolyagamma import pgdrawv, PyRNG

from pyglmdos.abstractions import Component
from pyglm.utils.utils import logistic

class _PolyaGammaAugmentedObservationsBase(Component):
    """
    Class to keep track of a set of spike count observations and the
    corresponding Polya-gamma auxiliary variables associated with them.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, population):
        self.population = population
        self.rng = PyRNG()

    @property
    def activation(self):
        return self.population.activation_model

    def augment_data(self, augmented_data):
        """
        Add a matrix of augmented counts
        :param augmented_data:
        :return:
        """
        S = augmented_data["S"]
        assert S.ndim == 1
        self.T = S.shape[0]

        # Initialize auxiliary variables
        augmented_data["omega"] = np.ones_like(S)
        pgdrawv(np.ones_like(S, dtype=np.int32),
                np.zeros_like(S),
                augmented_data["omega"], self.rng)

    @abc.abstractmethod
    def a(self, augmented_data):
        """
        The first parameter of the conditional Polya-gamma distribution
        p(\omega | \psi, s) = PG(b, \psi)
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def b(self, , augmented_data):
        """
        The exponent in the denominator of the logistic likelihood
            exp(\psi)^a / (1+exp(\psi)^b
        """
        raise NotImplementedError()

    def kappa(self, augmented_data):
        """
        Compute kappa = b-a/2
        :return:
        """
        return self.a(augmented_data) - self.b(augmented_data)/2.0

    def omega(self, augmented_data):
        return augmented_data["omega"]

    def resample(self, augmented_data):
        """
        Resample omega given xi and psi, then resample psi given omega, X, w, and sigma
        """
        psi = self.activation.compute_psi(augmented_data)

        # Resample the auxiliary variables, omega, in Python
        # self.omega = polya_gamma(self.conditional_b.reshape(self.T),
        #                          self.psi.reshape(self.T),
        #                          200).reshape((self.T,))

        # Create a PyPolyaGamma object and resample with the C code
        # seed = np.random.randint(2**16)
        # ppg = PyPolyaGamma(seed, self.model.trunc)
        # ppg.draw_vec(self.conditional_b, self.psi, self.omega)

        # Resample with Jesse Windle's ported code
        pgdrawv(self.b(augmented_data),
                psi,
                augmented_data["omega"],
                self.rng)


class BernoulliObservations(_PolyaGammaAugmentedObservationsBase):
    def a(self, augmented_data):
        return augmented_data["S"]

    def b(self, augmented_data):
        """
        The first parameter of the conditional Polya-gamma distribution
        p(\omega | \psi, s) = PG(b, \psi)
        """
        return np.ones_like(augmented_data["S"])

    def rvs(self, Psi):
        p = logistic(Psi)
        return np.random.rand(*p.shape) < p


class NegativeBinomialObservations(_PolyaGammaAugmentedObservationsBase):
    def __init__(self, population, xi=1.0):
        super(NegativeBinomialObservations, self).__init__(population)

        assert xi > 0, "Xi must greater than 0 for negative binomial NB(xi, p)"
        self.xi = xi

    def a(self, augmented_data):
        return augmented_data["S"]

    def b(self, augmented_data):
        """
        The first parameter of the conditional Polya-gamma distribution
        p(\omega | \psi, s) = PG(b, \psi)
        """
        return augmented_data["S"] + self.xi

    def rvs(self, Psi):
        p = logistic(Psi)
        return np.random.rand(*p.shape) < p
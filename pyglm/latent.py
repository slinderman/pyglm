"""
Latent variables associated with each neuron
"""
import numpy as np

from deps.pybasicbayes.abstractions import Distribution, GibbsSampling
from deps.pybasicbayes.distributions import Multinomial
from deps.pybasicbayes.util.cstats import sample_crp_tablecounts
from deps.pybasicbayes.util.stats import sample_discrete_from_log
from deps.pybasicbayes.util.general import ibincount

class _LatentVariableBase(Distribution):
    """
    Base class for latent variables
    """
    def __init__(self, population):
        self.population = population

    @property
    def N(self):
        return self.population.N

    @property
    def parameters(self):
        return ()

    @property
    def parameter_names(self):
        return ()

    def rvs(self, size=[]):
        return None

    def log_likelihood(self, data=[]):
        return 0

    def resample(self, data=[]):
        return

class LatentClass(_LatentVariableBase, GibbsSampling):
    """
    A latent class associated with each neuron. This
    may govern the connection probabilities and weight
    distributions of the network, for example. The class
    probabilities are stored in the vector, \beta.
    """
    def __init__(self, population, K=1, alpha=1.0, gamma=1.0, classes=None):
        super(LatentClass, self).__init__(population)
        self.K = K

        # Make a multinomial object for the latent classes
        self.alpha = alpha
        self.beta_obj = Multinomial(alpha_0=gamma,K=self.K)

        if classes is None:
            perm = np.argsort(np.random.rand(self.N))
            self.classes = ibincount(self.beta_obj.rvs(self.N))[perm]
        else:
            assert classes.size == self.N and np.amax(self.classes) < self.K

        print "Classes: ", self.classes

    @property
    def parameters(self):
        return (self.classes,)

    @property
    def parameter_names(self):
        return ("classes", )

    @property
    def beta(self):
        return self.beta_obj.weights

    def resample(self,data=[]):
        self.resample_classes()

        self.resample_class_probability()

    def resample_classes(self):
        """
        Resample each neuron's latent class
        :return:
        """
        # We have to do this sequentially since the classes
        # are not independent in the likelihood
        # import pdb; pdb.set_trace()

        # DEBUG
        # self.classes *= 0
        # self.classes[16:] = 1
        # print "Classes: ", self.classes
        # return

        for n in xrange(self.N):
            scores = np.zeros(self.K)
            for k in xrange(self.K):
                # Set the n-th neuron to class k and evaluate log likelihood
                self.classes[n] = k
                scores[k] = self.population.network.log_likelihood()
                # TODO: Add likelihood from other components, e.g. stimulus

                # Add the prior
                scores[k] += np.log(self.beta[k])
            scores = np.nan_to_num(scores)

            self.classes[n] = sample_discrete_from_log(scores)
        print "Classes: ", self.classes

    def resample_class_probability(self):
        counts = np.bincount(self.classes, minlength=self.K).reshape((1,self.K))
        ms = self._get_m(counts)
        self.beta_obj.resample(ms)
        self.alphav = self.alpha * self.beta

    def _get_m(self,counts):
        if not (0 == counts).all():
            # TODO: Copy over CRP count sampler
            m = sample_crp_tablecounts(self.alpha,counts,self.beta)
        else:
            m = np.zeros_like(counts)
        self.m = m
        return m
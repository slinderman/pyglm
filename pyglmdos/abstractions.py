"""
Some modeling abstractions.
"""
import abc

# from pyglm.deps.pybasicbayes.abstractions import GibbsSampling, MeanField, MeanFieldSVI

class _ComponentBase(object):
    """
    Wrapper for a component in our model, e.g. the weight model.
    """
    __metaclass__ = abc.ABCMeta

    def augment_data(self, augmented_data):
        """
        Augment the data with any local variables required by this component.

        :param augmented_data:  A data dictionary.
        :return:                None
        """
        pass


class _GibbsComponent(_ComponentBase):
    """
    Add Gibbs sampling functionality
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def resample(self, augmented_data):
        """
        Resample the component parameters with the given augmented data
        """
        pass


class _MeanFieldComponent(_ComponentBase):
    """
    Add mean field variational inference functionality
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def meanfieldupdate(self, augmented_data):
        """
        Update the mean field variational parameters given the augmented data.
        """
        pass

    @abc.abstractmethod
    def get_vlb(self, augmented_data):
        """
        Get the variational lower bound local to this component.
        :param augmented_data:
        :return:
        """
        pass


class _SVIComponent(_ComponentBase):
    """
    Add stochastic variational inference functionality
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def meanfieldupdate(self, augmented_data):
        """
        Update the mean field variational parameters given the augmented data.
        """
        pass

    @abc.abstractmethod
    def get_vlb(self, augmented_data):
        """
        Get the variational lower bound local to this component.
        :param augmented_data:
        :return:
        """
        pass


class Component(_GibbsComponent):
    """
    Combine all the
    """
    pass
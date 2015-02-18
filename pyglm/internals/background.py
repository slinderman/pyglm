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
        pass

    def resample(self, augmented_data):
        pass

    def meanfieldupdate(self, augmented_data):
        pass

    def get_vlb(self, augmented_data):
        return 0

    def resample_from_mf(self, augmented_data):
        pass


class LinearDynamicalSystemBackground(Component):
    """
    Linear Dynamical System model for the background activation.
    Since the potentials for the activation are of a Gaussian form,
    we can perform conjugate Gibbs sampling or variational inference
    for a Gaussian LDS model.
    """
    def __init__(self, population):
        raise NotImplementedError()

    def resample(self, augmented_data):
        pass
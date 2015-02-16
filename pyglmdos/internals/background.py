"""
Background models for neural spike train activations.

For example:
    - Linear Dynamical System (LDS) model for background rates

"""
from pyglmdos.abstractions import Component

class NoBackground(Component):
    """
    Null background model.
    """

    def resample(self, augmented_data):
        pass
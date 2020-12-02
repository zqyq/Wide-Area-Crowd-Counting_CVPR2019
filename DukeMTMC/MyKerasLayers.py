from __future__ import print_function
import tensorflow as tf
assert tf.__version__.startswith('1.')
from keras.engine import Layer


class AcrossChannelLRN(Layer):
    """
    Cross-channel Local Response Normalization for 2D feature maps.

    Aggregation is purely across channels, not within channels,
    and performed "pixelwise".

    If the value of the ith channel is :math:`x_i`, the output is

    .. math::

        x_i = \frac{x_i}{ (1 + \frac{\alpha}{n} \sum_j x_j^2 )^\beta }

    where the summation is performed over this position on :math:`n`
    neighboring channels.

    This code is adapted from Lasagne, which is from pylearn2.
    This layer is time consuming. Without this layer, it takes 4 sec for 100 iterations, with this layer, it takes 8 sec.
    """

    def __init__(self, local_size=5, alpha=1e-4, beta=0.75, k=1,
                 **kwargs):
        super(AcrossChannelLRN, self).__init__(**kwargs)
        self.local_size = local_size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        assert self.local_size % 2 == 1, "Only works with odd local_size!!!"

    def build(self, input_shape):
        print('No trainable weights for LRN layer.')

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        return tf.nn.local_response_normalization(x, depth_radius=self.local_size, bias=self.k, alpha=self.alpha, beta=self.beta, name=self.name)

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "local_size": self.local_size,
                  "alpha": self.alpha,
                  "beta": self.beta,
                  "k": self.k}
        base_config = super(AcrossChannelLRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# alias
LRN_across_channel = AcrossChannelLRN
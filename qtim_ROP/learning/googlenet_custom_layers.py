
from keras.layers.core import Layer
import keras.backend as K
import tensorflow as tf


class LRN(Layer):

    def __init__(self, alpha=0.0001,k=1,beta=0.75,n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)

    def call(self, x, mask=None):

        b, ch, r, c = x.shape
        print(x.shape)
        half_n = self.n // 2  # half the local region
        # orig keras code
        # input_sqr = T.sqr(x)  # square the input
        input_sqr = K.square(x)  # square the input
        # orig keras code
        # extra_channels = T.alloc(0., b, ch + 2 * half_n, r,c)  # make an empty tensor with zero pads along channel dimension
        # input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],input_sqr) # set the center to be the squared input

        # extra_channels = K.zeros((b, int(ch) + 2 * half_n, r, c))
        paddings = [[0, 0], [half_n, half_n], [0, 0], [0, 0]]
        input_sqr = tf.pad(input_sqr, paddings)
        print(input_sqr.shape)

        scale = self.k  # offset for the scale
        norm_alpha = self.alpha / self.n  # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i + int(ch), :, :]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PoolHelper(Layer):

    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:,:,1:,1:]

    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


from keras import backend as K
from keras.activations import selu
from keras.engine.topology import Layer
from keras.layers import PReLU


class LKReLU(Layer):
    """
    Implementation of Linked ReLU using keras layers: concat([relu(x), relu(-x)])
    From the paper: Solving internal covariate shift in deep learning with linked neurons.

    """

    def call(self, x):
        return K.concatenate((K.relu(x), K.relu(-x)), -1)

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape[:-1]) + [input_shape[-1] * 2, ])


class LKSELU(Layer):
    """
    Implementation of Linked SELU using keras layers: concat([selu(x), selu(-x)])
    From the paper: Solving internal covariate shift in deep learning with linked neurons.
    """

    def call(self, x):
        return K.concatenate((selu(x), selu(-x)), -1)

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape[:-1]) + [input_shape[-1] * 2, ])


class LKSwish(Layer):
    """
    Implementation of Linked Swish using keras layers: concat([swish(x), swish(-x)])
    From the paper: Solving internal covariate shift in deep learning with linked neurons.
    """

    def call(self, x):
        return K.concatenate((swish(x), swish(-x)), -1)

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape[:-1]) + [input_shape[-1] * 2, ])


class LKPReLU(Layer):
    """
    Implementation of Linked PReLU using keras layers: concat([prelu(x), prelu(-x)])
    From the paper: Solving internal covariate shift in deep learning with linked neurons.
    """

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.pos = PReLU()
        self.neg = PReLU()

        super(LKPReLU, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.concatenate((self.pos(x), self.neg(-x)), -1)

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape[:-1]) + [input_shape[-1] * 2, ])


def swish(x):
    """
    Implementation of the Swish activation function from the paper: Swish: a Self-Gated Activation Function
    :param x: Input tensor
    :return: x * sigmoid(x)
    """
    return x * K.sigmoid(x)

import keras.backend as K
from keras import Input
from keras.engine import get_source_inputs, Model
from keras.layers import Flatten, Dense, Conv2D, BatchNormalization, Activation, PReLU, LeakyReLU

from linked_neurons import LKReLU, LKSwish, LKSELU, LKPReLU, swish


def ReLUDepthNet(input_tensor=None, input_shape=None, width=4, depth=50,
                 classes=1000):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(width, (7, 7), strides=(2, 2), padding='same', name='convfirst', activation='relu')(img_input)
    for i in range(depth):
        x = Conv2D(width, (3, 3), padding='same', activation='relu', name='conv{}'.format(i))(x)

    x = Flatten()(x)

    x = Dense(classes, activation='softmax', name='fc')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='depthnet-{}-{}'.format(width, depth))

    return model


def LKReLUDepthNet(input_tensor=None, input_shape=None, width=4, depth=50,
                   classes=1000):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(width, (7, 7), strides=(2, 2), padding='same', name='convfirst')(img_input)
    x = LKReLU()(x)
    for i in range(depth):
        x = Conv2D(width, (3, 3), padding='same', name='conv{}'.format(i))(x)
        x = LKReLU()(x)

    x = Flatten()(x)

    x = Dense(classes, activation='softmax', name='fc')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='lkreludepthnet-{}-{}'.format(width, depth))

    return model


def PReLUDepthNet(input_tensor=None, input_shape=None, width=4, depth=50,
                  classes=1000):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(width, (7, 7), strides=(2, 2), padding='same', name='convfirst')(img_input)
    x = PReLU()(x)
    for i in range(depth):
        x = Conv2D(width, (3, 3), padding='same', name='conv{}'.format(i))(x)
        x = PReLU()(x)

    x = Flatten()(x)

    x = Dense(classes, activation='softmax', name='fc')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='preludepthnet-{}-{}'.format(width, depth))

    return model


def SeLUDepthNet(input_tensor=None, input_shape=None, width=4, depth=50,
                 classes=1000):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(width, (7, 7), strides=(2, 2), padding='same', name='convfirst', activation='selu')(img_input)
    for i in range(depth):
        x = Conv2D(width, (3, 3), padding='same', activation='selu', name='conv{}'.format(i))(x)

    x = Flatten()(x)

    x = Dense(classes, activation='softmax', name='fc')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='seludepthnet-{}-{}'.format(width, depth))
    return model


def BatchNormDepthNet(input_tensor=None, input_shape=None, width=4, depth=50,
                      classes=1000):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(width, (7, 7), strides=(2, 2), padding='same', name='convfirst', activation=None)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    for i in range(depth):
        x = Conv2D(width, (3, 3), padding='same', activation=None, name='conv{}'.format(i))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Flatten()(x)

    x = Dense(classes, activation='softmax', name='fc')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='batchnormdepthnet-{}-{}'.format(width, depth))

    return model


def LeakyDepthNet(input_tensor=None, input_shape=None, width=4, depth=50,
                  classes=1000):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(width, (7, 7), strides=(2, 2), padding='same', name='convfirst')(img_input)
    x = LeakyReLU()(x)
    for i in range(depth):
        x = Conv2D(width, (3, 3), padding='same', name='conv{}'.format(i))(x)
        x = LeakyReLU()(x)

    x = Flatten()(x)

    x = Dense(classes, activation='softmax', name='fc')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='leakydepthnet-{}-{}'.format(width, depth))

    return model


def SwishDepthNet(input_tensor=None, input_shape=None, width=4, depth=50,
                  classes=1000):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(width, (7, 7), strides=(2, 2), padding='same', name='convfirst', activation=swish)(img_input)
    for i in range(depth):
        x = Conv2D(width, (3, 3), padding='same', activation=swish, name='conv{}'.format(i))(x)

    x = Flatten()(x)

    x = Dense(classes, activation='softmax', name='fc')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='swishdepthnet-{}-{}'.format(width, depth))
    return model


def LKPReLUDepthNet(input_tensor=None, input_shape=None, width=4, depth=50,
                    classes=1000):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(width, (7, 7), strides=(2, 2), padding='same', name='convfirst')(img_input)
    x = LKPReLU()(x)
    for i in range(depth):
        x = Conv2D(width, (3, 3), padding='same', name='conv{}'.format(i))(x)
        x = LKPReLU()(x)

    x = Flatten()(x)

    x = Dense(classes, activation='softmax', name='fc')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='lkpreludepthnet-{}-{}'.format(width, depth))

    return model


def LKSELUDepthNet(input_tensor=None, input_shape=None, width=4, depth=50,
                   classes=1000):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(width, (7, 7), strides=(2, 2), padding='same', name='convfirst')(img_input)
    x = LKSELU()(x)
    for i in range(depth):
        x = Conv2D(width, (3, 3), padding='same', name='conv{}'.format(i))(x)
        x = LKSELU()(x)

    x = Flatten()(x)

    x = Dense(classes, activation='softmax', name='fc')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='lkseludepthnet-{}-{}'.format(width, depth))
    return model


def LKSwishDepthNet(input_tensor=None, input_shape=None, width=4, depth=50,
                    classes=1000):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(width, (7, 7), strides=(2, 2), padding='same', name='convfirst')(img_input)
    x = LKSwish()(x)
    for i in range(depth):
        x = Conv2D(width, (3, 3), padding='same', name='conv{}'.format(i))(x)
        x = LKSwish()(x)

    x = Flatten()(x)

    x = Dense(classes, activation='softmax', name='fc')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='lkswishdepthnet-{}-{}'.format(width, depth))
    return model

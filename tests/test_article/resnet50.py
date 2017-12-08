import keras.layers as layers
from keras import Input
from keras import backend as K
from keras.applications.resnet50 import conv_block, identity_block
from keras.engine import Model, get_source_inputs
from keras.layers import Conv2D, Flatten, Dense, Activation, \
    BatchNormalization, PReLU

from linked_neurons import LKReLU, LKSELU, LKSwish, LKPReLU, swish


def ReLUResNet50(input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000, include_top=False):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    # alpha = 1/8
    x = Conv2D(
        8, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='c')

    x = conv_block(x, 3, [16, 16, 64], stage=3, block='a')
    x = identity_block(x, 3, [16, 16, 64], stage=3, block='b')
    x = identity_block(x, 3, [16, 16, 64], stage=3, block='c')
    x = identity_block(x, 3, [16, 16, 64], stage=3, block='d')

    x = conv_block(x, 3, [32, 32, 128], stage=4, block='a')
    x = identity_block(x, 3, [32, 32, 128], stage=4, block='b')
    x = identity_block(x, 3, [32, 32, 128], stage=4, block='c')
    x = identity_block(x, 3, [32, 32, 128], stage=4, block='d')
    x = identity_block(x, 3, [32, 32, 128], stage=4, block='e')
    x = identity_block(x, 3, [32, 32, 128], stage=4, block='f')

    x = conv_block(x, 3, [32, 32, 128], stage=5, block='a')
    x = identity_block(x, 3, [32, 32, 128], stage=5, block='b')
    x = identity_block(x, 3, [32, 32, 128], stage=5, block='c')

    # x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='relu_resnet50')

    return model


def LKReLUResNet50(input_tensor=None, input_shape=None,
                   pooling=None,
                   classes=1000, include_top=False):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    # alpha = 1/8
    x = Conv2D(
        8, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = lkrelu_conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = lkrelu_identity_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = lkrelu_identity_block(x, 3, [16, 16, 64], stage=2, block='c')

    x = lkrelu_conv_block(x, 3, [16, 16, 64], stage=3, block='a')
    x = lkrelu_identity_block(x, 3, [16, 16, 64], stage=3, block='b')
    x = lkrelu_identity_block(x, 3, [16, 16, 64], stage=3, block='c')
    x = lkrelu_identity_block(x, 3, [16, 16, 64], stage=3, block='d')

    x = lkrelu_conv_block(x, 3, [32, 32, 128], stage=4, block='a')
    x = lkrelu_identity_block(x, 3, [32, 32, 128], stage=4, block='b')
    x = lkrelu_identity_block(x, 3, [32, 32, 128], stage=4, block='c')
    x = lkrelu_identity_block(x, 3, [32, 32, 128], stage=4, block='d')
    x = lkrelu_identity_block(x, 3, [32, 32, 128], stage=4, block='e')
    x = lkrelu_identity_block(x, 3, [32, 32, 128], stage=4, block='f')

    x = lkrelu_conv_block(x, 3, [32, 32, 128], stage=5, block='a')
    x = lkrelu_identity_block(x, 3, [32, 32, 128], stage=5, block='b')
    x = lkrelu_identity_block(x, 3, [32, 32, 128], stage=5, block='c')

    # x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='lkrelu_resnet50')

    return model


def LKReLUResNet50NoBatchnorm(input_tensor=None, input_shape=None,
                       pooling=None,
                       classes=1000, include_top=False):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    # alpha = 1/8
    x = Conv2D(
        8, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = Activation('relu')(x)

    x = lkrelu_conv_nobatchnorm_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = lkrelu_identity_nobatchnorm_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = lkrelu_identity_nobatchnorm_block(x, 3, [16, 16, 64], stage=2, block='c')

    x = lkrelu_conv_nobatchnorm_block(x, 3, [16, 16, 64], stage=3, block='a')
    x = lkrelu_identity_nobatchnorm_block(x, 3, [16, 16, 64], stage=3, block='b')
    x = lkrelu_identity_nobatchnorm_block(x, 3, [16, 16, 64], stage=3, block='c')
    x = lkrelu_identity_nobatchnorm_block(x, 3, [16, 16, 64], stage=3, block='d')

    x = lkrelu_conv_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='a')
    x = lkrelu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='b')
    x = lkrelu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='c')
    x = lkrelu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='d')
    x = lkrelu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='e')
    x = lkrelu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='f')

    x = lkrelu_conv_nobatchnorm_block(x, 3, [32, 32, 128], stage=5, block='a')
    x = lkrelu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=5, block='b')
    x = lkrelu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=5, block='c')

    # x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='lkrelu_nobatchnorm_resnet50')

    return model


def PReLUResNet50(input_tensor=None, input_shape=None,
                  pooling=None,
                  classes=1000, include_top=False):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    # alpha = 1/8
    x = Conv2D(
        8, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = PReLU()(x)

    x = prelu_conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = prelu_identity_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = prelu_identity_block(x, 3, [16, 16, 64], stage=2, block='c')

    x = prelu_conv_block(x, 3, [16, 16, 64], stage=3, block='a')
    x = prelu_identity_block(x, 3, [16, 16, 64], stage=3, block='b')
    x = prelu_identity_block(x, 3, [16, 16, 64], stage=3, block='c')
    x = prelu_identity_block(x, 3, [16, 16, 64], stage=3, block='d')

    x = prelu_conv_block(x, 3, [32, 32, 128], stage=4, block='a')
    x = prelu_identity_block(x, 3, [32, 32, 128], stage=4, block='b')
    x = prelu_identity_block(x, 3, [32, 32, 128], stage=4, block='c')
    x = prelu_identity_block(x, 3, [32, 32, 128], stage=4, block='d')
    x = prelu_identity_block(x, 3, [32, 32, 128], stage=4, block='e')
    x = prelu_identity_block(x, 3, [32, 32, 128], stage=4, block='f')

    x = prelu_conv_block(x, 3, [32, 32, 128], stage=5, block='a')
    x = prelu_identity_block(x, 3, [32, 32, 128], stage=5, block='b')
    x = prelu_identity_block(x, 3, [32, 32, 128], stage=5, block='c')

    # x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='prelu_resnet50')

    return model


def SELUResNet50(input_tensor=None, input_shape=None,
                 pooling=None,
                 classes=1000, include_top=False):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    # alpha = 1/8
    x = Conv2D(
        8, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('selu')(x)

    x = selu_conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = selu_identity_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = selu_identity_block(x, 3, [16, 16, 64], stage=2, block='c')

    x = selu_conv_block(x, 3, [16, 16, 64], stage=3, block='a')
    x = selu_identity_block(x, 3, [16, 16, 64], stage=3, block='b')
    x = selu_identity_block(x, 3, [16, 16, 64], stage=3, block='c')
    x = selu_identity_block(x, 3, [16, 16, 64], stage=3, block='d')

    x = selu_conv_block(x, 3, [32, 32, 128], stage=4, block='a')
    x = selu_identity_block(x, 3, [32, 32, 128], stage=4, block='b')
    x = selu_identity_block(x, 3, [32, 32, 128], stage=4, block='c')
    x = selu_identity_block(x, 3, [32, 32, 128], stage=4, block='d')
    x = selu_identity_block(x, 3, [32, 32, 128], stage=4, block='e')
    x = selu_identity_block(x, 3, [32, 32, 128], stage=4, block='f')

    x = selu_conv_block(x, 3, [32, 32, 128], stage=5, block='a')
    x = selu_identity_block(x, 3, [32, 32, 128], stage=5, block='b')
    x = selu_identity_block(x, 3, [32, 32, 128], stage=5, block='c')

    # x = selu_AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='selu_resnet50')

    return model


def LKSELUResNet50(input_tensor=None, input_shape=None,
                   pooling=None,
                   classes=1000, include_top=False):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    # alpha = 1/8
    x = Conv2D(
        8, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('selu')(x)

    x = lkselu_conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = lkselu_identity_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = lkselu_identity_block(x, 3, [16, 16, 64], stage=2, block='c')

    x = lkselu_conv_block(x, 3, [16, 16, 64], stage=3, block='a')
    x = lkselu_identity_block(x, 3, [16, 16, 64], stage=3, block='b')
    x = lkselu_identity_block(x, 3, [16, 16, 64], stage=3, block='c')
    x = lkselu_identity_block(x, 3, [16, 16, 64], stage=3, block='d')

    x = lkselu_conv_block(x, 3, [32, 32, 128], stage=4, block='a')
    x = lkselu_identity_block(x, 3, [32, 32, 128], stage=4, block='b')
    x = lkselu_identity_block(x, 3, [32, 32, 128], stage=4, block='c')
    x = lkselu_identity_block(x, 3, [32, 32, 128], stage=4, block='d')
    x = lkselu_identity_block(x, 3, [32, 32, 128], stage=4, block='e')
    x = lkselu_identity_block(x, 3, [32, 32, 128], stage=4, block='f')

    x = lkselu_conv_block(x, 3, [32, 32, 128], stage=5, block='a')
    x = lkselu_identity_block(x, 3, [32, 32, 128], stage=5, block='b')
    x = lkselu_identity_block(x, 3, [32, 32, 128], stage=5, block='c')

    # x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='lkselu_resnet50')

    return model


def SELUResNet50NoBatchnorm(input_tensor=None, input_shape=None,
                            pooling=None,
                            classes=1000, include_top=False):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    # alpha = 1/8
    x = Conv2D(
        8, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = Activation('selu')(x)

    x = selu_conv_nobatchnorm_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = selu_identity_nobatchnorm_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = selu_identity_nobatchnorm_block(x, 3, [16, 16, 64], stage=2, block='c')

    x = selu_conv_nobatchnorm_block(x, 3, [16, 16, 64], stage=3, block='a')
    x = selu_identity_nobatchnorm_block(x, 3, [16, 16, 64], stage=3, block='b')
    x = selu_identity_nobatchnorm_block(x, 3, [16, 16, 64], stage=3, block='c')
    x = selu_identity_nobatchnorm_block(x, 3, [16, 16, 64], stage=3, block='d')

    x = selu_conv_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='a')
    x = selu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='b')
    x = selu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='c')
    x = selu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='d')
    x = selu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='e')
    x = selu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='f')

    x = selu_conv_nobatchnorm_block(x, 3, [32, 32, 128], stage=5, block='a')
    x = selu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=5, block='b')
    x = selu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=5, block='c')

    # x = selu_AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='selu_nobatchnorm_resnet50')

    return model


def SwishResNet50(input_tensor=None, input_shape=None,
                  pooling=None,
                  classes=1000, include_top=False):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    # alpha = 1/8
    x = Conv2D(
        8, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation(swish)(x)

    x = swish_conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = swish_identity_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = swish_identity_block(x, 3, [16, 16, 64], stage=2, block='c')

    x = swish_conv_block(x, 3, [16, 16, 64], stage=3, block='a')
    x = swish_identity_block(x, 3, [16, 16, 64], stage=3, block='b')
    x = swish_identity_block(x, 3, [16, 16, 64], stage=3, block='c')
    x = swish_identity_block(x, 3, [16, 16, 64], stage=3, block='d')

    x = swish_conv_block(x, 3, [32, 32, 128], stage=4, block='a')
    x = swish_identity_block(x, 3, [32, 32, 128], stage=4, block='b')
    x = swish_identity_block(x, 3, [32, 32, 128], stage=4, block='c')
    x = swish_identity_block(x, 3, [32, 32, 128], stage=4, block='d')
    x = swish_identity_block(x, 3, [32, 32, 128], stage=4, block='e')
    x = swish_identity_block(x, 3, [32, 32, 128], stage=4, block='f')

    x = swish_conv_block(x, 3, [32, 32, 128], stage=5, block='a')
    x = swish_identity_block(x, 3, [32, 32, 128], stage=5, block='b')
    x = swish_identity_block(x, 3, [32, 32, 128], stage=5, block='c')

    # x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='swish_resnet50')

    return model


def LKReLUNoRes50(input_tensor=None, input_shape=None,
                  pooling=None,
                  classes=1000, include_top=False):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # alpha = 1/8
    x = Conv2D(
        8, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = LKReLU()(x)

    x = lkrelu_conv_nores_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = lkrelu_identity_nores_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = lkrelu_identity_nores_block(x, 3, [16, 16, 64], stage=2, block='c')

    x = lkrelu_conv_nores_block(x, 3, [16, 16, 64], stage=3, block='a')
    x = lkrelu_identity_nores_block(x, 3, [16, 16, 64], stage=3, block='b')
    x = lkrelu_identity_nores_block(x, 3, [16, 16, 64], stage=3, block='c')
    x = lkrelu_identity_nores_block(x, 3, [16, 16, 64], stage=3, block='d')

    x = lkrelu_conv_nores_block(x, 3, [32, 32, 128], stage=4, block='a')
    x = lkrelu_identity_nores_block(x, 3, [32, 32, 128], stage=4, block='b')
    x = lkrelu_identity_nores_block(x, 3, [32, 32, 128], stage=4, block='c')
    x = lkrelu_identity_nores_block(x, 3, [32, 32, 128], stage=4, block='d')
    x = lkrelu_identity_nores_block(x, 3, [32, 32, 128], stage=4, block='e')
    x = lkrelu_identity_nores_block(x, 3, [32, 32, 128], stage=4, block='f')

    x = lkrelu_conv_nores_block(x, 3, [32, 32, 128], stage=5, block='a')
    x = lkrelu_identity_nores_block(x, 3, [32, 32, 128], stage=5, block='b')
    x = lkrelu_identity_nores_block(x, 3, [32, 32, 128], stage=5, block='c')

    # x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='lkrelu_nores_resnet50')

    return model


def SELUNoRes50(input_tensor=None, input_shape=None,
                pooling=None,
                classes=1000, include_top=False):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # alpha = 1/8
    x = Conv2D(
        8, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = Activation('selu')(x)

    x = selu_conv_nores_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = selu_identity_nores_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = selu_identity_nores_block(x, 3, [16, 16, 64], stage=2, block='c')

    x = selu_conv_nores_block(x, 3, [16, 16, 64], stage=3, block='a')
    x = selu_identity_nores_block(x, 3, [16, 16, 64], stage=3, block='b')
    x = selu_identity_nores_block(x, 3, [16, 16, 64], stage=3, block='c')
    x = selu_identity_nores_block(x, 3, [16, 16, 64], stage=3, block='d')

    x = selu_conv_nores_block(x, 3, [32, 32, 128], stage=4, block='a')
    x = selu_identity_nores_block(x, 3, [32, 32, 128], stage=4, block='b')
    x = selu_identity_nores_block(x, 3, [32, 32, 128], stage=4, block='c')
    x = selu_identity_nores_block(x, 3, [32, 32, 128], stage=4, block='d')
    x = selu_identity_nores_block(x, 3, [32, 32, 128], stage=4, block='e')
    x = selu_identity_nores_block(x, 3, [32, 32, 128], stage=4, block='f')

    x = selu_conv_nores_block(x, 3, [32, 32, 128], stage=5, block='a')
    x = selu_identity_nores_block(x, 3, [32, 32, 128], stage=5, block='b')
    x = selu_identity_nores_block(x, 3, [32, 32, 128], stage=5, block='c')

    # x = selu_AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='selu_nores_resnet50')

    return model


def LKSELUNoRes50(input_tensor=None, input_shape=None,
                  pooling=None,
                  classes=1000, include_top=False):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    # alpha = 1/8
    x = Conv2D(
        8, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = LKSELU()(x)

    x = lkselu_conv_nores_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = lkselu_identity_nores_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = lkselu_identity_nores_block(x, 3, [16, 16, 64], stage=2, block='c')

    x = lkselu_conv_nores_block(x, 3, [16, 16, 64], stage=3, block='a')
    x = lkselu_identity_nores_block(x, 3, [16, 16, 64], stage=3, block='b')
    x = lkselu_identity_nores_block(x, 3, [16, 16, 64], stage=3, block='c')
    x = lkselu_identity_nores_block(x, 3, [16, 16, 64], stage=3, block='d')

    x = lkselu_conv_nores_block(x, 3, [32, 32, 128], stage=4, block='a')
    x = lkselu_identity_nores_block(x, 3, [32, 32, 128], stage=4, block='b')
    x = lkselu_identity_nores_block(x, 3, [32, 32, 128], stage=4, block='c')
    x = lkselu_identity_nores_block(x, 3, [32, 32, 128], stage=4, block='d')
    x = lkselu_identity_nores_block(x, 3, [32, 32, 128], stage=4, block='e')
    x = lkselu_identity_nores_block(x, 3, [32, 32, 128], stage=4, block='f')

    x = lkselu_conv_nores_block(x, 3, [32, 32, 128], stage=5, block='a')
    x = lkselu_identity_nores_block(x, 3, [32, 32, 128], stage=5, block='b')
    x = lkselu_identity_nores_block(x, 3, [32, 32, 128], stage=5, block='c')

    # x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='lkselu_nores_resnet50')

    return model


def LKSELUResNet50NoBatchnorm(input_tensor=None, input_shape=None,
                              pooling=None,
                              classes=1000, include_top=False):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    # alpha = 1/8
    x = Conv2D(
        8, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('selu')(x)

    x = lkselu_conv_nobatchnorm_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = lkselu_identity_nobatchnorm_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = lkselu_identity_nobatchnorm_block(x, 3, [16, 16, 64], stage=2, block='c')

    x = lkselu_conv_nobatchnorm_block(x, 3, [16, 16, 64], stage=3, block='a')
    x = lkselu_identity_nobatchnorm_block(x, 3, [16, 16, 64], stage=3, block='b')
    x = lkselu_identity_nobatchnorm_block(x, 3, [16, 16, 64], stage=3, block='c')
    x = lkselu_identity_nobatchnorm_block(x, 3, [16, 16, 64], stage=3, block='d')

    x = lkselu_conv_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='a')
    x = lkselu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='b')
    x = lkselu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='c')
    x = lkselu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='d')
    x = lkselu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='e')
    x = lkselu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=4, block='f')

    x = lkselu_conv_nobatchnorm_block(x, 3, [32, 32, 128], stage=5, block='a')
    x = lkselu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=5, block='b')
    x = lkselu_identity_nobatchnorm_block(x, 3, [32, 32, 128], stage=5, block='c')

    # x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='lkselu_resnet50_nobatchnorm')

    return model


def LKSwishResNet50(input_tensor=None, input_shape=None,
                    pooling=None,
                    classes=1000, include_top=False):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    # alpha = 1/8
    x = Conv2D(
        8, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation(swish)(x)

    x = lkswish_conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = lkswish_identity_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = lkswish_identity_block(x, 3, [16, 16, 64], stage=2, block='c')

    x = lkswish_conv_block(x, 3, [16, 16, 64], stage=3, block='a')
    x = lkswish_identity_block(x, 3, [16, 16, 64], stage=3, block='b')
    x = lkswish_identity_block(x, 3, [16, 16, 64], stage=3, block='c')
    x = lkswish_identity_block(x, 3, [16, 16, 64], stage=3, block='d')

    x = lkswish_conv_block(x, 3, [32, 32, 128], stage=4, block='a')
    x = lkswish_identity_block(x, 3, [32, 32, 128], stage=4, block='b')
    x = lkswish_identity_block(x, 3, [32, 32, 128], stage=4, block='c')
    x = lkswish_identity_block(x, 3, [32, 32, 128], stage=4, block='d')
    x = lkswish_identity_block(x, 3, [32, 32, 128], stage=4, block='e')
    x = lkswish_identity_block(x, 3, [32, 32, 128], stage=4, block='f')

    x = lkswish_conv_block(x, 3, [32, 32, 128], stage=5, block='a')
    x = lkswish_identity_block(x, 3, [32, 32, 128], stage=5, block='b')
    x = lkswish_identity_block(x, 3, [32, 32, 128], stage=5, block='c')

    # x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='lkswish_resnet50')

    return model


def LKPReLUResNet50(input_tensor=None, input_shape=None,
                    pooling=None,
                    classes=1000, include_top=False):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    # alpha = 1/8
    x = Conv2D(
        8, (3, 3), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = PReLU()(x)

    x = lkprelu_conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = lkprelu_identity_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = lkprelu_identity_block(x, 3, [16, 16, 64], stage=2, block='c')

    x = lkprelu_conv_block(x, 3, [16, 16, 64], stage=3, block='a')
    x = lkprelu_identity_block(x, 3, [16, 16, 64], stage=3, block='b')
    x = lkprelu_identity_block(x, 3, [16, 16, 64], stage=3, block='c')
    x = lkprelu_identity_block(x, 3, [16, 16, 64], stage=3, block='d')

    x = lkprelu_conv_block(x, 3, [32, 32, 128], stage=4, block='a')
    x = lkprelu_identity_block(x, 3, [32, 32, 128], stage=4, block='b')
    x = lkprelu_identity_block(x, 3, [32, 32, 128], stage=4, block='c')
    x = lkprelu_identity_block(x, 3, [32, 32, 128], stage=4, block='d')
    x = lkprelu_identity_block(x, 3, [32, 32, 128], stage=4, block='e')
    x = lkprelu_identity_block(x, 3, [32, 32, 128], stage=4, block='f')

    x = lkprelu_conv_block(x, 3, [32, 32, 128], stage=5, block='a')
    x = lkprelu_identity_block(x, 3, [32, 32, 128], stage=5, block='b')
    x = lkprelu_identity_block(x, 3, [32, 32, 128], stage=5, block='c')

    # x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='lkprelu_resnet50')

    return model


def lkrelu_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'lkrelu_res' + str(stage) + block + '_branch'
    bn_name_base = 'lkrelu_bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = LKReLU()(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = LKReLU()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    # x = LKReLU()(x)
    return x


def lkselu_identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'lkselu_res' + str(stage) + block + '_branch'
    bn_name_base = 'lkselu_bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = LKSELU()(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = LKSELU()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('selu')(x)
    # x = LKSELU()(x)
    return x


def lkselu_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'lkselu_res' + str(stage) + block + '_branch'
    bn_name_base = 'lkselu_bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = LKSELU()(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = LKSELU()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = Activation('selu')(x)
    # x = LKSELU()(x)
    return x


def lkrelu_identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'lkrelu_res' + str(stage) + block + '_branch'
    bn_name_base = 'lkrelu_bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = LKReLU()(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = LKReLU()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    # x = LKReLU()(x)
    return x


def lkselu_identity_nobatchnorm_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'purelkselu_res' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = LKSELU()(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = LKSELU()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('selu')(x)
    # x = LKSELU()(x)
    return x


def lkselu_conv_nobatchnorm_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'purelkselu_res' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = LKSELU()(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = LKSELU()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    x = layers.add([x, shortcut])
    x = Activation('selu')(x)
    # x = LKSELU()(x)
    return x


def lkrelu_conv_nobatchnorm_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'pure_lkrelu_res' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = LKReLU()(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = LKReLU()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    # x = LKReLU()(x)
    return x


def lkrelu_identity_nobatchnorm_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'pure_lkrelu_res' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = LKReLU()(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = LKReLU()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    # x = LKReLU()(x)
    return x


def prelu_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'prelu_res' + str(stage) + block + '_branch'
    bn_name_base = 'prelu_bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = PReLU()(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = PReLU()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = PReLU()(x)
    return x


def prelu_identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'prelu_res' + str(stage) + block + '_branch'
    bn_name_base = 'prelu_bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = PReLU()(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = PReLU()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = PReLU()(x)
    return x


def selu_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'selu_res' + str(stage) + block + '_branch'
    bn_name_base = 'selu_bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('selu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('selu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = Activation('selu')(x)
    return x


def selu_identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'selu_res' + str(stage) + block + '_branch'
    bn_name_base = 'selu_bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('selu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('selu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('selu')(x)
    return x


def selu_conv_nobatchnorm_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'selu_res' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = Activation('selu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = Activation('selu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    x = layers.add([x, shortcut])
    x = Activation('selu')(x)
    return x


def selu_identity_nobatchnorm_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'selu_res' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = Activation('selu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = Activation('selu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('selu')(x)
    return x


def swish_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'swish_res' + str(stage) + block + '_branch'
    bn_name_base = 'swish_bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation(swish)(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation(swish)(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = Activation(swish)(x)
    return x


def swish_identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'swish_res' + str(stage) + block + '_branch'
    bn_name_base = 'swish_bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation(swish)(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation(swish)(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation(swish)(x)
    return x


def lkrelu_conv_nores_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'lkrelu_nores' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = LKReLU()(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = LKReLU()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

    # x = Activation('relu')(x)
    x = LKReLU()(x)
    return x


def lkrelu_identity_nores_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'lkrelu_nores' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = LKReLU()(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = LKReLU()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

    x = LKReLU()(x)
    return x


def selu_conv_nores_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'nores_selu' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = Activation('selu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = Activation('selu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

    x = Activation('selu')(x)
    return x


def selu_identity_nores_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'nores_selu' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = Activation('selu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = Activation('selu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

    x = Activation('selu')(x)
    return x


def lkswish_identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'LKSwish_res' + str(stage) + block + '_branch'
    bn_name_base = 'LKSwish_bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = LKSwish()(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = LKSwish()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation(swish)(x)
    # x = LKSwish()(x)
    return x


def lkswish_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'LKSwish_res' + str(stage) + block + '_branch'
    bn_name_base = 'LKSwish_bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = LKSwish()(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = LKSwish()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = Activation(swish)(x)
    # x = LKSwish()(x)
    return x


def lkprelu_identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'lkprelu_res' + str(stage) + block + '_branch'
    bn_name_base = 'lkprelu_bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = LKPReLU()(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = LKPReLU()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = PReLU()(x)
    # x = LKPReLU()(x)
    return x


def lkprelu_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'lkprelu_res' + str(stage) + block + '_branch'
    bn_name_base = 'lkprelu_bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = LKPReLU()(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = LKPReLU()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = PReLU()(x)
    # x = LKPReLU()(x)
    return x


def lkselu_conv_nores_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'lkselu_nores' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = LKSELU()(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = LKSELU()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

    # x = Activation('relu')(x)
    x = LKSELU()(x)
    return x


def lkselu_identity_nores_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'lkselu_nores' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = LKSELU()(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = LKSELU()(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

    x = LKSELU()(x)
    return x

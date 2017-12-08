import datetime
import os
import random as rn
from unittest import TestCase

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K, Input
from keras.callbacks import TensorBoard
from keras.engine import Model
from keras.layers import Flatten, Dense, Conv2D, PReLU

from linked_neurons import LKReLU, LKPReLU, LKSELU, LKSwish, swish
from tests.util import load_cifar10


class TestDeep(TestCase):
    show = False

    def setUp(self):
        self.metrics = ['accuracy']
        self.epochs = 200
        self.width = 50
        self.width_histogram_freq = 0
        # Set seed
        self.seed = 100
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        rn.seed(self.seed)
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        tf.set_random_seed(self.seed)
        self.sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(self.sess)

    def test_relu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        inputs = Input(shape=x_train.shape[1:])
        x = Conv2D(self.width, (3, 3), activation='relu')(inputs)
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax', name='fc1000')(x)
        model = Model(inputs=inputs, outputs=x)
        print(model.summary())

        opt = keras.optimizers.sgd()

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/width-relu-{}-cifar10-{}-{}'.format(self.width, self.seed,
                                                                 datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                      # # ModelCheckpoint(
                      #                         'checkpoints/width-cifar10-{epoch:02d}-{val_loss:.2f}.hdf5')
                  ])
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def test_lkrelu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        inputs = Input(shape=x_train.shape[1:])
        x = Conv2D(self.width, (3, 3))(inputs)
        x = LKReLU()(x)
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax', name='fc1000')(x)
        model = Model(inputs=inputs, outputs=x)
        print(model.summary())

        opt = keras.optimizers.sgd()

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/width-lkrelu-{}-cifar10-{}-{}'.format(self.width, self.seed,
                                                                       datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                      # ModelCheckpoint(
                      # 'checkpoints/width-lkrelu-cifar10-{epoch:02d}-{val_loss:.2f}.hdf5')
                  ])
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def test_prelu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        inputs = Input(shape=x_train.shape[1:])
        x = Conv2D(self.width, (3, 3))(inputs)
        x = PReLU()(x)
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax', name='fc1000')(x)
        model = Model(inputs=inputs, outputs=x)
        print(model.summary())

        opt = keras.optimizers.sgd()

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/width-prelu-{}-cifar10-{}-{}'.format(self.width, self.seed,
                                                                  datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                      # ModelCheckpoint(
                      # 'checkpoints/width-prelu-cifar10-{epoch:02d}-{val_loss:.2f}.hdf5')
                  ])
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def test_selu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        inputs = Input(shape=x_train.shape[1:])
        x = Conv2D(self.width, (3, 3), activation='selu')(inputs)
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax', name='fc1000')(x)
        model = Model(inputs=inputs, outputs=x)
        print(model.summary())

        opt = keras.optimizers.sgd()

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/width-selu-{}-cifar10-{}-{}'.format(self.width, self.seed,
                                                                 datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                      # ModelCheckpoint(
                      # 'checkpoints/width-selu-cifar10-{epoch:02d}-{val_loss:.2f}.hdf5')
                  ])
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def test_swish(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        inputs = Input(shape=x_train.shape[1:])
        x = Conv2D(self.width, (3, 3), activation=swish)(inputs)
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax', name='fc1000')(x)
        model = Model(inputs=inputs, outputs=x)
        print(model.summary())

        opt = keras.optimizers.sgd()

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/width-swish-{}-cifar10-{}-{}'.format(self.width, self.seed,
                                                                  datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                      # ModelCheckpoint(
                      # 'checkpoints/width-swish-cifar10-{epoch:02d}-{val_loss:.2f}.hdf5')
                  ])
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def test_lkprelu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        inputs = Input(shape=x_train.shape[1:])
        x = Conv2D(self.width, (3, 3))(inputs)
        x = LKPReLU()(x)
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax', name='fc1000')(x)
        model = Model(inputs=inputs, outputs=x)
        print(model.summary())

        opt = keras.optimizers.sgd()

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/width-lkprelu-{}-cifar10-{}-{}'.format(self.width, self.seed,
                                                                        datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                      # ModelCheckpoint(
                      # 'checkpoints/width-lkprelu-cifar10-{epoch:02d}-{val_loss:.2f}.hdf5')
                  ])
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def test_lkselu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        inputs = Input(shape=x_train.shape[1:])
        x = Conv2D(self.width, (3, 3))(inputs)
        x = LKSELU()(x)
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax', name='fc1000')(x)
        model = Model(inputs=inputs, outputs=x)
        print(model.summary())

        opt = keras.optimizers.sgd()

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/width-lkselu-{}-cifar10-{}-{}'.format(self.width, self.seed,
                                                                       datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                      # ModelCheckpoint(
                      # 'checkpoints/width-lkselu-cifar10-{epoch:02d}-{val_loss:.2f}.hdf5')
                  ])
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def test_lkswish(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        inputs = Input(shape=x_train.shape[1:])
        x = Conv2D(self.width, (3, 3))(inputs)
        x = LKSwish()(x)
        x = Flatten()(x)
        x = Dense(num_classes, activation='softmax', name='fc1000')(x)
        model = Model(inputs=inputs, outputs=x)
        print(model.summary())

        opt = keras.optimizers.sgd()

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/width-lkswish-{}-cifar10-{}-{}'.format(self.width, self.seed,
                                                                        datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                      # ModelCheckpoint(
                      # 'checkpoints/width-lkswish-cifar10-{epoch:02d}-{val_loss:.2f}.hdf5')
                  ])
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

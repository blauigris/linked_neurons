import datetime
import os
import random as rn
from unittest import TestCase

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard

from tests.util import network_for_cifar10_factory, load_cifar10


class TestExtended(TestCase):

    def setUp(self):
        self.metrics = ['accuracy']
        self.epochs = 200

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

        model = network_for_cifar10_factory('relu', x_train.shape[1:], num_classes)

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/relu-extended-{}-{}'.format(self.seed, datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[TensorBoard(log_dir=log_dir), ])

    def test_prelu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = network_for_cifar10_factory('prelu', x_train.shape[1:], num_classes)

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/prelu-extended-{}-{}'.format(self.seed, datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[TensorBoard(log_dir=log_dir), ])

    def test_selu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = network_for_cifar10_factory('selu', x_train.shape[1:], num_classes)

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/selu-extended-{}-{}'.format(self.seed, datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[TensorBoard(log_dir=log_dir), ])

    def test_swish(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = network_for_cifar10_factory('swish', x_train.shape[1:], num_classes)

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/swish-extended-{}-{}'.format(self.seed, datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[TensorBoard(log_dir=log_dir), ])

    def test_lkrelu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = network_for_cifar10_factory('lkrelu', x_train.shape[1:], num_classes)

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/lkrelu-extended-{}-{}'.format(self.seed, datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[TensorBoard(log_dir=log_dir), ])

    def test_lkprelu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = network_for_cifar10_factory('lkprelu', x_train.shape[1:], num_classes)

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/lkprelu-extended-{}-{}'.format(self.seed, datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[TensorBoard(log_dir=log_dir), ])

    def test_lkselu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = network_for_cifar10_factory('lkselu', x_train.shape[1:], num_classes)

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/lkselu-extended-{}-{}'.format(self.seed, datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[TensorBoard(log_dir=log_dir), ])

    def test_lkswish(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = network_for_cifar10_factory('lkswish', x_train.shape[1:], num_classes)

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/lkswish-extended-{}-{}'.format(self.seed, datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[TensorBoard(log_dir=log_dir), ])

    def test_relu_sqrt2(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = network_for_cifar10_factory('relu', x_train.shape[1:], num_classes, width=np.sqrt(2))

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/relu-sqrt2-extended-{}-{}'.format(self.seed, datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[TensorBoard(log_dir=log_dir), ])

    def test_prelu_sqrt2(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = network_for_cifar10_factory('prelu', x_train.shape[1:], num_classes, width=np.sqrt(2))

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/prelu-sqrt2-extended-{}-{}'.format(self.seed, datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[TensorBoard(log_dir=log_dir), ])

    def test_selu_sqrt2(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = network_for_cifar10_factory('selu', x_train.shape[1:], num_classes, width=np.sqrt(2))

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/selu-sqrt2-extended-{}-{}'.format(self.seed, datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[TensorBoard(log_dir=log_dir), ])

    def test_swish_sqrt2(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = network_for_cifar10_factory('swish', x_train.shape[1:], num_classes, width=np.sqrt(2))

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)

        log_dir = 'summaries/swish-sqrt2-extended-{}-{}'.format(self.seed, datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[TensorBoard(log_dir=log_dir), ])

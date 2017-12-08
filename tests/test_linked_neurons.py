#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `linked_neurons` package."""

import os
import random as rn
from unittest import TestCase

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

from linked_neurons.linked_neurons import LKReLU, LKSELU, LKSwish
from .util import load_mnist


class TestLinkedNeurons(TestCase):
    def setUp(self):
        self.metrics = ['accuracy']
        self.epochs = 20
        self.batch_size = 128
        self.offset = 40
        self.verbose = 1
        # Set seed
        self.seed = 2
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        rn.seed(self.seed)
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        tf.set_random_seed(self.seed)
        self.sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(self.sess)

    def tearDown(self):
        self.sess.close()

    def test_seed(self):
        batch_size = 128
        num_classes = 10
        epochs = 1

        (x_train, y_train), (x_test, y_test) = load_mnist()
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=self.metrics)

        self.sess.run(tf.global_variables_initializer())
        vars1 = [self.sess.run(var) for var in tf.trainable_variables()]

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  validation_data=(x_test, y_test), shuffle=False)
        score1 = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score1[0])
        print('Test accuracy:', score1[1])
        self.tearDown()
        self.setUp()

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=self.metrics)

        self.sess.run(tf.global_variables_initializer())
        vars2 = [self.sess.run(var) for var in tf.trainable_variables()]
        self.assertTrue(np.all([np.all(vars1[i] == vars2[i]) for i in range(len(vars1))]))

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  validation_data=(x_test, y_test), shuffle=False)
        score2 = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score2[0])
        print('Test accuracy:', score2[1])
        self.assertAlmostEqual(score1[0], score2[0], places=2)
        self.assertAlmostEqual(score1[1], score2[1], places=2)

    def test_lkrelu_layer(self):
        (x_train, y_train), (x_test, y_test) = load_mnist()
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation=None, input_shape=x_train.shape[1:]))
        model.add(LKReLU())
        model.add(Conv2D(64, (3, 3), activation=None))
        model.add(LKReLU())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation=None))
        model.add(LKReLU())
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=self.metrics)

        model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=0,
                  validation_data=(x_test, y_test), shuffle=False)
        score = model.evaluate(x_test, y_test, verbose=self.verbose)

        self.assertAlmostEqual(score[1], 0.99, places=1)

    def test_lkselu_layer(self):
        (x_train, y_train), (x_test, y_test) = load_mnist()
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation=None, input_shape=x_train.shape[1:]))
        model.add(LKSELU())
        model.add(Conv2D(64, (3, 3), activation=None))
        model.add(LKSELU())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation=None))
        model.add(LKSELU())
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=self.metrics)

        model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=0,
                  validation_data=(x_test, y_test), shuffle=False)
        score = model.evaluate(x_test, y_test, verbose=self.verbose)

        self.assertAlmostEqual(score[1], 0.99, places=1)

    def test_lkswish_layer(self):
        (x_train, y_train), (x_test, y_test) = load_mnist()
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation=None, input_shape=x_train.shape[1:]))
        model.add(LKSwish())
        model.add(Conv2D(64, (3, 3), activation=None))
        model.add(LKSwish())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation=None))
        model.add(LKSwish())
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=self.metrics)

        model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=0,
                  validation_data=(x_test, y_test), shuffle=False)
        score = model.evaluate(x_test, y_test, verbose=self.verbose)

        self.assertAlmostEqual(score[1], 0.99, places=1)

import datetime
import os
import random as rn
from unittest import TestCase

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard

from tests.test_article.depth import ReLUDepthNet, LKReLUDepthNet, PReLUDepthNet, SeLUDepthNet, \
    BatchNormDepthNet, LeakyDepthNet, SwishDepthNet, LKPReLUDepthNet, LKSELUDepthNet, LKSwishDepthNet
from tests.util import load_cifar10


class TestDepth(TestCase):
    def setUp(self):
        self.metrics = ['accuracy']
        self.epochs = 200
        self.width = 4
        self.depth = 50
        self.lr = 0.001
        # Set seed
        self.seed = 100
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        rn.seed(self.seed)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                                      gpu_options=gpu_options
                                      )
        tf.set_random_seed(self.seed)
        self.sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(self.sess)

    def test_relu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = ReLUDepthNet(input_shape=x_train.shape[1:], classes=num_classes, width=self.width,
                             depth=self.depth)

        opt = keras.optimizers.adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)
        print(model.summary())

        log_dir = 'summaries/relu_depth{}-{}-{}-{}'.format(self.depth, self.seed, self.lr,
                                                           datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                  ])
        model.save('checkpoints/depth{}-relu.h5'.format(self.depth))

    def test_lkrelu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = LKReLUDepthNet(input_shape=x_train.shape[1:], classes=num_classes, width=self.width,
                               depth=self.depth)

        opt = keras.optimizers.adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)
        print(model.summary())

        log_dir = 'summaries/lkrelu_depth{}-{}-{}-{}'.format(self.depth, self.seed, self.lr,
                                                             datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                  ])
        model.save('checkpoints/depth{}-lkrelu.h5'.format(self.depth))

    def test_prelu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = PReLUDepthNet(input_shape=x_train.shape[1:], classes=num_classes, width=self.width,
                              depth=self.depth)

        opt = keras.optimizers.adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)
        print(model.summary())

        log_dir = 'summaries/prelu_depth{}-{}-{}-{}'.format(self.depth, self.seed, self.lr,
                                                            datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                  ])
        model.save('checkpoints/depth{}-prelu.h5'.format(self.depth))

    def test_selu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = SeLUDepthNet(input_shape=x_train.shape[1:], classes=num_classes, width=self.width,
                             depth=self.depth)

        opt = keras.optimizers.adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)
        print(model.summary())

        log_dir = 'summaries/selu_depth{}-{}-{}-{}'.format(self.depth, self.seed, self.lr,
                                                           datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                  ])
        model.save('checkpoints/depth{}-selu.h5'.format(self.depth))

    def test_batchnorm(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = BatchNormDepthNet(input_shape=x_train.shape[1:], classes=num_classes, width=self.width,
                                  depth=self.depth)

        opt = keras.optimizers.adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)
        print(model.summary())

        log_dir = 'summaries/batchnorm_depth{}-{}-{}-{}'.format(self.depth, self.seed, self.lr,
                                                                datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                  ])
        model.save('checkpoints/depth{}-batchnorm.h5'.format(self.depth))


    def test_swish(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = SwishDepthNet(input_shape=x_train.shape[1:], classes=num_classes, width=self.width,
                              depth=self.depth)

        opt = keras.optimizers.adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)
        print(model.summary())

        log_dir = 'summaries/swish_depth{}-{}-{}-{}'.format(self.depth, self.seed, self.lr,
                                                            datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                  ])
        model.save('checkpoints/depth{}-swish.h5'.format(self.depth))

    def test_lkprelu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = LKPReLUDepthNet(input_shape=x_train.shape[1:], classes=num_classes, width=self.width,
                                depth=self.depth)

        opt = keras.optimizers.adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)
        print(model.summary())

        log_dir = 'summaries/lkprelu_depth{}-{}-{}-{}'.format(self.depth, self.seed, self.lr,
                                                              datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                  ])
        model.save('checkpoints/depth{}-lkprelu.h5'.format(self.depth))

    def test_lkselu(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = LKSELUDepthNet(input_shape=x_train.shape[1:], classes=num_classes, width=self.width,
                               depth=self.depth)

        opt = keras.optimizers.adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)
        print(model.summary())

        log_dir = 'summaries/lkselu_depth{}-{}-{}-{}'.format(self.depth, self.seed, self.lr,
                                                             datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                  ])
        model.save('checkpoints/depth{}-lkselu.h5'.format(self.depth))

    def test_lkswish(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = LKSwishDepthNet(input_shape=x_train.shape[1:], classes=num_classes, width=self.width,
                                depth=self.depth)

        opt = keras.optimizers.adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)
        print(model.summary())

        log_dir = 'summaries/lkswish_depth{}-{}-{}-{}'.format(self.depth, self.seed, self.lr,
                                                              datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                  ])
        model.save('checkpoints/depth{}-lkswish.h5'.format(self.depth))

    def test_relu_sqrt2(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = ReLUDepthNet(input_shape=x_train.shape[1:], classes=num_classes, width=int(self.width * np.sqrt(2)),
                         depth=self.depth)

        opt = keras.optimizers.adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)
        print(model.summary())

        log_dir = 'summaries/relu_sqrt2_depth{}-{}-{}-{}'.format(self.depth, self.seed, self.lr,
                                                                  datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                  ])
        model.save('checkpoints/depth{}-relu_sqrt2.h5'.format(self.depth))

    def test_prelu_sqrt2(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = PReLUDepthNet(input_shape=x_train.shape[1:], classes=num_classes, width=int(self.width * np.sqrt(2)),
                              depth=self.depth)

        opt = keras.optimizers.adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)
        print(model.summary())

        log_dir = 'summaries/prelu_sqrt2_depth{}-{}-{}-{}'.format(self.depth, self.seed, self.lr,
                                                               datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                  ])
        model.save('checkpoints/depth_{}-prelu_sqrt2.h5'.format(self.depth))

    def test_selu_sqrt2(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = SeLUDepthNet(input_shape=x_train.shape[1:], classes=num_classes, width=int(self.width * np.sqrt(2)),
                             depth=self.depth)

        opt = keras.optimizers.adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)
        print(model.summary())

        log_dir = 'summaries/selu_sqrt2_depth{}-{}-{}-{}'.format(self.depth, self.seed, self.lr,
                                                              datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                  ])
        model.save('checkpoints/depth{}-selu_sqrt2.h5'.format(self.depth))

    def test_swish_sqrt2(self):
        batch_size = 32
        num_classes = 10

        (x_train, y_train), (x_test, y_test) = load_cifar10()

        model = SwishDepthNet(input_shape=x_train.shape[1:], classes=num_classes, width=int(self.width * np.sqrt(2)),
                              depth=self.depth)

        opt = keras.optimizers.adam(lr=self.lr)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=self.metrics)
        print(model.summary())

        log_dir = 'summaries/swish_sqrt2_depth{}-{}-{}-{}'.format(self.depth, self.seed, self.lr,
                                                               datetime.datetime.now())
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=self.epochs,
                  validation_data=(x_test, y_test),
                  shuffle=False,
                  callbacks=[
                      TensorBoard(log_dir=log_dir),
                  ])
        model.save('checkpoints/depth{}-swish_sqrt2.h5'.format(self.depth))

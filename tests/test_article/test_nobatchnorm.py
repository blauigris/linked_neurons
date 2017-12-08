import datetime
import os
import random as rn
from unittest import TestCase

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from tests.test_article.resnet50 import LKReLUResNet50NoBatchnorm, LKSELUResNet50NoBatchnorm, SELUResNet50NoBatchnorm


class TestNoBatchnorm(TestCase):
    def setUp(self):
        self.metrics = ['accuracy']
        self.epochs = 2000
        # Set seed
        self.seed = 100
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        rn.seed(self.seed)
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        tf.set_random_seed(self.seed)
        self.sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(self.sess)

    def test_lkrelu(self):
        batch_size = 128
        classes = 10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # x_train = block_reduce(x_train, (1, 5, 5, 1), func=np.mean)
        # x_test = block_reduce(x_test, (1, 5, 5, 1), func=np.mean)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = np_utils.to_categorical(y_train, classes)
        y_test = np_utils.to_categorical(y_test, classes)

        model = LKReLUResNet50NoBatchnorm(input_shape=[32, 32, 3], classes=10, include_top=True)
        print(model.summary())
        opt = Adam(lr=1e-3, decay=1e-6, )
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=opt,
                      metrics=self.metrics)

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total basicnet_width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)

        datagen.fit(x_train)

        now = datetime.datetime.now()
        filepath = 'checkpoints/resnet50-nobatchnorm-lkrelu-{epoch:02d}-%s.hdf5' % now
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False,
                                     save_weights_only=False, mode='max', period=100)

        log_dir = 'summaries/resnet50-nobatchnorm-lkrelu-{}-{}'.format(self.seed, now)
        tensorboard = TensorBoard(log_dir=log_dir)

        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=self.epochs,
                            steps_per_epoch=int(x_train.shape[0] / batch_size),
                            verbose=1,
                            validation_data=(x_test, y_test),
                            callbacks=[checkpoint, tensorboard])
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def test_lkselu(self):
        batch_size = 128
        classes = 10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # x_train = block_reduce(x_train, (1, 5, 5, 1), func=np.mean)
        # x_test = block_reduce(x_test, (1, 5, 5, 1), func=np.mean)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = np_utils.to_categorical(y_train, classes)
        y_test = np_utils.to_categorical(y_test, classes)

        model = LKSELUResNet50NoBatchnorm(input_shape=[32, 32, 3], classes=10, include_top=True)
        print(model.summary())
        opt = Adam(lr=1e-3, decay=1e-6, )
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=opt,
                      metrics=self.metrics)

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total basicnet_width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)

        datagen.fit(x_train)

        now = datetime.datetime.now()
        filepath = 'checkpoints/resnet50-nobatchnorm-lkselu-{epoch:02d}-%s.hdf5' % now
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False,
                                     save_weights_only=False, mode='max', period=100)

        log_dir = 'summaries/resnet50-nobatchnorm-lkselu-{}-{}'.format(self.seed, now)
        tensorboard = TensorBoard(log_dir=log_dir)

        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=self.epochs,
                            steps_per_epoch=int(x_train.shape[0] / batch_size),
                            verbose=1,
                            validation_data=(x_test, y_test),
                            callbacks=[checkpoint, tensorboard])
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def test_selu(self):
        batch_size = 128
        classes = 10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # x_train = block_reduce(x_train, (1, 5, 5, 1), func=np.mean)
        # x_test = block_reduce(x_test, (1, 5, 5, 1), func=np.mean)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = np_utils.to_categorical(y_train, classes)
        y_test = np_utils.to_categorical(y_test, classes)

        model = SELUResNet50NoBatchnorm(input_shape=[32, 32, 3], classes=10, include_top=True)
        print(model.summary())
        opt = Adam(lr=1e-3, decay=1e-6)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=opt,
                      metrics=self.metrics)

        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total basicnet_width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)

        datagen.fit(x_train)

        now = datetime.datetime.now()
        filepath = 'checkpoints/resnet50-nobatchnorm-selu-{epoch:02d}-%s.hdf5' % now
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False,
                                     save_weights_only=False, mode='max', period=100)

        log_dir = 'summaries/resnet50-nobatchnorm-selu-{}-{}'.format(self.seed, now)
        tensorboard = TensorBoard(log_dir=log_dir)

        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=self.epochs,
                            steps_per_epoch=int(x_train.shape[0] / batch_size),
                            verbose=1,
                            validation_data=(x_test, y_test),
                            callbacks=[checkpoint, tensorboard])
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

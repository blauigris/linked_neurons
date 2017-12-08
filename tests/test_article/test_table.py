import glob
import os
from unittest import TestCase

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError

from tests.util import network_for_cifar10_factory, load_cifar10, get_parameter_counts


class TestTable(TestCase):
    def test_width(self):
        resultdir = 'summaries/'
        table_outputdir = '../../docs/article/tables/'
        table_data = {}
        activations = ['relu', 'lkrelu', 'prelu', 'lkprelu', 'selu', 'lkselu', 'swish', 'lkswish',
                       ]
        colnames = {'relu': 'ReLU', 'lkrelu': 'LK-ReLU', 'prelu': 'PReLU', 'lkprelu': 'LK-PReLU',
                    'selu': 'SELU', 'lkselu': 'LK-SELU', 'swish': 'Swish', 'lkswish': 'LK-Swish'}

        tags = ['val_acc']
        widths = [50, 400]

        tag_data = {}
        for tag in tags:
            width_data = []
            for width in widths:
                activation_data = []

                for activation in activations:
                    template = os.path.join(resultdir, 'width-{}-{}*/events*'.format(activation, width))
                    filenames = glob.glob(template)
                    if not filenames:
                        raise RuntimeError('Files not found for {}'.format(template))

                    maxes = []
                    for filename in filenames:
                        maxes.append(get_best(filename, tag))

                    median = np.median(maxes)
                    activation_data.append(median)

                width_data.append(activation_data)

            tag_data[tag] = pd.DataFrame(width_data, index=widths,
                                         columns=[colnames[activation] for activation in activations])

        accuracy_table = tag_data['val_acc']
        accuracy_table.to_latex(os.path.join(table_outputdir, 'width.tex'))

    def test_depth50(self):
        resultdir = 'summaries/'
        table_outputdir = '../../docs/article/tables/'
        table_data = {}
        activations = ['relu', 'lkrelu', 'relu_sqrt2',
                       'prelu', 'lkprelu', 'prelu_sqrt2',
                       'selu', 'lkselu', 'selu_sqrt2',
                       'swish', 'lkswish', 'swish_sqrt2',
                       'batchnorm']
        colnames = {'relu': 'ReLU', 'lkrelu': 'LK-ReLU', 'relu_sqrt2': 'ReLU $\sqrt{2}$',
                    'prelu': 'PReLU', 'lkprelu': 'LK-PReLU', 'prelu_sqrt2': 'PReLU $\sqrt{2}$',
                    'selu': 'SELU', 'lkselu': 'LK-SELU', 'selu_sqrt2': 'SELU $\sqrt{2}$',
                    'swish': 'Swish', 'lkswish': 'LK-Swish', 'swish_sqrt2': 'Swish $\sqrt{2}$',
                    'batchnorm': 'BN'}
        tags = ['val_acc']
        for tag in tags:
            table_medians_tag = []
            table_max_tag = []

            for activation in activations:
                template = os.path.join(resultdir, '{}_depth*/events*'.format(activation))
                filenames = glob.glob(template)
                if not filenames:
                    raise RuntimeError('Files matching {} not found'.format(template))

                maxes = []
                for filename in filenames:
                    maxes.append(get_best(filename, tag))

                median = np.median(maxes)
                table_medians_tag.append(median)
                table_max_tag.append(max(maxes))

            table_tag = pd.DataFrame([table_max_tag, table_medians_tag],
                                     index=['Max', 'Median'],
                                     columns=[colnames[activation] for activation in activations])
            table_data[tag] = table_tag

        accuracy_table = table_data['val_acc']
        accuracy_table.T.to_latex(os.path.join(table_outputdir, 'depth50.tex'))

    def test_allcnn(self):
        resultdir = 'summaries/'
        table_outputdir = '../../docs/article/tables/'
        table_data = {}
        lrs = ['0.01', 'schedule-0.01', 'schedule-0.001']
        activations = ['relu', 'lkrelu', 'prelu', 'lkprelu', 'selu', 'lkselu', 'swish',
                       'lkswish',
                       ]
        colnames = {'relu': 'ReLU', 'lkrelu': 'LK-ReLU', 'prelu': 'PReLU', 'lkprelu': 'LK-PReLU',
                    'selu': 'SELU', 'lkselu': 'LK-SELU', 'swish': 'Swish', 'lkswish': 'LK-Swish',
                    }
        order = [colnames[activation] for activation in activations]

        tags = ['val_acc']
        for tag in tags:
            table_tag = []
            for lr in lrs:
                lr_data = []
                for activation in activations:
                    template = os.path.join(resultdir, 'allcnn_{}-cifar10-{}-*/event*'.format(activation, lr))
                    filenames = glob.glob(template)
                    if not filenames:
                        raise RuntimeError('Files matching {} not found'.format(template))

                    lr_data.append(get_best(filenames[0], tag))
                lr_series = pd.Series(lr_data, index=order)
                table_tag.append(lr_series)
            table_tag = pd.DataFrame(table_tag, index=lrs)

            table_data[tag] = table_tag

        accuracy_table = table_data['val_acc']
        pd.DataFrame(accuracy_table.max()).T.to_latex(os.path.join(table_outputdir, 'allcnn.tex'))

    def test_resnet50(self):
        resultdir = 'summaries/'
        table_outputdir = '../../docs/article/tables/'
        table_data = {}
        activations = ['relu', 'lkrelu', 'prelu', 'lkprelu', 'selu', 'lkselu', 'swish', 'lkswish',
                       ]
        colnames = {'relu': 'ReLU', 'lkrelu': 'LK-ReLU', 'prelu': 'PReLU', 'lkprelu': 'LK-PReLU',
                    'selu': 'SELU', 'lkselu': 'LK-SELU', 'swish': 'Swish', 'lkswish': 'LK-Swish',
                    }
        tags = ['val_acc']
        for tag in tags:
            table_tag = []

            for activation in activations:
                template = os.path.join(resultdir, 'resnet50-{}*/events*'.format(activation, tag))
                filenames = glob.glob(template)
                if not filenames:
                    raise RuntimeError('Files matching {} not found'.format(template))

                table_tag.append(get_best(filenames[0], tag))

            table_tag = pd.DataFrame([table_tag],
                                     index=['Best'],
                                     columns=[colnames[activation] for activation in activations])
            table_data[tag] = table_tag

        accuracy_table = table_data['val_acc']
        accuracy_table.to_latex(os.path.join(table_outputdir, 'resnet50.tex'))

    def test_resnet50_nobatchnorm(self):
        resultdir = 'summaries/'
        table_outputdir = '../../docs/article/tables/'
        activations = ['lkrelu', 'nobatchnorm-lkrelu',
                       'selu', 'nobatchnorm-selu',
                       'lkselu', 'nobatchnorm-lkselu'
                       ]
        colnames = {'lkrelu': 'LK-ReLU BN', 'nobatchnorm-lkrelu': 'LK-ReLU No BN',
                    'selu': 'SELU BN', 'nobatchnorm-selu': 'SELU No BN',
                    'lkselu': 'LK-SELU BN', 'nobatchnorm-lkselu': 'LK-SELU No BN'
                    }
        tags = ['val_acc']
        table_data = {}

        for tag in tags:
            table_tag_max = []
            table_tag_time = []
            template = os.path.join(resultdir, 'resnet50-nobatchnorm-lkrelu*/events*')
            filename = glob.glob(template)[0]
            _, relu_time = get_best_and_time(filename, tag)
            for activation in activations:
                template = os.path.join(resultdir, 'resnet50-{}*/events*'.format(activation))
                try:
                    filename = glob.glob(template)[0]
                except IndexError:
                    raise RuntimeError('File matching {} not found'.format(template))
                max_val, time = get_best_and_time(filename, tag)
                table_tag_max.append(max_val)
                table_tag_time.append(time / relu_time)

            table_tag = pd.DataFrame([table_tag_max, table_tag_time],
                                     index=['Accuracy', 'Time ratio'],
                                     columns=[colnames[activation] for activation in activations])
            table_data[tag] = table_tag

        accuracy_table = table_data['val_acc']
        accuracy_table.to_latex(os.path.join(table_outputdir, 'resnet50-nobatchnorm.tex'))

    def test_extended(self):
        resultdir = 'summaries/'
        table_outputdir = '../../docs/article/tables/'
        table_data = {}
        activations = ['relu', 'lkrelu', 'relu-sqrt2',
                       'prelu', 'lkprelu', 'prelu-sqrt2',
                       'selu', 'lkselu', 'selu-sqrt2',
                       'swish', 'lkswish', 'swish-sqrt2'
                       ]
        colnames = {'relu': 'ReLU', 'lkrelu': 'LK-ReLU', 'relu-sqrt2': 'ReLU sqrt2',
                    'prelu': 'PReLU', 'lkprelu': 'LK-PReLU', 'prelu-sqrt2': 'PReLU sqrt2',
                    'selu': 'SELU', 'lkselu': 'LK-SELU', 'selu-sqrt2': 'SELU sqrt2',
                    'swish': 'Swish', 'lkswish': 'LK-Swish', 'swish-sqrt2': 'Swish sqrt2'
                    }

        num_classes = 10
        (x_train, y_train), (x_test, y_test) = load_cifar10()
        networks = {'relu': network_for_cifar10_factory('relu', x_train.shape[1:], num_classes),
                    'prelu': network_for_cifar10_factory('prelu', x_train.shape[1:], num_classes),
                    'selu': network_for_cifar10_factory('selu', x_train.shape[1:], num_classes),
                    'swish': network_for_cifar10_factory('swish', x_train.shape[1:], num_classes),
                    'lkrelu': network_for_cifar10_factory('lkrelu', x_train.shape[1:], num_classes),
                    'lkprelu': network_for_cifar10_factory('lkprelu', x_train.shape[1:], num_classes),
                    'lkselu': network_for_cifar10_factory('lkselu', x_train.shape[1:], num_classes),
                    'lkswish': network_for_cifar10_factory('lkswish', x_train.shape[1:], num_classes),
                    'relu-sqrt2': network_for_cifar10_factory('relu', x_train.shape[1:], num_classes,
                                                              width=np.sqrt(2)),
                    'prelu-sqrt2': network_for_cifar10_factory('prelu', x_train.shape[1:], num_classes,
                                                               width=np.sqrt(2)),
                    'selu-sqrt2': network_for_cifar10_factory('selu', x_train.shape[1:], num_classes, width=np.sqrt(2)),
                    'swish-sqrt2': network_for_cifar10_factory('swish', x_train.shape[1:], num_classes,
                                                               width=np.sqrt(2)),
                    }

        parameters = pd.DataFrame([get_parameter_counts(networks[activation])['total'] for activation in activations],
                                  index=[colnames[activation] for activation in activations])
        tags = ['val_acc']
        for tag in tags:
            table_tag = []

            for activation in activations:
                template = os.path.join(resultdir, '{}-extended*/events*'.format(activation, tag))
                filenames = glob.glob(template)
                if not filenames:
                    raise RuntimeError('Files not found for {}'.format(template))

                maxes = []
                for filename in filenames:
                    maxes.append(get_best(filename, tag))

                median = np.median(maxes)
                table_tag.append(median)

            table_tag = pd.DataFrame([table_tag],
                                     index=['Best'],
                                     columns=[colnames[activation] for activation in activations])
            table_data[tag] = table_tag

        accuracy_table = table_data['val_acc'].T

        singles = accuracy_table.loc[['ReLU', 'PReLU', 'SELU', 'Swish']].values
        lks = accuracy_table.loc[['LK-ReLU', 'LK-PReLU', 'LK-SELU', 'LK-Swish']].values
        sqrt2 = accuracy_table.loc[['ReLU sqrt2', 'PReLU sqrt2', 'SELU sqrt2', 'Swish sqrt2']].values

        table = np.concatenate([singles, sqrt2, lks], axis=1)
        table = pd.DataFrame(table, index=['ReLU', 'PReLU', 'SELU', 'Swish'],
                             columns=['Single', '$sqrt{2}$', 'LK-inked'])
        table.to_latex(os.path.join(table_outputdir, 'extended.tex'))

        param_singles = parameters.loc[['ReLU', 'PReLU', 'SELU', 'Swish']].values
        param_lks = parameters.loc[['LK-ReLU', 'LK-PReLU', 'LK-SELU', 'LK-Swish']].values
        param_sqrt2 = parameters.loc[['ReLU sqrt2', 'PReLU sqrt2', 'SELU sqrt2', 'Swish sqrt2']].values

        param_table = np.concatenate([param_singles, param_sqrt2, param_lks], axis=1)
        param_table = pd.DataFrame(param_table, index=['ReLU', 'PReLU', 'SELU', 'Swish'],
                                   columns=['Single', '$\sqrt{2}$', 'Linked'])
        param_table.to_latex(os.path.join(table_outputdir, 'param_extended.tex'))


def get_best(filename, tag):
    best = float('-inf')
    try:
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for e in tf.train.summary_iterator(filename):
                for v in e.summary.value:
                    if v.tag == tag:
                        best = max(best, v.simple_value)
    except DataLossError as ex:
        print('Exception when file: {}, {}'.format(filename, ex))

    return best

def get_best_and_time(filename, tag):
    best = float('-inf')
    try:
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            start = next(tf.train.summary_iterator(filename))

            for e in tf.train.summary_iterator(filename):
                for v in e.summary.value:
                    if v.tag == tag:
                        best = max(best, v.simple_value)

            time = e.wall_time - start.wall_time
    except DataLossError as ex:
        print('Exception when file: {}, {}'.format(filename, ex))

    return best, time

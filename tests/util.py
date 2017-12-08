import keras
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras import backend as K
from keras.activations import selu
from keras.applications import mobilenet
from keras.datasets import mnist, cifar10, cifar100
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, \
    LeakyReLU, PReLU, ELU
from keras.models import Sequential
from matplotlib import ticker
from sklearn.decomposition import PCA

from linked_neurons.linked_neurons import LKReLU, LKPReLU, LKSELU, LKSwish


class SolutionPlot:
    def __init__(self, decision_function, *, coef=None, basis_functions=None, fig=None, ax=None, scale=1,
                 title=None, show_ticks=True, resolution=100, show_colorbar=True, margin=0):
        self.margin = margin
        self.show_colorbar = show_colorbar
        self.scale = scale
        self.resolution = resolution
        self.ax = ax
        self.fig = fig
        self.show_ticks = show_ticks
        self.coef = coef
        self.basis_functions = basis_functions
        self.decision_function = decision_function
        self.title = title
        self.pca = None

    def draw(self, X, y):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(1)

        self._prepare_plot()

        if X.shape[1] > 2:
            self.pca = PCA(n_components=2)
            if X.ndim > 2:
                self.pca.fit(X.reshape(X.shape[0], -1))
            else:
                self.pca.fit(X)

        self.draw_contourf(X, y)
        self.draw_prediction(X, y)

    def draw_contourf(self, X, y):

        num = self.resolution // X.ndim
        min_ = X.min() * self.scale - self.margin
        max_ = X.max() * self.scale + self.margin
        xxx = [np.linspace(min_, max_, num=num) for _ in range(X.ndim)]
        XXX = np.meshgrid(*xxx)

        img = np.c_[[xxx.ravel() for xxx in XXX]].T
        if X.shape[1] > 2:
            XX, YY = self.pca.inverse_transform(img)
        else:
            XX, YY = XXX

        Z = self.decision_function(img)
        if X.shape[1] > 2:
            if Z.ndim > 2:
                Z = self.pca.transform(Z.reshape(Z.shape[0], -1))
            else:
                Z = self.pca.transform(Z)
        else:
            Z = Z.reshape(XXX[0].shape)

        sns.set_palette('coolwarm')
        CS = self.ax.contourf(XX, YY, Z,
                              # vmin=-np.max(np.abs(Z)), vmax=np.max(np.abs(Z)),
                              cmap=cm.coolwarm,
                              alpha=0.8)

        def fmt(x, pos):
            return '{:.2E}'.format(x)

        if self.show_colorbar:
            self.fig.colorbar(CS, ax=self.ax, alpha=0.7, format=ticker.FuncFormatter(fmt))

        self.ax.contour(XX, YY, Z, [-1, 0, 1], linewidth=(5, 10, 5),
                        colors=('blue', 'black', 'red'))

    def draw_prediction(self, X, y):

        self.ax.scatter(X[:, 0], X[:, 1], c=y, vmin=-np.max(np.abs(y)), vmax=np.max(np.abs(y)),
                        cmap=cm.coolwarm, linewidths=1, edgecolors='black')
        if self.basis_functions is not None:
            try:
                self.ax.scatter(self.basis_functions[:, 0], self.basis_functions[:, 1], c=self.coef,
                                s=120, alpha=0.6, cmap=cm.coolwarm)
            except ValueError:
                pass

    def _prepare_plot(self):
        self.ax.set_title(self.title)
        if self.show_ticks:
            self.ax.set_xlabel('$x_1$')
            self.ax.set_ylabel('$x_2$')
            self.ax.spines["top"].set_visible(False)
            self.ax.spines["right"].set_visible(False)
            self.ax.spines['bottom'].set_color('gray')
            self.ax.spines['left'].set_color('gray')
            self.ax.get_xaxis().tick_bottom()
            self.ax.get_yaxis().tick_left()
            self.ax.set_xlabel('$x_1$')
            self.ax.set_ylabel('$x_2$')
        else:
            self.ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom='off',  # ticks along the bottom edge are off
                top='off',  # ticks along the top edge are off
                left='off',
                right='off',
                labelleft='off',
                labelbottom='off')  # labels along the bottom edge are off

    def save(self, X, y, filepath):
        self.draw(X, y)
        self.fig.savefig(filepath)

    def show(self, X, y):
        self.draw(X, y)
        plt.show()


def get_activation(activation):
    """
    returns activation object from string
    Leaky, Parametric, RELU6, ELU i SELU, doublerelu
    :param activation:
    :return:
    """
    if activation == 'relu':
        return Activation('relu')
    elif activation == 'leaky':
        return LeakyReLU()
    elif activation == 'prelu' or activation == 'parametric':
        return PReLU()
    elif activation == 'relu6':
        return Activation(mobilenet.relu6)
    elif activation == 'elu':
        return ELU()
    elif activation == 'selu':
        return Activation(selu)
    elif activation == 'swish':
        return Activation(swish)
    elif activation == 'lkrelu':
        return LKReLU()
    elif activation == 'lkprelu':
        return LKPReLU()
    elif activation == 'lkselu':
        return LKSELU()
    elif activation == 'lkswish':
        return LKSwish()
    else:
        raise ValueError('Invalid activation')


def network_for_mnist_factory(activation, input_shape, num_classes, width=1, kernel_initializer='glorot_normal',
                              bias_initializer='zeros', last_activation='softmax'):
    model = Sequential()
    model.add(Conv2D(int(32 * width), kernel_size=(3, 3),
                     activation=None,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     input_shape=input_shape))
    model.add(get_activation(activation))
    model.add(Conv2D(int(64 * width), (3, 3), activation=None, kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer))
    model.add(get_activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(int(128 * width), activation=None, kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer))
    model.add(get_activation(activation))

    model.add(Dense(num_classes, activation=last_activation))
    return model


def network_for_dying_mnist_factory(activation, input_shape, num_classes, last_activation=None):
    model = Sequential()
    model.add(Conv2D(11, kernel_size=(3, 3),
                     activation=None,
                     input_shape=input_shape,
                     ))
    model.add(get_activation(activation))
    model.add(Conv2D(11, kernel_size=(3, 3),
                     activation=None,
                     input_shape=input_shape,
                     ))
    model.add(get_activation(activation))
    model.add(Flatten())
    model.add(Dense(num_classes, activation=last_activation))
    return model


def network_for_cifar10_factory(activation, input_shape, num_classes, width=1, kernel_initializer='glorot_normal',
                                bias_initializer='zeros'):
    model = Sequential()

    model.add(Conv2D(int(32 * width), (3, 3), padding='same', kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     input_shape=input_shape))
    model.add(get_activation(activation))
    model.add(Conv2D(int(32 * width), (3, 3), kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer, ))
    model.add(get_activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(int(64 * width), (3, 3), padding='same', kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer, ))
    model.add(get_activation(activation))
    model.add(Conv2D(int(64 * width), (3, 3), kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer, ))
    model.add(get_activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(int(512 * width), kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer, ))
    model.add(get_activation(activation))
    model.add(Dense(num_classes, kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer, ))
    model.add(Activation('softmax'))
    return model


def load_mnist():
    # input image dimensions
    img_rows, img_cols = 28, 28
    num_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train -= 0.5
    x_test -= 0.5
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


def load_cifar10():
    num_classes = 10
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train), (x_test, y_test)


def load_cifar100():
    num_classes = 100
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train), (x_test, y_test)


def swish(x):
    return x * K.sigmoid(x)


def get_parameter_counts(model, line_length=None, positions=None):
    """Returns the number of parameters of keras model

    # Arguments
        model: Keras model instance.
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements in each line.
            If not provided, defaults to `[.33, .55, .67, 1.]`.
        print_fn: Print function to use.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.
    """
    if model.__class__.__name__ == 'Sequential':
        sequential_like = True
    else:
        sequential_like = True
        for v in model.nodes_by_depth.values():
            if (len(v) > 1) or (len(v) == 1 and len(v[0].inbound_layers) > 1):
                # if the model has multiple nodes or if the nodes have multiple inbound_layers
                # the model is no longer sequential
                sequential_like = False
                break

    if sequential_like:
        line_length = line_length or 65
        positions = positions or [.45, .85, 1.]
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        # header names for the different log elements
        to_display = ['Layer (type)', 'Output Shape', 'Param #']
    else:
        line_length = line_length or 100
        positions = positions or [.33, .55, .67, 1.]
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        # header names for the different log elements
        to_display = ['Layer (type)', 'Output Shape', 'Param #', 'Connected to']
        relevant_nodes = []
        for v in model.nodes_by_depth.values():
            relevant_nodes += v

    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    return {'total': trainable_count + non_trainable_count, 'trainable': trainable_count,
            'non_trainable': non_trainable_count}

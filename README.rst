==============
Linked Neurons
==============


.. image:: https://img.shields.io/pypi/v/linked_neurons.svg
        :target: https://pypi.python.org/pypi/linked_neurons

.. image:: https://img.shields.io/travis/blauigris/linked_neurons.svg
        :target: https://travis-ci.org/blauigris/linked_neurons

.. image:: https://readthedocs.org/projects/linked-neurons/badge/?version=latest
        :target: https://linked-neurons.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/blauigris/linked_neurons/shield.svg
     :target: https://pyup.io/repos/github/blauigris/linked_neurons/
     :alt: Updates


Keras implementation of the article "Solving internal covariate shift in deep learning with linked neurons"

Linked neurons is a deep learning framework where two or more activations are coupled in order to have gradient in
all input space. This improves covariate shift robustness and enables to forfeit other techniques such Batch Normalization
which incur in additional computational cost. More info in the arxiv paper HERE.


* Free software: MIT license
* Documentation: https://linked-neurons.readthedocs.io.


Features
--------

* 200% faster than Batchnorm.
* 40% faster than SELU.
* Use your favorite activation function without worrying about covariate shift.

QuickStart
----------

Install using pip::

    pip install linked_neurons

Import Linked Neurons in a project::

    import linked_neurons

Inside there are the implementations as a keras layer.Layer subclasses.
For the moment being there are LK-ReLU, LK-PReLU, LK-SELU and LK-Swish available.


Example using Sequential::

    model = Sequential()
    (x_train, y_train), (x_test, y_test) = load_mnist()
    model.add(Conv2D(32, kernel_size=(3, 3), activation=None, input_shape=x_train.shape[1:]))
    model.add(LKReLU())
    model.add(Conv2D(64, (3, 3), activation=None))
    model.add(LKReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=None))
    model.add(LKReLU())
    model.add(Dense(10, activation='softmax'))

Example using functional api::

    (x_train, y_train), (x_test, y_test) = load_mnist()
    img_input = Input(shape=x_train.shape[1:])
    x = Conv2D(width, (7, 7), strides=(2, 2), padding='same', name='convfirst')(img_input)
    x = LKReLU()(x)
    x = Conv2D(width, (3, 3), padding='same')(x)
    x = LKReLU()(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc')(x)
    model = Model(inputs, x)

For more examples see the tests or keras documentation.


Replicating results
===================

Reproducibility is of paramount importance in science, so we provide all the code needed to replicate our results.

Running the experiments
-----------------------

In order to replicate the results displayed in the paper and the paper itself first download the entire repo so
the scripts used to generate them are also included::

    git clone git@github.com:blauigris/linked_neurons.git

Each of the experiments are implemented separately using nosetests for ease of use. They are located inside tests/test_article
In order to run them, just call::

    nosetests test_<experiment>

For instance, for the entire depth experiment one may use::

    nosetests test_depth.py

or if only one activation is required::

    nosetests test_depth:TestDepth.test_lkrelu

It will store the results in form of tensorboard summaries at tests/test_article/summaries and the checkpoints
in tests/test_article/checkpoints in some cases.

In case one might want to play with the hyperparameters, they are set into the setUp method of each test.

The experiments are quite fast to run especially the ones from the sections of covariate shift, depth and width.
Allcnn* and resnet50 take a day using a GTX 1080Ti.


Generating the tables
---------------------

Once run, the scripts
for creating the tables used in the paper are in tests/test_article/test_table. Each of the tests generates a table.
For instance, in order to generate the table of the depth experiment::

    nosetests test_table:TestTable.test_depth

This will create a .tex file containing the table at docs/article/tables. This table is included into paper.tex, so
recompiling it should update the paper with the new results.


Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage


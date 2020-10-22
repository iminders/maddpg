# -*- coding:utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
from tensorflow.keras import layers, models


def mpl(num_units, input_size, output_size, dropout=None):
    model = models.Sequential()
    model.add(layers.Dense(num_units, activation='relu',
                           input_shape=(input_size,)))
    if dropout is not None:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_units, activation='relu'))
    if dropout is not None:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(output_size, activation='relu'))
    return model


if __name__ == '__main__':
    print("tensorflow version:", tf.__version__)
    num_units, input_size, output_size = 64, 10, 2
    m = mpl(num_units, input_size, output_size)
    m.summary()

    num_units, input_size, output_size, dropout = 64, 10, 2, 0.1
    m = mpl(num_units, input_size, output_size, dropout)
    m.summary()

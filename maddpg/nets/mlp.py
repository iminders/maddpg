# -*- coding:utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.regularizers import L2


class MLP(tf.keras.Model):
    def __init__(self, num_units, input_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = layers.Dense(
            num_units, activation='relu', input_shape=(input_size,),
            kernel_regularizer=L2(l2=0.01),
            kernel_constraint=MinMaxNorm(min_value=-5.0, max_value=5.0)
            )
        self.layer2 = layers.Dense(
            num_units, activation='relu', input_shape=(num_units,),
            kernel_regularizer=L2(l2=0.01),
            kernel_constraint=MinMaxNorm(min_value=-5.0, max_value=5.0)
            )
        self.layer3 = layers.Dense(
            output_size, input_shape=(num_units,),
            kernel_regularizer=L2(l2=0.01)
            )

    def call(self, inputs):
        o1 = self.layer1(inputs)
        o2 = self.layer2(o1)
        return self.layer3(o2)


if __name__ == '__main__':
    import numpy as np
    print("tensorflow version:", tf.__version__)
    num_units, input_size, output_size = 64, 10, 2
    m = MLP(num_units, input_size, output_size)
    x = np.random.random((2, 10))
    y = m(x)
    print(y)
    m.summary()

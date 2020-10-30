import tensorflow as tf


def sum(x, axis=None, keepdims=False):
    return tf.reduce_sum(
        x, axis=None if axis is None else [axis], keep_dims=keepdims)


def mean(x, axis=None, keepdims=False):
    return tf.reduce_mean(
        x, axis=None if axis is None else [axis], keep_dims=keepdims)


def var(x, axis=None, keepdims=False):
    meanx = mean(x, axis=axis, keepdims=keepdims)
    return mean(tf.square(x - meanx), axis=axis, keepdims=keepdims)


def std(x, axis=None, keepdims=False):
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))


def max(x, axis=None, keepdims=False):
    return tf.reduce_max(
        x, axis=None if axis is None else [axis], keep_dims=keepdims)


def min(x, axis=None, keepdims=False):
    return tf.reduce_min(
        x, axis=None if axis is None else [axis], keep_dims=keepdims)


def concatenate(arrs, axis=0):
    return tf.concat(axis=axis, values=arrs)


def argmax(x, axis=None):
    return tf.argmax(x, axis=axis)


def softmax(x, axis=None):
    return tf.nn.softmax(x, axis=axis)

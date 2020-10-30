import tensorflow as tf

from maddpg.distributions.tf_utils import max, softmax, sum


class Pd(object):
    """
    A particular probability distribution
    """

    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def logp(self, x):
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class SoftCategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.nn.softmax(self.logits, axis=-1)

    def logp(self, x):
        return -tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=x)

    def kl(self, other):
        a0 = self.logits - max(self.logits, axis=1, keepdims=True)
        a1 = other.logits - max(other.logits, axis=1, keepdims=True)
        ea0 = tf.math.exp(a0)
        ea1 = tf.math.exp(a1)
        z0 = sum(ea0, axis=1, keepdims=True)
        z1 = sum(ea1, axis=1, keepdims=True)
        p0 = ea0 / z0
        return sum(p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=1)

    def entropy(self):
        a0 = self.logits - max(self.logits, axis=1, keepdims=True)
        ea0 = tf.math.exp(a0)
        z0 = sum(ea0, axis=1, keepdims=True)
        p0 = ea0 / z0
        return sum(p0 * (tf.math.log(z0) - a0), axis=1)

    def sample(self):
        u = tf.random.uniform(tf.shape(self.logits))
        return softmax(self.logits - tf.math.log(-tf.math.log(u)), axis=-1)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

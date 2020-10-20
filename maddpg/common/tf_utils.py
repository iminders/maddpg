import random

import numpy as np
import tensorflow as tf


def set_global_seeds(seed):
    if seed is None:
        seed = np.random.randint(int(1e6))
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

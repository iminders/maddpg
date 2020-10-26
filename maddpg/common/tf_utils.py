import random

import numpy as np
import tensorflow as tf


def set_global_seeds(seed):
    if seed is None:
        seed = np.random.randint(int(1e6))
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def update_target_variables(target_variables,
                            source_variables,
                            tau=1.0,
                            use_locking=False,
                            name="update_target_variables"):

    def update_op(target_variable, source_variable, tau):
        if tau == 1.0:
            return target_variable.assign(source_variable, use_locking)
        else:
            return target_variable.assign(
                tau * source_variable + (1.0 - tau) * target_variable,
                use_locking)

    # with tf.name_scope(name, values=target_variables + source_variables):
    update_ops = [update_op(target_var, source_var, tau)
                  for target_var, source_var
                  in zip(target_variables, source_variables)]
    return tf.group(name="update_all_variables", *update_ops)

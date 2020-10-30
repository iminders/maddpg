# -*- coding:utf-8 -*-

from maddpg.common.logger import logger
from maddpg.nets.mlp import MLP


def get_critic_model(id, args, act_shapes, obs_shapes):
    logger.info("create critic nets for agent: %d" % id)
    input_size = sum(obs_shapes) + act_shapes[id]
    output_size = 1
    model = MLP(args.num_units, input_size, output_size)
    if args.print_net:
        model.summary()
    return model


if __name__ == '__main__':
    from maddpg.arguments import parse_experiment_args
    args = parse_experiment_args()
    m = get_critic_model(1, args, [2, 2, 2], [4, 4, 4])

    m = get_critic_model(0, args, [5, 2, 2], [4, 4, 4])

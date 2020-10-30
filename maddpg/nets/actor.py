# -*- coding:utf-8 -*-

from maddpg.common.logger import logger
from maddpg.nets.mlp import MLP


def get_actor_model(id, args, act_shapes, obs_shapes):
    logger.info("create actor nets for agent: %d" % id)
    input_size = sum(obs_shapes)
    output_size = act_shapes[id]
    model = MLP(args.num_units, input_size, output_size)
    if args.print_net:
        model.summary()
    return model


if __name__ == '__main__':
    from maddpg.arguments import parse_experiment_args
    args = parse_experiment_args()
    m = get_actor_model(1, args, [2, 2, 2], [4, 4, 4])

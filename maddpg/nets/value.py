# -*- coding:utf-8 -*-

from maddpg.common.logger import logger
from maddpg.nets.mpl import mpl


def get_value(id, args, act_shapes, obs_shapes):
    logger.info("create value nets for agent: %d" % id)
    input_size = sum(obs_shapes) + act_shapes[id]
    output_size = 1
    model = mpl(args.num_units, input_size, output_size, dropout=args.dropout)
    if args.print_net:
        model.summary()


if __name__ == '__main__':
    from maddpg.arguments import parse_experiment_args
    args = parse_experiment_args()
    m = get_value(1, args, [2, 2, 2], [4, 4, 4])

    m = get_value(0, args, [5, 2, 2], [4, 4, 4])

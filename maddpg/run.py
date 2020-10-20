# -*- coding:utf-8 -*-

from maddpg.arguments import parse_experiment_args
from maddpg.common.const import EXPLORER, LEARNER
from maddpg.common.explore_client import parallel_explore
from maddpg.common.logger import logger
from maddpg.common.tf_utils import set_global_seeds

if __name__ == '__main__':
    args = parse_experiment_args()
    if args.debug:
        import logging
        logger.setLevel(logging.DEBUG)

    if args.role == EXPLORER:
        parallel_explore(args)

    if args.role == LEARNER:
        logger.info("set global_seeds: %s" % str(args.seed))
        set_global_seeds(args.seed)

# -*- coding:utf-8 -*-

import gc

from maddpg.arguments import parse_experiment_args
from maddpg.common.const import EXPLORER, LEARNER
from maddpg.common.env_utils import make_env
from maddpg.common.logger import logger
from maddpg.common.tf_utils import set_global_seeds
from maddpg.explorer import parallel_explore
from maddpg.learner import make_learner_agent

if __name__ == '__main__':
    args = parse_experiment_args()
    if args.debug:
        import logging
        logger.setLevel(logging.DEBUG)

    if args.role == EXPLORER:
        parallel_explore(args)

    if args.role == LEARNER:
        logger.info("parameters start" + "*" * 100)
        logger.info(str(args))
        logger.info("parameters end  " + "*" * 100)

        logger.info("set global_seeds: %s" % str(args.seed))
        set_global_seeds(args.seed)
        env = make_env(args=args, id=0)
        agent = make_learner_agent(args, env.n, env.action_space,
                                   env.observation_space)
        env = None
        gc.collect()

        agent.serve()
        logger.info("Finished, tensorboard --logdir=%s" % agent.tb_dir)

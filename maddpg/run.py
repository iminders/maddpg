# -*- coding:utf-8 -*-

from maddpg.arguments import parse_experiment_args
from maddpg.common.const import EXPLORER, LEARNER
from maddpg.common.env_utils import get_act_shapes, get_obs_shapes, make_env
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
        act_shapes = get_act_shapes(env)
        obs_shapes = get_obs_shapes(env)
        agent = make_learner_agent(args, act_shapes, obs_shapes)
        agent.serve()
        logger.info("Finished, tensorboard --logdir=%s" % args.tensorboard_dir)

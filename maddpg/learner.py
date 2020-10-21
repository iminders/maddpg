from maddpg.common.logger import logger


def make_learner_agent(args=None, act_shapes=[2, 2, 2], obs_shapes=[4, 4, 4]):
    logger.info("act_shapes:" + str(act_shapes))
    logger.info("obs_shapes:" + str(obs_shapes))
    return None

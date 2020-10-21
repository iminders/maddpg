# -*- coding:utf-8 -*-

import time

import numpy as np

from maddpg.agents.base.agent import BaseAgent
from maddpg.common.logger import logger


class Agent(BaseAgent):
    def __init__(self, args, agent_num, act_spaces, obs_spaces):
        super(Agent, self).__init__(args, agent_num, act_spaces, obs_spaces)
        logger.info("policys:act_shapes:%s, obs_shapes:%s" %
                    (str(self.act_shapes), str(self.obs_shapes)))
        self.policys = []
        logger.info("values:act_shapes:%s, obs_shapes:%s" %
                    (str(self.act_shapes), str(self.obs_shapes)))
        self.values = []
        self.target_policys = []
        self.target_values = []

    def add_graph(self):
        return None

    def action(self, obs):
        # TODO:
        return self.random_action()

    def update_params(self, obs, act, rew, obs_next, done):
        # TODO
        time.sleep(1)
        [q_value, p_loss, q_loss, p_reg,
            act_reg] = np.random.random(5).tolist()
        return q_value, p_loss, q_loss, p_reg, act_reg

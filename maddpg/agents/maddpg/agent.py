# -*- coding:utf-8 -*-

import time

from maddpg.agents.base.agent import BaseAgent
from maddpg.common.logger import logger


class Agent(BaseAgent):
    def __init__(self, args, agent_num, act_spaces, obs_spaces):
        self.policys = []
        self.values = []
        self.target_policys = []
        self.target_values = []

    def action(self, obs):
        # TODO:
        return self.random_action()

    def learn(self, iter=0):
        logger.info("iter: %d" % iter)
        time.sleep(1)

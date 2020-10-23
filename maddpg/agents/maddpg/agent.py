# -*- coding:utf-8 -*-

import time

import numpy as np

from maddpg.agents.base.agent import BaseAgent
from maddpg.common.logger import logger
from maddpg.distributions.util import get_distribution
from maddpg.nets.policy import get_policy_model
from maddpg.nets.value import get_value_model


class Agent(BaseAgent):
    def __init__(self, args, agent_num, act_spaces, obs_spaces):
        super(Agent, self).__init__(args, agent_num, act_spaces, obs_spaces)
        logger.info("policys:act_shapes:%s, obs_shapes:%s" %
                    (str(self.act_shapes), str(self.obs_shapes)))
        self.policys = self.create_policys()
        self.target_policys = self.create_policys()
        logger.info("values:act_shapes:%s, obs_shapes:%s" %
                    (str(self.act_shapes), str(self.obs_shapes)))
        self.values = self.create_values()
        self.target_values = self.create_values()
        self.noise_pds = [get_distribution(
            self.act_spaces[i], args.noise_pd) for i in range(self.n)]

    def create_policys(self):
        policys = []
        for i in range(self.n):
            m = get_policy_model(
                i, self.args, self.act_shapes, self.obs_shapes)
            policys.append(m)
        return policys

    def create_values(self):
        values = []
        for i in range(self.n):
            m = get_value_model(i, self.args, self.act_shapes, self.obs_shapes)
            values.append(m)
        return values

    def add_graph(self):
        return None

    def action(self, obs):
        # TODO(liuwen): 合并运行，加快inference速度
        acts = [self.policys[i].predict(obs)[0] for i in range(self.n)]
        return [self.noise_pds[i].sample() + acts[i] for i in range(self.n)]

    def update_params(self, obs, act, rew, obs_next, done):
        start = time.time()

        # TODO: remove
        [q_value, p_loss, q_loss, p_reg,
            act_reg] = np.random.random(5).tolist()
        update_time = time.time() - start
        logger.debug("update_params use %.3 seconds" % update_time)
        return q_value, p_loss, q_loss, p_reg, act_reg

    def learn(self, iter=0):
        logger.debug("ddpg agent learn iter: %d" % iter)
        obs, act, rew, obs_t, done = self.memory.sample(self.args.batch_size)
        q_value, p_loss, q_loss, p_reg, act_reg, u_t = self.update_params(
            obs, act, rew, obs_t, done)
        avg_rew = np.mean(rew)
        if iter <= 1:
            return
        self.writer.add_scalar(
            '1.performance/3.sample_avg_reward', avg_rew, iter)
        self.writer.add_scalar('1.performance/4.q_value', q_value, iter)
        self.writer.add_scalar('2.train/p_loss', p_loss, iter)
        self.writer.add_scalar('2.train/q_loss', q_loss, iter)
        self.writer.add_scalar('2.train/reg_action', act_reg, iter)
        self.writer.add_scalar('2.train/reg_policy', p_reg, iter)
        self.writer.add_scalar('3.time/1.update', u_t, iter)

        if iter % 100 == 0:
            clear_memory()

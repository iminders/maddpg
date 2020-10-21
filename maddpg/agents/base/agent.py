# -*- coding:utf-8 -*-

import gc
import os
import time

import numpy as np
from tensorflow.summary import SummaryWriter

from maddpg.common.env_utils import get_shapes, uniform_action
from maddpg.common.logger import logger
from maddpg.common.storage import Storage


class BaseAgent:
    def __init__(self, args, agent_num, act_spaces, obs_spaces):
        self.args = args
        self.act_spaces = act_spaces
        self.obs_spaces = obs_spaces
        self.act_shapes = get_shapes(act_spaces)
        self.obs_shapes = get_shapes(obs_spaces)
        logger.info("init agent, act_shapes=%s, obs_shapes=%s" (
            self.act_shapes, self.obs_shapes))
        self.n = agent_num
        # 初始化目录
        self.tb_dir = self.must_get_dir(os.path.join(
            args.tb_dir, args.runner, args.run_id))
        self.model_dir = self.must_get_dir(os.path.join(
            args.model_dir, args.runner, args.run_id))
        self.writer = SummaryWriter(self.tb_dir)
        # s3云存储
        self.stoarge = Storage(args)

    def random_action(self):
        return uniform_action(self.act_spaces)

    def action(self):
        raise NotImplementedError

    def update_params(self, obs, act, rew, obs_t, done):
        raise NotImplementedError

    def learn(self, iter=0):
        start = time.time()
        logger.info("iter: %d" % iter)
        obs, act, rew, obs_t, done = self.memory.sample(self.args.batch_size)
        q_value, p_loss, q_loss, p_reg, act_reg = self.update_params()
        update_time = time.time() - start
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
        self.writer.add_scalar('3.time/2.update', update_time, iter)
        if iter % 100 == 0:
            gc.collect()

    def must_get_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

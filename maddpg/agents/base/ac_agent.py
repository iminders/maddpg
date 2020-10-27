# -*- coding:utf-8 -*-

import gc
import os
import time

import numpy as np
import tensorflow as tf

from maddpg.common.env_utils import get_shapes, uniform_action
from maddpg.common.logger import logger
from maddpg.common.replay_buffer import ReplayBuffer
from maddpg.common.storage import Storage


class ACAgent:
    def __init__(self, args, agent_num, act_spaces, obs_spaces):
        self.args = args
        self.act_spaces = act_spaces
        self.obs_spaces = obs_spaces
        self.act_shapes = get_shapes(act_spaces)
        self.obs_shapes = get_shapes(obs_spaces)
        self.n = agent_num
        self.act_starts = []
        self.act_ends = []
        start, end = 0, 0
        for i in range(self.n):
            self.act_starts.append(start)
            end = start + self.act_shapes[i]
            self.act_ends.append(end)
            start = end

        logger.info("act_starts:" + str(self.act_starts))
        logger.info("act_ends:" + str(self.act_ends))
        # 初始化目录
        self.tb_dir = self.must_get_dir(os.path.join(
            args.tb_dir, args.runner, args.run_id))
        self.model_dir = self.must_get_dir(os.path.join(
            args.model_dir, args.runner, args.run_id))
        self.writer = tf.summary.create_file_writer(self.tb_dir)
        # s3云存储
        self.stoarge = Storage(args)
        self.buffer = ReplayBuffer(1e6)

    def random_action(self):
        return uniform_action(self.act_spaces)

    def action(self):
        raise NotImplementedError

    def update_params(self, obs, act, rew, obs_next, done):
        raise NotImplementedError

    def learn(self, iter=0):
        start = time.time()
        obs, act, rew, obs_next, done = self.buffer.sample(
            self.args.batch_size)
        actor_loss, critic_loss, action_reg = \
            self.update_params(obs, act, rew, obs_next, done)
        update_time = time.time() - start
        avg_rew = np.mean(rew)
        if iter <= 1:
            return
        tf.summary.scalar('1.performance/2.sample_reward', avg_rew, iter)
        tf.summary.scalar('2.train/actor_loss', actor_loss, iter)
        tf.summary.scalar('2.train/critic_loss', critic_loss, iter)
        tf.summary.scalar('2.train/action_reg', action_reg, iter)
        tf.summary.scalar('3.time/2.update', update_time, iter)
        if iter % 1000 == 0:
            gc.collect()

    def must_get_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

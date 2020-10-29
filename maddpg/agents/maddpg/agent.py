# -*- coding:utf-8 -*-

import time

import numpy as np
import tensorflow as tf

from maddpg.agents.base.ac_agent import ACAgent
from maddpg.common.logger import logger
from maddpg.common.tf_utils import update_target_variables
from maddpg.nets.actor import get_actor_model
from maddpg.nets.critic import get_critic_model


class Agent(ACAgent):
    def __init__(self, args, agent_num, act_spaces, obs_spaces):
        super(Agent, self).__init__(args, agent_num, act_spaces, obs_spaces)
        logger.info("actors:act_shapes:%s, obs_shapes:%s" %
                    (str(self.act_shapes), str(self.obs_shapes)))

        self.actors = self.create_actors()
        self.target_actors = self.create_actors()
        logger.info("critics:act_shapes:%s, obs_shapes:%s" %
                    (str(self.act_shapes), str(self.obs_shapes)))
        self.critics = self.create_critics()
        self.target_critics = self.create_critics()
        self.sigma = args.sigma
        self.decay_step = args.decay_step
        self.decay_rate = args.decay_rate
        self.min_sigma = args.min_sigma
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.plr, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
            amsgrad=False, name='Adam')
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.qlr, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
            amsgrad=False, name='Adam')

    def create_actors(self):
        actors = []
        for i in range(self.n):
            m = get_actor_model(
                i, self.args, self.act_shapes, self.obs_shapes)
            actors.append(m)
        return actors

    def create_critics(self):
        critics = []
        for i in range(self.n):
            m = get_critic_model(
                i, self.args, self.act_shapes, self.obs_shapes)
            critics.append(m)
        return critics

    def add_graph(self):
        return None

    def action(self, obs):
        self.step += len(obs)
        if self.step % self.decay_step == 0:
            self.sigma = max(self.sigma * self.decay_rate, self.min_sigma)
            logger.info("sigma decay to: %.3f" % self.sigma)
        batch_obs = np.asarray(obs)

        acts = [self.actors[i](batch_obs) for i in range(self.n)]
        for i in range(self.n):
            acts[i] += tf.random.normal(
                shape=acts[i].shape, mean=0., stddev=self.sigma,
                dtype=tf.float32)

        return tf.stack(acts, axis=1)

    def update_params(self, obs, act_n, rew_n, next_obs, done_n):
        start = time.time()

        obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)
        obs_next_tf = tf.convert_to_tensor(next_obs, dtype=tf.float32)
        act_n_tf = tf.convert_to_tensor(act_n, dtype=tf.float32)

        critic_loss, actor_loss, action_reg = 0.0, 0.0, 0.0

        next_target_acts = [self.target_actors[i](
            next_obs) for i in range(self.n)]

        next_target_qs = []
        for i in range(self.n):
            critic_input = tf.concat([obs_next_tf, next_target_acts[i]], 1)
            next_target_qs.append(self.target_critics[i](critic_input))

        target_qs = []
        for i in range(self.n):
            target_q = rew_n[i] + self.args.gamma * \
                (1.0 - done_n[i]) * next_target_qs[i]
            target_qs.append(target_q)

        for i in range(self.n):
            critic_input = tf.concat([obs_tf, act_n_tf[:, i, :]], 1)
            with tf.GradientTape() as tape:
                current_q = self.critics[i](critic_input)
                loss = tf.reduce_mean(
                    tf.keras.losses.MSE(current_q, target_qs[i]))
            critic_grad = tape.gradient(
                loss, self.critics[i].trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critics[i].trainable_variables))
            critic_loss += loss

        for i in range(self.n):
            with tf.GradientTape() as tape:
                action = self.actors[i](obs)
                reg = tf.norm(action, ord=2) * 1e-3
                critic_input = tf.concat([obs_tf, action], 1)
                loss = reg - tf.reduce_mean(self.critics[i](critic_input))
                action_reg += reg

            actor_grad = tape.gradient(
                loss, self.actors[i].trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actors[i].trainable_variables))
            actor_loss += loss

        for i in range(self.n):
            update_target_variables(
                self.actors[i].trainable_variables,
                self.target_actors[i].trainable_variables, self.args.tau)
            update_target_variables(
                self.critics[i].trainable_variables,
                self.target_critics[i].trainable_variables, self.args.tau)

        update_time = time.time() - start
        logger.debug("update_params use %.3f seconds" % update_time)
        return actor_loss, critic_loss, action_reg

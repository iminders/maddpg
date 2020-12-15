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
        self.actor_optimizers = [tf.keras.optimizers.Adam(
            learning_rate=args.plr, name='Adam') for i in range(self.n)]
        self.critic_optimizers = [tf.keras.optimizers.Adam(
            learning_rate=args.qlr,  name='Adam') for i in range(self.n)]

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
            logger.info("sigma decay to: %.3f,at %d" % (self.sigma, self.step))
        batch_obs = tf.convert_to_tensor(np.asarray(obs), dtype=tf.float32)

        acts = []
        for i in range(self.n):
            act = self.actors[i](batch_obs[:, i, :])
            noised_act = act + tf.random.normal(
                shape=act.shape, stddev=self.sigma, dtype=tf.float32)
            # TODO(liuwen): 根据act_space来clip
            acts.append(tf.clip_by_value(noised_act, -1.0, 1.0))
        acts_tf = tf.stack(acts, axis=1)
        return acts_tf

    def update_params(self, obs_n, act_n, rew_n, next_obs_n, done_n, isw):
        start = time.time()
        batch_size = obs_n.shape[0]
        # batch_size * n * obs_size
        obs_n_tf = tf.convert_to_tensor(obs_n, dtype=tf.float32)
        # batch_size  * n * obs_size
        obs_next_n_tf = tf.convert_to_tensor(next_obs_n, dtype=tf.float32)
        # batch_size * n * act_size
        act_n_tf = tf.convert_to_tensor(act_n, dtype=tf.float32)
        # batch_size * n * 1
        rew_n_tf = tf.expand_dims(tf.convert_to_tensor(
            rew_n, dtype=tf.float32), axis=2)
        # batch_size * n * 1
        done_n_tf = tf.expand_dims(tf.convert_to_tensor(
            done_n, dtype=tf.float32), axis=2)

        critic_loss, actor_loss, action_reg = 0.0, 0.0, 0.0

        for i in range(self.n):
            next_target_act_n = [self.target_actors[j](obs_next_n_tf[:, j, :])
                                 for j in range(self.n)]
            next_target_act_n = tf.concat(next_target_act_n, axis=1)
            # batch_size * (obs_n_size + act_n_szie)
            critic_input = tf.concat(
                [tf.reshape(obs_next_n_tf, [batch_size, -1]),
                 tf.reshape(next_target_act_n, [batch_size, -1])], 1)
            # batch_size * 1
            next_target_q = self.target_critics[i](critic_input)
            # batch_size * 1
            done = done_n_tf[:, i, :]
            # batch_size * 1
            rew = rew_n_tf[:, i, :]
            target_q = rew + self.args.gamma * \
                (tf.ones(done.shape) - done) * next_target_q

            # critic train
            with tf.GradientTape() as tape:
                current_q = self.critics[i](tf.concat(
                    [tf.reshape(obs_n_tf, [batch_size, -1]),
                     tf.reshape(act_n_tf, [batch_size, -1])], 1))
                abs_error = tf.math.abs(current_q - target_q)
                loss = tf.reduce_mean(tf.square(abs_error))

            critic_grad = tape.gradient(
                loss, self.critics[i].trainable_variables)
            self.critic_optimizers[i].apply_gradients(
                zip(critic_grad, self.critics[i].trainable_variables))
            critic_loss += loss

            # actor train
            with tf.GradientTape() as tape:
                # batch_size * act_size
                act = self.actors[i](obs_n_tf[:, i, :])
                act_n = tf.Variable(act_n_tf)
                act_n[:, i, :].assign(act)

                reg = tf.norm(act, ord=2) * 1e-3
                loss = reg - tf.reduce_mean(
                    self.critics[i](tf.concat([
                        tf.reshape(obs_n_tf, [batch_size, -1]),
                        tf.reshape(act_n, [batch_size, -1])], 1)))
                action_reg += reg

            actor_grad = tape.gradient(
                loss, self.actors[i].trainable_variables)
            self.actor_optimizers[i].apply_gradients(
                zip(actor_grad, self.actors[i].trainable_variables))
            actor_loss += loss

            update_target_variables(
                self.target_actors[i].weights,
                self.actors[i].weights, self.args.tau)
            update_target_variables(
                self.target_critics[i].weights,
                self.critics[i].weights, self.args.tau)

        update_time = time.time() - start
        logger.debug("update_params use %.3f seconds" % update_time)
        return actor_loss, critic_loss, action_reg, abs_error

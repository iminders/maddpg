# -*- coding:utf-8 -*-
import pickle
import time
import zlib

import numpy as np
import tensorflow as tf
import zmq

from maddpg.agents.maddpg.agent import Agent
from maddpg.common.logger import logger


def make_learner_agent(args=None, n=3, act_spaces=None, obs_spaces=None):
    logger.info("act_spaces:" + str(act_spaces))
    logger.info("obs_spaces:" + str(obs_spaces))
    agent = Agent(args, n, act_spaces, obs_spaces)
    return agent


def get_explore_log(i, warm_up, episode, rew, explore_time, total_time):
    progress_pct = episode * 100.0 / warm_up
    progress_msg = "[warm_up %5.2f%%]episode:%-6d step:%-9d" % (
        progress_pct, episode, i)
    time_msg = "rew:%-8.3f batch explore time:%.2fs, total:%.2fs" % (
        rew, explore_time, total_time)
    return progress_msg + time_msg


def get_train_log(i, episode, max_episode_num, rew, batch_time, total_time):
    progress_pct = episode * 100.0 / max_episode_num
    progress_msg = "[%5.2f%%]episode:%-8d" % (progress_pct, episode)
    score_msg = " rew:%-8.3f, batch_time:%.2fs total:%.2fs" % (
        rew, batch_time, total_time)
    return progress_msg + score_msg


def serve(agent):
    logger.info("serve")
    c = zmq.Context()
    s = c.socket(zmq.REP)
    s.bind('tcp://127.0.0.1:%d' % agent.args.port)
    logger.info("zmq bind at tcp://0.0.0.0:%d" % agent.args.port)

    explore_size = agent.args.explore_size
    env_batch_size = agent.args.env_batch_size

    i, iter, episode, stop_client_num, record_i = 0, 0, 0, 0, 0

    episode_rews = [0] * agent.args.save_rate
    mean_reward = 0.0

    start = time.time()
    batch_start = time.time()
    log_start = time.time()

    with agent.writer.as_default():
        while True:
            z = s.recv_pyobj()
            p = zlib.decompress(z)
            data = pickle.loads(p)
            [obs, action, next_obs, rew, done, terminal] = data
            for j in range(env_batch_size):
                agent.buffer.add(obs[j], action[j], rew[j],
                                 next_obs[j], done[j])
            i += env_batch_size

            if i % (explore_size * 100) == 0 and episode <= agent.args.warm_up:
                t = time.time()
                if episode < agent.args.save_rate:
                    mean_reward = 0.0
                else:
                    mean_reward = np.mean(episode_rews)
                logger.info(get_explore_log(i, agent.args.warm_up,
                            episode, mean_reward, t-batch_start, t-start))
                batch_start = t

            for j in range(env_batch_size):
                if all(done[j]) or terminal[j]:
                    episode += 1
                    loc = episode % agent.args.save_rate
                    episode_rews[loc] = np.sum(rew[j])
                    if episode % agent.args.save_rate == 0:
                        record_i += 1
                        mean_reward = np.mean(episode_rews)
                        if mean_reward > agent.best_score:
                            agent.best_score = mean_reward
                            agent.save()
                        tf.summary.scalar(
                            '1.performance/2.episode_reward',
                            mean_reward, record_i)
                        if episode > agent.args.warm_up:
                            batch_end = time.time()
                            log_msg = get_train_log(
                                i, episode, agent.args.num_episodes,
                                mean_reward, batch_end-log_start,
                                batch_end-start)
                            log_start = batch_end
                            logger.info(log_msg)

            if i % explore_size == 0 and episode > agent.args.warm_up:
                iter += 1
                explore_time = time.time() - batch_start
                logger.debug(
                    "serve collect %d explore samples spend %.3f secs" % (
                        agent.args.batch_size, explore_time))
                tf.summary.scalar('3.time/2.explore', explore_time, iter)
                agent.learn(iter)
                t = time.time()
                batch_start = t
            action = agent.action(next_obs)
            p = pickle.dumps(action)

            if episode >= agent.args.num_episodes:
                stop_client_num += 1
                logger.info("i=%d, episode=%d" % (i, episode))
                s.send_pyobj(pickle.dumps("stop"))
                if stop_client_num >= agent.args.num_env:
                    agent.writer.close()
                    break
            else:
                s.send_pyobj(p)
    s.close()

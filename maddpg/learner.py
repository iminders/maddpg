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
    progress_pct = i * 100.0 / warm_up
    progress_msg = "[explore:%5.2f%%]episode:%-5d step:%-8d" % (
        progress_pct, episode, i)
    time_msg = "rew:%-8.3f batch explore time:%.2fs, total:%.2fs" % (
        rew, explore_time, total_time)
    return progress_msg + time_msg


def get_train_log(i, episode, max_episode_num, rew, total_time):
    progress_pct = episode * 100.0 / max_episode_num
    progress_msg = "[%5.2f%%]episode:%-8d" % (progress_pct, episode)
    score_msg = " rew:%-8.3f total:%10.2fsecs" % (rew, total_time)
    return progress_msg + score_msg


def serve(agent):
    logger.info("serve")
    c = zmq.Context()
    s = c.socket(zmq.REP)
    s.bind('tcp://127.0.0.1:%d' % agent.args.port)
    logger.info("zmq bind at tcp://0.0.0.0:%d" % agent.args.port)
    i, iter, episode, stop_client_num, record_i = 0, 0, 0, 0, 0
    start = time.time()
    batch_start = time.time()
    episode_rews = [0] * agent.args.save_rate
    with agent.writer.as_default():
        while True:
            z = s.recv_pyobj()
            p = zlib.decompress(z)
            data = pickle.loads(p)
            [obs, action, next_obs, rew, done, terminal] = data
            agent.buffer.add(obs, action, rew, next_obs, done)

            mean_reward = np.mean(episode_rews)
            if i % agent.args.batch_size == 0 and i <= agent.args.warm_up:
                t = time.time()
                if episode < agent.args.save_rate:
                    mean_reward = 0.0
                logger.info(get_explore_log(i, agent.args.warm_up,
                            episode, mean_reward, t - batch_start, t-start))
                batch_start = t

            if all(done) or terminal:
                episode += 1
                episode_rews[episode % agent.args.save_rate] = np.mean(rew)
                if episode % agent.args.save_rate == 0:
                    record_i += 1
                    tf.summary.scalar(
                        '1.performance/2.avg_episode_rew',
                        mean_reward, record_i)
                    if i > agent.args.warm_up:
                        log_msg = get_train_log(
                            i, episode, agent.args.num_episodes,
                            mean_reward, time.time()-start)
                        logger.info(log_msg)

            if i % agent.args.batch_size == 0 and i > agent.args.warm_up:
                iter += 1
                explore_time = time.time() - batch_start
                logger.debug(
                    "serve collect %d explore samples spend %.3f secs" % (
                        agent.args.batch_size, explore_time))
                tf.summary.scalar('3.time/2.explore', explore_time, iter)
                agent.learn(iter)
                t = time.time()
                batch_start = t
            action = agent.action(obs)
            p = pickle.dumps(action)
            i += 1
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

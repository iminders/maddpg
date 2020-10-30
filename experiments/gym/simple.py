import time

from maddpg.arguments import parse_experiment_args
from maddpg.common.env_utils import make_env, uniform_action

args = parse_experiment_args()
env = make_env(args=args, id=0)


for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = uniform_action(env.action_space)
        observation, reward, done, info = env.step(action)
        for i in range(env.n):
            print(i, action[i], reward[i])
        time.sleep(5)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

import argparse
import os
import socket
from datetime import datetime


def parse_experiment_args():
    parser = argparse.ArgumentParser("Experiments for maddpg environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple",
                        help="name of the scenario script")
    parser.add_argument("--max_episode_len", type=int, default=25,
                        help="maximum episode length")
    parser.add_argument("--num_episodes", type=int, default=100000,
                        help="number of episodes")
    parser.add_argument("--num_adversaries", type=int, default=0,
                        help="number of adversaries")
    parser.add_argument("--good_policy", type=str, default="maddpg",
                        help="policy for good agents")
    parser.add_argument("--adv_policy", type=str, default="maddpg",
                        help="policy of adversaries")
    parser.add_argument("--print_net", action="store_true", default=False)

    # Core training parameters
    parser.add_argument("--qlr", type=float, default=3e-3,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--plr", type=float, default=1e-3,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="discount factor")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="explore noise sigma")
    parser.add_argument("--decay_rate", type=float, default=0.98,
                        help="explore noise sigma decay rate")
    parser.add_argument("--min_sigma", type=float, default=0.2,
                        help="explore minimal sigma")
    parser.add_argument("--decay_step", type=int, default=50000,
                        help="explore noise sigma")

    parser.add_argument("--tau", type=float, default=0.97,
                        help="discount factor")

    parser.add_argument("--batch_size", type=int, default=1024,
                        help="number of episodes to optimize at the same time")
    parser.add_argument("--explore_size", type=int, default=100,
                        help="number of episodes to optimize at the same time")

    parser.add_argument("--num_units", type=int, default=64,
                        help="number of units in the mlp")
    parser.add_argument("--dropout", type=float, default=None,
                        help="dropout")
    parser.add_argument("--noise_pd", type=str, default="gaussian",
                        help="noise probability distribution 探索时使用的随机方法")
    # Checkpointing
    parser.add_argument("--exp_name", type=str, default="sample",
                        help="name of the experiment")
    parser.add_argument("--model_dir", type=str,
                        default=os.path.join("exp", "models"),
                        help="directory for save state and model")
    parser.add_argument("--tb_dir", type=str,
                        default=os.path.join("exp", "tensorboard"),
                        help="directory where tensorboard data is saved")
    parser.add_argument('--minio_host', help='minio host',
                        default="49.234.229.193:5001",
                        type=str)
    parser.add_argument('--minio_key', help='minio key',
                        default="maddpg",
                        type=str)
    parser.add_argument('--minio_secret',
                        help='minio secret, ask liuwen.w@qq.com to get secret',
                        default=os.getenv("MADDPG_MINIO_SECRET"),
                        type=str)
    parser.add_argument('--minio_bucket', help='minio bucket',
                        default="maddpg",
                        type=str)
    parser.add_argument("--save_rate", type=int, default=1000,
                        help="save model every this episodes are completed")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000,
                        help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str,
                        default="./benchmark_files/",
                        help="directory where benchmark data is saved")

    # worker settings
    parser.add_argument("--num_env", type=int, default=2,
                        help="explore environments number")
    parser.add_argument("--num_agent", type=int, default=3,
                        help="explore environments number")
    parser.add_argument("--env_batch_size", type=int, default=10,
                        help="explore batch environment size")
    parser.add_argument('--warm_up', type=int, default=1500)

    parser.add_argument('--role', type=str, default="learner",
                        help='learner/explorer')
    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=4932)
    parser.add_argument("--device", type=str, default="cpu",
                        help="run with gpu or only cpu")
    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--runner', type=str, default=socket.gethostname(),
                        help="which machine runner experiment")
    parser.add_argument('--run_id', type=str,
                        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                        help="the experiment run id")
    parser.add_argument("--debug", action="store_true", default=False)

    return parser.parse_args()

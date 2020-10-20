import argparse


def parse_args():
    parser = argparse.ArgumentParser("Experiments for maddpg environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple",
                        help="name of the scenario script")
    parser.add_argument("--max_episode_len", type=int, default=25,
                        help="maximum episode length")
    parser.add_argument("--num_episodes", type=int, default=60000,
                        help="number of episodes")
    parser.add_argument("--num_adversaries", type=int, default=0,
                        help="number of adversaries")
    parser.add_argument("--good_policy", type=str, default="maddpg",
                        help="policy for good agents")
    parser.add_argument("--adv_policy", type=str, default="maddpg",
                        help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--qlr", type=float, default=3e-3,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--plr", type=float, default=1e-3,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="discount factor")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="number of episodes to optimize at the same time")
    parser.add_argument("--num_units", type=int, default=64,
                        help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="sample",
                        help="name of the experiment")
    parser.add_argument("--mode_dir", type=str, default="models",
                        help="directory for save state and model")
    parser.add_argument("--tb_dir", type=str, default="tensorboard",
                        help="directory where tensorboard data is saved")

    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cpu",
                        help="run with gpu or only cpu")
    parser.add_argument("--debug", action="store_true", default=False)

    # worker settings
    parser.add_argument("--num_explore", type=int, default=2,
                        help="explore worker number")

    return parser.parse_args()

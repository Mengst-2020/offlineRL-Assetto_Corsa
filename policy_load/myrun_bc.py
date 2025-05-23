import argparse
import random

import gym
import d4rl

import numpy as np
import torch
print("------------------------$$$$$$$$$$$$$$$$$$$$$$----------------------------")
import h5py
import minari
import deepdish as dd
import json
from gymnasium.spaces import Box
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent 
sys.path.insert(0, str(project_root))


from offlinerlkit.nets import MLP
from offlinerlkit.modules import Actor
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import BCPolicy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="bc")
    parser.add_argument("--task", type=str, default="ac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=3000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    dataset = d4rl.qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # create buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    actor = Actor(actor_backbone, args.action_dim, max_action=args.max_action, device=args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    # create policy
    policy = BCPolicy(actor, actor_optim)

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes
    )

    # train
    policy_trainer.train()


def train_mydataset(args=get_args(),Path=None):
    # create env and dataset
    # env = gym.make(args.task)
    from getdata import get_mydataset,make_env
    env=make_env()
    dataset = get_mydataset(Path)
    with open(Path+'metadata.json', 'r', encoding='utf-8') as f:
        data_info = json.load(f)
    args.obs_shape = tuple(json.loads(data_info["observation_space"])["shape"])
    args.action_dim = json.loads(data_info["action_space"])["shape"][0]
    args.max_action = json.loads(data_info["action_space"])["high"][0]

    # create buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    # env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    actor = Actor(actor_backbone, args.action_dim, max_action=args.max_action, device=args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    # create policy
    policy = BCPolicy(actor, actor_optim)

    # log
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_flag=False,
        eval_episodes=args.eval_episodes,
    )

    # train
    policy_trainer.train()


def PolicyNetwork(args=get_args(),Path='/home/mengst/.minari/datasets/SAC/monza-v4/data/'):
    with open(Path+'metadata.json', 'r', encoding='utf-8') as f:
        data_info = json.load(f)
    args.obs_shape = tuple(json.loads(data_info["observation_space"])["shape"])
    args.action_dim = json.loads(data_info["action_space"])["shape"][0]
    args.max_action = json.loads(data_info["action_space"])["high"][0]

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    actor = Actor(actor_backbone, args.action_dim, max_action=args.max_action, device=args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    # create policy
    policy = BCPolicy(actor, actor_optim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_filepath = "model_offline/monza_log/bc/seed_0&timestamp_25-0408-212657/model/policy.pth"
    load_filepath = "log/ac/bc/seed_0&timestamp_25-0421-181938/model/policy.pth"
    checkpoint = torch.load(load_filepath, map_location=device)
    policy.load_state_dict(checkpoint)
    return policy


if __name__ == "__main__":
    train_mydataset(Path='/home/mengst/.minari/datasets/SAC/barcelona-v4/data/')
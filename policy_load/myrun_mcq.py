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

from offlinerlkit.nets import MLP, VAE
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import MCQPolicy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mcq")
    parser.add_argument("--task", type=str, default="ac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[400, 400])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--lmbda", type=float, default=0.9)
    parser.add_argument("--num-sampled-actions", type=int, default=10)
    parser.add_argument("--behavior-policy-lr", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    dataset = qlearning_dataset(env)
    if 'antmaze' in args.task:
        dataset["rewards"] -= 1.0
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims, dropout_rate=0.1)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    behavior_policy = VAE(
        input_dim=np.prod(args.obs_shape),
        output_dim=args.action_dim,
        hidden_dim=750,
        latent_dim=args.action_dim*2,
        max_action=args.max_action,
        device=args.device
    )
    behavior_policy_optim = torch.optim.Adam(behavior_policy.parameters(), lr=args.behavior_policy_lr)

    # create policy
    policy = MCQPolicy(
        actor,
        critic1,
        critic2,
        behavior_policy,
        actor_optim,
        critic1_optim,
        critic2_optim,
        behavior_policy_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        lmbda=args.lmbda,
        num_sampled_actions=args.num_sampled_actions
    )

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
    from getdata import get_mydataset,make_env
    env=make_env()
    dataset = get_mydataset(Path)
    if 'antmaze' in args.task:
        dataset["rewards"] -= 1.0
    with open(Path+'metadata.json', 'r', encoding='utf-8') as f:
        data_info = json.load(f)
    args.obs_shape = tuple(json.loads(data_info["observation_space"])["shape"])
    args.action_dim = json.loads(data_info["action_space"])["shape"][0]
    args.max_action = json.loads(data_info["action_space"])["high"][0]
    action_shape=tuple(json.loads(data_info["action_space"])["shape"])
    action_space=Box(low=np.array([-1.0,  -1.0,  -1.0]), high=np.array([1.0,  1.0,  1.0]))

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims, dropout_rate=0.1)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(action_shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    behavior_policy = VAE(
        input_dim=np.prod(args.obs_shape),
        output_dim=args.action_dim,
        hidden_dim=750,
        latent_dim=args.action_dim*2,
        max_action=args.max_action,
        device=args.device
    )
    behavior_policy_optim = torch.optim.Adam(behavior_policy.parameters(), lr=args.behavior_policy_lr)

    # create policy
    policy = MCQPolicy(
        actor,
        critic1,
        critic2,
        behavior_policy,
        actor_optim,
        critic1_optim,
        critic2_optim,
        behavior_policy_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        lmbda=args.lmbda,
        num_sampled_actions=args.num_sampled_actions
    )

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
        # eval_flag=False,
        eval_episodes=args.eval_episodes
    )

    # train
    policy_trainer.train()


def PolicyNetwork(args=get_args(),Path='/home/mengst/.minari/datasets/SAC/monza-v4/data/'):
    with open(Path+'metadata.json', 'r', encoding='utf-8') as f:
        data_info = json.load(f)
    args.obs_shape = tuple(json.loads(data_info["observation_space"])["shape"])
    args.action_dim = json.loads(data_info["action_space"])["shape"][0]
    args.max_action = json.loads(data_info["action_space"])["high"][0]
    action_shape=tuple(json.loads(data_info["action_space"])["shape"])
    action_space=Box(low=np.array([-1.0,  -1.0,  -1.0]), high=np.array([1.0,  1.0,  1.0]))

    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims, dropout_rate=0.1)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args.max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(action_shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    behavior_policy = VAE(
        input_dim=np.prod(args.obs_shape),
        output_dim=args.action_dim,
        hidden_dim=750,
        latent_dim=args.action_dim*2,
        max_action=args.max_action,
        device=args.device
    )
    behavior_policy_optim = torch.optim.Adam(behavior_policy.parameters(), lr=args.behavior_policy_lr)

    # create policy
    policy = MCQPolicy(
        actor,
        critic1,
        critic2,
        behavior_policy,
        actor_optim,
        critic1_optim,
        critic2_optim,
        behavior_policy_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        lmbda=args.lmbda,
        num_sampled_actions=args.num_sampled_actions
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_filepath = "model_offline/monza_log/mcq/seed_0&timestamp_25-0410-134852/model/policy.pth"
    checkpoint = torch.load(load_filepath, map_location=device)
    policy.load_state_dict(checkpoint)
    return policy


if __name__ == "__main__":
    train_mydataset(Path='/home/mengst/.minari/datasets/SAC/monza-v4/data/')
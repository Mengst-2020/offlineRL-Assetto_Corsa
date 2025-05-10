import matplotlib.pyplot as plt
import sys
import os

import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from rl_zoo3.train import train
from stable_baselines3 import PPO
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datetime import datetime

import minari
from minari import DataCollector
from omegaconf import OmegaConf

from gymnasium.spaces import Box


sys.path.extend([os.path.abspath('./assetto_corsa_gym'), './algorithm/discor'])
import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from discor.agent_dataset import Agent
from discor.algorithm import SAC
import logging


import json
import argparse
from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian,DiagGaussian
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import IQLPolicy
from policy_load.myrun_bc import PolicyNetwork
#monza bc 4弯道会冲出去,有些赛道低速也无法通过
#monza combo 5勉强会跑,直道乱走                              #####函数有错
#monza cql 直接出赛道 
#monza edac 刚开始还好，之后动作选择比较极端
#monza iql 1表现较好，但弯道有一些有时候过不去 动作[0]
#monza mcq 不动，极端动作
#monza mobile 不动,动作较好                          #####函数有错
#monza mopo 动，但摆动较大，行驶距离短                 #####函数有错
#monza rambo  2略好，有些赛道低速也无法通过    #####函数有错
#monza td3bc  3动，重置后能低速通过赛道

if __name__ == "__main__":
    config = OmegaConf.load("config.yml")
    work_dir = "outputs" + os.sep + datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
    work_dir = os.path.abspath(work_dir) + os.sep
    env = assettoCorsa.make_ac_env(cfg=config, work_dir=work_dir)  

    device = torch.device("cpu")
    algo = SAC(
                state_dim=125,
                action_dim=3,
                device=device, seed=config.seed,
                **OmegaConf.to_container(config.SAC))
    agent = Agent(env=env, test_env=env, algo=algo, log_dir="output",
                    device=device, seed=config.seed, **config.Agent, wandb_logger=None)
    agent.load("model_sac/model_monza",False)
    
    policy_net=PolicyNetwork()
    policy_net.eval()

    max_step=6000
    max_epo=50
    num_epo=0
    dataset= None
    while num_epo<max_epo:
        num_epo=num_epo+1
        obs, _ = env.reset()
        done = False
        accumulated_rew = 0
        step=0
        rand_pos=random.random()
        while not done and step<max_step:
            step=step+1
            action = policy_net.select_action(torch.Tensor(obs))
            action_good, _ = agent._algo.exploit(obs)

            if not isinstance(action[0], np.float32):
                action=action[0]

            # if num_epo>3 and step<100:
            #     print("!!!!!!!!!!!!!!!!!!!!!$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            #     action=action_good

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            accumulated_rew += reward
        print("Accumulated rew: ", accumulated_rew)
        # if dataset is None:
        #     dataset = env.create_dataset(
        #         dataset_id="SAC/india-v4",
        #         algorithm_name="SAC-Policy",
        #         code_permalink="https://github.com/Farama-Foundation/Minari",
        #         author="Farama",
        #         author_email="contact@farama.org"
        #     )
        # else:
        #     env.add_to_dataset(dataset)

    env.close()
    
    # high = np.full((125, ), 1e8)
    # low = np.full((125, ), -1e8)
    # observation_space = Box(low=low, high=high)
    # action_space = Box(low=np.array([-1.0,  -1.0,  -1.0]), high=np.array([1.0,  1.0,  1.0]))
    # assert isinstance(observation_space, spaces.Box)
    # assert isinstance(action_space, spaces.Box)
    # device = torch.device("cpu")
    # policy_net = PolicyNetwork(np.prod(observation_space.shape), action_space.shape[0])
    # policy_net.eval()  # 将模型设置为评估模式
    # policy_net.load_state_dict(torch.load('cql_monza.pth', map_location=torch.device('cpu')))

    # checkpoint = torch.load('model_offline/log/cql_256_2/monza/model/policy.pth', map_location='cpu')
    # actor_weights = {k.replace('actor.', ''): v for k, v in checkpoint.items() if k.startswith('actor.')}
    # policy_net.load_state_dict(actor_weights, strict=False) 

    # action = policy_net(torch.Tensor(obs)).detach().numpy()
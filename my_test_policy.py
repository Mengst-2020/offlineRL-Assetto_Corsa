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
import time
from minari import DataCollector
from omegaconf import OmegaConf

from gymnasium.spaces import Box


sys.path.extend([os.path.abspath('./assetto_corsa_gym'), './algorithm/discor'])
import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from discor.agent_dataset import Agent
from discor.algorithm import SAC
import logging
import socket

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":

    config = OmegaConf.load("config.yml")
    work_dir = "outputs" + os.sep + datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
    work_dir = os.path.abspath(work_dir) + os.sep
    env = assettoCorsa.make_ac_env(cfg=config, work_dir=work_dir)
    os.environ["MINARI_DATASETS_PATH"] = "F:/code/assetto_corsa_gym-main/mydata"
    dataset= None

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
    


    device = torch.device("cpu")
    algo = SAC(
                state_dim=env.env.observation_space.shape[0],
                action_dim=3,
                device=device, seed=config.seed,
                **OmegaConf.to_container(config.SAC))
    agent = Agent(env=env, test_env=env, algo=algo, log_dir="output",
                    device=device, seed=config.seed, **config.Agent, wandb_logger=None)
    # agent.load("model_ac/model_barcelona",False)
    # agent.load("model_ac/model_monza",False)
    agent.load("model_ac/model_redbull",False)


    max_epo=50
    num_epo=0
  
    while num_epo<max_epo:
        num_epo=num_epo+1
        obs, _ = env.reset()
        done = False
        accumulated_rew = 0
        step=0

        while not done :
            step=step+1
            action_good, _ = agent._algo.exploit(obs)
            obs, reward, terminated, truncated, _ = env.step(action_good)

            # print(action_good)

            # msg = ",".join([f"{a:.4f}" for a in action_good])
            # data=msg.encode()
            # s.sendto(data,('192.168.137.34',2346))
            
            done = terminated or truncated
            accumulated_rew += reward

        print("Accumulated rew: ", accumulated_rew)
        # if dataset is None:
        #     dataset = env.create_dataset(
        #         dataset_id="barcelona/sac-v3",
        #         algorithm_name="SAC-Policy",
        #         code_permalink="https://github.com/Farama-Foundation/Minari",
        #         author="Farama",
        #         author_email="contact@farama.org"
        #     )
        # else:
        #     env.add_to_dataset(dataset)

    env.close()
    
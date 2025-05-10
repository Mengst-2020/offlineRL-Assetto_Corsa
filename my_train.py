import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import pandas as pd
import glob as glob
import time
import pickle
from omegaconf import OmegaConf
from datetime import datetime

import minari
from minari import DataCollector
import torch

import ray
from ray.rllib.algorithms.ppo import PPOConfig

# add custom paths
sys.path.extend([os.path.abspath('./assetto_corsa_gym'), './algorithm/discor'])
import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from discor.agent_dataset import Agent
from discor.algorithm import SAC

# Configure the logging system
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format of the log messages
    datefmt='%Y-%m-%d %H:%M:%S',  # Format of the timestamp
)

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
agent = Agent(env=env, test_env=env, algo=algo, log_dir="output_barcelona",
                  device=device, seed=config.seed, **config.Agent, wandb_logger=None)

os.environ["MINARI_DATASETS_PATH"] = "F:/code/assetto_corsa_gym-main/mydata"
dataset= None
while(True):
    agent.train_episode()
    if agent._steps > agent._num_steps:
        break
    # if dataset is None:
    #     dataset = env.create_dataset(
    #         dataset_id="SAC/monza-v0",
    #         algorithm_name="SAC-Policy",
    #         code_permalink="https://github.com/Farama-Foundation/Minari",
    #         author="Farama",
    #         author_email="contact@farama.org"
    #     )
    # else:
    #     env.add_to_dataset(dataset)

env.close()
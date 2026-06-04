import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import pandas as pd
import glob as glob
import time
import pickle
from omegaconf import OmegaConf

import minari
from minari import DataCollector
import torch

# add custom paths
sys.path.extend([os.path.abspath('./assetto_corsa_gym'), './algorithm/discor'])
import AssettoCorsaEnv.assettoCorsa as assettoCorsa

# Configure the logging system
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format of the log messages
    datefmt='%Y-%m-%d %H:%M:%S',  # Format of the timestamp
)

import ray
import copy
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from gymnasium.envs.registration import register
sys.path.append(r"F:/code/assetto_corsa_gym-main/assetto_corsa_gym")
from AssettoCorsaEnv.ac_env import AssettoCorsaEnv

register(
    id='AssettoCorsaEnv',
    entry_point="AssettoCorsaEnv.ac_env:AssettoCorsaEnv",
)

# env = AssettoCorsaEnv()
# env.reset()
# for i in range(100):
#     if i % 2 == 0:
#         steer = .1
#     else:
#         steer = -.1
#     next_state, reward, done, _,info = env.step(np.array([steer, 0.5, -1.]))  # action is already applied
#     time.sleep(0.01)
#     if done:
#         break

ray.init()
algo_config=PPOConfig()
# algo_config=SACConfig()
checkpoint_dir = os.path.abspath("./ray_checkpoints/PPO_405")
os.makedirs(checkpoint_dir, exist_ok=True)
algo_config = algo_config.training(gamma=0.992,
                                   lr=0.0003,
                                #    actor_lr=0.0003,
                                #    critic_lr=0.0003,
                                #    alpha_lr=0.003,
                                #    n_step=3,
                                #    policy_model_config={'fcnet_hiddens': [256, 256,256]},
                                #    q_model_config={'fcnet_hiddens': [256, 256,256]},
                                #    replay_buffer_config={
                                #        "_enable_replay_buffer_api": True,
                                #         # "type": "MultiAgentReplayBuffer",
                                #         "capacity": 10000000,
                                #         "replay_batch_size": 128,
                                #         "replay_sequence_length": 1,
                                #    },
                                #    tau=0.005,
                                #    num_steps_sampled_before_learning_starts=2000
                                )
algo_config = algo_config.resources(num_gpus=0)
algo_config = algo_config.env_runners(num_env_runners=0) 
algo_config = algo_config.environment(env="AssettoCorsaEnv")
# algo_config.replay_buffer_config["capacity"] = 20000  # reduce replay buffer
algo_config = algo_config.framework('torch')

algo = algo_config.build()

checkpoint_path = os.path.abspath("./ray_checkpoints/PPO_405")
algo.restore(checkpoint_path)

while(True):
    result = algo.train()
    checkpoint = algo.save(checkpoint_dir)
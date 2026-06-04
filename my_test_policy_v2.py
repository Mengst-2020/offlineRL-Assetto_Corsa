import matplotlib.pyplot as plt
import sys
import os
import json

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
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

from my_image import VisionStudentPolicy,VisionPolicyFastRunner
from collections import deque

from mpc_assetto_corsa import AssettoCorsaMPCController

sys.path.extend([os.path.abspath('./assetto_corsa_gym'), './algorithm/discor'])
import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from discor.agent_dataset import Agent
from discor.algorithm import SAC
import logging

from my_policy_load import IQL_Network,TD3BC_Network,BC_Network,RAMBO_Network


def to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def append_jsonl(log_path, payload):
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(to_serializable(payload), ensure_ascii=False) + "\n")

def load_config():
    config_path = "config.yaml" if os.path.exists("config.yaml") else "config.yml"
    return OmegaConf.load(config_path), config_path

def build_image_policy(device):
    model = VisionStudentPolicy(8).to(device)
    model.load_state_dict(torch.load("model_image/vision_student_policy.pth", map_location=device))
    model.eval()

    algo = SAC(
                state_dim=125,
                action_dim=3,
                device=device, seed=config.seed,
                **OmegaConf.to_container(config.SAC))
    agent = Agent(env=env, test_env=env, algo=algo, log_dir="output",
                    device=device, seed=config.seed, **config.Agent, wandb_logger=None)
    agent.load("model_ac/model_monza", False)

    return VisionPolicyFastRunner(model=model, history_len=8, device=device),agent

def build_agent_policy(device):
    policy_net = IQL_Network()
    policy_net.eval()

    algo = SAC(
                state_dim=125,
                action_dim=3,
                device=device, seed=config.seed,
                **OmegaConf.to_container(config.SAC))
    agent = Agent(env=env, test_env=env, algo=algo, log_dir="output",
                    device=device, seed=config.seed, **config.Agent, wandb_logger=None)
    agent.load("model_ac/model_monza", False)

    # agent.load("model_ac/model_barcelona",False)
    # agent.load("model_ac/model_redbull",False)
    # agent.load("output_nstep3_628/model/checkpoints/step_00800000",False)

    return policy_net, agent

if __name__ == "__main__":
    config = OmegaConf.load("config.yml")
    test_policy_cfg = getattr(config, "TestPolicy", {})
    control_mode = str(getattr(test_policy_cfg, "control_mode", "mpc")).lower()

    work_dir = "outputs" + os.sep + datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
    work_dir = os.path.abspath(work_dir) + os.sep
    os.makedirs(work_dir, exist_ok=True)
    mpc_log_path = os.path.join(work_dir, "mpc_debug.jsonl")

    env = assettoCorsa.make_ac_env(cfg=config, work_dir=work_dir)  

    device = torch.device("cuda")

    if control_mode == "agent":
        policy_net, agent = build_agent_policy(device)
    elif control_mode == "image":
        image_policy, agent = build_image_policy(device)
    elif control_mode == "mpc":
        origin_env=env.env.env
        origin_env.enable_mpc = True
        mpc = AssettoCorsaMPCController(
            env=origin_env,
            horizon=int(getattr(config.AssettoCorsa, "mpc_horizon", 15)),
            dt=float(getattr(config.AssettoCorsa, "mpc_dt", origin_env.dt)),
            vehicle_params=dict(getattr(config.AssettoCorsa, "mpc_vehicle_params", {})),
        )
    else:
        raise ValueError(f"Unsupported control_mode: {control_mode}")


    dataset= None
    os.environ["MINARI_DATASETS_PATH"] = "F:/code/assetto_corsa_gym-main/mydata"

    controls_rate_limit = np.array([[-600/180, 600/180],
                                    [-1200/100, 1200/100], # the first is the falling edge of the pedal -> checked both
                                    [-1200/100, 1200/100],  # the first is the brake release; the second the brake press;
                                    ]) * (1/25)
    controls_min_values = np.array([-1.0,  -1.0,  -1.0], dtype=np.float32)
    controls_max_values = np.array([ 1.0,   1.0,   1.0], dtype=np.float32)

    print(f"MPC debug log: {mpc_log_path}")

    episode_idx = 0
    # while  not origin_env.end_flag:
    while True:
        episode_idx += 1
        obs, _ = env.reset()
        if control_mode == "mpc":
            mpc.reset()
        done = False
        accumulated_rew = 0
        step=0
        while not done:
            step=step+1

            if control_mode == "mpc":
                action, planning_info = mpc.select_action()
            elif control_mode == "image":
                action = image_policy.act(obs["image"])
                planning_info = None
            elif control_mode == "agent":
                action, _ = agent._algo.exploit(obs["state"])
                planning_info = None
            else:
                action = policy_net.select_action(torch.as_tensor(obs["state"], dtype=torch.float32))
                if not isinstance(action[0], float):
                    action = action[0]
                planning_info = None

            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)
            abs_action_before_step = origin_env.current_actions.copy() if control_mode == "mpc" else None

            # if not origin_env.use_relative_actions:  #绝对动作
            #     max_delta_action = Box(low=controls_rate_limit[:,0], high=controls_rate_limit[:,1])
            #     action = action * max_delta_action.high+origin_env.current_actions
            #     action = np.clip(action, controls_min_values, controls_max_values)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            accumulated_rew += reward
            if planning_info is not None:
                append_jsonl(
                    mpc_log_path,
                    {
                        "episode": episode_idx,
                        "step": step,
                        "mode": planning_info.get("message"),
                        "planning_time": planning_info.get("planning_time"),
                        "objective": planning_info.get("objective"),
                        "rel_action": action.copy(),
                        "abs_action_before_step": abs_action_before_step,
                        "abs_action_after_step": origin_env.current_actions.copy(),
                        "mpc_u0": planning_info.get("u0"),
                        "state": planning_info.get("state"),
                        "debug": planning_info.get("debug"),
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                    },
                )
            if planning_info is not None and (step % 10) == 0:
                print(
                    f"step={step} plan_t={planning_info['planning_time']:.3f}s "
                    f"mode={planning_info['message']} \n"
                    f"rel_action={action} abs_action={origin_env.current_actions}"
                )
            # if origin_env.failmode is not None and step>125:
            #     if origin_env.failmode ["failure_reason"]=="Oversteer":
            #         obs, _ = env.reset()
            #         print("################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            #         import winsound
            #         winsound.Beep(1000, 2500)  # 1000Hz，0.5秒
            #         sys.exit()
        print("Accumulated rew: ", accumulated_rew)
        # if dataset is None:
        #     dataset = env.create_dataset(
        #         dataset_id="monza_image_relative/sac-v2",
        #         algorithm_name="SAC-Policy",
        #         code_permalink="https://github.com/Farama-Foundation/Minari",
        #         author="Farama",
        #         author_email="contact@farama.org"
        #     )
        # else:
        #     env.add_to_dataset(dataset)

    env.close()
    
    

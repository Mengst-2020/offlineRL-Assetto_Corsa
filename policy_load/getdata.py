import numpy as np
import deepdish as dd
import time

def get_mydataset(Path=None, terminate_on_end=False, **kwargs):
    dataset_all = dd.io.load(Path+'main_data.hdf5')
    episode_N = len(dataset_all)
    # N = dataset_all["episode_1"]['rewards'].shape[0]
    # ['actions', 'infos', 'observations', 'rewards', 'terminations', 'truncations', 'id', 'total_steps']
    # print(N)
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # episode_step = 0
    for i in range(episode_N -1):
        episode_dataset=dataset_all["episode_"+ str(i)]
        episode_step_sum=episode_dataset['rewards'].shape[0]
        for step in range(episode_step_sum-1):
            obs = episode_dataset['observations'][step].astype(np.float32)
            new_obs = episode_dataset['observations'][step+1].astype(np.float32)
            action = episode_dataset['actions'][step].astype(np.float32)
            reward = episode_dataset['rewards'][step].astype(np.float32)
            done_bool = bool(episode_dataset['terminations'][step])

            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)


    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


def make_env():
    import os
    import sys
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
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format of the log messages
        datefmt='%Y-%m-%d %H:%M:%S',  # Format of the timestamp
    )
    from omegaconf import OmegaConf
    config = OmegaConf.load("config.yml")
    env = assettoCorsa.make_ac_env(cfg=config, work_dir="outputs")
    return env


def termination_fn_point2dwallenv(obs, act, next_obs):#################
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done


if __name__ == "__main__":
    dataset_all = dd.io.load('/home/mengst/.minari/datasets/SAC/monza-v0/data/main_data.hdf5')
    episode_N = len(dataset_all)
    print(episode_N)

    # env=make_env()
    # env.reset()
    # for i in range(100):
    #     if i % 2 == 0:
    #         steer = .1
    #     else:
    #         steer = -.1
    #     next_state, reward, done, _,info = env.step(action=np.array([steer, 0.5, -1.]))  # action is already applied
    #     time.sleep(0.01)
    #     if done:
    #         break
import os
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pickle
from pathlib import Path
from tqdm import tqdm
import random, torch
from discor.replay_buffer import ReplayBuffer, EnsembleBuffer
from discor.utils import RunningMeanStats
from AssettoCorsaEnv.data_loader import DataLoader

from minari import DataCollector
import copy
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import time

class Agent:
    def __init__(self, env, test_env, algo, log_dir, device, num_steps=3_000_000,
                 batch_size=256, memory_size=1_000_000,
                 update_interval=1, start_steps=10000, log_interval=10, checkpoint_freq=0,checkpoint_ep_freq=0,
                 eval_interval=5000, num_eval_episodes=5, seed=0, use_offline_buffer=False, offline_buffer_size=1_000_000, wandb_logger=None):

        # Environment.
        self._env = env
        # self._env = DataCollector(env)#########################3

        self._test_env = test_env
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_ep_freq = checkpoint_ep_freq
        self.wandb_logger = wandb_logger

        # self._env.seed(seed)
        self._env.reset(seed=seed)

        # self._test_env.reset(2**31-1-seed)
        self._test_env.reset()

        # Algorithm.
        self._algo = algo

        state_shape=self._env.observation_space["state"].shape#######################

        if use_offline_buffer:
            self._replay_buffer = EnsembleBuffer(memory_size=memory_size, state_shape=state_shape,
                                                 action_shape=self._env.action_space.shape, gamma=self._algo.gamma, nstep=self._algo.nstep, offline_buffer_size=offline_buffer_size)
        else:
            # Replay buffer with n-step return.
            self._replay_buffer = ReplayBuffer(memory_size=memory_size, state_shape=state_shape,
                                               action_shape=self._env.action_space.shape, gamma=self._algo.gamma, nstep=self._algo.nstep)
        
        self.segment_buffers = [
            ReplayBuffer(memory_size=1000, state_shape=state_shape,
                        action_shape=self._env.action_space.shape, gamma=self._algo.gamma, nstep=self._algo.nstep)
            for _ in range(7)]
        
        # Directory to log.
        self._log_dir = log_dir
        self._model_dir = os.path.join(log_dir, 'model')
        self._summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        if not os.path.exists(self._summary_dir):
            os.makedirs(self._summary_dir)

        self.episodes_stats = []
        self._steps = 0
        self._episodes = 0
        self._train_return = RunningMeanStats(log_interval)
        self._writer = SummaryWriter(log_dir=self._summary_dir)
        self._best_eval_score = -np.inf

        self._device = device
        self._num_steps = num_steps
        self._batch_size = batch_size
        self._update_interval = update_interval
        self._start_steps = start_steps
        self._log_interval = log_interval
        self._eval_interval = eval_interval
        self._num_eval_episodes = num_eval_episodes
        self._start_time = time.time()

        self.best_lap_time = np.inf
        self.best_reward = -np.inf

        logger.info(f'num_steps: {num_steps}')
        logger.info(f'batch_size: {batch_size}')
        logger.info(f'update_interval: {update_interval}')
        logger.info(f'start_steps: {start_steps}')
        logger.info(f'log_interval: {log_interval}')
        logger.info(f'eval_interval: {eval_interval}')
        logger.info(f'num_eval_episodes: {num_eval_episodes}')
        logger.info(f'seed: {seed}')
        logger.info(f'gamma: {self._algo.gamma}')
        logger.info(f'nstep: {self._algo.nstep}')
        logger.info(f'memory_size: {memory_size}')

        self.dataset_id = "hello/test-v3"
        self.dataset = None

        

    def save(self, path, save_buffer=True):
        self._algo.save_models(path)
        if save_buffer:
            with open(os.path.join(path, 'replay_buffer.pkl'), 'wb') as f:
                pickle.dump(self._replay_buffer, f)
            logger.info("saved replay buffer to {}".format(path))
        logger.info("saved models to {}".format(path))

    def load(self, path, load_buffer=True):
        self._algo.load_models(path)
        logger.info(f"loaded model from {path}")
        if load_buffer:
            with open(path + "replay_buffer.pkl", 'rb') as f:
                self._replay_buffer = pickle.load(f)
            self._steps = self._replay_buffer._n
            logger.info(f"loaded buffer from {path}. Number of steps: {len(self._replay_buffer)}")

    def run(self):
        try:
            while True:
                self.train_episode()
                if self._steps > self._num_steps:
                    break
                if self._eval_interval and (self._steps % self._eval_interval == 0):
                    logger.info("Evaluating")
                    self.evaluate()

        finally:
            if self.dataset is None:
                self.dataset = self._env.create_dataset(
                    dataset_id=self.dataset_id,
                    algorithm_name="Random-Policy",
                    code_permalink="https://github.com/Farama-Foundation/Minari",
                    author="Farama",
                    author_email="contact@farama.org"
                )
            else:
                self._env.add_to_dataset(self.dataset)
            # self.save(os.path.join(self._model_dir, 'final'), save_buffer=True)

    def update_model(self):####################################
        train_stats = None
        # Update online networks.
        if self._steps % self._update_interval == 0:
            batch = self._replay_buffer.sample(self._batch_size, self._device)
            train_stats = self._algo.update_online_networks(batch, self._writer)

        # Update target networks.
        self._algo.update_target_networks()
        return train_stats

    def train_episode(self):
        """
        Train only one episode
        """
        self._episodes += 1
        episode_return = 0.
        episode_steps = 0

        # ep_start_time = time.time()
        # ep_stats = {}
        # train_stats = None

        try:
            done = False
            step_perf, action_perf, update_model_perf = [], [], []
            state,_ = self._env.reset()
            step_start_time = time.perf_counter()

            while (not done):
                start_profile = time.perf_counter()
                if self._start_steps > self._steps:
                    action = self._env.action_space.sample()
                else:
                    action, _ = self._algo.explore(state)
                action_perf.append(time.perf_counter() - start_profile)

                # apply actions right away without blocking
                original_env = self._env.env 
                original_env = original_env.env
                # original_env.set_actions(action)
                # self._env.set_actions(action)##################################
                # update model·
                start_profile = time.perf_counter()
                if self._steps >= self._start_steps:
                    train_stats = self.update_model()
                update_model_perf.append(time.perf_counter() - start_profile)

                # get observations
                next_state, reward, done, truncated,info = self._env.step(action)  # action is already applied
                
                step_perf.append(time.perf_counter() - step_start_time)
                step_start_time = time.perf_counter()

                # Set done=True only when the agent fails, ignoring done signal
                # if the agent reach time horizons.
                if (episode_steps + 1 >= original_env.max_episode_steps):
                    masked_done = False
                else:
                    masked_done = done

                if info['terminated']:
                    rb_done = True
                else:
                    rb_done = False

                # original_env.read_segment()###############################
                # if original_env.segment_flag:
                #     self.segment_buffers[0].append(
                #         state, action, reward, next_state, masked_done,
                #         episode_done=rb_done)
                # if not original_env.segment_flag and self.segment_buffers[0]._n>0:
                #     if self.segment_buffers[original_env.segment_num]._n==0 or self.segment_buffers[original_env.segment_num]._n>self.segment_buffers[0]._n:
                #         self.segment_buffers[original_env.segment_num], self.segment_buffers[0] = self.segment_buffers[0], self.segment_buffers[original_env.segment_num]
                #     self.segment_buffers[0]._reset()

                
                self._replay_buffer.append(
                    state, action, reward, next_state, masked_done,
                    episode_done=rb_done)

                self._steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

                if self.checkpoint_freq and (self._steps % self.checkpoint_freq == 0):
                    logger.info(f"checkpointing model {self._steps} steps")
                    self.save(os.path.join(self._model_dir, "checkpoints", f"step_{self._steps:08d}"), save_buffer=False)
                #     self.save_remake(os.path.join(self._model_dir, "checkpoints", f"step_{self._steps:08d}"), save_buffer=True)

            # if self.checkpoint_ep_freq and (self._episodes % self.checkpoint_ep_freq == 0):
            #     ckpt_dir = os.path.join(self._model_dir, "checkpoints", f"ep_{self._episodes:08d}")
            #     os.makedirs(ckpt_dir, exist_ok=True)
            #     logger.info(f"checkpointing model at episode {self._episodes} -> {ckpt_dir}")
            #     self.save(ckpt_dir, save_buffer=False)

        except TimeoutError:
            logger.exception("Agent TimeoutError")
        finally:
            pass
            # env_ep_stats = self._env.close()

        # We log running mean of training rewards.
        # self._train_return.append(episode_return)

        # if self._episodes % self._log_interval == 0:
        #     self._writer.add_scalar(
        #         'reward/train', self._train_return.get(), self._steps)

        # print(f'Episode: {self._episodes:<4}  '
        #       f'Episode steps: {episode_steps:<4}  '
        #       f'Return: {episode_return:<5.1f}')

        # ep_time = time.time() - ep_start_time
        # ep_stats['total_steps'] = self._steps
        # ep_stats['episode'] = self._episodes
        # ep_stats['ep_reward'] = episode_return
        # ep_stats['ep_steps'] = episode_steps
        # ep_stats.update(env_ep_stats if isinstance(env_ep_stats, dict) else {})

        # # if env_ep_stats["BestLap"] < self.best_lap_time:
        # #     logger.info(f"new best lap time {env_ep_stats['BestLap']}")
        # #     self.best_lap_time = env_ep_stats["BestLap"]
        # #     self.save(os.path.join(self._model_dir, 'best_lap_time'), save_buffer=False)

        # # if env_ep_stats["ep_reward"] > self.best_reward:
        # #     logger.info(f"new best reward {env_ep_stats['ep_reward']}")
        # #     self.best_reward = env_ep_stats["ep_reward"]
        # #     self.save(os.path.join(self._model_dir, 'best_reward'), save_buffer=False)

        # eval_metrics = self.common_metrics()
        # eval_metrics.update(ep_stats)
        # if train_stats:
        #     eval_metrics.update(train_stats)
        # eval_metrics["update_model_perf_mean"] = np.array(update_model_perf).mean()
        # eval_metrics["update_model_perf_max"] = np.array(update_model_perf).max()
        # eval_metrics["update_model_perf_std"] = np.array(update_model_perf).std()
        # eval_metrics["step_perf_mean"] = np.array(step_perf).mean()
        # eval_metrics["step_perf_max"] = np.array(step_perf).max()
        # eval_metrics["step_perf_std"] = np.array(step_perf).std()
        # eval_metrics["step_perf_q99"] = np.quantile(np.array(step_perf), 0.99)
        # eval_metrics["step_perf_> thres"] = np.sum(np.array(step_perf) > 0.041)
        # eval_metrics["action_perf_mean"] = np.array(action_perf).mean()
        # eval_metrics["action_perf_max"] = np.array(action_perf).max()
        # eval_metrics["action_perf_std"] = np.array(action_perf).std()
        # logger.info(f"Avr step time: {eval_metrics['step_perf_mean']:.3f}s, actions: {eval_metrics['action_perf_mean']:.4f}s, update: {eval_metrics['update_model_perf_mean']:.3f}s")
        # logger.info(f"Max step time: {eval_metrics['step_perf_max']:.3f}s, actions: {eval_metrics['action_perf_max']:.4f}s, update: {eval_metrics['update_model_perf_max']:.3f}s")
        # logger.info(f"std step time: {eval_metrics['step_perf_std']:.3f}s, actions: {eval_metrics['action_perf_std']:.4f}s, update: {eval_metrics['update_model_perf_std']:.3f}s")
        # logger.info(f"step_perf_> thres: {eval_metrics['step_perf_> thres']} / {len(step_perf)}")
        # if self.wandb_logger:
        #     self.wandb_logger.log(eval_metrics, 'episodes')
        # self.episodes_stats.append(eval_metrics)
        # pd.DataFrame(self.episodes_stats).to_csv(os.path.join(self._log_dir, 'summary.csv'), index=None)
        # logger.info(f'Episode done. Took {ep_time:.2f}s.  Steps per episode: {episode_steps}. Buffer size: {len(self._replay_buffer)} fps: {episode_steps/ep_time:.2f}')

    def train_segment_episode(self,lapdist_start=None,lapdist_end=None,):
        """
        Train only one episode
        """
        self._episodes += 1
        episode_return = 0.
        episode_steps = 0

        state_all=self._env.env.env.client.step_sim()
        Lapdist=self._env.env.env.track_length*state_all["NormalizedSplinePosition"]

        d1=(lapdist_start-Lapdist) % self._env.env.env.track_length
        d2=(Lapdist-lapdist_start) % self._env.env.env.track_length
        if d1>d2:
            backward=1
        else:
            backward=0
        if backward:
            self._env.env.env.client.controls.local_controls.vj.setButton(2,1)
            time.sleep(0.2)
            self._env.env.env.client.controls.local_controls.vj.setButton(2,0)

        while abs(Lapdist-lapdist_start)>30:
            if state_all['numberOfTyresOut'] > 2.:
                self._env.env.env.client.reset()
                
                if backward:
                    self._env.env.env.client.controls.local_controls.vj.setButton(2,1)
                    time.sleep(0.2)
                    self._env.env.env.client.controls.local_controls.vj.setButton(2,0)

            self._env.env.env.client.controls.set_controls(steer=0, acc=.5, brake=0.)
            self._env.env.env.client.respond_to_server()
            state_all=self._env.env.env.client.step_sim()
            Lapdist=self._env.env.env.track_length*state_all["NormalizedSplinePosition"]
            print(Lapdist)

        self._env.env.env.client.reset()

        try:
            done = False
            step_perf, action_perf, update_model_perf = [], [], []
            state,_ = self._env.reset()

            step_start_time = time.perf_counter()

            while (not done):
                start_profile = time.perf_counter()
                if self._start_steps > self._steps:
                    action = self._env.action_space.sample()
                else:
                    action, _ = self._algo.explore(state)
                action_perf.append(time.perf_counter() - start_profile)

                # apply actions right away without blocking
                original_env = self._env.env 
                original_env = original_env.env
                # original_env.set_actions(action)
                # self._env.set_actions(action)##################################

                # update model
                start_profile = time.perf_counter()
                if self._steps >= self._start_steps:
                    train_stats = self.update_model()
                update_model_perf.append(time.perf_counter() - start_profile)

                # get observations
                next_state, reward, done, truncated,info = self._env.step(action)  # action is already applied
                
                step_perf.append(time.perf_counter() - step_start_time)
                step_start_time = time.perf_counter()

                # Set done=True only when the agent fails, ignoring done signal
                # if the agent reach time horizons.
                if (episode_steps + 1 >= original_env.max_episode_steps):
                    masked_done = False
                else:
                    masked_done = done

                if info['terminated']:
                    rb_done = True
                else:
                    rb_done = False

                
                self._replay_buffer.append(
                    state, action, reward, next_state, masked_done,
                    episode_done=rb_done)

                self._steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

                if self.checkpoint_freq and (self._steps % self.checkpoint_freq == 0):
                    logger.info(f"checkpointing model {self._steps} steps")
                    self.save(os.path.join(self._model_dir, "checkpoints", f"step_{self._steps:08d}"), save_buffer=False)
        except TimeoutError:
            logger.exception("Agent TimeoutError")
        finally:
            pass

    def evaluate(self):
        try:
            total_return = 0.0
            for _ in range(self._num_eval_episodes):
                state = self._test_env.reset()
                episode_return = 0.0
                done = False

                while (not done):
                    action, entropies = self._algo.exploit(state)
                    next_state, reward, done, info = self._test_env.step(action)
                    self._test_env.states[-1]["entropies"] = entropies.cpu().numpy().item()
                    episode_return += reward
                    state = next_state
                total_return += episode_return
        except TimeoutError:
            logger.exception("Agent TimeoutError")
        finally:
            env_ep_stats = self._env.close()
            pd.DataFrame([env_ep_stats]).to_csv(os.path.join(self._log_dir, 'eval_summary.csv'), index=None)

    def __del__(self):
        self._env.close()
        self._test_env.close()
        self._writer.close()
        if self.wandb_logger:
            self.wandb_logger.finish()

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._steps,
            episode=self._episodes,
            buffer_size=len(self._replay_buffer),
            total_time=time.time() - self._start_time,
        )

    def load_pre_train_data(self, trajs_path, env):
        brake_map_file = Path(self._env.ac_configs_path) / "cars" / self._env.config.car / 'brake_map.csv'
        steer_map_file = Path(self._env.ac_configs_path) / "cars" / self._env.config.car / 'steer_map.csv'

        total_added_episodes = 0

        env_data = DataLoader(env, trajs_path, steer_map_file, brake_map_file)
        for ep in tqdm(range(env_data.trajectories_count)[:]):
            state = env_data.reset()

            total_added_episodes += 1
            for i in range(len(env_data.trajectory) - 1):
                action = env_data.act()
                next_state, reward, done, info = env_data.step(action)

                if info['terminated']:
                    terminated = True
                else:
                    terminated = False

                # end of trajectory
                if i >= len(env_data.trajectory) - 2:
                    episode_done = True # will add done as zero to the RB
                else:
                    episode_done = False # use the termination signal from the environment

                self._replay_buffer.append(state, action, reward, next_state, terminated=terminated, episode_done=episode_done)
                state = next_state
                if episode_done:
                    break
        logger.info(f"Loaded {trajs_path} Buffer size: {len(self._replay_buffer)}")

    def pre_train(self):
        self._algo.update_entropy = False
        logger.info("Pre-training...")
        for _ in tqdm(range(self._replay_buffer._n)):
            self.update_model()
        self._algo.update_entropy = True

    def _capture_rng_states(self):
        return {
            "py": random.getstate(),
            "np": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def _restore_rng_states(self, rng):
        if rng.get("py") is not None: random.setstate(rng["py"])
        if rng.get("np") is not None: np.random.set_state(rng["np"])
        if rng.get("torch") is not None: torch.set_rng_state(rng["torch"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
    
    def save_remake(self, path, save_buffer=False):
        os.makedirs(path, exist_ok=True)
        self._algo.save_checkpoint(path, "checkpoint.pt")

        training_state = {
            "steps": getattr(self, "_steps", 0),
            "episodes": getattr(self, "_episodes", 0),
            "start_steps": getattr(self, "_start_steps", 0),
            "best_eval": getattr(self, "_best_eval", None),
            "rng": self._capture_rng_states(),
        }
        tmp_ts = os.path.join(path, "training_state.tmp.pt")
        ts_path = os.path.join(path, "training_state.pt")
        torch.save(training_state, tmp_ts)
        os.replace(tmp_ts, ts_path)

        if save_buffer and hasattr(self, "_replay_buffer"):
            tmp_rb = os.path.join(path, "replay_buffer.tmp.pkl")
            rb_path = os.path.join(path, "replay_buffer.pkl")
            with open(tmp_rb, "wb") as f:
                pickle.dump(self._replay_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_rb, rb_path) 

            seg_path = os.path.join(path, "segment_buffer.pkl")
            with open(seg_path, "wb") as f:
                pickle.dump(self.segment_buffers, f, protocol=pickle.HIGHEST_PROTOCOL)


    def load_remake(self, path, load_buffer=True, map_location=None):
        import os, pickle, torch
        self._algo.load_checkpoint(path, "checkpoint.pt", map_location=map_location)

        # 2) 训练进度 + RNG
        ts_path = os.path.join(path, "training_state.pt")

        if os.path.exists(ts_path) and load_buffer:
            tr = torch.load(ts_path, map_location="cpu", weights_only=False)
            self._steps = int(tr.get("steps", 0))
            self._episodes = int(tr.get("episodes", 0))
            self._start_steps = int(tr.get("start_steps", 0))
            self._best_eval = tr.get("best_eval", None)
            rng = tr.get("rng", {})
            self._restore_rng_states(rng)

        # 3) Replay Buffer
        if load_buffer:
            rb_path = os.path.join(path, "replay_buffer.pkl")
            if os.path.exists(rb_path):
                with open(rb_path, "rb") as f:
                    self._replay_buffer = pickle.load(f)
                    self._replay_buffer._n = len(self._replay_buffer)
                # 可用 buffer 长度校正步数（按需）
                self._steps = max(getattr(self, "_steps", 0), len(self._replay_buffer))
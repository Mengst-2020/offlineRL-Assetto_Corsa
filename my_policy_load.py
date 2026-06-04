import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import argparse
from typing import Dict, List, Union, Tuple, Optional, Callable
from copy import deepcopy
import gymnasium as gym
from gymnasium.spaces import Box
import os.path as path
import os
import deepdish as dd
from collections import defaultdict
from operator import itemgetter
import cv2
import matplotlib.pyplot as plt

class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(125,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # 改成连续动作
        self.obs_shape = (125,)
        self.action_dim = 3
        self.max_action = 1.0
        self.action_shape =(3,)
        self.action_space = Box(low=np.array([-1.0,  -1.0,  -1.0]), high=np.array([1.0,  1.0,  1.0]))

    def reset(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        done = True
        info = {}
        return obs, reward, done, info

def get_mydataset(Path=None, image_flag=False, **kwargs):
    # dataset_all = dd.io.load(Path+'main_data.hdf5')
    # episode_N = len(dataset_all)
    # # N = ldataset_all.keys()
    # N = dataset_all["episode_1"]['observations']['image'][1]
    # plt.imshow(cv2.cvtColor(N, cv2.COLOR_BGR2RGB))
    # plt.show()

    action_abs_flag=False #是否使用绝对动作

    if isinstance(Path, str):
        paths = [Path]
    elif isinstance(Path, (list, tuple)):
        paths = list(Path)
    else:
        paths = [
        "mydata/barcelona/sac-v2/data/",
        "mydata/monza/sac-v2/data/",
        "mydata/redbull/sac-v2/data/",
        ]

    obs_ = []
    next_obs_ = []
    action_ = []
    action_abs_ = []
    reward_ = []
    done_ = []
    info_=[]

    ctrl_rate=25
    controls_rate_limit = np.array([[-600/180, 600/180],
                                    [-1200/100, 1200/100], # the first is the falling edge of the pedal -> checked both
                                    [-1200/100, 1200/100],  # the first is the brake release; the second the brake press;
                                    ]) * (1/ctrl_rate)
    max_delta_action = Box(low=controls_rate_limit[:,0], high=controls_rate_limit[:,1])
    controls_min_values = np.array([-1.0,  -1.0,  -1.0], dtype=np.float32)
    controls_max_values = np.array([ 1.0,   1.0,   1.0], dtype=np.float32)

    # episode_step = 0
    for base_path in paths:
        dataset_all = dd.io.load(base_path+'main_data.hdf5')
        episode_N = len(dataset_all)

        if image_flag:
            N = dataset_all["episode_0"]['observations']['image'][500]
            dataset_all.key()
            plt.imshow(cv2.cvtColor(N, cv2.COLOR_BGR2RGB))
            plt.show()
        
        for i in range(episode_N -1):
            episode_dataset=dataset_all["episode_"+ str(i)]
            episode_step_sum=episode_dataset['rewards'].shape[0]
            if episode_step_sum<1000:
                continue
            elif episode_dataset['infos']['terminated'][0]:
                episode_step_sum=episode_step_sum-250
            current_action = np.array([0.0, -1.0, -1.0], dtype=np.float32)
            for step in range(episode_step_sum-1):
                if image_flag:
                    obs = episode_dataset['observations']['state'][step].astype(np.float32)
                    new_obs = episode_dataset['observations']['state'][step+1].astype(np.float32)
                else:
                    obs = episode_dataset['observations'][step].astype(np.float32)
                    new_obs = episode_dataset['observations'][step+1].astype(np.float32)

                action = episode_dataset['actions'][step].astype(np.float32)

                current_action += action * max_delta_action.high
                current_action=np.clip(current_action, controls_min_values, controls_max_values)

                # reward = episode_dataset['rewards'][step].astype(np.float32)
                reward = obs[0]*80*3.6*(1-obs[1]*10./12.)
                done_bool = bool(episode_dataset['terminations'][step])
                info=episode_dataset['infos']['type'][0][step]

                obs_.append(obs)
                next_obs_.append(new_obs)
                action_.append(action)
                action_abs_.append(current_action.copy())
                reward_.append(reward)
                done_.append(done_bool)
                info_.append(info)

    if action_abs_flag:
        return {
            'observations': np.array(obs_),
            'actions_delta': np.array(action_),
            'actions': np.array(action_abs_),
            'next_observations': np.array(next_obs_),
            'rewards': np.array(reward_),
            'terminals': np.array(done_),
            'info': np.array(info_),
        }
    else:
        return {
            'observations': np.array(obs_),
            'actions': np.array(action_),
            'actions_abs': np.array(action_abs_),
            'next_observations': np.array(next_obs_),
            'rewards': np.array(reward_),
            'terminals': np.array(done_),
            'info': np.array(info_),
        }

class GaussianNoise:
    """The vanilla Gaussian process, for exploration in DDPG by default."""

    def __init__(self, mu=0.0, sigma=1.0):
        self._mu = mu
        assert 0 <= sigma, "Noise std should not be negative."
        self._sigma = sigma

    def __call__(self, size):
        return np.random.normal(self._mu, self._sigma, size)

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation()]
            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [nn.Linear(hidden_dims[-1], output_dim)]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ActorProb(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        dist_net: nn.Module,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        self.dist_net = dist_net.to(device)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Normal:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist

class Actor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        action_dim: int,
        max_action: float = 1.0,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        output_dim = action_dim
        self.last = nn.Linear(latent_dim, output_dim).to(device)
        self._max = max_action

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        actions = self._max * torch.tanh(self.last(logits))
        return actions

class Critic(nn.Module):
    def __init__(self, backbone: nn.Module, device: str = "cpu") -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        self.last = nn.Linear(latent_dim, 1).to(device)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
            obs = torch.cat([obs, actions], dim=1)
        logits = self.backbone(obs)
        values = self.last(logits)
        return values

class NormalWrapper(torch.distributions.Normal):
    def log_prob(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

class TanhNormalWrapper(torch.distributions.Normal):
    def __init__(self, loc, scale, max_action):
        super().__init__(loc, scale)
        self._max_action = max_action

    def log_prob(self, action, raw_action=None):
        squashed_action = action/self._max_action
        if raw_action is None:
            raw_action = self.arctanh(squashed_action)
        log_prob = super().log_prob(raw_action).sum(-1, keepdim=True)
        eps = 1e-6
        log_prob = log_prob - torch.log(self._max_action*(1 - squashed_action.pow(2)) + eps).sum(-1, keepdim=True)
        return log_prob

    def mode(self):
        raw_action = self.mean
        action = self._max_action * torch.tanh(self.mean)
        return action, raw_action

    def arctanh(self, x):
        one_plus_x = (1 + x).clamp(min=1e-6)
        one_minus_x = (1 - x).clamp(min=1e-6)
        return 0.5 * torch.log(one_plus_x / one_minus_x)

    def rsample(self):
        raw_action = super().rsample()
        action = self._max_action * torch.tanh(raw_action)
        return action, raw_action

class DiagGaussian(nn.Module):
    def __init__(
        self,
        latent_dim,
        output_dim,
        unbounded=False,
        conditioned_sigma=False,
        max_mu=1.0,
        sigma_min=-5.0,
        sigma_max=2.0
    ):
        super().__init__()
        self.mu = nn.Linear(latent_dim, output_dim)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = nn.Linear(latent_dim, output_dim)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))
        self._unbounded = unbounded
        self._max = max_mu
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def forward(self, logits):
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=self._sigma_min, max=self._sigma_max).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return NormalWrapper(mu, sigma)

class TanhDiagGaussian(DiagGaussian):
    def __init__(
        self,
        latent_dim,
        output_dim,
        unbounded=False,
        conditioned_sigma=False,
        max_mu=1.0,
        sigma_min=-5.0,
        sigma_max=2.0
    ):
        super().__init__(
            latent_dim=latent_dim,
            output_dim=output_dim,
            unbounded=unbounded,
            conditioned_sigma=conditioned_sigma,
            max_mu=max_mu,
            sigma_min=sigma_min,
            sigma_max=sigma_max
        )

    def forward(self, logits):
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=self._sigma_min, max=self._sigma_max).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return TanhNormalWrapper(mu, sigma, self._max)

class StandardScaler(object):
    def __init__(self, mu=None, std=None):
        self.mu = mu
        self.std = std

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu
    
    def save_scaler(self, save_path):
        mu_path = path.join(save_path, "mu.npy")
        std_path = path.join(save_path, "std.npy")
        np.save(mu_path, self.mu)
        np.save(std_path, self.std)
    
    def load_scaler(self, load_path):
        mu_path = path.join(load_path, "mu.npy")
        std_path = path.join(load_path, "std.npy")
        self.mu = np.load(mu_path)
        self.std = np.load(std_path)

    def transform_tensor(self, data: torch.Tensor):
        device = data.device
        data = self.transform(data.cpu().numpy())
        data = torch.tensor(data, device=device)
        return data

class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations)
     
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
        }
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }

class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x

def soft_clamp(
    x : torch.Tensor,
    _min: Optional[torch.Tensor] = None,
    _max: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

class EnsembleLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        weight_decay: float = 0.0
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble

        self.register_parameter("weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))

        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))

        self.register_parameter("saved_weight", nn.Parameter(self.weight.detach().clone()))
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias

        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias

        return x

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = self.weight_decay * (0.5*((self.weight**2).sum()))
        return decay_loss

class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        num_ensemble: int = 7,
        num_elites: int = 5,
        activation: nn.Module = Swish,
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        with_reward: bool = True,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_reward = with_reward
        self.device = torch.device(device)

        self.activation = activation()

        assert len(weight_decays) == (len(hidden_dims) + 1)

        module_list = []
        hidden_dims = [obs_dim+action_dim] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
        for in_dim, out_dim, weight_decay in zip(hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1]):
            module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
        self.backbones = nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(
            hidden_dims[-1],
            2 * (obs_dim + self._with_reward),
            num_ensemble,
            weight_decays[-1]
        )

        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * -10, requires_grad=True)
        )

        self.register_parameter(
            "elites",
            nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False)
        )

        self.to(self.device)

    def forward(self, obs_action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.as_tensor(obs_action, dtype=torch.float32).to(self.device)
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.backbones:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
    
    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs

class BaseDynamics(object):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer
    ) -> None:
        super().__init__()
        self.model = model
        self.optim = optim
    
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        raise NotImplementedError


class EnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric"
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode

    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += obs
        std = np.sqrt(np.exp(logvar))

        ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)

        # choose one model from ensemble
        num_models, batch_size, _ = ensemble_samples.shape
        model_idxs = self.model.random_elite_idxs(batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        
        next_obs = samples[..., :-1]
        reward = samples[..., -1:]
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward

        if self._penalty_coef:
            if self._uncertainty_mode == "aleatoric":
                penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
            elif self._uncertainty_mode == "pairwise-diff":
                next_obses_mean = mean[..., :-1]
                next_obs_mean = np.mean(next_obses_mean, axis=0)
                diff = next_obses_mean - next_obs_mean
                penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
            elif self._uncertainty_mode == "ensemble_std":
                next_obses_mean = mean[..., :-1]
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
            else:
                raise ValueError
            penalty = np.expand_dims(penalty, 1).astype(np.float32)
            assert penalty.shape == reward.shape
            reward = reward - self._penalty_coef * penalty
            info["penalty"] = penalty
        
        return next_obs, reward, terminal, info
    
    @ torch.no_grad()
    def sample_next_obss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, action], dim=-1)
        obs_act = self.scaler.transform_tensor(obs_act)
        mean, logvar = self.model(obs_act)
        mean[..., :-1] += obs
        std = torch.sqrt(torch.exp(logvar))

        mean = mean[self.model.elites.data.cpu().numpy()]
        std = std[self.model.elites.data.cpu().numpy()]

        samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
        next_obss = samples[..., :-1]
        return next_obss

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        return inputs, targets
 
    def learn(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            
            mean, logvar = self.model(inputs_batch)
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.model.get_decay_loss()
            loss = loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)
    
    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)


class BasePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def train() -> None:
        raise NotImplementedError
    
    def eval() -> None:
        raise NotImplementedError
    
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        raise NotImplementedError
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        raise NotImplementedError

class IQLPolicy(BasePolicy):
    """
    Implicit Q-Learning <Ref: https://arxiv.org/abs/2110.06169>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic_q1: nn.Module,
        critic_q2: nn.Module,
        critic_v: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic_q1_optim: torch.optim.Optimizer,
        critic_q2_optim: torch.optim.Optimizer,
        critic_v_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Space,
        tau: float = 0.005,
        gamma: float  = 0.99,
        expectile: float = 0.8,
        temperature: float = 0.1
    ) -> None:
        super().__init__()

        self.actor = actor
        self.critic_q1, self.critic_q1_old = critic_q1, deepcopy(critic_q1)
        self.critic_q1_old.eval()
        self.critic_q2, self.critic_q2_old = critic_q2, deepcopy(critic_q2)
        self.critic_q2_old.eval()
        self.critic_v = critic_v

        self.actor_optim = actor_optim
        self.critic_q1_optim = critic_q1_optim
        self.critic_q2_optim = critic_q2_optim
        self.critic_v_optim = critic_v_optim

        self.action_space = action_space
        self._tau = tau
        self._gamma = gamma
        self._expectile = expectile
        self._temperature = temperature

    def train(self) -> None:
        self.actor.train()
        self.critic_q1.train()
        self.critic_q2.train()
        self.critic_v.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic_q1.eval()
        self.critic_q2.eval()
        self.critic_v.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic_q1_old.parameters(), self.critic_q1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic_q2_old.parameters(), self.critic_q2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        with torch.no_grad():
            dist = self.actor(obs)
            if deterministic:
                action = dist.mode().cpu().numpy()
            else:
                action = dist.sample().cpu().numpy()
        action = np.clip(action, self.action_space.low[0], self.action_space.high[0])
        return action
    
    def _expectile_regression(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self._expectile, (1 - self._expectile))
        return weight * (diff**2)
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        
        # update value net
        with torch.no_grad():
            q1, q2 = self.critic_q1_old(obss, actions), self.critic_q2_old(obss, actions)
            q = torch.min(q1, q2)
        v = self.critic_v(obss)
        critic_v_loss = self._expectile_regression(q-v).mean()
        self.critic_v_optim.zero_grad()
        critic_v_loss.backward()
        self.critic_v_optim.step()

        # update critic
        q1, q2 = self.critic_q1(obss, actions), self.critic_q2(obss, actions)
        with torch.no_grad():
            next_v = self.critic_v(next_obss)
            target_q = rewards + self._gamma * (1 - terminals) * next_v
        
        critic_q1_loss = ((q1 - target_q).pow(2)).mean()
        critic_q2_loss = ((q2 - target_q).pow(2)).mean()

        self.critic_q1_optim.zero_grad()
        critic_q1_loss.backward()
        self.critic_q1_optim.step()

        self.critic_q2_optim.zero_grad()
        critic_q2_loss.backward()
        self.critic_q2_optim.step()

        # update actor
        with torch.no_grad():
            q1, q2 = self.critic_q1_old(obss, actions), self.critic_q2_old(obss, actions)
            q = torch.min(q1, q2)
            v = self.critic_v(obss)
            exp_a = torch.exp((q - v) * self._temperature)
            exp_a = torch.clip(exp_a, None, 100.0)
        dist = self.actor(obss)
        log_probs = dist.log_prob(actions)
        actor_loss = -(exp_a * log_probs).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self._sync_weight()

        return {
            "loss/actor": actor_loss.item(),
            "loss/q1": critic_q1_loss.item(),
            "loss/q2": critic_q2_loss.item(),
            "loss/v": critic_v_loss.item()
        }

class BCPolicy(BasePolicy):

    def __init__(
        self,
        actor: nn.Module,
        actor_optim: torch.optim.Optimizer
    ) -> None:

        super().__init__()
        self.actor = actor
        self.actor_optim = actor_optim
    
    def train(self) -> None:
        self.actor.train()

    def eval(self) -> None:
        self.actor.eval()
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()
        return action
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions = batch["observations"], batch["actions"]
        
        a = self.actor(obss)
        actor_loss = ((a - actions).pow(2)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return {
            "loss/actor": actor_loss.item()
        }

class TD3Policy(BasePolicy):
    """
    Twin Delayed Deep Deterministic policy gradient <Ref: https://arxiv.org/abs/1802.09477>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        max_action: float = 1.0,
        exploration_noise: Callable = GaussianNoise,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        update_actor_freq: int = 2,
    ) -> None:
        super().__init__()

        self.actor = actor
        self.actor_old = deepcopy(actor)
        self.actor_old.eval()
        self.actor_optim = actor_optim

        self.critic1 = critic1
        self.critic1_old = deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim

        self.critic2 = critic2
        self.critic2_old = deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim

        self._tau = tau
        self._gamma = gamma

        self._max_action = max_action
        self.exploration_noise = exploration_noise
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        self._freq = update_actor_freq

        self._cnt = 0
        self._last_actor_loss = 0
    
    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()
        if not deterministic:
            action = action + self.exploration_noise(action.shape)
            action = np.clip(action, -self._max_action, self._max_action)
        return action
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        
        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            next_actions = (self.actor_old(next_obss) + noise).clamp(-self._max_action, self._max_action)
            next_q = torch.min(self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions))
            target_q = rewards + self._gamma * (1 - terminals) * next_q
        
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        if self._cnt % self._freq == 0:
            a = self.actor(obss)
            q = self.critic1(obss, a)
            actor_loss = -q.mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self._last_actor_loss = actor_loss.item()
            self._sync_weight()
        
        self._cnt += 1

        return {
            "loss/actor": self._last_actor_loss,
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

class TD3BCPolicy(TD3Policy):
    """
    TD3+BC <Ref: https://arxiv.org/abs/2106.06860>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        max_action: float = 1.0,
        exploration_noise: Callable = GaussianNoise,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        update_actor_freq: int = 2,
        alpha: float = 2.5,
        scaler: StandardScaler = None
    ) -> None:

        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            max_action=max_action,
            exploration_noise=exploration_noise,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            update_actor_freq=update_actor_freq
        )

        self._alpha = alpha
        self.scaler = scaler
    
    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.scaler is not None:
            obs = self.scaler.transform(obs)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()
        if not deterministic:
            action = action + self.exploration_noise(action.shape)
            action = np.clip(action, -self._max_action, self._max_action)
        return action
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]
        
        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)
            next_actions = (self.actor_old(next_obss) + noise).clamp(-self._max_action, self._max_action)
            next_q = torch.min(self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions))
            target_q = rewards + self._gamma * (1 - terminals) * next_q
        
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        if self._cnt % self._freq == 0:
            a = self.actor(obss)
            q = self.critic1(obss, a)
            lmbda = self._alpha / q.abs().mean().detach()
            actor_loss = -lmbda * q.mean() + ((a - actions).pow(2)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self._last_actor_loss = actor_loss.item()
            self._sync_weight()
        
        self._cnt += 1

        return {
            "loss/actor": self._last_actor_loss,
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

class SACPolicy(BasePolicy):
    """
    Soft Actor Critic <Ref: https://arxiv.org/abs/1801.01290>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2
    ) -> None:
        super().__init__()

        self.actor = actor
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()

        self.actor_optim = actor_optim
        self.critic1_optim = critic1_optim
        self.critic2_optim = critic2_optim

        self._tau = tau
        self._gamma = gamma

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def actforward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        return squashed_action, log_prob

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            action, _ = self.actforward(obs, deterministic)
        return action.cpu().numpy()

    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]

        # update critic
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obss)
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions)
            ) - self._alpha * next_log_probs
            target_q = rewards + self._gamma * (1 - terminals) * next_q

        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)

        actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result

class MOPOPolicy(SACPolicy):
    """
    Model-based Offline Policy Optimization <Ref: https://arxiv.org/abs/2005.13239>
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self.dynamics = dynamics

    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            actions = self.select_action(observations)
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
            {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}
        return super().learn(mix_batch)

class RAMBOPolicy(MOPOPolicy):
    """
    RAMBO-RL: Robust Adversarial Model-Based Offline Reinforcement Learning <Ref: https://arxiv.org/abs/2204.12581>
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        dynamics_adv_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        adv_weight: float = 0,
        adv_train_steps: int = 1000,
        adv_rollout_batch_size: int = 256,
        adv_rollout_length: int = 5,
        include_ent_in_adv: bool = False,
        scaler: StandardScaler = None,
        device="cpu"
    ) -> None:
        super().__init__(
            dynamics, 
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self._dynmics_adv_optim = dynamics_adv_optim
        self._adv_weight = adv_weight
        self._adv_train_steps = adv_train_steps
        self._adv_rollout_batch_size = adv_rollout_batch_size
        self._adv_rollout_length = adv_rollout_length
        self._include_ent_in_adv = include_ent_in_adv
        self.scaler = scaler
        self.device = device
        
    def load(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "rambo_pretrain.pth"), map_location=self.device))

    def pretrain(self, data: Dict, n_epoch, batch_size, lr, logger) -> None:
        self._bc_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        observations = data["observations"]
        actions = data["actions"]
        sample_num = observations.shape[0]
        idxs = np.arange(sample_num)

        logger.log("Pretraining policy")
        self.actor.train()
        for i_epoch in range(n_epoch):
            np.random.shuffle(idxs)
            sum_loss = 0
            for i_batch in range(sample_num // batch_size):
                batch_obs = observations[i_batch * batch_size: (i_batch + 1) * batch_size]
                batch_act = actions[i_batch * batch_size: (i_batch + 1) * batch_size]
                batch_obs = torch.from_numpy(batch_obs).to(self.device)
                batch_act = torch.from_numpy(batch_act).to(self.device)
                dist = self.actor(batch_obs)
                pred_actions, _ = dist.rsample()
                bc_loss = ((pred_actions - batch_act) ** 2).mean()

                self._bc_optim.zero_grad()
                bc_loss.backward()
                self._bc_optim.step()
                sum_loss += bc_loss.cpu().item()
            print(f"Epoch {i_epoch}, mean bc loss {sum_loss/i_batch}")
        torch.save(self.state_dict(), os.path.join(logger.model_dir, "rambo_pretrain.pth"))

    def update_dynamics(
        self, 
        real_buffer, 
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        all_loss_info = {
            "adv_dynamics_update/all_loss": 0, 
            "adv_dynamics_update/sl_loss": 0, 
            "adv_dynamics_update/adv_loss": 0, 
            "adv_dynamics_update/adv_advantage": 0, 
            "adv_dynamics_update/adv_log_prob": 0, 
        }
        self.dynamics.model.train()
        steps = 0
        while steps < self._adv_train_steps:
            init_obss = real_buffer.sample(self._adv_rollout_batch_size)["observations"].cpu().numpy()
            observations = init_obss
            for t in range(self._adv_rollout_length):
                actions = super().select_action(observations)
                sl_observations, sl_actions, sl_next_observations, sl_rewards = \
                    itemgetter("observations", "actions", "next_observations", "rewards")(real_buffer.sample(self._adv_rollout_batch_size))
                next_observations, terminals, loss_info = self.dynamics_step_and_forward(observations, actions, sl_observations, sl_actions, sl_next_observations, sl_rewards)
                for _key in loss_info:
                    all_loss_info[_key] += loss_info[_key]
                # nonterm_mask = (~terminals).flatten()
                steps += 1
                # observations = next_observations[nonterm_mask]
                observations = next_observations.copy()
                # if nonterm_mask.sum() == 0:
                    # break
                if steps == 1000:
                    break
        self.dynamics.model.eval()
        return {_key: _value/steps for _key, _value in all_loss_info.items()}

    def dynamics_step_and_forward(
        self,
        observations,
        actions,
        sl_observations,
        sl_actions,
        sl_next_observations,
        sl_rewards,
    ):
        obs_act = np.concatenate([observations, actions], axis=-1)
        obs_act = self.dynamics.scaler.transform(obs_act)
        diff_mean, logvar = self.dynamics.model(obs_act)
        observations = torch.from_numpy(observations).to(diff_mean.device)
        diff_obs, diff_reward = torch.split(diff_mean, [diff_mean.shape[-1]-1, 1], dim=-1)
        mean = torch.cat([diff_obs + observations, diff_reward], dim=-1)
        std = torch.sqrt(torch.exp(logvar))
        
        dist = torch.distributions.Normal(mean, std)
        ensemble_sample = dist.sample()
        ensemble_size, batch_size, _ = ensemble_sample.shape
        
        # select the next observations
        selected_indexes = self.dynamics.model.random_elite_idxs(batch_size)
        sample = ensemble_sample[selected_indexes, np.arange(batch_size)]
        next_observations = sample[..., :-1]
        rewards = sample[..., -1:]
        terminals = self.dynamics.terminal_fn(observations.detach().cpu().numpy(), actions, next_observations.detach().cpu().numpy())

        # compute logprob
        log_prob = dist.log_prob(sample).sum(-1, keepdim=True)
        log_prob = log_prob[self.dynamics.model.elites.data, ...]
        prob = log_prob.double().exp()
        prob = prob * (1/len(self.dynamics.model.elites.data))
        log_prob = prob.sum(0).log().type(torch.float32)

        # compute the advantage
        with torch.no_grad():
            next_actions, next_policy_log_prob = self.actforward(next_observations, deterministic=True)
            next_q = torch.minimum(
                self.critic1(next_observations, next_actions), 
                self.critic2(next_observations, next_actions)
            )
            if self._include_ent_in_adv:
                next_q = next_q - self._alpha * next_policy_log_prob

            value = rewards + (1-torch.from_numpy(terminals).to(mean.device).float()) * self._gamma * next_q

            value_baseline = torch.minimum(
                self.critic1(observations, actions), 
                self.critic2(observations, actions)
            )
            advantage = value - value_baseline
            advantage = (advantage - advantage.mean()) / (advantage.std()+1e-6)
        adv_loss = (log_prob * advantage).mean()

        # compute the supervised loss
        sl_input = torch.cat([sl_observations, sl_actions], dim=-1).cpu().numpy()
        sl_target = torch.cat([sl_next_observations-sl_observations, sl_rewards], dim=-1)
        sl_input = self.dynamics.scaler.transform(sl_input)
        sl_mean, sl_logvar = self.dynamics.model(sl_input)
        sl_inv_var = torch.exp(-sl_logvar)
        sl_mse_loss_inv = (torch.pow(sl_mean - sl_target, 2) * sl_inv_var).mean(dim=(1, 2))
        sl_var_loss = sl_logvar.mean(dim=(1, 2))
        sl_loss = sl_mse_loss_inv.sum() + sl_var_loss.sum()
        sl_loss = sl_loss + self.dynamics.model.get_decay_loss()
        sl_loss = sl_loss + 0.001 * self.dynamics.model.max_logvar.sum() - 0.001 * self.dynamics.model.min_logvar.sum()

        all_loss = self._adv_weight * adv_loss + sl_loss
        self._dynmics_adv_optim.zero_grad()
        all_loss.backward()
        self._dynmics_adv_optim.step()

        return next_observations.cpu().numpy(), terminals, {
            "adv_dynamics_update/all_loss": all_loss.cpu().item(), 
            "adv_dynamics_update/sl_loss": sl_loss.cpu().item(), 
            "adv_dynamics_update/adv_loss": adv_loss.cpu().item(), 
            "adv_dynamics_update/adv_advantage": advantage.mean().cpu().item(), 
            "adv_dynamics_update/adv_log_prob": log_prob.mean().cpu().item(), 
        }

    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        for _ in range(rollout_length):
            actions = super().select_action(observations)
            next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
            {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.scaler is not None:
            obs = self.scaler.transform(obs)
        return super().select_action(obs, deterministic)

def get_bc_args():
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

def get_td3bc_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="td3bc")
    parser.add_argument("--task", type=str, default="ac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--update-actor-freq", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=3000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

def get_iql_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="iql")
    parser.add_argument("--task", type=str, default="ac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-q-lr", type=float, default=3e-4)
    parser.add_argument("--critic-v-lr", type=float, default=3e-4)
    parser.add_argument("--dropout_rate", type=float, default=None)
    parser.add_argument("--lr-decay", type=bool, default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--epoch", type=int, default=6000)
    parser.add_argument("--step-per-epoch", type=int, default=500)
    parser.add_argument("--eval_episodes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

def get_rambo_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="rambo")
    parser.add_argument("--task", type=str, default="ac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-5)
    parser.add_argument("--dynamics-lr", type=float, default=3e-5)
    parser.add_argument("--dynamics-adv-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-freq", type=int, default=250)
    parser.add_argument("--dynamics-update-freq", type=int, default=1000)
    parser.add_argument("--adv-batch-size", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--adv-weight", type=float, default=3e-4)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--load-dynamics-path", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=3000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--include-ent-in-adv", type=bool, default=False)
    parser.add_argument("--load-bc-path", type=str, default=None)
    parser.add_argument("--bc-lr", type=float, default=1e-4)
    parser.add_argument("--bc-epoch", type=int, default=50)
    parser.add_argument("--bc-batch-size", type=int, default=256)

    return parser.parse_args()

def BC_Network(args=get_bc_args()):
    env = DummyEnv()
    args.obs_shape = env.obs_shape
    args.action_dim = env.action_dim
    args.max_action = env.max_action
    action_shape=env.action_shape
    action_space=env.action_space

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    actor = Actor(actor_backbone, args.action_dim, max_action=args.max_action, device=args.device)

    # create policy
    policy = BCPolicy(actor, actor_optim=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load_filepath = "model_offline/monza_log/bc/seed_0&timestamp_25-0408-212657/model/policy.pth"
    load_filepath = 'model_offline/ac-v1/bc/seed_0&timestamp_25-1126-141616/model/policy.pth'
    checkpoint = torch.load(load_filepath, map_location=device)
    policy.load_state_dict(checkpoint)
    return policy

def IQL_Network(args=get_iql_args()):
    env = DummyEnv()
    args.obs_shape = env.obs_shape
    args.action_dim = env.action_dim
    args.max_action = env.max_action
    action_shape=env.action_shape
    action_space=env.action_space

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims, dropout_rate=args.dropout_rate)
    critic_q1_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=args.hidden_dims)
    critic_q2_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=args.hidden_dims)
    critic_v_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=False,
        conditioned_sigma=False,
        max_mu=args.max_action
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic_q1 = Critic(critic_q1_backbone, args.device)
    critic_q2 = Critic(critic_q2_backbone, args.device)
    critic_v = Critic(critic_v_backbone, args.device)
    
    # create IQL policy
    policy = IQLPolicy(
        actor,
        critic_q1,
        critic_q2,
        critic_v,
        actor_optim=None,
        critic_q1_optim=None,
        critic_q2_optim=None,
        critic_v_optim=None,
        action_space=action_space,
        tau=args.tau,
        gamma=args.gamma,
        expectile=args.expectile,
        temperature=args.temperature
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load_filepath = "model_offline/monza_log/iql/seed_0&timestamp_25-0410-122810/model/policy.pth"
    load_filepath = 'model_offline/ac-v1/iql/seed_0&timestamp_25-1125-223815/model/policy.pth'
    # load_filepath = "log/ac/iql/seed_0&timestamp_25-0421-103210/model/policy.pth"
    # load_filepath = "log/ac/iql/seed_0&timestamp_25-0424-204509/model/policy.pth"
    
    
    checkpoint = torch.load(load_filepath, map_location=device)
    print("Keys in the checkpoint dictionary:", checkpoint.keys())
    policy.load_state_dict(checkpoint)
    return policy

def TD3BC_Network(args=get_td3bc_args()):
    env = DummyEnv()
    args.obs_shape = env.obs_shape
    args.action_dim = env.action_dim
    args.max_action = env.max_action
    action_shape=env.action_shape
    action_space=env.action_space

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape)+args.action_dim, hidden_dims=[256, 256])
    actor = Actor(actor_backbone, args.action_dim, max_action=args.max_action, device=args.device)

    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    
    dataset = get_mydataset()
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)
    obs_mean, obs_std = buffer.normalize_obs()
    # scaler for normalizing observations
    scaler = StandardScaler(mu=obs_mean, std=obs_std)

    # create policy
    policy = TD3BCPolicy(
        actor,
        critic1,
        critic2,
        actor_optim=None,
        critic1_optim=None,
        critic2_optim=None,
        tau=args.tau,
        gamma=args.gamma,
        max_action=args.max_action,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        update_actor_freq=args.update_actor_freq,
        alpha=args.alpha,
        scaler=scaler 
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_filepath = "model_offline/log/td3bc/seed_0&timestamp_25-0414-102105/model/policy.pth"
    load_filepath = 'model_offline/ac-v1/td3bc/seed_0&timestamp_25-1126-174957/model/policy.pth'
    # load_filepath = 'model_offline/ac-v1/td3bc/new_seed_0&timestamp_25-1205-112805/model/policy.pth'
    # load_filepath = 'model_offline/ac-v1/td3bc/seed_0&timestamp_25-1126-174957/checkpoint/policy_epoch_2000.pth'
    checkpoint = torch.load(load_filepath, map_location=device)
    policy.load_state_dict(checkpoint)
    return policy

def RAMBO_Network(args=get_rambo_args()):
    env = DummyEnv()
    args.obs_shape = env.obs_shape
    args.action_dim = env.action_dim
    args.max_action = env.max_action
    action_shape=env.action_shape
    action_space=env.action_space

    dataset = get_mydataset()
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dataset)
    obs_mean, obs_std = real_buffer.normalize_obs()
    fake_buffer_size = args.step_per_epoch // args.rollout_freq * args.model_retain_epochs * args.rollout_batch_size * args.rollout_length
    fake_buffer = ReplayBuffer(
        buffer_size=fake_buffer_size, 
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )

    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
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

    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args.obs_shape),
        action_dim=args.action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )
    dynamics_adv_optim = torch.optim.Adam(
        dynamics_model.parameters(), 
        lr=args.dynamics_adv_lr
    )
    dynamics_scaler = StandardScaler()
    def termination_fn_point2dwallenv(obs, act, next_obs):#################
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        done = np.array([False]).repeat(len(obs))
        done = done[:,None]
        return done
    def obs_unnormalization(termination_fn, obs_mean, obs_std):
        def thunk(obs, act, next_obs):
            obs = obs*obs_std + obs_mean
            next_obs = next_obs*obs_std + obs_mean
            return termination_fn(obs, act, next_obs)
        return thunk
    termination_fn = obs_unnormalization(termination_fn_point2dwallenv, obs_mean, obs_std)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        dynamics_scaler,
        termination_fn,
    )

    policy_scaler = StandardScaler(mu=obs_mean, std=obs_std)

    # create policy
    policy = RAMBOPolicy(
        dynamics, 
        actor, 
        critic1, 
        critic2, 
        actor_optim, 
        critic1_optim, 
        critic2_optim, 
        dynamics_adv_optim,
        tau=args.tau, 
        gamma=args.gamma, 
        alpha=alpha, 
        adv_weight=args.adv_weight, 
        adv_rollout_length=args.rollout_length, 
        adv_rollout_batch_size=args.adv_batch_size,
        include_ent_in_adv=args.include_ent_in_adv,
        scaler=policy_scaler,
        device=args.device
    ).to(args.device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_filepath = "model_offline/log/rambo/seed_0&timestamp_25-0413-214639/model/rambo_pretrain.pth"
    # load_filepath = 'model_offline/ac-v1/rambo/model/rambo_pretrain.pth'
    checkpoint = torch.load(load_filepath, map_location=device)
    policy.load_state_dict(checkpoint)
    return policy


if __name__ == "__main__":
    get_mydataset(Path='mydata/barcelona_image_relative/sac-v1/data/',image_flag=True)
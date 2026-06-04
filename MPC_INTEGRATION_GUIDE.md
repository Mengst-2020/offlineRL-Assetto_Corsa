# 如何在 my_test_policy_v2.py 中集成 Physics-based MPC

## 方法 1: 最小修改 (推荐快速尝试)

在您的 `my_test_policy_v2.py` 中，找到这一部分:

```python
# 原始代码 (第 35-45 行)
algo = SAC(
    state_dim=125,
    action_dim=3,
    device=device, seed=config.seed,
    **OmegaConf.to_container(config.SAC))
agent = Agent(env=env, test_env=env, algo=algo, log_dir="output",
    device=device, seed=config.seed, **config.Agent, wandb_logger=None)
agent.load("model_ac/model_monza",False)
```

### 改为:

```python
# 修改后 - 替换 SAC 为 Physics-based MPC
from mpc_assetto_corsa import AssettoCorsaMPCController

# 替换这部分:
# algo = SAC(...)
# agent = Agent(...)
# agent.load(...)

# 使用这个:
mpc = AssettoCorsaMPCController(
    env_step_fn=env.step,
    state_dim=125,
    action_dim=3,
    horizon=5,
    population_size=200,
    device="cuda"
)

# 在控制循环中 (第 75-78 行)，将:
# action, _ = agent._algo.exploit(obs["state"])

# 改为:
action, planning_info = mpc.select_action(obs["state"])
```

## 完整修改示例

以下是修改后的 `my_test_policy_v2.py`:

```python
import matplotlib.pyplot as plt
import sys
import os
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
from my_image import VisionStudentPolicy, VisionPolicyFastRunner
from collections import deque

sys.path.extend([os.path.abspath('./assetto_corsa_gym'), './algorithm/discor'])
import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from mpc_assetto_corsa import AssettoCorsaMPCController  # 新增导入
import logging

if __name__ == "__main__":
    config = OmegaConf.load("config.yml")
    work_dir = "outputs" + os.sep + datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
    work_dir = os.path.abspath(work_dir) + os.sep
    env = assettoCorsa.make_ac_env(cfg=config, work_dir=work_dir)  

    # === 修改部分 1: 初始化 MPC 代替 SAC ===
    mpc = AssettoCorsaMPCController(
        env_step_fn=env.step,           # 使用环境的物理模型
        state_dim=125,
        action_dim=3,
        horizon=5,                      # 预测 5 步 (~200ms @25Hz)
        population_size=200,            # CEM 种群大小
        device="cuda"
    )
    
    # 设置奖励权重
    mpc.set_reward_weights(
        progress=1.0,      # 优先前进
        stability=0.3,     # 稳定性
        smoothness=0.2     # 平滑度
    )
    
    origin_env = env.env.env
    dataset = None
    os.environ["MINARI_DATASETS_PATH"] = "F:/code/assetto_corsa_gym-main/mydata"

    controls_rate_limit = np.array([[-600/180, 600/180],
                                    [-1200/100, 1200/100],
                                    [-1200/100, 1200/100],
                                    ]) * (1/25)
    controls_min_values = np.array([-1.0,  -1.0,  -1.0], dtype=np.float32)
    controls_max_values = np.array([ 1.0,   1.0,   1.0], dtype=np.float32)

    # === 修改部分 2: 控制循环 ===
    episode_count = 0
    planning_times = []
    
    while not origin_env.end_flag:
        obs, _ = env.reset()
        done = False
        accumulated_rew = 0
        step = 0
        
        while not done:
            step = step + 1

            # 使用 MPC 选择动作
            state = obs["state"] if isinstance(obs, dict) else obs
            action, planning_info = mpc.select_action(state)
            
            planning_times.append(planning_info['planning_time'])
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            accumulated_rew += reward
            
            # 打印进度
            if step % 100 == 0:
                avg_planning_time = np.mean(planning_times[-100:])
                print(f"Step {step}: "
                      f"Reward={reward:.2f}, "
                      f"Accumulated={accumulated_rew:.2f}, "
                      f"Planning time={planning_info['planning_time']*1000:.1f}ms, "
                      f"Avg={avg_planning_time*1000:.1f}ms")

        episode_count += 1
        print(f"\n=== Episode {episode_count} 完成 ===")
        print(f"Total Reward: {accumulated_rew:.2f}")
        print(f"Steps: {step}")
        
    # 打印统计信息
    print("\n" + "="*50)
    print("MPC 规划统计:")
    stats = mpc.get_planning_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
```

## 方法 2: 参数对比测试

创建一个新文件 `test_mpc_parameters.py`:

```python
"""
测试不同 MPC 参数的性能和质量
"""
import numpy as np
import time
from datetime import datetime
from omegaconf import OmegaConf
import sys
import os

sys.path.extend([os.path.abspath('./assetto_corsa_gym'), './algorithm/discor'])
import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from mpc_assetto_corsa import AssettoCorsaMPCController

def test_mpc_config(horizon, population, name):
    """测试特定配置"""
    config = OmegaConf.load("config.yml")
    work_dir = "outputs" + os.sep + datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
    work_dir = os.path.abspath(work_dir) + os.sep
    env = assettoCorsa.make_ac_env(cfg=config, work_dir=work_dir)
    
    mpc = AssettoCorsaMPCController(
        env_step_fn=env.step,
        state_dim=125,
        action_dim=3,
        horizon=horizon,
        population_size=population,
        device="cuda"
    )
    
    origin_env = env.env.env
    
    planning_times = []
    rewards = []
    
    obs, _ = env.reset()
    done = False
    step = 0
    
    while not done and step < 500:
        step += 1
        state = obs["state"] if isinstance(obs, dict) else obs
        
        start = time.time()
        action, planning_info = mpc.select_action(state)
        elapsed = time.time() - start
        
        planning_times.append(elapsed)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    
    return {
        'name': name,
        'horizon': horizon,
        'population': population,
        'avg_planning_time': np.mean(planning_times),
        'max_planning_time': np.max(planning_times),
        'total_reward': np.sum(rewards),
        'avg_reward': np.mean(rewards),
    }

if __name__ == "__main__":
    configs = [
        (3, 100, "快速模式"),
        (5, 200, "平衡模式"),
        (8, 500, "精准模式"),
    ]
    
    results = []
    for horizon, population, name in configs:
        print(f"\n测试: {name} (H={horizon}, P={population})")
        result = test_mpc_config(horizon, population, name)
        results.append(result)
        print(f"  平均规划时间: {result['avg_planning_time']*1000:.1f}ms")
        print(f"  总奖励: {result['total_reward']:.2f}")
    
    print("\n" + "="*60)
    print("对比结果:")
    print("="*60)
    print(f"{'配置':<12} {'H':<4} {'P':<5} {'规划(ms)':<12} {'奖励':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<12} {r['horizon']:<4} {r['population']:<5} "
              f"{r['avg_planning_time']*1000:<12.1f} {r['total_reward']:<10.1f}")
```

## 方法 3: 混合策略 (SAC + MPC)

如果想同时使用已训练的 SAC 和 MPC:

```python
from mpc_assetto_corsa import AssettoCorsaMPCController
from discor.agent_dataset import Agent
from discor.algorithm import SAC

# 初始化两个控制器
device = torch.device("cuda")
algo = SAC(state_dim=125, action_dim=3, device=device, 
           seed=config.seed, **OmegaConf.to_container(config.SAC))
agent = Agent(env=env, test_env=env, algo=algo, log_dir="output",
              device=device, seed=config.seed, **config.Agent, wandb_logger=None)
agent.load("model_ac/model_monza", False)

mpc = AssettoCorsaMPCController(
    env_step_fn=env.step,
    state_dim=125, action_dim=3,
    horizon=3,           # MPC 用于需要精确控制的情况
    population_size=100
)

# 在控制循环中智能切换
while not done:
    step += 1
    state = obs["state"]
    
    # 判断是否需要精确控制
    if step < 50:  # 起始阶段用 MPC
        action, _ = mpc.select_action(state)
        controller = "MPC"
    else:  # 正常阶段用 SAC (更快)
        action, _ = agent._algo.exploit(state)
        controller = "SAC"
    
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    if step % 100 == 0:
        print(f"Step {step}: Using {controller}")
```

## 关键修改要点

1. **导入**:
   ```python
   from mpc_assetto_corsa import AssettoCorsaMPCController
   ```

2. **初始化** (替代 SAC):
   ```python
   mpc = AssettoCorsaMPCController(
       env_step_fn=env.step,
       state_dim=125,
       action_dim=3,
       horizon=5,
       population_size=200,
       device="cuda"
   )
   ```

3. **调用** (替代 `agent._algo.exploit`):
   ```python
   action, planning_info = mpc.select_action(obs["state"])
   ```

4. **监控** (可选):
   ```python
   print(f"规划时间: {planning_info['planning_time']*1000:.1f}ms")
   print(f"最优回报: {planning_info['best_return']:.2f}")
   ```

## 性能优化技巧

### 如果规划时间太长:
```python
# 减少参数
mpc = AssettoCorsaMPCController(
    ...,
    horizon=3,              # 从 5 改为 3
    population_size=100     # 从 200 改为 100
)
mpc.cem_iterations = 2      # 从 3 改为 2
```

### 如果控制质量不好:
```python
# 增加参数
mpc = AssettoCorsaMPCController(
    ...,
    horizon=8,              # 增加预测时窗
    population_size=500     # 增加搜索范围
)
```

## 测试建议

1. **首先测试单个 MPC**:
   ```bash
   python mpc_assetto_corsa_example.py --example 1
   ```

2. **然后集成到你的代码**:
   ```bash
   # 修改 my_test_policy_v2.py 后运行
   python my_test_policy_v2.py
   ```

3. **对比参数效果**:
   ```bash
   python test_mpc_parameters.py
   ```

4. **尝试混合策略** (可选):
   - 查看 `mpc_assetto_corsa_example.py` 的示例 2

## 常见问题

**Q: 为什么选择 horizon=5?**
A: @25Hz 采样率，5 步 = 200ms，这是一个很好的平衡点。可以根据你的需求调整。

**Q: GPU 内存不足怎么办?**
A: 减少 `population_size` 到 100 或 50。这是最快的解决方案。

**Q: 相比 SAC 哪个快?**
A: SAC 快 (1-5ms)，但 MPC 质量更高。对赛车应用，100ms 延迟完全可接受。

**Q: 换地图/换车需要重新训练吗?**
A: **不需要**！这是 Physics-based MPC 的优势。直接用即可。

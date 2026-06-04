"""
集成示例: 在 my_test_policy_v2.py 中使用 Physics-based MPC
直接使用 Assetto Corsa 物理模型进行控制，无需训练任何神经网络模型
"""

import sys
import os
import numpy as np
import torch
from datetime import datetime
from omegaconf import OmegaConf
from gymnasium.spaces import Box

sys.path.extend([os.path.abspath('./assetto_corsa_gym'), './algorithm/discor'])
import AssettoCorsaEnv.assettoCorsa as assettoCorsa
from mpc_assetto_corsa import AssettoCorsaMPCController
from discor.agent_dataset import Agent
from discor.algorithm import SAC


def example_mpc_only():
    """
    示例 1: 仅使用 MPC 进行控制（基于物理模型）
    无需任何预训练的神经网络
    """
    print("=" * 70)
    print("示例 1: 仅使用 Physics-based MPC 控制")
    print("=" * 70)
    
    config = OmegaConf.load("config.yml")
    work_dir = "outputs" + os.sep + datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
    work_dir = os.path.abspath(work_dir) + os.sep
    env = assettoCorsa.make_ac_env(cfg=config, work_dir=work_dir)
    
    # 创建 MPC 控制器
    # 核心特点: 直接使用 env.step 作为物理模型，无需训练
    mpc = AssettoCorsaMPCController(
        env_step_fn=env.step,          # 使用 Assetto Corsa 的真实物理
        state_dim=125,                  # AC 环境的状态维度
        action_dim=3,                   # 动作维度: [steering, gas, brake]
        horizon=5,                      # 预测 5 步 (~200ms @25Hz)
        population_size=200,            # CEM 种群大小
        use_cem=True                    # 使用交叉熵方法优化
    )
    
    # 设置奖励函数权重
    mpc.set_reward_weights(
        progress=1.0,      # 优化前进速度
        stability=0.3,     # 优化稳定性
        smoothness=0.2     # 优化平滑度
    )
    
    origin_env = env.env.env
    
    # 控制循环
    num_episodes = 3
    for episode in range(num_episodes):
        if origin_env.end_flag:
            break
        
        obs, _ = env.reset()
        done = False
        accumulated_reward = 0
        step = 0
        episode_times = []
        
        while not done and step < 1000:
            step += 1
            
            # 使用 MPC 选择动作
            state = obs["state"] if isinstance(obs, dict) else obs
            action, planning_info = mpc.select_action(state)
            
            # 执行动作
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            accumulated_reward += reward
            
            episode_times.append(planning_info['planning_time'])
            
            # 定期打印信息
            if step % 50 == 0:
                avg_time = np.mean(episode_times[-50:])
                print(f"Episode {episode}, Step {step}:")
                print(f"  Action: steering={action[0]:6.3f}, "
                      f"gas={action[1]:6.3f}, brake={action[2]:6.3f}")
                print(f"  Reward: {reward:7.2f}, Best return: {planning_info['best_return']:7.2f}")
                print(f"  Avg planning time: {avg_time*1000:5.1f}ms")
        
        print(f"\n✓ Episode {episode} completed!")
        print(f"  Total reward: {accumulated_reward:.2f}")
        print(f"  Total steps: {step}")
        print(f"  Avg planning time: {np.mean(episode_times)*1000:.1f}ms\n")
    
    # 打印统计信息
    stats = mpc.get_planning_statistics()
    print("\n" + "=" * 70)
    print("MPC Planning Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def example_hybrid_mpc_learned_policy():
    """
    示例 2: 混合策略
    在不同情况下使用 MPC 和学习策略
    - 正常情况: 使用学习策略(SAC)更快
    - 危险情况(靠近赛道边界): 使用 MPC 进行精细控制
    """
    print("\n" + "=" * 70)
    print("示例 2: 混合策略 (SAC + Physics-based MPC)")
    print("=" * 70)
    
    config = OmegaConf.load("config.yml")
    work_dir = "outputs" + os.sep + datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]
    work_dir = os.path.abspath(work_dir) + os.sep
    env = assettoCorsa.make_ac_env(cfg=config, work_dir=work_dir)
    
    # 加载已训练的 SAC 模型
    device = torch.device("cuda")
    algo = SAC(
        state_dim=125,
        action_dim=3,
        device=device,
        seed=config.seed,
        **OmegaConf.to_container(config.SAC)
    )
    agent = Agent(
        env=env, 
        test_env=env, 
        algo=algo, 
        log_dir="output",
        device=device, 
        seed=config.seed, 
        **config.Agent, 
        wandb_logger=None
    )
    agent.load("model_ac/model_monza", False)
    
    # 创建 MPC 控制器（用于危险情况）
    mpc = AssettoCorsaMPCController(
        env_step_fn=env.step,
        state_dim=125,
        action_dim=3,
        horizon=3,              # 更短的时间窗口以保持实时性
        population_size=100,    # 较少的样本以加快计算
        use_cem=True
    )
    
    origin_env = env.env.env
    
    # 控制循环
    num_episodes = 2
    mpc_usage_count = 0
    sac_usage_count = 0
    
    for episode in range(num_episodes):
        if origin_env.end_flag:
            break
        
        obs, _ = env.reset()
        done = False
        accumulated_reward = 0
        step = 0
        
        while not done and step < 1000:
            step += 1
            state = obs["state"] if isinstance(obs, dict) else obs
            
            # 简单的切换策略
            # 假设状态的第一个分量代表到赛道边界的距离
            distance_to_edge = state[0] if len(state) > 0 else 1.0
            
            # 根据距离选择策略
            if distance_to_edge < 0.3:  # 接近边界
                # 使用 MPC 进行精细控制
                action, _ = mpc.select_action(state)
                mpc_usage_count += 1
                strategy = "MPC"
            else:
                # 使用学习策略进行快速控制
                action, _ = agent._algo.exploit(state)
                sac_usage_count += 1
                strategy = "SAC"
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            accumulated_reward += reward
            
            if step % 100 == 0:
                print(f"Episode {episode}, Step {step}: Using {strategy}, "
                      f"Distance to edge: {distance_to_edge:.3f}, "
                      f"Accumulated reward: {accumulated_reward:.2f}")
        
        print(f"\n✓ Episode {episode} completed!")
        print(f"  Total reward: {accumulated_reward:.2f}")
    
    print(f"\n统计信息:")
    print(f"  MPC 使用次数: {mpc_usage_count}")
    print(f"  SAC 使用次数: {sac_usage_count}")
    print(f"  MPC 使用比例: {100*mpc_usage_count/(mpc_usage_count+sac_usage_count):.1f}%")


def example_mpc_parameter_tuning():
    """
    示例 3: MPC 参数调优指南
    展示不同参数对性能的影响
    """
    print("\n" + "=" * 70)
    print("示例 3: MPC 参数对性能的影响")
    print("=" * 70)
    
    configs = [
        {
            'name': '快速模式 (低延迟)',
            'horizon': 3,
            'population_size': 100,
            'cem_iterations': 2,
        },
        {
            'name': '平衡模式 (推荐)',
            'horizon': 5,
            'population_size': 200,
            'cem_iterations': 3,
        },
        {
            'name': '精准模式 (高质量)',
            'horizon': 8,
            'population_size': 500,
            'cem_iterations': 5,
        },
    ]
    
    print("\n参数对比:")
    print("-" * 70)
    print(f"{'模式':<20} {'Horizon':<12} {'Population':<15} {'Iterations':<12}")
    print("-" * 70)
    
    for cfg in configs:
        print(f"{cfg['name']:<20} {cfg['horizon']:<12} {cfg['population_size']:<15} "
              f"{cfg['cem_iterations']:<12}")
    
    print("\n建议:")
    print("  • 快速模式: 实时性要求高，低延迟应用")
    print("  • 平衡模式: 通常推荐使用，兼衡质量和速度")
    print("  • 精准模式: 对控制质量要求高，可以容忍延迟")
    
    print("\n调优建议:")
    print("  1. Horizon (规划时间窗口):")
    print("     - 增大: 更长期规划，但计算成本增加")
    print("     - 降低: 快速反应，但短视")
    print("     - 推荐: 5-10 步 (@25Hz = 200-400ms)")
    print()
    print("  2. Population size (CEM种群大小):")
    print("     - 增大: 更好的搜索覆盖")
    print("     - 降低: 计算更快，但可能陷入局部最优")
    print("     - 推荐: 200-500 取决于硬件")
    print()
    print("  3. CEM iterations (优化迭代):")
    print("     - 增大: 更好地收敛")
    print("     - 降低: 更快的计算")
    print("     - 推荐: 3-5 次")


def example_mpc_with_vision():
    """
    示例 4: 结合视觉信息的 MPC
    """
    print("\n" + "=" * 70)
    print("示例 4: MPC + 视觉信息")
    print("=" * 70)
    
    print("\n实现思路:")
    print("  1. 提取图像中的重要特征:")
    print("     - 赛道边界位置")
    print("     - 前方弯道曲率")
    print("     - 对手位置")
    print()
    print("  2. 将视觉特征融入 MPC 奖励函数:")
    print("     - 避免赛道边界的奖励")
    print("     - 根据曲率调整速度的奖励")
    print("     - 避免碰撞的奖励")
    print()
    print("  3. 动态调整 MPC 参数:")
    print("     - 在弯道时增加 horizon")
    print("     - 在直道时减少计算")
    print()
    print("示例代码:")
    print("""
    # 提取视觉特征
    vision_features = extract_features_from_image(obs['image'])
    distance_to_edge = vision_features['distance_to_edge']
    next_curve_radius = vision_features['curve_radius']
    
    # 动态调整 MPC 参数
    if next_curve_radius < 100:  # 急弯
        mpc.horizon = 8
        mpc.population_size = 300
    else:  # 直道
        mpc.horizon = 3
        mpc.population_size = 100
    
    # 调整奖励权重
    if distance_to_edge < 0.2:
        mpc.set_reward_weights(stability=0.8)  # 优先稳定性
    else:
        mpc.set_reward_weights(progress=1.0)   # 优先速度
    
    action, _ = mpc.select_action(obs["state"])
    """)


if __name__ == "__main__":
    # 运行示例
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=int, default=1,
                       help='示例编号 (1-4)')
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("Assetto Corsa Physics-based MPC 集成示例")
    print("使用真实 Assetto Corsa 物理模型，无需训练神经网络")
    print("=" * 70)
    
    if args.example == 1:
        example_mpc_only()
    elif args.example == 2:
        example_hybrid_mpc_learned_policy()
    elif args.example == 3:
        example_mpc_parameter_tuning()
    elif args.example == 4:
        example_mpc_with_vision()
    else:
        print(f"未知示例: {args.example}")
        print("请选择 1-4")

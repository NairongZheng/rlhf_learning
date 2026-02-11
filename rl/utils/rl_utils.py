"""
RL辅助工具函数

包含advantage计算、reward归一化等常用函数
"""

import torch
from typing import Optional


def compute_advantages(
    rewards: torch.Tensor,
    baseline: str = "mean",
    scale: bool = True,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    计算advantages（组内归一化）

    在GRPO中，对每个prompt的G个生成样本，计算advantage：
        A_i = (R_i - baseline) / (std + epsilon)

    这样可以让高于平均的样本获得正advantage，低于平均的获得负advantage

    Args:
        rewards: shape [batch, G] 或 [batch * G]
            每个prompt的G个样本的reward
        baseline: baseline类型
            - "mean": 使用组内均值
            - "median": 使用组内中位数
            - "none": 不使用baseline（advantage = reward）
        scale: 是否进行标准化（除以std）
        epsilon: 防止除零的小常数

    Returns:
        advantages: 与rewards相同shape的advantage值
    """
    if rewards.dim() == 1:
        # 如果是1D，假设已经是展平的batch*G
        # 这种情况下无法进行组内归一化，只能全局归一化
        if baseline == "mean":
            base = rewards.mean()
        elif baseline == "median":
            base = rewards.median()
        else:
            base = 0.0

        advantages = rewards - base

        if scale:
            std = rewards.std()
            advantages = advantages / (std + epsilon)

        return advantages

    elif rewards.dim() == 2:
        # shape [batch, G]，可以进行组内归一化
        batch_size, num_generations = rewards.shape

        # 计算baseline
        if baseline == "mean":
            base = rewards.mean(dim=1, keepdim=True)  # [batch, 1]
        elif baseline == "median":
            base = rewards.median(dim=1, keepdim=True)[0]  # [batch, 1]
        else:
            base = 0.0

        # 计算advantages
        advantages = rewards - base

        # 组内标准化
        if scale:
            std = advantages.std(dim=1, keepdim=True)  # [batch, 1]
            advantages = advantages / (std + epsilon)

        return advantages

    else:
        raise ValueError(f"rewards应该是1D或2D tensor，但得到了{rewards.dim()}D")


def normalize_rewards(
    rewards: torch.Tensor,
    method: str = "standardize",
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    归一化rewards

    Args:
        rewards: 任意shape的reward tensor
        method: 归一化方法
            - "standardize": (r - mean) / std  # 标准化到均值0，方差1
            - "normalize": (r - min) / (max - min)  # 归一化到[0, 1]
            - "center": r - mean  # 中心化到均值0
        epsilon: 防止除零

    Returns:
        归一化后的rewards
    """
    if method == "standardize":
        mean = rewards.mean()
        std = rewards.std()
        return (rewards - mean) / (std + epsilon)

    elif method == "normalize":
        min_r = rewards.min()
        max_r = rewards.max()
        return (rewards - min_r) / (max_r - min_r + epsilon)

    elif method == "center":
        mean = rewards.mean()
        return rewards - mean

    else:
        raise ValueError(f"未知的归一化方法: {method}")


def whiten_advantages(
    advantages: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    白化advantages（标准化到均值0，方差1）

    这是PPO中常用的技巧，可以提高训练稳定性

    Args:
        advantages: advantage值
        epsilon: 防止除零

    Returns:
        白化后的advantages
    """
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + epsilon)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> tuple:
    """
    计算Generalized Advantage Estimation (GAE)

    用于PPO算法

    GAE公式：
        δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = Σ_{l=0}^{T-t} (γλ)^l * δ_{t+l}

    Args:
        rewards: shape [batch, T]，每个时间步的reward
        values: shape [batch, T]，每个时间步的value估计
        dones: shape [batch, T]，1表示episode结束，0表示继续
        gamma: 折扣因子
        gae_lambda: GAE的λ参数（权衡bias-variance）

    Returns:
        advantages: shape [batch, T]，每个时间步的advantage
        returns: shape [batch, T]，每个时间步的return（用于训练value model）
    """
    batch_size, T = rewards.shape
    advantages = torch.zeros_like(rewards)

    # 从后向前计算GAE
    gae = 0
    for t in reversed(range(T)):
        # 计算TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        if t == T - 1:
            # 最后一个时间步，没有next_value
            next_value = 0
        else:
            next_value = values[:, t + 1]

        # 如果episode结束，next_value设为0
        next_value = next_value * (1 - dones[:, t])

        delta = rewards[:, t] + gamma * next_value - values[:, t]

        # GAE递归：A_t = δ_t + γλ * A_{t+1}
        gae = delta + gamma * gae_lambda * (1 - dones[:, t]) * gae
        advantages[:, t] = gae

    # 计算returns: R_t = A_t + V(s_t)
    returns = advantages + values

    return advantages, returns


def test_rl_utils():
    """
    测试RL工具函数
    """
    print("\n" + "="*60)
    print("测试RL工具函数")
    print("="*60)

    # 测试1: compute_advantages - 2D情况
    print("\n[测试1] compute_advantages - 组内归一化")
    batch_size = 3
    num_generations = 4
    rewards = torch.randn(batch_size, num_generations) * 2 + 5  # 均值约5

    print(f"  原始rewards shape: {rewards.shape}")
    print(f"  原始rewards:\n{rewards}")

    advantages = compute_advantages(rewards, baseline="mean", scale=True)
    print(f"\n  Advantages:\n{advantages}")
    print(f"  每组均值: {advantages.mean(dim=1)}")  # 应该接近0
    print(f"  每组标准差: {advantages.std(dim=1)}")  # 应该接近1

    # 测试2: compute_advantages - 1D情况
    print("\n[测试2] compute_advantages - 全局归一化")
    rewards_1d = rewards.flatten()
    advantages_1d = compute_advantages(rewards_1d, baseline="mean", scale=True)
    print(f"  Advantages 1D shape: {advantages_1d.shape}")
    print(f"  全局均值: {advantages_1d.mean():.4f}")  # 应该接近0
    print(f"  全局标准差: {advantages_1d.std():.4f}")  # 应该接近1

    # 测试3: normalize_rewards
    print("\n[测试3] normalize_rewards")
    test_rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    standardized = normalize_rewards(test_rewards, method="standardize")
    print(f"  原始: {test_rewards}")
    print(f"  Standardize: {standardized}")
    print(f"    均值: {standardized.mean():.4f}, 标准差: {standardized.std():.4f}")

    normalized = normalize_rewards(test_rewards, method="normalize")
    print(f"  Normalize [0,1]: {normalized}")
    print(f"    最小值: {normalized.min():.4f}, 最大值: {normalized.max():.4f}")

    centered = normalize_rewards(test_rewards, method="center")
    print(f"  Center: {centered}")
    print(f"    均值: {centered.mean():.4f}")

    # 测试4: whiten_advantages
    print("\n[测试4] whiten_advantages")
    test_adv = torch.randn(10) * 5 + 3
    whitened = whiten_advantages(test_adv)
    print(f"  原始advantage均值: {test_adv.mean():.4f}, 标准差: {test_adv.std():.4f}")
    print(f"  白化后均值: {whitened.mean():.4f}, 标准差: {whitened.std():.4f}")

    # 测试5: 验证组内归一化的正确性
    print("\n[测试5] 验证组内归一化")
    # 创建一个简单的例子
    rewards_simple = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],  # 第1组
        [10.0, 20.0, 30.0, 40.0]  # 第2组，值更大
    ])
    advantages_simple = compute_advantages(rewards_simple, baseline="mean", scale=True)
    print(f"  原始rewards:\n{rewards_simple}")
    print(f"  Advantages:\n{advantages_simple}")
    print(f"  第1组的advantage应该和第2组的相似（因为都是递增的）")
    print(f"    第1组: {advantages_simple[0]}")
    print(f"    第2组: {advantages_simple[1]}")

    print("\n测试完成!")


if __name__ == "__main__":
    test_rl_utils()

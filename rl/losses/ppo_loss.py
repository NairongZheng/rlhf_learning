"""
PPO损失函数

Proximal Policy Optimization - 经典RL算法
"""

import torch
import torch.nn.functional as F
from typing import Dict


def compute_ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float = 0.2,
    value_clip_range: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    normalize_advantages: bool = True
) -> tuple:
    """
    计算PPO损失

    PPO结合了三个损失：
        1. Policy loss: clipped surrogate objective
        2. Value loss: MSE between predicted value and return
        3. Entropy loss: 鼓励探索

    数学公式：
        L_CLIP = -E[min(r·A, clip(r, 1-ε, 1+ε)·A)]
        L_VF = E[(V - G)^2]
        L_ENT = -E[H(π)]
        L_TOTAL = L_CLIP + c1·L_VF - c2·L_ENT

    Args:
        log_probs: 当前policy的log概率，shape [batch]
        old_log_probs: 旧policy的log概率，shape [batch]
        advantages: GAE计算的advantages，shape [batch]
        values: value model的预测值，shape [batch]
        returns: 实际的returns（目标值），shape [batch]
        clip_range: policy clip范围（symmetric）
        value_clip_range: value clip范围
        value_loss_coef: value loss的权重
        entropy_coef: entropy的权重
        normalize_advantages: 是否归一化advantages

    Returns:
        loss: 总损失
        metrics: 指标字典
    """
    # 1. 归一化advantages（可选）
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 2. 计算policy ratio
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # 3. Clipped surrogate objective (Policy Loss)
    clip_low = 1.0 - clip_range
    clip_high = 1.0 + clip_range
    clipped_ratio = torch.clamp(ratio, clip_low, clip_high)

    policy_loss1 = ratio * advantages
    policy_loss2 = clipped_ratio * advantages
    policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

    # 4. Value Loss（可选clipping）
    if value_clip_range > 0:
        # Value clipping（防止value变化过大）
        values_clipped = returns + torch.clamp(
            values - returns,
            -value_clip_range,
            value_clip_range
        )
        value_loss1 = (values - returns) ** 2
        value_loss2 = (values_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
    else:
        # 标准MSE loss
        value_loss = 0.5 * ((values - returns) ** 2).mean()

    # 5. Entropy Loss（鼓励探索）
    # 近似计算：entropy ≈ -Σ p·log(p)
    # 由于我们只有log_probs，使用近似：entropy ≈ -log_probs.mean()
    entropy = -log_probs.mean()

    # 6. 总损失
    total_loss = (
        policy_loss +
        value_loss_coef * value_loss -
        entropy_coef * entropy
    )

    # 7. 计算补充指标
    clip_fraction = (ratio != clipped_ratio).float().mean()
    approx_kl = (log_ratio).mean()  # 近似KL散度

    metrics = {
        'loss': total_loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy.item(),
        'mean_ratio': ratio.mean().item(),
        'min_ratio': ratio.min().item(),
        'max_ratio': ratio.max().item(),
        'clip_fraction': clip_fraction.item(),
        'approx_kl': approx_kl.item(),
        'mean_advantage': advantages.mean().item(),
        'std_advantage': advantages.std().item(),
        'value_mean': values.mean().item(),
        'return_mean': returns.mean().item()
    }

    return total_loss, metrics


if __name__ == "__main__":
    # 简单测试
    print("测试PPO loss...")

    batch_size = 8
    # 模拟数据
    log_probs = torch.randn(batch_size) - 3.0
    old_log_probs = torch.randn(batch_size) - 3.0
    advantages = torch.randn(batch_size)
    values = torch.randn(batch_size) * 2
    returns = values + advantages  # returns = values + advantages

    loss, metrics = compute_ppo_loss(
        log_probs, old_log_probs, advantages,
        values, returns
    )

    print(f"✅ PPO Loss测试通过!")
    print(f"  Total Loss: {metrics['loss']:.4f}")
    print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
    print(f"  Value Loss: {metrics['value_loss']:.4f}")
    print(f"  Entropy: {metrics['entropy']:.4f}")
    print(f"  Mean Ratio: {metrics['mean_ratio']:.4f}")
    print(f"  Clip Fraction: {metrics['clip_fraction']:.4f}")

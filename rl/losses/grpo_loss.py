"""
GRPO损失函数

Group Relative Policy Optimization - DeepSeek-V3论文使用的方法
"""

import torch
import torch.nn.functional as F
from typing import Dict


def compute_grpo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range_low: float = 0.1,
    clip_range_high: float = 0.2,
    kl_coef: float = 0.0
) -> tuple:
    """
    计算GRPO损失

    核心思想：
        1. 计算importance sampling ratio: r = π_new / π_old
        2. 使用asymmetric clip防止policy变化过大
        3. 最大化advantage加权的目标函数

    数学公式：
        L_GRPO = -E[min(r·A, clip(r, 1-ε_low, 1+ε_high)·A)]

    可选添加KL散度惩罚：
        L = L_GRPO + λ_KL · KL(π_new || π_old)

    Args:
        log_probs: 当前policy的log概率，shape [batch*G]
        ref_log_probs: reference policy的log概率，shape [batch*G]
        advantages: advantage值（组内归一化的reward），shape [batch*G]
        clip_range_low: clip下界（控制policy下降幅度）
        clip_range_high: clip上界（控制policy上升幅度）
        kl_coef: KL惩罚系数（0表示不使用）

    Returns:
        loss: 总损失
        metrics: 指标字典（policy_loss, kl, mean_ratio, clip_fraction等）
    """
    # 1. 计算ratio = π_new / π_old
    log_ratio = log_probs - ref_log_probs
    ratio = torch.exp(log_ratio)

    # 2. Clip ratio (asymmetric clipping)
    clip_low = 1.0 - clip_range_low
    clip_high = 1.0 + clip_range_high
    clipped_ratio = torch.clamp(ratio, clip_low, clip_high)

    # 3. Surrogate loss
    loss1 = ratio * advantages
    loss2 = clipped_ratio * advantages
    policy_loss = -torch.min(loss1, loss2).mean()

    # 4. KL惩罚（可选）
    kl_penalty = 0.0
    kl = 0.0
    if kl_coef > 0:
        kl = log_ratio.mean()
        kl_penalty = kl_coef * kl

    # 5. 总损失
    total_loss = policy_loss + kl_penalty

    # 6. 计算补充指标
    clip_fraction = (ratio != clipped_ratio).float().mean()

    # 收集所有指标
    metrics = {
        'loss': total_loss.item(),
        'policy_loss': policy_loss.item(),
        'kl': kl.item() if kl_coef > 0 else 0.0,
        'kl_penalty': kl_penalty.item() if kl_coef > 0 else 0.0,
        'mean_ratio': ratio.mean().item(),
        'min_ratio': ratio.min().item(),
        'max_ratio': ratio.max().item(),
        'clip_fraction': clip_fraction.item(),
        'mean_advantage': advantages.mean().item(),
        'std_advantage': advantages.std().item()
    }

    return total_loss, metrics


if __name__ == "__main__":
    # 简单测试
    print("测试GRPO loss...")
    log_probs = torch.randn(8) * 0.5 - 5.0
    ref_log_probs = torch.randn(8) * 0.5 - 5.0
    advantages = torch.randn(8)
    
    loss, metrics = compute_grpo_loss(log_probs, ref_log_probs, advantages)
    print(f"✅ Loss: {metrics['loss']:.4f}")
    print(f"✅ Mean Ratio: {metrics['mean_ratio']:.4f}")

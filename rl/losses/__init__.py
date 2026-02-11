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
    # log(r) = log(π_new) - log(π_old)
    log_ratio = log_probs - ref_log_probs
    ratio = torch.exp(log_ratio)

    # 2. Clip ratio (asymmetric clipping)
    # 下界: 1 - ε_low
    # 上界: 1 + ε_high
    clip_low = 1.0 - clip_range_low
    clip_high = 1.0 + clip_range_high
    clipped_ratio = torch.clamp(ratio, clip_low, clip_high)

    # 3. Surrogate loss
    # 对比两个项，取最小值（保守的估计）
    loss1 = ratio * advantages
    loss2 = clipped_ratio * advantages
    policy_loss = -torch.min(loss1, loss2).mean()  # 负号因为我们要最大化

    # 4. KL惩罚（可选）
    kl_penalty = 0.0
    kl = 0.0
    if kl_coef > 0:
        # 近似KL散度：KL(π_new || π_old) ≈ log_ratio
        kl = log_ratio.mean()
        kl_penalty = kl_coef * kl

    # 5. 总损失
    total_loss = policy_loss + kl_penalty

    # 6. 计算补充指标
    # Clip fraction: 被ratio clip的比例（用于监控policy变化程度）
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


def test_grpo_loss():
    """
    测试GRPO损失函数
    """
    print("\n" + "="*60)
    print("测试GRPO Loss")
    print("="*60)

    # 模拟数据
    batch_size = 2
    num_generations = 4
    total_samples = batch_size * num_generations

    # 随机生成log probs
    log_probs = torch.randn(total_samples) * 0.5 - 5.0  # 均值约-5
    ref_log_probs = torch.randn(total_samples) * 0.5 - 5.0

    # 生成advantages（组内归一化后的reward）
    advantages = torch.randn(total_samples)

    print(f"\n输入数据:")
    print(f"  Log probs shape: {log_probs.shape}")
    print(f"  Log probs mean: {log_probs.mean():.4f}")
    print(f"  Advantages mean: {advantages.mean():.4f}, std: {advantages.std():.4f}")

    # 测试1: 基础GRPO loss（不KL惩罚）
    print("\n[测试1] 基础GRPO Loss")
    loss, metrics = compute_grpo_loss(
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        advantages=advantages,
        clip_range_low=0.1,
        clip_range_high=0.2,
        kl_coef=0.0
    )

    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
    print(f"  Mean Ratio: {metrics['mean_ratio']:.4f}")
    print(f"  Clip Fraction: {metrics['clip_fraction']:.4f}")

    # 测试2: 带KL惩罚
    print("\n[测试2] 带KL惩罚（kl_coef=0.1）")
    loss, metrics = compute_grpo_loss(
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        advantages=advantages,
        clip_range_low=0.1,
        clip_range_high=0.2,
        kl_coef=0.1
    )

    print(f"  Total Loss: {metrics['loss']:.4f}")
    print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
    print(f"  KL: {metrics['kl']:.4f}")
    print(f"  KL Penalty: {metrics['kl_penalty']:.4f}")

    # 测试3: 极端情况 - 所有advantages都为正
    print("\n[测试3] 所有advantages为正")
    positive_advantages = torch.abs(torch.randn(total_samples))
    loss, metrics = compute_grpo_loss(
        log_probs=log_probs,
        ref_log_probs=ref_log_probs,
        advantages=positive_advantages,
        clip_range_low=0.1,
        clip_range_high=0.2
    )

    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Mean Advantage: {metrics['mean_advantage']:.4f}")

    # 测试4: 极端情况 - policy大幅偏离
    print("\n[测试4] Policy大幅偏离（ratio >> 1）")
    large_log_probs = ref_log_probs + 2.0  # ratio = exp(2.0) ≈ 7.4
    loss, metrics = compute_grpo_loss(
        log_probs=large_log_probs,
        ref_log_probs=ref_log_probs,
        advantages=advantages,
        clip_range_low=0.1,
        clip_range_high=0.2
    )

    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Mean Ratio: {metrics['mean_ratio']:.4f}")
    print(f"  Clip Fraction: {metrics['clip_fraction']:.4f}  # 应该接近1.0")

    print("\n测试完成!")


if __name__ == "__main__":
    test_grpo_loss()
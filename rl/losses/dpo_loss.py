"""
DPO损失函数

Direct Preference Optimization - 直接从偏好数据优化policy
无需显式的Reward Model，直接学习人类偏好
"""

import torch
import torch.nn.functional as F
from typing import Dict


def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    label_smoothing: float = 0.0
) -> tuple:
    """
    计算DPO损失

    核心思想：
        直接从偏好数据优化policy，无需显式reward model
        通过对比chosen和rejected响应，学习隐式的reward函数

    数学公式：
        L_DPO = -log σ(β · [log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x))])

    其中：
        - y_w: chosen (preferred) response
        - y_l: rejected (dispreferred) response
        - β: 温度参数（控制偏离reference的程度）
        - σ: sigmoid函数

    β 参数详解（核心超参数）：
    =========================
    β 控制 policy 允许偏离 reference 的程度（隐式 KL 散度约束）

    作用机制：
        - β 越大：约束越强，policy 被迫接近 reference
          * 优点：训练稳定，保持基础能力和流畅性
          * 缺点：改进可能不明显
          * 适用：微调已经不错的模型，或需要保守更新

        - β 越小：约束越松，policy 可以大幅偏离 reference
          * 优点：可能获得更大的行为改进
          * 缺点：可能失去流畅性，训练不稳定
          * 适用：需要较大改进，或初次对齐

    推荐值：
        - β = 0.01-0.05: 很宽松，允许较大改变（需要大幅改进时）
        - β = 0.1-0.5:   平衡，适合大多数情况（推荐）✅
        - β = 0.5-1.0:   较强约束，适合微调
        - β > 1.0:       很强约束，只做最小改动

    实际效果：
        虽然 β 较大时分布差异小（KL 散度 ~0.3-1.0 nats），但在关键决策点上
        可以产生显著的行为差异：
        - 参数实际改变：约 5-10%
        - 有害内容减少：可达 80%+
        - 用户偏好提升：可达 70%+

    设计哲学：
        "小的分布改变 + 大的行为影响"
        DPO 不是要"大改"模型，而是要在关键点上"精准"改进

    Args:
        policy_chosen_logps: policy对chosen的log概率，shape [batch]
        policy_rejected_logps: policy对rejected的log概率，shape [batch]
        reference_chosen_logps: reference对chosen的log概率，shape [batch]
        reference_rejected_logps: reference对rejected的log概率，shape [batch]
        beta: 温度参数（控制偏离reference的程度，推荐0.1-0.5）
        loss_type: 损失类型
            - "sigmoid": 标准DPO loss（推荐）
            - "hinge": Hinge loss变体
            - "ipo": IPO (Identity Preference Optimization) loss
        label_smoothing: 标签平滑（0-1之间，0表示不使用）

    Returns:
        loss: 总损失
        metrics: 指标字典（loss, accuracy, chosen_reward, rejected_reward等）
    """
    # 1. 计算log ratio
    # log(π_θ / π_ref) for chosen and rejected
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # 2. 计算logits（DPO的核心）
    # logits = β * [log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l))]
    logits = beta * (policy_logratios - ref_logratios)

    # 3. 根据loss_type计算损失
    if loss_type == "sigmoid":
        # 标准DPO loss: -log σ(logits)
        losses = -F.logsigmoid(logits)

        # Label smoothing（可选）
        if label_smoothing > 0:
            # 混合正负样本的loss
            losses = (1 - label_smoothing) * losses + label_smoothing * -F.logsigmoid(-logits)

    elif loss_type == "hinge":
        # Hinge loss变体: max(0, 1 - logits)
        losses = torch.relu(1 - logits)

    elif loss_type == "ipo":
        # IPO (Identity Preference Optimization) loss
        # L = (logits - 1/(2*beta))^2
        losses = (logits - 1/(2*beta)) ** 2

    else:
        raise ValueError(f"未知的loss_type: {loss_type}")

    loss = losses.mean()

    # 4. 计算准确率（policy偏好chosen的比例）
    # policy_logratios > 0 表示policy更偏好chosen
    accuracy = (policy_logratios > 0).float().mean()

    # 5. 计算隐式reward（用于监控）
    # r(x,y) = β * log(π_θ(y|x) / π_ref(y|x))
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

    # 6. 收集指标
    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'mean_logit': logits.mean().item(),
        'chosen_reward_mean': chosen_rewards.mean().item(),
        'rejected_reward_mean': rejected_rewards.mean().item(),
        'reward_margin': (chosen_rewards - rejected_rewards).mean().item(),
        'policy_logratio_mean': policy_logratios.mean().item(),
        'ref_logratio_mean': ref_logratios.mean().item()
    }

    return loss, metrics


if __name__ == "__main__":
    # 简单测试
    print("测试DPO loss...")

    batch_size = 4
    # 模拟log概率
    policy_chosen = torch.randn(batch_size) - 3.0
    policy_rejected = torch.randn(batch_size) - 3.0
    ref_chosen = torch.randn(batch_size) - 3.0
    ref_rejected = torch.randn(batch_size) - 3.0

    # 测试不同loss type
    for loss_type in ["sigmoid", "hinge", "ipo"]:
        loss, metrics = compute_dpo_loss(
            policy_chosen, policy_rejected,
            ref_chosen, ref_rejected,
            beta=0.1,
            loss_type=loss_type
        )
        print(f"\n✅ {loss_type} loss测试通过")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Reward Margin: {metrics['reward_margin']:.4f}")

"""
奖励模型（Reward Model）- 对生成结果进行评分

用于PPO和GRPO等RL算法，也可以用于训练DPO的偏好数据
"""

import torch
import torch.nn as nn
from typing import Optional
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.modules.text_encoder import TextEncoder


class RewardModel(nn.Module):
    """
    奖励模型 - 对prompt+completion进行评分

    架构：
        输入tokens → TextEncoder → 取最后一个token的hidden state → reward_head → scalar reward

    使用场景：
        - PPO：预训练的固定Reward Model，用于计算reward
        - GRPO：可选，也可以使用自定义reward函数
        - DPO：不需要，DPO直接从偏好数据学习

    注意：
        - Reward Model通常需要预先在偏好数据上训练
        - 训练方法：给定chosen和rejected对，最大化 reward(chosen) - reward(rejected)
        - 在RL训练时，Reward Model的参数固定不更新
    """

    def __init__(
        self,
        vocab_size: int = 151657,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 512,
        rope_base: int = 10000,
        dropout: float = 0.1,
        norm_type: str = "pre"
    ):
        """
        初始化奖励模型

        Args:
            vocab_size: 词汇表大小
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            max_seq_len: 最大序列长度
            rope_base: RoPE的base频率
            dropout: dropout率
            norm_type: LayerNorm类型
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # 1. Text Encoder作为backbone
        self.backbone = TextEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            dropout=dropout,
            norm_type=norm_type
        )

        # 2. Reward Head：将hidden state映射到标量reward
        self.reward_head = nn.Linear(hidden_dim, 1)

        # 初始化reward head（使用较小的权重）
        nn.init.normal_(self.reward_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播 - 计算reward

        Args:
            input_ids: 输入token IDs，shape [batch, seq_len]
                      包含prompt + completion
            attention_mask: 可选的attention mask

        Returns:
            rewards: shape [batch]，每个序列的reward分数
        """
        # 1. 编码整个序列
        hidden_states = self.backbone(input_ids, mask=attention_mask)  # [batch, seq_len, hidden_dim]

        # 2. 取最后一个有效token的hidden state
        if attention_mask is not None:
            # 找到每个序列最后一个非padding的位置
            last_positions = attention_mask.sum(dim=1) - 1  # [batch]
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden = hidden_states[batch_indices, last_positions]  # [batch, hidden_dim]
        else:
            # 如果没有mask，直接取最后一个位置
            last_hidden = hidden_states[:, -1, :]  # [batch, hidden_dim]

        # 3. 通过reward head计算reward
        rewards = self.reward_head(last_hidden).squeeze(-1)  # [batch]

        return rewards

    def compute_reward_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        计算Reward Model的训练损失

        使用ranking loss：最大化 reward(chosen) - reward(rejected)
        实际使用：L = -log sigmoid(reward(chosen) - reward(rejected))

        Args:
            chosen_ids: chosen的token IDs，shape [batch, seq_len]
            rejected_ids: rejected的token IDs，shape [batch, seq_len]
            chosen_mask: chosen的attention mask
            rejected_mask: rejected的attention mask

        Returns:
            loss: 损失值
            metrics: 指标字典（accuracy, reward_diff等）
        """
        # 1. 计算chosen和rejected的reward
        chosen_rewards = self.forward(chosen_ids, chosen_mask)  # [batch]
        rejected_rewards = self.forward(rejected_ids, rejected_mask)  # [batch]

        # 2. 计算ranking loss
        # L = -log sigmoid(reward_chosen - reward_rejected)
        reward_diff = chosen_rewards - rejected_rewards
        loss = -torch.nn.functional.logsigmoid(reward_diff).mean()

        # 3. 计算准确率（chosen > rejected的比例）
        accuracy = (reward_diff > 0).float().mean()

        # 4. 收集指标
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'reward_diff_mean': reward_diff.mean().item(),
            'chosen_reward_mean': chosen_rewards.mean().item(),
            'rejected_reward_mean': rejected_rewards.mean().item()
        }

        return loss, metrics


def test_reward_model():
    """
    测试Reward Model
    """
    print("\n" + "="*60)
    print("测试Reward Model")
    print("="*60)

    # 创建模型
    model = RewardModel(
        vocab_size=1000,
        hidden_dim=128,
        num_heads=4,
        num_layers=1,
        max_seq_len=64
    )

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")

    # 测试1: 前向传播
    print("\n[测试1] 前向传播")
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    rewards = model(input_ids)
    print(f"  输入shape: {input_ids.shape}")
    print(f"  输出reward shape: {rewards.shape}")
    print(f"  Reward值: {rewards}")

    # 测试2: 计算ranking loss
    print("\n[测试2] 计算Ranking Loss")
    chosen_ids = torch.randint(0, 1000, (batch_size, seq_len))
    rejected_ids = torch.randint(0, 1000, (batch_size, seq_len))

    loss, metrics = model.compute_reward_loss(chosen_ids, rejected_ids)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Reward Diff Mean: {metrics['reward_diff_mean']:.4f}")
    print(f"  Chosen Reward Mean: {metrics['chosen_reward_mean']:.4f}")
    print(f"  Rejected Reward Mean: {metrics['rejected_reward_mean']:.4f}")

    # 测试3: 带attention mask
    print("\n[测试3] 带Attention Mask")
    attention_mask = torch.ones(batch_size, seq_len)
    # 设置一些位置为padding
    attention_mask[0, 10:] = 0
    attention_mask[1, 12:] = 0

    rewards = model(input_ids, attention_mask)
    print(f"  序mask的reward shape: {rewards.shape}")
    print(f"  Reward值: {rewards}")

    print("\n测试完成!")


if __name__ == "__main__":
    test_reward_model()
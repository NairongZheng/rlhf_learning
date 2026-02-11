"""
价值模型（Value Model）- 用于PPO算法

估计状态价值V(s)，用于计算advantage
"""

import torch
import torch.nn as nn
from typing import Optional
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.modules.text_encoder import TextEncoder


class ValueModel(nn.Module):
    """
    价值模型 - 估计状态价值V(s)
    
    架构：
        输入tokens → TextEncoder → 取最后一个token的hidden state → value_head → scalar value
    
    使用场景：
        - PPO: 与Policy Model配合，用于计算advantage
        - 训练时：用当前状态预测未来累积奖励
    
    与Reward Model的区别：
        - Reward Model: 评估完整的(prompt, completion)对，输出奖励分数
        - Value Model: 评估状态（部分生成的序列），输出预期累积奖励
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
        初始化价值模型
        
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
        
        # 2. Value Head：将hidden state映射到标量value
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # 初始化value head（使用较小的权重）
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value_head.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播 - 计算状态价值
        
        Args:
            input_ids: 输入token IDs，shape [batch, seq_len]
            attention_mask: 可选的attention mask
        
        Returns:
            values: shape [batch]，每个状态的价值估计
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
        
        # 3. 通过value head计算value
        values = self.value_head(last_hidden).squeeze(-1)  # [batch]
        
        return values
    
    def compute_value_loss(
        self,
        input_ids: torch.Tensor,
        returns: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        计算Value Model的训练损失
        
        使用MSE loss: L = (V(s) - G)^2
        其中G是实际的return（累积奖励）
        
        Args:
            input_ids: 输入token IDs，shape [batch, seq_len]
            returns: 实际的returns（目标值），shape [batch]
            attention_mask: 可选的attention mask
        
        Returns:
            loss: 损失值
            metrics: 指标字典
        """
        # 1. 计算预测的value
        predicted_values = self.forward(input_ids, attention_mask)  # [batch]
        
        # 2. 计算MSE loss
        loss = nn.functional.mse_loss(predicted_values, returns)
        
        # 3. 计算补充指标
        mae = torch.abs(predicted_values - returns).mean()  # 平均绝对误差
        
        metrics = {
            'loss': loss.item(),
            'mae': mae.item(),
            'value_mean': predicted_values.mean().item(),
            'value_std': predicted_values.std().item(),
            'return_mean': returns.mean().item(),
            'return_std': returns.std().item()
        }
        
        return loss, metrics


if __name__ == "__main__":
    # 简单测试
    print("测试Value Model...")
    
    model = ValueModel(
        vocab_size=1000,
        hidden_dim=128,
        num_heads=4,
        num_layers=1,
        max_seq_len=64
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 参数量: {total_params:,}")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    values = model(input_ids)
    print(f"✅ Values shape: {values.shape}")
    print(f"✅ Values: {values}")
    
    # 测试value loss
    returns = torch.randn(batch_size)
    loss, metrics = model.compute_value_loss(input_ids, returns)
    print(f"✅ Loss: {metrics['loss']:.4f}")
    print(f"✅ MAE: {metrics['mae']:.4f}")

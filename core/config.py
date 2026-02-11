"""
核心配置基类

所有训练配置的基类，包含通用参数
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseConfig:
    """
    所有配置的基类

    包含模型和训练的基础参数，SFT和RL配置都会继承这个类
    """
    # 模型基础参数
    hidden_dim: int = 128  # 隐藏层维度
    num_heads: int = 4  # 注意力头数
    num_layers: int = 1  # Transformer层数
    dropout: float = 0.1  # Dropout率

    # 训练参数
    learning_rate: float = 1e-4  # 学习率
    weight_decay: float = 0.01  # 权重衰减
    batch_size: int = 4  # 批次大小
    max_grad_norm: float = 1.0  # 梯度裁剪范数

    # 其他
    debug: bool = False  # 是否开启调试模式
    seed: int = 42  # 随机种子

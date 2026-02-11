"""
RL训练配置

包含GRPO、DPO、PPO等算法的配置类
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import BaseConfig


@dataclass
class RLConfig(BaseConfig):
    """
    RL训练基础配置

    所有RL算法的通用配置基类
    """
    # 算法选择
    algorithm: str = "grpo"  # "grpo" | "dpo" | "ppo"

    # RL特定的训练参数
    learning_rate: float = 1e-5  # RL通常使用更小的学习率
    kl_coef: float = 0.0  # KL散度惩罚系数（防止policy偏离reference太远）

    # 生成参数
    max_new_tokens: int = 128  # 最大生成长度
    temperature: float = 1.0  # 生成温度
    top_k: int = 0  # Top-K采样（0表示不使用）
    top_p: float = 1.0  # Top-P（nucleus）采样

    # 模型相关
    vocab_size: int = 151657  # Qwen2.5词表大小
    max_seq_len: int = 512  # 最大序列长度


@dataclass
class GRPOConfig(RLConfig):
    """
    GRPO算法配置

    Group Relative Policy Optimization - DeepSeek-V3论文中使用的方法
    核心思想：在一组生成样本中进行组内相对比较，无需Value Model
    """
    algorithm: str = "grpo"

    # GRPO核心参数
    num_generations: int = 4  # 每个prompt生成的样本数（G值），越大variance越小但计算量越大
    clip_range_low: float = 0.1  # clip下界，控制policy下降幅度
    clip_range_high: float = 0.2  # clip上界，控制policy上升幅度

    # Advantage计算
    scale_rewards: bool = True  # 是否对reward进行组内归一化（强烈推荐True）
    reward_baseline: str = "mean"  # baseline类型："mean" | "median" | "none"

    # Reward设置
    use_reward_model: bool = False  # False时使用自定义reward函数
    reward_func: Optional[Callable] = None  # 自定义reward函数：(prompt, completion) -> float

    # 生成多样性参数
    temperature: float = 1.0  # 建议保持1.0或略高，增加多样性
    top_p: float = 0.9  # 可以适当降低增加多样性


@dataclass
class DPOConfig(RLConfig):
    """
    DPO算法配置

    Direct Preference Optimization - 直接从偏好数据优化policy
    优势：无需Reward Model，实现简单，内存效率高
    """
    algorithm: str = "dpo"

    # DPO核心参数
    beta: float = 0.1  # 温度参数（控制KL散度约束强度）
    # β 参数选择指南：
    #   - 0.01-0.05: 宽松约束，允许较大改变（需要大幅改进时）
    #   - 0.1-0.5:   平衡约束，适合大多数情况（推荐）✅
    #   - 0.5-1.0:   较强约束，适合微调已经不错的模型
    #   - >1.0:      很强约束，只做最小改动
    # 原理：β越大→policy越接近reference，β越小→policy自由度越大
    # 效果：虽然分布差异小，但在关键决策点上可产生显著行为差异

    loss_type: str = "sigmoid"  # 损失类型："sigmoid"（推荐）| "hinge" | "ipo"
    label_smoothing: float = 0.0  # 标签平滑（0-1之间，0表示不使用）

    # Reference model设置
    reference_free: bool = False  # 是否使用reference-free模式（实验性）


@dataclass
class PPOConfig(RLConfig):
    """
    PPO算法配置

    Proximal Policy Optimization - 经典RL算法
    优势：理论保证强，性能最优
    劣势：实现复杂，需要Value Model，内存占用大
    """
    algorithm: str = "ppo"

    # PPO核心参数
    clip_range: float = 0.2  # PPO clip范围（symmetric），控制policy更新幅度
    value_clip_range: float = 0.2  # Value function clip范围

    # GAE (Generalized Advantage Estimation)参数
    # GAE理论：在bias和variance之间取得平衡
    gamma: float = 0.99  # 折扣因子γ，控制未来奖励的权重
    gae_lambda: float = 0.95  # GAE的λ参数（权衡bias-variance）
                               # λ=0时为TD(0)高bias低variance，λ=1时为Monte Carlo低bias高variance

    # 多步训练相关
    num_chunks: int = 3  # 将completion分成多少个chunk（T），用于多步训练
                         # T=1: 单步训练（传统方式）
                         # T=3: 多步训练（可以观察GAE的时序计算过程）
    chunk_strategy: str = "fixed"  # chunk划分策略："fixed"=固定长度划分

    # 损失函数权重
    value_loss_coef: float = 0.5  # Value loss权重 c1
    entropy_coef: float = 0.01  # 熵正则化权重 c2（鼓励探索）

    # 训练策略
    ppo_epochs: int = 4  # 每批数据更新几次（提高样本效率）
    num_mini_batches: int = 4  # 将一批数据分成几个mini-batch
    max_grad_norm: float = 1.0  # 梯度裁剪阈值（防止梯度爆炸）

    # Reward设置
    use_reward_model: bool = True  # PPO通常需要Reward Model
    normalize_advantages: bool = True  # 是否归一化advantages（推荐开启）

    # 生成参数（调整以适配多步训练）
    max_new_tokens: int = 60  # 增加生成长度，确保每个chunk有足够的tokens
                               # 60 tokens分3个chunk，每个约20 tokens，比较合理

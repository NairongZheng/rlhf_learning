"""
RL (强化学习) 模块

包含GRPO、DPO、PPO等RL算法的实现
"""

from .config_rl import RLConfig, GRPOConfig, DPOConfig, PPOConfig

__version__ = '0.1.0'
__all__ = ['RLConfig', 'GRPOConfig', 'DPOConfig', 'PPOConfig']
"""
核心共享组件

包含所有SFT和RL训练共享的基础组件
"""

from .config import BaseConfig
from .tokenizers import QwenTokenizerWrapper

__all__ = ['BaseConfig', 'QwenTokenizerWrapper']

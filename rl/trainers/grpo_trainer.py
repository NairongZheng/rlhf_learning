"""
GRPO Trainer - Group Relative Policy Optimization训练器

DeepSeek-V3论文中使用的RL算法，特点是无需Value Model，通过组内比较计算advantage
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Callable, Union
import copy
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.models.policy_model import PolicyModel
from rl.models.reward_model import RewardModel
from rl.losses.grpo_loss import compute_grpo_loss
from rl.utils.rl_utils import compute_advantages
from rl.config_rl import GRPOConfig


class GRPOTrainer:
    """
    GRPO训练器

    训练流程：
        1. 生成阶段：给定prompts，用policy生成G个completions
        2. 评分阶段：用reward model或自定义函数计算每个completion的reward
        3. Advantage计算：组内归一化得到advantages
        4. 优化阶段：用GRPO loss更新policy

    核心特点：
        - 无需Value Model（相比PPO更简单）
        - 组内相对比较（减少variance）
        - 灵活的reward设计（可以用reward model或自定义函数）
    """

    def __init__(
        self,
        config: GRPOConfig,
        policy_model: PolicyModel,
        tokenizer,  # 用于将text转换为token IDs
        ref_model: Optional[PolicyModel] = None,
        reward_model: Optional[RewardModel] = None,
        reward_func: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        初始化GRPO Trainer

        Args:
            config: GRPO配置
            policy_model: 策略模型（需要训练）
            tokenizer: tokenizer，用于将prompt转换为token IDs
            ref_model: reference model（固定，不更新）
                      如果为None，会复制policy_model作为reference
            reward_model: 奖励模型（可选）
            reward_func: 自定义reward函数：(prompt: str, completion: str) -> float
                        如果config.use_reward_model=False，必须提供
            optimizer: 优化器，如果为None会使用AdamW
        """
        self.config = config
        self.policy = policy_model
        self.tokenizer = tokenizer
        self.device = next(policy_model.parameters()).device

        # Reference model（固定，不更新）
        if ref_model is None:
            print("  [初始化] 复制policy model作为reference model...")
            self.ref_model = copy.deepcopy(policy_model)
        else:
            self.ref_model = ref_model

        # 冻结reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Reward设置
        if config.use_reward_model:
            if reward_model is None:
                raise ValueError("config.use_reward_model=True但未提供reward_model")
            self.reward_model = reward_model
            self.reward_model.eval()
            for param in self.reward_model.parameters():
                param.requires_grad = False
            self.reward_func = None
        else:
            if reward_func is None and config.reward_func is None:
                raise ValueError(
                    "config.use_reward_model=False但未提供reward_func\u6216config.reward_func"
                )
            self.reward_model = None
            self.reward_func = reward_func or config.reward_func

        # 优化器
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                policy_model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer

        # 训练统计
        self.step = 0

    def _safe_encode(self, text: str) -> torch.Tensor:
        """
        安全的 encode 方法，兼容不同 tokenizer

        处理两种返回格式：
        1. SimpleTokenizer: 直接返回 torch.Tensor
        2. QwenTokenizerWrapper: 返回 Dict 或 BatchEncoding（类似dict），包含 'input_ids' 和 'attention_mask'

        Args:
            text: 输入文本

        Returns:
            token IDs tensor [1, seq_len]
        """
        result = self.tokenizer.encode(text, return_tensors="pt")

        # 尝试作为 dict-like 对象访问（支持 dict、BatchEncoding 等）
        # BatchEncoding 不是 dict 的子类，但支持 dict-like 访问
        try:
            return result['input_ids'].to(self.device)
        except (KeyError, TypeError):
            # 如果失败，假设是直接返回的 tensor
            return result.to(self.device)

    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """
        单步训练

        Args:
            prompts: 一批prompt文本，List[str]

        Returns:
            metrics: 训练指标字典
        """
        batch_size = len(prompts)
        self.step += 1

        # =====================================
        # 阶段1: 生成阶段
        # =====================================
        # 对每个prompt生成G个completions
        all_completions = []  # List[List[str]]，长度为batch_size，每个元素是G个completion
        all_completion_ids = []  # 存储token IDs用于后续计算

        self.policy.eval()
        with torch.no_grad():
            for prompt in prompts:
                prompt_completions = []
                prompt_completion_ids = []

                # 将prompt转换为token IDs
                prompt_ids = self._safe_encode(prompt)

                # 生成G个样本
                for _ in range(self.config.num_generations):
                    # 修复：接收attention_mask，确保序列长度一致
                    generated_ids, attention_mask = self.policy.generate(
                        prompt_ids,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p,
                        do_sample=True,
                        return_attention_mask=True  # 获取attention_mask
                    )

                    # 解码得到completion文本（只要新生成的部分）
                    completion_ids = generated_ids[:, prompt_ids.shape[1]:]
                    completion_text = self.tokenizer.decode(
                        completion_ids[0],
                        skip_special_tokens=True
                    )

                    prompt_completions.append(completion_text)
                    prompt_completion_ids.append(generated_ids)  # 保存完整的序列（prompt+completion）

                all_completions.append(prompt_completions)
                all_completion_ids.append(prompt_completion_ids)

        # =====================================
        # 阶段2: 评分阶段
        # =====================================
        # 计算每个completion的reward
        rewards_list = []

        for i, prompt in enumerate(prompts):
            prompt_rewards = []

            for j in range(self.config.num_generations):
                completion = all_completions[i][j]

                if self.reward_model is not None:
                    # 使用Reward Model
                    completion_ids = all_completion_ids[i][j]
                    with torch.no_grad():
                        reward = self.reward_model(completion_ids).item()
                else:
                    # 使用自定义reward函数
                    reward = self.reward_func(prompt, completion)

                prompt_rewards.append(reward)

            rewards_list.append(prompt_rewards)

        # 转换为tensor: [batch_size, G]
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)

        # =====================================
        # 阶段3: Advantage计算
        # =====================================
        advantages = compute_advantages(
            rewards=rewards,
            baseline=self.config.reward_baseline,
            scale=self.config.scale_rewards
        )  # [batch_size, G]

        # =====================================
        # 阶段4: 优化阶段
        # =====================================
        self.policy.train()

        # 将所有样本展平成batch维度
        all_input_ids = []
        for i in range(batch_size):
            for j in range(self.config.num_generations):
                all_input_ids.append(all_completion_ids[i][j])

        # 合并为一个batch：[batch_size * G, seq_len]
        # 注意：不同样本可能生成不同长度的序列，需要padding到统一长度
        if len(all_input_ids) > 1:
            # 找到最大长度
            max_len = max(seq.shape[1] for seq in all_input_ids)

            # Padding到最大长度
            padded_input_ids = []
            for seq in all_input_ids:
                if seq.shape[1] < max_len:
                    pad_len = max_len - seq.shape[1]
                    padding = torch.zeros(
                        (seq.shape[0], pad_len),
                        dtype=seq.dtype,
                        device=self.device
                    )
                    padded_seq = torch.cat([seq, padding], dim=1)
                    padded_input_ids.append(padded_seq)
                else:
                    padded_input_ids.append(seq)

            input_ids_batch = torch.cat(padded_input_ids, dim=0)
        else:
            input_ids_batch = all_input_ids[0]
        advantages_flat = advantages.flatten()  # [batch_size * G]

        # 计算policy的log概率
        log_probs = self.policy.get_log_probs(input_ids_batch)  # [batch_size * G]

        # 计算reference的log概率
        with torch.no_grad():
            ref_log_probs = self.ref_model.get_log_probs(input_ids_batch)  # [batch_size * G]

        # 计算GRPO loss
        loss, loss_metrics = compute_grpo_loss(
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages_flat,
            clip_range_low=self.config.clip_range_low,
            clip_range_high=self.config.clip_range_high,
            kl_coef=self.config.kl_coef
        )

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.config.max_grad_norm
        )

        self.optimizer.step()

        # =====================================
        # 收集指标
        # =====================================
        metrics = {
            'step': self.step,
            'reward_mean': rewards.mean().item(),
            'reward_std': rewards.std().item(),
            'reward_max': rewards.max().item(),
            'reward_min': rewards.min().item(),
            **loss_metrics  # 添加loss相关的指标
        }

        return metrics

    def train(
        self,
        train_prompts: List[str],
        num_epochs: int = 10,
        log_interval: int = 1
    ) -> List[Dict[str, float]]:
        """
        完整的训练循环

        Args:
            train_prompts: 训练用的prompt列表
            num_epochs: 训练轮数
            log_interval: 打印日志的间隔

        Returns:
            所有epoch的metrics列表
        """
        all_metrics = []

        print("\n" + "="*60)
        print("GRPO 训练开始")
        print("="*60)
        print(f"  Prompts数量: {len(train_prompts)}")
        print(f"  训练轮数: {num_epochs}")
        print(f"  每个prompt生成数: {self.config.num_generations}")
        print(f"  学习率: {self.config.learning_rate}")
        print("="*60)

        for epoch in range(num_epochs):
            # 训练一步
            metrics = self.train_step(train_prompts)
            all_metrics.append(metrics)

            # 打印日志
            if (epoch + 1) % log_interval == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"  Loss: {metrics['loss']:.4f}")
                print(f"  Reward: {metrics['reward_mean']:.4f} ± {metrics['reward_std']:.4f}")
                print(f"  Reward Range: [{metrics['reward_min']:.4f}, {metrics['reward_max']:.4f}]")
                print(f"  KL: {metrics['kl']:.4f}")
                print(f"  Mean Ratio: {metrics['mean_ratio']:.4f}")
                print(f"  Clip Fraction: {metrics['clip_fraction']:.4f}")

        print("\n" + "="*60)
        print("GRPO 训练完成")
        print("="*60)

        return all_metrics

    @torch.no_grad()
    def generate_samples(
        self,
        prompts: List[str],
        num_samples: int = 1,
        temperature: float = 0.8
    ) -> List[List[str]]:
        """
        使用当前policy生成样本（用于评估）

        Args:
            prompts: prompt列表
            num_samples: 每个prompt生成的样本数
            temperature: 生成温度

        Returns:
            生成的样本，List[List[str]]，外层是prompts，内层是每个prompt的多个样本
        """
        self.policy.eval()
        all_samples = []

        for prompt in prompts:
            prompt_ids = self._safe_encode(prompt)
            samples = []

            for _ in range(num_samples):
                # 修复：接收attention_mask
                generated_ids, attention_mask = self.policy.generate(
                    prompt_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    return_attention_mask=True  # 获取attention_mask
                )

                completion_ids = generated_ids[:, prompt_ids.shape[1]:]
                completion_text = self.tokenizer.decode(
                    completion_ids[0],
                    skip_special_tokens=True
                )
                samples.append(completion_text)

            all_samples.append(samples)

        return all_samples
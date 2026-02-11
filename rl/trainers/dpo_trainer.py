"""
DPO Trainer - Direct Preference Optimization训练器

相比GRPO更简单，无需在线生成，直接从偏好数据学习
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import copy
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.models.policy_model import PolicyModel
from rl.losses.dpo_loss import compute_dpo_loss
from rl.config_rl import DPOConfig


class DPOTrainer:
    """
    DPO训练器
    
    训练流程（相比GRPO更简单）：
        1. 读取偏好数据：(prompt, chosen, rejected)
        2. 计算policy和reference的log概率
        3. 用DPO损失更新policy
    
    核心特点：
        - 无需Reward Model
        - 无需在线生成（离线学习）
        - 直接从人类偏好数据学习
        - 实现简单，内存效率高
    """
    
    def __init__(
        self,
        config: DPOConfig,
        policy_model: PolicyModel,
        tokenizer,
        ref_model: Optional[PolicyModel] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        初始化DPO Trainer
        
        Args:
            config: DPO配置
            policy_model: 策略模型（需要训练）
            tokenizer: tokenizer，用于将text转换为token IDs
            ref_model: reference model（固定，不更新）
                      如果为None，会复制policy_model作为reference
            optimizer: 优化器，如果为None会使用AdamW
        """
        self.config = config
        self.policy = policy_model
        self.tokenizer = tokenizer
        self.device = next(policy_model.parameters()).device
        
        # Reference model（固定，不更新）
        # DPO算法理论：
        # - Reference model 是训练开始时的"快照"，作为对比基准
        # - 防止policy偏离太远（类似正则化）
        # - 绝对不能更新，否则失去约束作用
        if ref_model is None:
            print("  [初始化] 复制policy model作为reference model...")
            self.ref_model = copy.deepcopy(policy_model)

            # 警告：如果policy_model是随机初始化的
            print("\n" + "="*60)
            print("⚠️  警告：使用 policy_model 的副本作为 ref_model")
            print("="*60)
            print("如果 policy_model 是随机初始化的，DPO 效果会很差！")
            print("")
            print("DPO 算法假设：")
            print("  - ref_model 应该是经过 SFT 训练的模型")
            print("  - ref_model 作为'对比基准'，必须能生成合理文本")
            print("")
            print("建议流程：")
            print("  1. 先进行 SFT 训练（sft/train_sft.py）")
            print("  2. 加载 SFT 模型作为 policy_model")
            print("  3. 然后运行 DPO 训练")
            print("")
            print("详见：rl/examples/train_dpo.py 顶部的文档说明")
            print("="*60 + "\n")
        else:
            self.ref_model = ref_model

        # 冻结reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # 验证ref_model的冻结状态（防御性编程）
        assert all(not p.requires_grad for p in self.ref_model.parameters()), \
            "ref_model参数应该是requires_grad=False"
        assert not self.ref_model.training, \
            "ref_model应该处于eval模式"
        
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
    
    def train_step(
        self,
        prompts: List[str],
        chosen: List[str],
        rejected: List[str]
    ) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            prompts: prompt列表
            chosen: 被偏好的回复列表
            rejected: 被拒绝的回复列表
        
        Returns:
            metrics: 训练指标字典
        """
        batch_size = len(prompts)
        self.step += 1
        
        # =====================================
        # 阶段1: 准备输入数据
        # =====================================
        # 将prompt + response拼接成完整序列
        chosen_texts = [p + c for p, c in zip(prompts, chosen)]
        rejected_texts = [p + r for p, r in zip(prompts, rejected)]
        
        # Tokenize
        chosen_ids_list = []
        rejected_ids_list = []

        for c_text, r_text in zip(chosen_texts, rejected_texts):
            chosen_ids = self._safe_encode(c_text)
            rejected_ids = self._safe_encode(r_text)

            chosen_ids_list.append(chosen_ids)
            rejected_ids_list.append(rejected_ids)
        
        # 合并成batch（假设长度相同，实际使用中需要padding）
        chosen_ids_batch = torch.cat(chosen_ids_list, dim=0)  # [batch, seq_len]
        rejected_ids_batch = torch.cat(rejected_ids_list, dim=0)  # [batch, seq_len]
        
        # =====================================
        # 阶段2: 计算log概率
        # =====================================
        self.policy.train()
        
        # Policy的log概率
        policy_chosen_logps = self.policy.get_log_probs(chosen_ids_batch)  # [batch]
        policy_rejected_logps = self.policy.get_log_probs(rejected_ids_batch)  # [batch]
        
        # Reference的log概率（不计算梯度）
        with torch.no_grad():
            ref_chosen_logps = self.ref_model.get_log_probs(chosen_ids_batch)  # [batch]
            ref_rejected_logps = self.ref_model.get_log_probs(rejected_ids_batch)  # [batch]
        
        # =====================================
        # 阶段3: 计算DPO loss
        # =====================================
        loss, loss_metrics = compute_dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=ref_chosen_logps,
            reference_rejected_logps=ref_rejected_logps,
            beta=self.config.beta,
            loss_type=self.config.loss_type,
            label_smoothing=self.config.label_smoothing
        )

        # 计算 KL 散度（用于监控）
        # KL(π_policy || π_ref) ≈ E[log π_policy - log π_ref]
        with torch.no_grad():
            # 对chosen和rejected分别计算KL，然后取平均
            kl_chosen = (policy_chosen_logps - ref_chosen_logps).mean()
            kl_rejected = (policy_rejected_logps - ref_rejected_logps).mean()
            kl_divergence = (kl_chosen + kl_rejected) / 2

            # KL散度过大时发出警告
            if kl_divergence > 1.5:
                print(f"    ⚠️  KL散度={kl_divergence:.3f} 较大，policy偏离reference较远")
                print(f"        建议：增加β参数（当前β={self.config.beta}）")
            elif kl_divergence > 1.0:
                print(f"    ℹ️  KL散度={kl_divergence:.3f} 适中")

        # =====================================
        # 阶段4: 反向传播和优化
        # =====================================
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
            'kl_divergence': kl_divergence.item(),  # 添加KL散度监控
            'beta': self.config.beta,  # 记录当前β值
            **loss_metrics  # 添加loss相关的指标
        }

        return metrics
    
    def train(
        self,
        train_data: List[Dict[str, str]],
        num_epochs: int = 10,
        log_interval: int = 1
    ) -> List[Dict[str, float]]:
        """
        完整的训练循环
        
        Args:
            train_data: 训练数据，List[{"prompt": str, "chosen": str, "rejected": str}]
            num_epochs: 训练轮数
            log_interval: 打印日志的间隔
        
        Returns:
            所有epoch的metrics列表
        """
        all_metrics = []
        
        print("\n" + "="*60)
        print("DPO 训练开始")
        print("="*60)
        print(f"  数据量: {len(train_data)}")
        print(f"  训练轮数: {num_epochs}")
        print(f"  Beta: {self.config.beta}")
        print(f"  Loss Type: {self.config.loss_type}")
        print(f"  学习率: {self.config.learning_rate}")
        print("="*60)
        
        for epoch in range(num_epochs):
            # 提取prompts, chosen, rejected
            prompts = [item['prompt'] for item in train_data]
            chosen = [item['chosen'] for item in train_data]
            rejected = [item['rejected'] for item in train_data]
            
            # 训练一步
            metrics = self.train_step(prompts, chosen, rejected)
            all_metrics.append(metrics)
            
            # 打印日志
            if (epoch + 1) % log_interval == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"  Loss: {metrics['loss']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  KL Divergence: {metrics['kl_divergence']:.4f}  (β={metrics['beta']})")
                print(f"  Chosen Reward: {metrics['chosen_reward_mean']:.4f}")
                print(f"  Rejected Reward: {metrics['rejected_reward_mean']:.4f}")
                print(f"  Reward Margin: {metrics['reward_margin']:.4f}")
        
        print("\n" + "="*60)
        print("DPO 训练完成")
        print("="*60)
        
        return all_metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        eval_data: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """
        评估模型在验证集上的表现

        Args:
            eval_data: 评估数据

        Returns:
            评估指标
        """
        self.policy.eval()
        self.ref_model.eval()  # 防御性编程：显式确保ref_model处于eval模式

        prompts = [item['prompt'] for item in eval_data]
        chosen = [item['chosen'] for item in eval_data]
        rejected = [item['rejected'] for item in eval_data]
        
        # 准备数据
        chosen_texts = [p + c for p, c in zip(prompts, chosen)]
        rejected_texts = [p + r for p, r in zip(prompts, rejected)]
        
        chosen_ids_list = []
        rejected_ids_list = []

        for c_text, r_text in zip(chosen_texts, rejected_texts):
            chosen_ids = self._safe_encode(c_text)
            rejected_ids = self._safe_encode(r_text)

            chosen_ids_list.append(chosen_ids)
            rejected_ids_list.append(rejected_ids)
        
        chosen_ids_batch = torch.cat(chosen_ids_list, dim=0)
        rejected_ids_batch = torch.cat(rejected_ids_list, dim=0)
        
        # 计算log概率
        policy_chosen_logps = self.policy.get_log_probs(chosen_ids_batch)
        policy_rejected_logps = self.policy.get_log_probs(rejected_ids_batch)
        
        ref_chosen_logps = self.ref_model.get_log_probs(chosen_ids_batch)
        ref_rejected_logps = self.ref_model.get_log_probs(rejected_ids_batch)
        
        # 计算DPO loss
        loss, metrics = compute_dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=ref_chosen_logps,
            reference_rejected_logps=ref_rejected_logps,
            beta=self.config.beta,
            loss_type=self.config.loss_type
        )
        
        return metrics

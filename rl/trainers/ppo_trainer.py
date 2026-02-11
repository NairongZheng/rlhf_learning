"""
PPO Trainer - Proximal Policy Optimization训练器

经典RL算法，理论保证强，但实现较复杂
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass
import copy
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.models.policy_model import PolicyModel
from rl.models.value_model import ValueModel
from rl.models.reward_model import RewardModel
from rl.losses.ppo_loss import compute_ppo_loss
from rl.utils.rl_utils import compute_gae, whiten_advantages
from rl.config_rl import PPOConfig


@dataclass
class RolloutData:
    """
    PPO采样数据的容器（多步版本，支持T=1或T>1）

    存储在当前策略下采样得到的trajectory数据，用于后续的策略更新

    多步训练说明（T=3）：
    ========================================
    - 将一个completion分成T个chunk
    - 每个chunk计算独立的reward、value和log_prob
    - GAE会反向计算，利用"未来"的value估计advantage

    为什么可以用"未来数据"？
    - Rollout阶段：完整生成completion，所有数据都已确定
    - GAE阶段：反向遍历已有数据，使用V(s_{t+1})来估计advantage
    - 这不是"作弊"，而是离线RL的标准做法

    Attributes:
        input_ids: 完整序列（prompt + completion）[batch, seq_len]
        rewards: 每个chunk的奖励值 [batch, T]，T=1时为[batch, 1]
        old_log_probs: 旧策略的对数概率 log π_θ_old(a|s) [batch, T]
        old_values: Value model的估计 V_φ(s) [batch, T]
        dones: episode结束标记 [batch, T]，只有最后一个时间步为1
        chunk_boundaries: 每个chunk的结束位置 List[List[int]]，长度为[batch][T]
        attention_mask: 可选的attention mask [batch, seq_len]
    """
    input_ids: torch.Tensor           # [batch, seq_len]
    rewards: torch.Tensor              # [batch, T]
    old_log_probs: torch.Tensor        # [batch, T]
    old_values: torch.Tensor           # [batch, T]
    dones: torch.Tensor                # [batch, T]
    chunk_boundaries: List[List[int]]  # [batch][T] - 每个chunk的结束位置
    attention_mask: Optional[torch.Tensor] = None  # [batch, seq_len]


class PPOTrainer:
    """
    PPO训练器
    
    训练流程：
        1. 生成阶段：收集trajectories（状态、动作、奖励序列）
        2. 计算advantages：使用GAE
        3. 多轮优化：对同一批数据进行多次更新
        4. 更新policy和value model
    
    特点：
        - 需要Value Model
        - 使用GAE计算advantages
        - 多轮优化（提高样本效率）
        - 理论保证强
    """
    
    def __init__(
        self,
        config: PPOConfig,
        policy_model: PolicyModel,
        value_model: ValueModel,
        tokenizer,
        reward_model: Optional[RewardModel] = None,
        reward_func: Optional[Callable] = None,
        policy_optimizer: Optional[torch.optim.Optimizer] = None,
        value_optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        初始化PPO Trainer
        
        Args:
            config: PPO配置
            policy_model: 策略模型
            value_model: 价值模型
            tokenizer: tokenizer
            reward_model: 奖励模型（可选）
            reward_func: 自定义reward函数
            policy_optimizer: policy优化器
            value_optimizer: value优化器
        """
        self.config = config
        self.policy = policy_model
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.device = next(policy_model.parameters()).device
        
        # Reward设置
        if config.use_reward_model:
            if reward_model is None:
                raise ValueError("use_reward_model=True但未提供reward_model")
            self.reward_model = reward_model
            self.reward_model.eval()
            for param in self.reward_model.parameters():
                param.requires_grad = False
            self.reward_func = None
        else:
            if reward_func is None:
                raise ValueError("use_reward_model=False但未提供reward_func")
            self.reward_model = None
            self.reward_func = reward_func
        
        # 优化器
        if policy_optimizer is None:
            self.policy_optimizer = torch.optim.AdamW(
                policy_model.parameters(),
                lr=config.learning_rate
            )
        else:
            self.policy_optimizer = policy_optimizer
        
        if value_optimizer is None:
            self.value_optimizer = torch.optim.AdamW(
                value_model.parameters(),
                lr=config.learning_rate
            )
        else:
            self.value_optimizer = value_optimizer

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
    
    def _collect_rollout_data(self, prompts: List[str]) -> RolloutData:
        """
        收集训练数据（多步版本，T=num_chunks）

        PPO理论对应：
        ========================================
        在当前策略 π_θ_old 下采样trajectories
        收集：(state, action, reward, log_prob, value)

        多步改造说明：
        - 原本每个prompt生成1个completion，得到1个reward
        - 现在将completion分成T个chunk，每个chunk计算独立的reward/value/log_prob
        - 这样可以体验GAE的时序计算过程

        GAE时序计算说明：
        - 在此阶段，我们完整生成了completion，所有数据都已确定
        - 后续GAE计算时，可以"反向"使用V(s_{t+1})，因为轨迹已完成
        - 这不是"作弊"，而是离线优化的标准做法

        Args:
            prompts: 输入的prompt列表

        Returns:
            RolloutData: 包含所有采样数据的数据类
                - input_ids: 完整序列（prompt + completion）[batch, seq_len]
                - rewards: 每个chunk的奖励值 [batch, T]
                - old_log_probs: 旧策略的对数概率 [batch, T]
                - old_values: Value model的估计 [batch, T]
                - dones: episode结束标记 [batch, T]
                - chunk_boundaries: 每个chunk的结束位置
        """
        batch_size = len(prompts)
        T = self.config.num_chunks  # 时间步数

        all_input_ids = []
        all_rewards = []          # [batch, T]
        all_log_probs = []        # [batch, T]
        all_values = []           # [batch, T]
        all_dones = []            # [batch, T]
        all_chunk_boundaries = [] # [batch][T]
        all_attention_masks = []  # [batch, seq_len]

        self.policy.eval()  # 采样阶段不需要梯度
        self.value_model.eval()

        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                prompt_ids = self._safe_encode(prompt)
                prompt_len = prompt_ids.shape[1]

                # ========================================
                # 步骤1：生成完整completion
                # ========================================
                # 重要：先完整生成，再分chunk
                # 这样不影响生成过程，只是后续分析时用多步
                generated_ids, attention_mask = self.policy.generate(
                    prompt_ids,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,  # PPO需要采样
                    return_attention_mask=True
                )

                # ========================================
                # 步骤2：确定chunk边界（固定长度划分）
                # ========================================
                # 将completion等分成T个chunk
                completion_ids = generated_ids[:, prompt_len:]  # 只取completion部分
                completion_len = attention_mask[0, prompt_len:].sum().item()  # 实际长度

                # 将completion等分成T个chunk
                chunk_size = max(1, completion_len // T)
                chunk_boundaries_i = []
                for t in range(T):
                    end_pos = min(prompt_len + (t + 1) * chunk_size, generated_ids.shape[1])
                    chunk_boundaries_i.append(end_pos)

                # 确保最后一个chunk包含所有剩余tokens
                chunk_boundaries_i[-1] = prompt_len + completion_len

                # ========================================
                # 步骤3：为每个chunk计算reward, value, log_prob
                # ========================================
                chunk_rewards = []
                chunk_values = []
                chunk_log_probs = []

                for t in range(T):
                    # 当前时间步的状态：prompt + chunk_0 + ... + chunk_t
                    state_ids = generated_ids[:, :chunk_boundaries_i[t]]

                    # ========================================
                    # 计算reward（使用到当前chunk的部分completion）
                    # ========================================
                    partial_completion = self.tokenizer.decode(
                        state_ids[0, prompt_len:],
                        skip_special_tokens=True
                    )

                    if self.reward_model is not None:
                        reward = self.reward_model(state_ids).item()
                    else:
                        reward = self.reward_func(prompt, partial_completion)

                    chunk_rewards.append(reward)

                    # ========================================
                    # 计算value（V(s_t) = 从当前状态继续能获得的期望回报）
                    # ========================================
                    value = self.value_model(state_ids).item()
                    chunk_values.append(value)

                    # ========================================
                    # 计算log_prob（为每个chunk独立计算）
                    # ========================================
                    if t == 0:
                        start_pos = prompt_len
                    else:
                        start_pos = chunk_boundaries_i[t-1]
                    end_pos = chunk_boundaries_i[t]

                    # 为当前chunk计算log_prob
                    log_prob = self.policy.get_log_probs_partial(
                        generated_ids,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        attention_mask=attention_mask
                    ).item()
                    chunk_log_probs.append(log_prob)

                # ========================================
                # 步骤4：设置dones（只有最后一步done=1）
                # ========================================
                dones = [0.0] * (T - 1) + [1.0]  # 前T-1步为0，最后一步为1

                # ========================================
                # 步骤5：记录数据
                # ========================================
                all_input_ids.append(generated_ids[0])
                all_rewards.append(chunk_rewards)
                all_log_probs.append(chunk_log_probs)
                all_values.append(chunk_values)
                all_dones.append(dones)
                all_chunk_boundaries.append(chunk_boundaries_i)
                all_attention_masks.append(attention_mask[0])

        # ============================================
        # 步骤6：Padding处理（统一序列长度）
        # ============================================
        # 将所有序列padding到batch内最大长度
        max_len = max(seq.shape[0] for seq in all_input_ids)

        padded_input_ids = []
        padded_attention_masks = []

        for i in range(batch_size):
            seq = all_input_ids[i]
            mask = all_attention_masks[i]
            seq_len = seq.shape[0]

            if seq_len < max_len:
                pad_len = max_len - seq_len
                # Padding input_ids
                padding = torch.zeros(
                    (pad_len,),
                    dtype=seq.dtype,
                    device=self.device
                )
                padded_seq = torch.cat([seq, padding], dim=0)
                padded_input_ids.append(padded_seq)

                # Padding attention_mask
                mask_padding = torch.zeros(
                    (pad_len,),
                    dtype=mask.dtype,
                    device=self.device
                )
                padded_mask = torch.cat([mask, mask_padding], dim=0)
                padded_attention_masks.append(padded_mask)
            else:
                padded_input_ids.append(seq)
                padded_attention_masks.append(mask)

        # Stack成batch
        input_ids_batch = torch.stack(padded_input_ids, dim=0)  # [batch, max_len]
        attention_mask_batch = torch.stack(padded_attention_masks, dim=0)  # [batch, max_len]

        # 转换为tensor [batch, T]
        rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32, device=self.device)
        log_probs_tensor = torch.tensor(all_log_probs, dtype=torch.float32, device=self.device)
        values_tensor = torch.tensor(all_values, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(all_dones, dtype=torch.float32, device=self.device)

        # 返回数据类
        return RolloutData(
            input_ids=input_ids_batch,
            rewards=rewards_tensor,
            old_log_probs=log_probs_tensor,
            old_values=values_tensor,
            dones=dones_tensor,
            chunk_boundaries=all_chunk_boundaries,
            attention_mask=attention_mask_batch
        )

    def _compute_gae_advantages(
        self,
        rewards: torch.Tensor,      # [batch, T]
        values: torch.Tensor,       # [batch, T]
        dones: torch.Tensor         # [batch, T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用GAE（Generalized Advantage Estimation）计算advantages

        GAE理论（Schulman et al. 2016）：
        ========================================
        目标：在bias和variance之间取得平衡

        核心公式：
        1. TD error: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        2. GAE: A_t = Σ_{l=0}^∞ (γλ)^l · δ_{t+l}

        参数说明：
        - γ (gamma): 折扣因子，控制未来奖励的权重
        - λ (lambda): GAE参数，控制bias-variance trade-off
          * λ=0: A_t = δ_t（高bias，低variance）
          * λ=1: A_t = Σ γ^l·r_{t+l} - V(s_t)（低bias，高variance）

        注意：对于文本生成任务，T=1（单步生成），但我们仍使用GAE接口保持代码统一性

        Args:
            rewards: 每个时间步的奖励 [batch, T]
            values: Value model的估计值 [batch, T]
            dones: 每个时间步是否结束 [batch, T]

        Returns:
            advantages: GAE优势函数 [batch, T]
            returns: 目标值（advantages + values） [batch, T]
        """
        # ============================================
        # 调用GAE实现
        # ============================================
        # 使用rl_utils.compute_gae计算完整的GAE
        # 即使T=1时，也保持代码的统一性和正确性
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )

        # ============================================
        # Advantage归一化（可选，默认开启）
        # ============================================
        # 作用：标准化到均值0，方差1，提高训练稳定性
        if self.config.normalize_advantages:
            advantages = whiten_advantages(advantages)

        return advantages, returns

    def _update_policy_and_value(
        self,
        rollout_data: RolloutData,
        advantages: torch.Tensor,  # [batch, T]
        returns: torch.Tensor      # [batch, T]
    ) -> Dict[str, float]:
        """
        策略和价值函数更新（多步版本）

        PPO理论对应：
        ========================================
        目标：最大化 J(θ) = E[L_CLIP + c1·L_VF - c2·L_ENT]

        核心创新：Clipped Surrogate Objective
        - 防止policy更新过大（trust region的近似）
        - 保证训练稳定性

        损失函数：
        1. Policy Loss (L_CLIP):
           L_CLIP = -E[min(r·A, clip(r, 1-ε, 1+ε)·A)]
           其中 r = π_new / π_old

        2. Value Loss (L_VF):
           L_VF = E[(V - G)^2]
           其中 G = returns（GAE计算的目标值）

        3. Entropy Loss (L_ENT):
           L_ENT = -E[H(π)]
           鼓励探索，防止策略过早收敛

        超参数：
        - ppo_epochs: 每批数据重复使用次数（提高样本效率）
        - clip_range: ε，裁剪范围（默认0.2）
        - value_loss_coef: c1，value loss权重（默认0.5）
        - entropy_coef: c2，entropy权重（默认0.01）

        Args:
            rollout_data: 采样的数据
            advantages: GAE计算的优势函数 [batch, T]
            returns: 目标值 [batch, T]

        Returns:
            metrics: 训练指标（loss, clip_fraction等）
        """
        batch_size, T = advantages.shape

        # ============================================
        # Flatten到 [batch*T] 维度
        # ============================================
        # 原因：loss计算需要一维的advantages和returns
        # 每个(sample, timestep)对应一个独立的训练样本
        advantages_flat = advantages.flatten()  # [batch*T]
        returns_flat = returns.flatten()        # [batch*T]
        old_log_probs_flat = rollout_data.old_log_probs.flatten()  # [batch*T]

        # ============================================
        # 扩展input_ids（简化方案）
        # ============================================
        # 教学简化：所有时间步使用相同的完整序列
        # 实际应该为每个时间步使用对应的partial sequence
        # 但这样可以最小化代码改动，专注于理解GAE原理
        input_ids_expanded = rollout_data.input_ids.unsqueeze(1).expand(-1, T, -1)  # [batch, T, seq_len]
        input_ids_flat = input_ids_expanded.reshape(batch_size * T, -1)  # [batch*T, seq_len]

        all_policy_losses = []
        all_value_losses = []
        all_entropies = []
        all_clip_fractions = []

        # ============================================
        # 多轮优化循环（提高样本效率）
        # ============================================
        # PPO关键：对同一批数据进行多次优化
        # 通过clipping防止更新过大

        for epoch in range(self.config.ppo_epochs):
            # ========================================
            # 计算新策略的输出
            # ========================================
            self.policy.train()
            self.value_model.train()

            # 新策略的对数概率
            # log π_θ_new(a|s)
            new_log_probs_flat = self.policy.get_log_probs(input_ids_flat)  # [batch*T]

            # 新的value估计
            # V_φ_new(s)
            new_values_flat = self.value_model(input_ids_flat)  # [batch*T]

            # ========================================
            # 计算PPO损失
            # ========================================
            # 调用ppo_loss.py中的compute_ppo_loss
            # 这个函数实现了完整的PPO损失计算
            loss, loss_metrics = compute_ppo_loss(
                log_probs=new_log_probs_flat,           # π_new
                old_log_probs=old_log_probs_flat,       # π_old
                advantages=advantages_flat.detach(),    # A (detach防止梯度回传)
                values=new_values_flat,                 # V_new
                returns=returns_flat,                   # G (GAE targets)
                clip_range=self.config.clip_range,      # ε
                value_clip_range=self.config.value_clip_range,
                value_loss_coef=self.config.value_loss_coef,  # c1
                entropy_coef=self.config.entropy_coef,         # c2
                normalize_advantages=False  # 已经在GAE阶段归一化
            )

            # ========================================
            # 反向传播和梯度更新
            # ========================================
            # 更新policy参数：θ ← θ + α·∇_θ J(θ)
            self.policy_optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.config.max_grad_norm
            )

            self.policy_optimizer.step()

            # 收集每轮的metrics
            all_policy_losses.append(loss_metrics['policy_loss'])
            all_value_losses.append(loss_metrics['value_loss'])
            all_entropies.append(loss_metrics['entropy'])
            all_clip_fractions.append(loss_metrics['clip_fraction'])

        # ============================================
        # 汇总训练指标
        # ============================================
        # 计算每个时间步的平均reward（用于监控）
        reward_per_timestep = rollout_data.rewards.mean(dim=0)  # [T]

        metrics = {
            'step': self.step,
            'reward_mean': rollout_data.rewards.mean().item(),
            'reward_std': rollout_data.rewards.std().item(),
            'policy_loss': sum(all_policy_losses) / len(all_policy_losses),
            'value_loss': sum(all_value_losses) / len(all_value_losses),
            'entropy': sum(all_entropies) / len(all_entropies),
            'clip_fraction': sum(all_clip_fractions) / len(all_clip_fractions),
            **loss_metrics  # 包含其他详细指标
        }

        # 添加每个时间步的reward（用于分析时序变化）
        for t in range(T):
            metrics[f'reward_t{t}'] = reward_per_timestep[t].item()

        return metrics

    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """
        单步PPO训练 - 主控函数

        PPO训练流程：
        ========================================
        1. 数据收集（Sampling）：在当前策略下采样trajectories
        2. 计算Advantage（GAE）：使用Generalized Advantage Estimation
        3. 策略更新（Optimization）：多轮优化policy和value model

        理论参考：
        - PPO论文: Schulman et al. 2017
        - GAE论文: Schulman et al. 2016

        Args:
            prompts: 训练用的prompt列表

        Returns:
            metrics: 训练指标字典
                - step: 训练步数
                - reward_mean: 平均奖励
                - reward_std: 奖励标准差
                - policy_loss: 策略损失
                - value_loss: 价值损失
                - entropy: 策略熵
                - clip_fraction: 被裁剪的比例
        """
        self.step += 1

        # ============================================
        # 阶段1：数据收集（对应PPO理论的采样阶段）
        # ============================================
        # 在当前策略 π_θ_old 下采样trajectories
        # 收集：(state, action, reward, log_prob, value)
        rollout_data = self._collect_rollout_data(prompts)

        # ============================================
        # 阶段2：计算Advantages（使用GAE公式）
        # ============================================
        # 多步版本（T=num_chunks），直接使用多步数据
        # rollout_data.rewards 已经是 [batch, T]，无需unsqueeze

        # 调用GAE计算
        advantages, returns = self._compute_gae_advantages(
            rewards=rollout_data.rewards,       # [batch, T]
            values=rollout_data.old_values,     # [batch, T]
            dones=rollout_data.dones            # [batch, T]
        )
        # 返回 advantages [batch, T], returns [batch, T]

        # GAE时序计算详解（关键理解点）：
        # ========================================
        # GAE从 t=T-1 反向计算到 t=0：
        #
        # t=2 (最后一步，done=1):
        #   next_value = 0（因为 done=1，episode结束）
        #   δ_2 = r_2 + γ*0 - V(s_2) = r_2 - V(s_2)
        #   A_2 = δ_2
        #
        # t=1 (中间步，done=0):
        #   next_value = V(s_2)（使用"未来"的value）
        #   δ_1 = r_1 + γ*V(s_2) - V(s_1)
        #   A_1 = δ_1 + γλ*A_2
        #      = δ_1 + γλ*(r_2 - V(s_2))
        #
        # t=0 (第一步，done=0):
        #   next_value = V(s_1)（使用"未来"的value）
        #   δ_0 = r_0 + γ*V(s_1) - V(s_0)
        #   A_0 = δ_0 + γλ*A_1
        #      = δ_0 + γλ*(δ_1 + γλ*A_2)
        #
        # 关键理解：
        # - 在Rollout阶段，我们已经完整生成了T个chunk
        # - 所有 r_0, r_1, r_2, V(s_0), V(s_1), V(s_2) 都已经计算好
        # - GAE只是"反向遍历"这些已有数据，计算advantages
        # - 使用V(s_{t+1})不是"预测未来"，而是利用已完成轨迹的信息
        #
        # 这是离线RL的核心特点：
        # - 先完整采样轨迹（Rollout）
        # - 再利用完整轨迹优化策略（使用未来信息是合理的）
        # - 而在线RL（如SARSA）必须实时决策，不能用未来信息
        # ========================================

        # ============================================
        # 阶段3：策略和价值函数更新（多轮优化）
        # ============================================
        # 使用Clipped Surrogate Objective进行多轮优化
        # 防止policy更新过大，保证训练稳定性
        metrics = self._update_policy_and_value(
            rollout_data=rollout_data,
            advantages=advantages,
            returns=returns
        )

        return metrics
    
    def train(
        self,
        train_prompts: List[str],
        num_epochs: int = 10,
        log_interval: int = 1
    ) -> List[Dict[str, float]]:
        """
        完整训练循环
        
        Args:
            train_prompts: 训练prompts
            num_epochs: 训练轮数
            log_interval: 日志打印间隔
        
        Returns:
            所有epoch的metrics
        """
        all_metrics = []
        
        print("\n" + "="*60)
        print("PPO 训练开始")
        print("="*60)
        print(f"  Prompts数量: {len(train_prompts)}")
        print(f"  训练轮数: {num_epochs}")
        print(f"  PPO Epochs: {self.config.ppo_epochs}")
        print(f"  学习率: {self.config.learning_rate}")
        print("="*60)
        
        for epoch in range(num_epochs):
            metrics = self.train_step(train_prompts)
            all_metrics.append(metrics)
            
            if (epoch + 1) % log_interval == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"  Reward: {metrics['reward_mean']:.4f} ± {metrics['reward_std']:.4f}")
                print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                print(f"  Value Loss: {metrics['value_loss']:.4f}")
                print(f"  Entropy: {metrics['entropy']:.4f}")
                print(f"  Clip Fraction: {metrics['clip_fraction']:.4f}")
        
        print("\n" + "="*60)
        print("PPO 训练完成")
        print("="*60)
        
        return all_metrics

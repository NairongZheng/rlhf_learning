"""
策略模型（Policy Model）- RL训练的核心组件

复用SFT的TextEncoder和TextDecoder，添加生成和对数概率计算功能
用于GRPO、DPO、PPO等RL算法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.modules.text_encoder import TextEncoder
from core.modules.text_decoder import TextDecoder


class PolicyModel(nn.Module):
    """
    策略模型 - 用于RL训练

    架构：
        输入tokens → TextEncoder → hidden states → TextDecoder → logits

    核心功能：
        1. forward(): 计算logits和log_probs（用于训练）
        2. generate(): 生成completions（用于采样）
        3. get_log_probs(): 计算给定completion的log概率（用于计算ratio）

    与SFT模型的区别：
        - SFT：使用teacher forcing，给定完整的input_ids计算loss
        - RL Policy：需要采样生成，并计算生成序列的log概率用于policy gradient
    """

    def __init__(
        self,
        vocab_size: int = 151657,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        max_seq_len: int = 512,
        rope_base: int = 10000,
        dropout: float = 0.1,
        norm_type: str = "pre"
    ):
        """
        初始化策略模型

        Args:
            vocab_size: 词汇表大小
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            max_seq_len: 最大序列长度
            rope_base: RoPE的base频率
            dropout: dropout率
            norm_type: LayerNorm类型
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # 1. Text Encoder（backbone）
        self.encoder = TextEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            dropout=dropout,
            norm_type=norm_type
        )

        # 2. Text Decoder（生成头）
        self.decoder = TextDecoder(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            dropout=dropout,
            norm_type=norm_type
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> Union[torch.Tensor, dict]:
        """
        前向传播 - 用于训练

        Args:
            input_ids: 输入token IDs，shape [batch, seq_len]
            attention_mask: 可选的attention mask
            return_dict: 是否返回字典格式

        Returns:
            logits: shape [batch, seq_len, vocab_size]
            或包含logits和log_probs的字典
        """
        # 1. Encoder编码
        hidden_states = self.encoder(input_ids, mask=attention_mask)

        # 2. Decoder解码（生成causal mask用于自回归）
        seq_len = input_ids.shape[1]
        causal_mask = self.decoder.generate_causal_mask(seq_len, input_ids.device)

        # 3. 计算logits
        logits = self.decoder(hidden_states, causal_mask=causal_mask)

        if return_dict:
            return {
                'logits': logits,
                'hidden_states': hidden_states
            }

        return logits

    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算给定序列的对数概率 - 用于RL算法计算ratio

        重要：这个函数计算的是整个序列的平均log概率
        log π(y|x) = (1/T) * Σ log P(y_t | x, y_<t)

        Args:
            input_ids: shape [batch, seq_len]
            attention_mask: 可选的mask

        Returns:
            log_probs: shape [batch]，每个样本的平均log概率
        """
        # 1. 前向传播得到logits
        logits = self.forward(input_ids, attention_mask)  # [batch, seq_len, vocab_size]

        # 2. 计算每个位置的log概率
        # 对于位置t，我们需要P(token_t | tokens_0:t-1)
        # 所以使用logits[t-1]来预测token[t]

        # 取logits的前seq_len-1个位置，对应预测input_ids的后seq_len-1个token
        logits_for_pred = logits[:, :-1, :]  # [batch, seq_len-1, vocab_size]
        target_ids = input_ids[:, 1:]  # [batch, seq_len-1]

        # 3. 计算log softmax
        log_softmax = F.log_softmax(logits_for_pred, dim=-1)  # [batch, seq_len-1, vocab_size]

        # 4. 收集target token的log概率
        # gather: 从log_softmax中选出target_ids对应的log概率
        target_log_probs = torch.gather(
            log_softmax,
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)  # [batch, seq_len-1]

        # 5. 如果有attention_mask，只计算有效位置的平均
        if attention_mask is not None:
            mask = attention_mask[:, 1:].float()  # [batch, seq_len-1]
            # 计算加权平均（只考虑非padding位置）
            log_probs = (target_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1)
        else:
            # 简单平均
            log_probs = target_log_probs.mean(dim=-1)

        return log_probs  # [batch]

    def get_log_probs_partial(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        end_pos: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算部分序列的log概率（用于多步PPO）

        用途：
        - 为每个chunk计算独立的log_prob
        - 例如：chunk_0是tokens [10:20]，计算这部分的平均log_prob

        为什么需要这个方法？
        - 多步PPO中，每个时间步对应一个chunk
        - 需要为每个chunk计算独立的log_prob，用于计算policy ratio
        - get_log_probs()只能计算整个序列的平均值，无法区分chunk

        Args:
            input_ids: 完整序列 [batch, full_seq_len]
            start_pos: 要计算log_prob的起始位置（包含）
            end_pos: 要计算log_prob的结束位置（不包含）
            attention_mask: 注意力掩码 [batch, full_seq_len]

        Returns:
            log_probs: [batch] - 该部分序列的平均log概率

        示例：
            # 假设序列是：[prompt (0-9)] + [chunk_0 (10-19)] + [chunk_1 (20-29)]
            # 计算chunk_0的log_prob：
            log_prob_chunk0 = model.get_log_probs_partial(input_ids, start_pos=10, end_pos=20)

        原理：
            1. 前向传播得到完整序列的logits
            2. 只提取[start_pos:end_pos]区间的logits和targets
            3. 计算该区间的平均log_prob

        注意：
            - logits对应的是预测下一个token
            - 所以logits[:, start_pos:end_pos-1]用于预测input_ids[:, start_pos+1:end_pos]
        """
        # 前向传播得到完整序列的logits
        logits = self.forward(input_ids, attention_mask)  # [batch, seq_len, vocab_size]

        # 只计算[start_pos:end_pos)区间的log_prob
        # logits对应的是预测下一个token，所以：
        # logits[:, start_pos:end_pos-1]用于预测input_ids[:, start_pos+1:end_pos]
        logits_for_pred = logits[:, start_pos:end_pos-1, :]  # [batch, chunk_len-1, vocab_size]
        target_ids = input_ids[:, start_pos+1:end_pos]        # [batch, chunk_len-1]

        # 计算log probabilities
        log_softmax = F.log_softmax(logits_for_pred, dim=-1)  # [batch, chunk_len-1, vocab_size]

        # 提取target tokens的log_probs
        target_log_probs = torch.gather(
            log_softmax,
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)  # [batch, chunk_len-1]

        # 如果提供了attention_mask，只计算有效tokens的平均
        if attention_mask is not None:
            mask = attention_mask[:, start_pos+1:end_pos]  # [batch, chunk_len-1]
            masked_log_probs = target_log_probs * mask
            # 避免除以0
            mask_sum = mask.sum(dim=-1).clamp(min=1)
            log_probs = masked_log_probs.sum(dim=-1) / mask_sum
        else:
            # 没有mask，直接平均
            log_probs = target_log_probs.mean(dim=-1)  # [batch]

        return log_probs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        return_attention_mask: bool = True
    ) -> Union[torch.Tensor, tuple]:
        """
        自回归生成 - 用于采样completions

        支持多种采样策略：
            - Greedy: temperature=0 或 do_sample=False
            - Temperature Sampling: temperature > 0
            - Top-K Sampling: top_k > 0
            - Top-P (Nucleus) Sampling: top_p < 1.0

        Args:
            input_ids: prompt tokens，shape [batch, prompt_len]
            max_new_tokens: 最大生成长度
            temperature: 温度参数（越高越随机）
            top_k: Top-K采样（0表示不使用）
            top_p: Top-P采样（1.0表示不使用）
            do_sample: 是否采样（False时使用greedy）
            pad_token_id: padding token ID
            eos_token_id: 结束token ID
            return_attention_mask: 是否返回attention_mask（True时返回tuple）

        Returns:
            如果return_attention_mask=True:
                (generated_ids, attention_mask):
                    - generated_ids: shape [batch, max_len]，padding到batch内最大长度
                    - attention_mask: shape [batch, max_len]，1表示真实token，0表示padding
            如果return_attention_mask=False:
                generated_ids: shape [batch, max_len]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 使用列表存储每个样本的生成序列（允许不同长度）
        generated_ids_list = [input_ids[i].clone() for i in range(batch_size)]

        # 记录哪些序列已经结束（遇到EOS）
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 自回归生成
        for _ in range(max_new_tokens):
            # 检查是否所有序列都已结束
            if finished.all():
                break

            # 将当前所有序列padding到相同长度（用于batch forward）
            current_lengths = [len(seq) for seq in generated_ids_list]
            max_current_len = max(current_lengths)

            # Padding到当前最大长度
            padded_seqs = []
            for i in range(batch_size):
                seq = generated_ids_list[i]
                if len(seq) < max_current_len:
                    padding = torch.full(
                        (max_current_len - len(seq),),
                        pad_token_id,
                        dtype=seq.dtype,
                        device=device
                    )
                    padded_seq = torch.cat([seq, padding], dim=0)
                else:
                    padded_seq = seq
                padded_seqs.append(padded_seq)

            current_batch = torch.stack(padded_seqs, dim=0)  # [batch, max_current_len]

            # 前向传播得到logits
            logits = self.forward(current_batch)  # [batch, max_current_len, vocab_size]

            # 只需要最后一个位置的logits
            next_token_logits = logits[:, -1, :]  # [batch, vocab_size]

            # 应用temperature
            if temperature > 0 and do_sample:
                next_token_logits = next_token_logits / temperature

            # 应用Top-K过滤
            if top_k > 0 and do_sample:
                next_token_logits = self._top_k_filtering(next_token_logits, top_k)

            # 应用Top-P过滤
            if top_p < 1.0 and do_sample:
                next_token_logits = self._top_p_filtering(next_token_logits, top_p)

            # 计算概率分布
            probs = F.softmax(next_token_logits, dim=-1)

            # 采样或贪婪选择
            if do_sample and temperature > 0:
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = torch.argmax(probs, dim=-1)

            # 对于已结束的序列，用pad token替代
            next_token = torch.where(finished, pad_token_id, next_token)

            # 拼接到各自的序列
            for i in range(batch_size):
                if not finished[i]:
                    generated_ids_list[i] = torch.cat(
                        [generated_ids_list[i], next_token[i:i+1]],
                        dim=0
                    )

            # 更新finished状态
            finished = finished | (next_token == eos_token_id)

        # =====================================
        # 修复：Padding到batch内最大长度
        # =====================================
        # 问题：不同样本可能在不同时间点生成EOS，导致长度不一致
        # 解决：在返回前将所有序列padding到统一长度，同时返回attention_mask

        # 获取当前batch内的最大长度
        current_lengths = [generated_ids_list[i].shape[0] for i in range(batch_size)]
        max_length = max(current_lengths)

        # Padding到统一长度
        padded_ids = []
        attention_masks = []

        for i in range(batch_size):
            seq = generated_ids_list[i]
            seq_len = seq.shape[0]
            pad_len = max_length - seq_len

            if pad_len > 0:
                # 需要padding
                padding = torch.full(
                    (pad_len,),
                    pad_token_id,
                    dtype=seq.dtype,
                    device=device
                )
                padded_seq = torch.cat([seq, padding], dim=0)

                # Attention mask：1表示真实token，0表示padding
                mask = torch.cat([
                    torch.ones(seq_len, dtype=torch.long, device=device),
                    torch.zeros(pad_len, dtype=torch.long, device=device)
                ], dim=0)
            else:
                # 不需要padding
                padded_seq = seq
                mask = torch.ones(seq_len, dtype=torch.long, device=device)

            padded_ids.append(padded_seq)
            attention_masks.append(mask)

        # Stack成batch：[batch, max_length]
        generated_ids = torch.stack(padded_ids, dim=0)
        attention_mask = torch.stack(attention_masks, dim=0)

        # 根据参数决定返回格式
        if return_attention_mask:
            return generated_ids, attention_mask
        return generated_ids

    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        Top-K过滤：只保留概率最高的K个token

        Args:
            logits: shape [batch, vocab_size]
            top_k: 保留的token数量

        Returns:
            过滤后的logits
        """
        # 找到top-k的阈值
        top_k = min(top_k, logits.size(-1))
        top_k_values, _ = torch.topk(logits, top_k)
        threshold = top_k_values[:, -1].unsqueeze(-1)

        # 将低于阈值的设为-inf
        logits = torch.where(logits < threshold, float('-inf'), logits)

        return logits

    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Top-P (Nucleus) 过滤：保留累积概率达到p的最小token集合

        Args:
            logits: shape [batch, vocab_size]
            top_p: 累积概率阈值

        Returns:
            过滤后的logits
        """
        # 排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # 计算累积概率
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 找到累积概率超过top_p的位置
        sorted_indices_to_remove = cumulative_probs > top_p

        # 保留至少一个token（将第一个设为False）
        sorted_indices_to_remove[:, 0] = False

        # 将需要移除的位置设为-inf
        sorted_logits[sorted_indices_to_remove] = float('-inf')

        # 恢复原始顺序
        logits = torch.gather(sorted_logits, dim=-1, index=sorted_indices.argsort(dim=-1))

        return logits


def test_policy_model():
    """
    测试Policy Model
    """
    print("\n" + "="*60)
    print("测试Policy Model")
    print("="*60)

    # 创建模型
    model = PolicyModel(
        vocab_size=1000,
        hidden_dim=128,
        num_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
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

    logits = model(input_ids)
    print(f"  输入shape: {input_ids.shape}")
    print(f"  输出logits shape: {logits.shape}")

    # 测试2: 计算log概率
    print("\n[测试2] 计算log概率")
    log_probs = model.get_log_probs(input_ids)
    print(f"  Log概率shape: {log_probs.shape}")
    print(f"  Log概率值: {log_probs}")

    # 测试3: 生成（greedy）
    print("\n[测试3] Greedy生成")
    prompt = torch.randint(0, 1000, (1, 8))
    generated = model.generate(
        prompt,
        max_new_tokens=16,
        do_sample=False
    )
    print(f"  Prompt长度: {prompt.shape[1]}")
    print(f"  生成后长度: {generated.shape[1]}")
    print(f"  生成的tokens: {generated[0, prompt.shape[1]:].tolist()[:10]}...")

    # 测试4: 采样生成
    print("\n[测试4] 采样生成（temperature=1.0, top_p=0.9）")
    generated = model.generate(
        prompt,
        max_new_tokens=16,
        temperature=1.0,
        top_p=0.9,
        do_sample=True
    )
    print(f"  生成后长度: {generated.shape[1]}")
    print(f"  生成的tokens: {generated[0, prompt.shape[1]:].tolist()[:10]}...")

    print("\n测试完成!")


if __name__ == "__main__":
    test_policy_model()

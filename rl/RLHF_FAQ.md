# RLHF 训练常见问题和关键知识点

本文档汇总 RLHF (Reinforcement Learning from Human Feedback) 训练过程中的关键知识点、常见问题和最佳实践。

---

## 目录

1. [训练流程相关](#1-训练流程相关)
2. [Reward Model 相关](#2-reward-model-相关)
3. [算法对比](#3-算法对比)
4. [实际工程问题](#4-实际工程问题)
5. [常见误区](#5-常见误区)
6. [最佳实践](#6-最佳实践)

---

## 1. 训练流程相关

### Q1.1: RLHF 的完整流程是什么？

**A**: RLHF 包含三个主要阶段：

```
阶段1: SFT (Supervised Fine-Tuning)
  ├─ 输入：预训练模型 + 指令数据集
  ├─ 目标：学习基本的指令遵循能力
  └─ 输出：SFT模型（作为policy初始化）

阶段2: Reward Model Training
  ├─ 输入：SFT模型 + 偏好数据
  ├─ 目标：学习人类偏好的打分函数
  └─ 输出：Reward Model（RM）

阶段3: RL Fine-tuning
  ├─ 输入：SFT模型（policy） + RM
  ├─ 算法：PPO / GRPO / DPO 等
  └─ 输出：最终对齐模型
```

### Q1.2: 各阶段需要什么样的数据？

**A**: 不同阶段的数据要求：

| 阶段 | 数据格式 | 数据量 | 数据来源 |
|-----|---------|--------|---------|
| **SFT** | `{prompt, completion}` | 数万-数十万 | 人工标注、合成数据 |
| **RM训练** | `{prompt, chosen, rejected}` | 数千-数万对 | 人工偏好标注、模型排序 |
| **RL训练** | `{prompts}` 列表 | 数千-数万 | 与SFT数据分布一致 |

**关键区别**：
- SFT 需要单个高质量的回答
- RM 需要成对的好/坏对比
- RL 只需要 prompts（completion 由模型生成）

### Q1.3: 模型在各阶段是否会更新？

**A**:

| 模型 | 阶段1:SFT | 阶段2:RM训练 | 阶段3:RL训练 |
|-----|----------|-------------|------------|
| **Policy Model** | ✅ 训练 | ❌ 冻结 | ✅ 训练 |
| **Reward Model** | ❌ 不存在 | ✅ 训练 | ❌ **冻结使用** |
| **Reference Model** | ❌ 不存在 | ❌ 不存在 | ❌ **冻结**（policy的副本） |

**重点**：
- RM 在 RL 阶段**不更新**，始终使用训练好的权重
- Reference model 是 policy 的初始副本，用于计算 KL 惩罚，**永不更新**

---

## 2. Reward Model 相关

### Q2.1: Reward Model 什么时候训练？

**A**: RM 在 SFT 之后、RL 训练之前**单独训练一次**。

```
1. 完成 SFT ✓
2. 训练 Reward Model ← 此时训练
3. RL 训练（使用冻结的 RM）
```

### Q2.2: RM 是否在 RL 过程中更新？

**A**: **不会更新**。原因：

1. **避免 Reward Hacking**：
   - 如果 RM 和 policy 同时训练，policy 会学会"欺骗" RM
   - 例如：生成 RM 喜欢但人类不喜欢的文本

2. **稳定性考虑**：
   - RM 变化会导致 reward 不稳定
   - 训练难以收敛

3. **工程简化**：
   - RM 可以预先训练好并共享
   - 不需要额外的计算资源

### Q2.3: GRPO 需要 Reward Model 吗？

**A**: **不强制需要**，GRPO 非常灵活：

**方式1：使用 Reward Model**
```python
config = GRPOConfig(use_reward_model=True)
trainer = GRPOTrainer(..., reward_model=rm)
```

**方式2：使用自定义函数**
```python
def my_reward(prompt, completion):
    return score  # 基于规则计算

config = GRPOConfig(use_reward_model=False, reward_func=my_reward)
trainer = GRPOTrainer(..., reward_func=my_reward)
```

**方式3：混合使用**
```python
def hybrid_reward(prompt, completion):
    rm_score = reward_model(...)
    rule_bonus = check_format(completion)
    return rm_score + rule_bonus
```

**对比**：
- **DPO**：不需要 RM（直接从偏好数据学习）
- **PPO**：通常需要 RM
- **GRPO**：可选，根据需求灵活配置

### Q2.4: 如何评估 Reward Model 的质量？

**A**:

**训练时指标**：
- **Accuracy**：RM 正确区分 chosen/rejected 的比例
  - 目标：> 70%（越高越好）
- **Loss**：Ranking loss，应该持续下降

**测试时评估**：
```python
# 1. 准备测试集（held-out 偏好数据）
test_pairs = [
    {"prompt": "...", "chosen": "...", "rejected": "..."},
    ...
]

# 2. 计算准确率
correct = 0
for pair in test_pairs:
    score_chosen = rm(prompt + chosen)
    score_rejected = rm(prompt + rejected)
    if score_chosen > score_rejected:
        correct += 1

accuracy = correct / len(test_pairs)
print(f"Test Accuracy: {accuracy:.2%}")
```

**人工评估**：
- 对同一 prompt 的多个 completions 排序
- 检查 RM 的排序是否与人类一致

---

## 3. 算法对比

### Q3.1: GRPO vs DPO vs PPO 的区别是什么？

**A**:

| 特性 | **GRPO** | **DPO** | **PPO** |
|-----|---------|---------|---------|
| **是否需要RM** | 可选 | ❌ 不需要 | ✅ 需要 |
| **是否需要Value Model** | ❌ 不需要 | ❌ 不需要 | ✅ 需要 |
| **训练数据** | Prompts | 偏好对 | Prompts |
| **采样策略** | 在线生成 | 离线数据 | 在线生成 |
| **实现复杂度** | ⭐⭐ 简单 | ⭐ 最简单 | ⭐⭐⭐ 复杂 |
| **内存占用** | 中 | 低 | 高 |
| **灵活性** | ⭐⭐⭐ 最灵活 | ⭐ 受限于数据 | ⭐⭐ 中等 |
| **理论保证** | 中 | 强 | 最强 |

### Q3.2: 何时使用哪个算法？

**A**:

**推荐 GRPO 的场景**：
- ✅ 快速实验和迭代
- ✅ 需要灵活的 reward 设计（规则、RM、混合）
- ✅ 多目标优化（如：质量+安全+长度）
- ✅ 资源有限（无需 Value Model）

**推荐 DPO 的场景**：
- ✅ 已有高质量偏好数据
- ✅ 追求简单稳定（无需采样和 RL 训练）
- ✅ 数据质量比模型质量更重要

**推荐 PPO 的场景**：
- ✅ 追求最优性能
- ✅ 有充足的计算资源
- ✅ 需要严格的理论保证

**DeepSeek-V3 的选择**：GRPO（实用主义）

### Q3.3: GRPO 的核心创新是什么？

**A**:

1. **Group-based Baseline**：
   - 为每个 prompt 生成 G 个样本
   - 使用组内平均 reward 作为 baseline
   - 好处：自动归一化，无需 Value Model

2. **双重 Clipping**：
   ```python
   ratio = π_new / π_old
   clip_range = (0.1, 0.2)  # 不对称

   # 允许小幅度下降（0.1），限制大幅度上升（0.2）
   loss = -min(ratio * advantage, clip(ratio, 0.1, 0.2) * advantage)
   ```

3. **灵活的 Reward**：
   - 不强制使用 RM
   - 支持任意 Python 函数
   - 易于多目标优化

---

## 4. 实际工程问题

### Q4.1: PPO 的生成和训练是如何组织的？

**A**: 采用**生产者-消费者模式**：

```
生产阶段（Generation）：
  ├─ Policy Model 生成大批量样本（1K-8K）
  ├─ Reward Model 对样本打分
  └─ 存入 Experience Buffer

消费阶段（Training）：
  ├─ 从 Buffer 采样 mini-batches
  ├─ 多轮遍历（K epochs）进行梯度更新
  └─ 清空 Buffer，回到生成阶段
```

**参数设置**（OpenAI 实践）：
- Batch size（生成）：1K-8K 样本
- Mini-batch size（训练）：64-256
- K epochs：3-5 轮
- Buffer 更新频率：每生成一批新样本

**为什么分离**：
- 生成阶段：需要高吞吐，可并行
- 训练阶段：需要多次复用数据，提升效率

### Q4.2: 序列中途截断如何处理？

**A**:

**场景**：训练时固定序列长度（如512），但生成可能在第300步就结束（遇到 EOS）。

**处理方式**：

**方式1：固定长度截断**（简单但不理想）
```python
# 强制计算到 max_length
sequence = generated_tokens[:max_length]
returns = compute_returns(rewards)  # 长度 = max_length
```
问题：EOS 之后的 padding tokens 也参与计算

**方式2：EOS 感知截断**（推荐）
```python
# 找到 EOS 位置
eos_pos = find_eos(generated_tokens)

# 只计算到 EOS
valid_length = eos_pos + 1
rewards = rewards[:valid_length]
returns = compute_returns(rewards)

# Padding 部分的 loss 设为 0
mask = torch.arange(max_length) < valid_length
loss = (loss * mask).sum() / mask.sum()
```

**方式3：Bootstrap Value**（PPO中）
```python
# 如果在 max_length 处截断（非自然结束）
if not is_done:
    # 用 Value Model 估计未来收益
    bootstrap_value = value_model(last_state)
    returns[-1] += gamma * bootstrap_value
```

**工业实践**：方式2 + 方式3 结合

### Q4.3: SFT 阶段需要生成文本吗？

**A**: **不需要**，SFT 使用 **Teacher Forcing**。

**Teacher Forcing 原理**：
```python
# 输入：完整的 prompt + completion
input_ids = tokenizer("用户: 介绍Python\n助手: Python是...")

# 前向传播（并行计算所有位置）
logits = model(input_ids)  # [seq_len, vocab_size]

# 计算 loss（只在 completion 部分）
loss_mask = (input_ids == completion_tokens)  # 标记哪些是要学习的
loss = cross_entropy(logits[loss_mask], input_ids[loss_mask])
```

**关键点**：
- ✅ **完全并行**：所有 token 同时计算
- ✅ **无需生成**：ground truth 已知
- ✅ **高效**：比自回归生成快 100x

**对比 RL 训练**：
- RL 阶段：需要生成（因为没有 ground truth completion）
- SFT 阶段：有标注数据，直接用 teacher forcing

### Q4.4: 训练时会 padding 到 128k 吗？

**A**: **不会**，没有模型训练时 padding 到 max context length。

**实际做法**：

**1. Dynamic Batching**
```python
# 每个 batch 内的序列长度相近
batch1 = [seq of length ~500]  # padding to 512
batch2 = [seq of length ~2000] # padding to 2048
batch3 = [seq of length ~7800] # padding to 8192
```

**2. Bucketing**
```python
# 预定义长度桶
buckets = [512, 1024, 2048, 4096, 8192]

# 将数据按长度分组
for data in dataset:
    length = len(data)
    bucket = find_closest_bucket(length)
    bucket_datasets[bucket].append(data)
```

**3. Sequence Packing**（更高级）
```python
# 把多个短序列拼接到一起
seq1 = [tokens1] + [EOS]
seq2 = [tokens2] + [EOS]
seq3 = [tokens3] + [EOS]

packed_seq = seq1 + seq2 + seq3  # 总长度 ≈ 2048
```

**实际长度分布**：
- 训练数据：80% < 2k, 15% 在 2-4k, 5% > 4k
- Padding 到：2048, 4096 居多
- 极少 padding 到：16k, 32k, 128k（太浪费）

### Q4.5: RL 训练时的序列长度如何设置？

**A**:

**推荐设置**：
```python
# Prompt 长度：灵活
# Generation 长度：固定或有上限
config = GRPOConfig(
    max_new_tokens=256,  # 生成的最大长度
    # 实际总长度 = prompt_len + max_new_tokens
)
```

**处理策略**：

**策略1：固定生成长度**
```python
# 所有 prompt 都生成相同长度
for prompt in prompts:
    completion = model.generate(prompt, max_new_tokens=256)
    # completion 长度 = 256（或遇到 EOS 提前停止）
```

**策略2：Early Stopping**
```python
# 遇到 EOS 停止，但 padding 到 max_new_tokens
completion = model.generate(prompt, max_new_tokens=256, pad_to_max=True)
# 实际生成可能 150 tokens，padding 106 个 PAD tokens
```

**内存优化**：
- 用 Flash Attention：支持变长序列
- Gradient Checkpointing：减少中间激活

---

## 5. 常见误区

### 误区1: "RM 需要在 RL 过程中一起训练"

**❌ 错误**：RM 在 RL 时需要更新，否则不准确。

**✅ 正确**：
- RM 预先训练，RL 时冻结
- 原因：防止 reward hacking
- 如果 RM 不够好，应该回到阶段2重新训练 RM

### 误区2: "Reference Model 会更新"

**❌ 错误**：Reference model 在训练中也会更新。

**✅ 正确**：
- Reference model 是 policy 的初始副本
- **永不更新**，用于计算 KL 散度
- 代码：`ref_model.eval()` 且不在 optimizer 中

### 误区3: "DPO 训练后 policy 变化很小"

**❌ 错误**：DPO 只是微调，不会有大的变化。

**✅ 正确**：
- DPO 可以产生显著的行为改变
- 关键在于 β 参数（KL 惩罚系数）
- β 小 → 变化大；β 大 → 变化小
- 典型值：β = 0.1-0.5

### 误区4: "SFT 需要从头训练"

**❌ 错误**：SFT 要随机初始化模型权重。

**✅ 正确**：
- 从预训练模型（如 Qwen2-7B）开始
- SFT 是 fine-tuning，不是 pre-training
- 随机初始化会导致灾难性遗忘

### 误区5: "RL 训练需要 Value Model"

**❌ 错误**：所有 RL 算法都需要 Value Model。

**✅ 正确**：
- PPO：需要（用于估计优势函数）
- GRPO：不需要（用 group baseline）
- DPO：不需要（不是在线 RL）

---

## 6. 最佳实践

### 6.1 数据准备

**SFT 数据**：
- ✅ 质量 > 数量（1万条高质量 > 10万条低质量）
- ✅ 多样性（覆盖不同任务类型）
- ✅ 格式一致（统一 prompt template）
- ✅ 过滤有害内容

**RM 偏好数据**：
- ✅ 对比度明显（chosen 明显好于 rejected）
- ✅ 人工标注优先（模型排序次之）
- ✅ 数据量：5k-50k 对（取决于任务）
- ✅ 测试集：留出 10-20% 评估 RM

**RL Prompts**：
- ✅ 与 SFT 数据分布一致
- ✅ 定期更新（避免 mode collapse）
- ✅ 包含多种难度（简单、中等、困难）

### 6.2 超参数调优

**SFT 阶段**：
```python
learning_rate = 1e-5  # 较小的学习率
batch_size = 32       # 根据显存调整
num_epochs = 3        # 不要过拟合
warmup_ratio = 0.1    # 10% 步数用于 warmup
```

**RM 训练**：
```python
learning_rate = 1e-4  # 可以稍大
batch_size = 16       # 偏好对
num_epochs = 3-5      # 监控测试 accuracy
```

**GRPO 训练**：
```python
learning_rate = 1e-5 to 1e-6  # 非常小
num_generations = 4-8         # G 值，越大越稳定
clip_range = (0.1, 0.2)       # DeepSeek 推荐
temperature = 1.0             # 生成时温度
kl_coef = 0.0-0.1            # KL 惩罚（可选）
```

### 6.3 训练监控

**SFT 阶段关注**：
- Loss 持续下降
- 验证集 loss 不上升（防止过拟合）
- 定期生成样本，人工评估质量

**RM 训练关注**：
- Accuracy > 70%（测试集）
- Loss 稳定下降
- 对测试样本打分符合直觉

**RL 训练关注**：
- ✅ **Reward Mean**：应该逐渐上升
- ✅ **KL Divergence**：保持较小（< 5.0）
- ⚠️ **Clip Fraction**：0.1-0.3 正常
- ⚠️ **Mean Ratio**：应该接近 1.0
- 定期生成样本，人工评估（最重要！）

### 6.4 训练稳定性

**如果 Reward 不上升**：
1. 检查 reward 函数（是否有区分度）
2. 增加 `num_generations`（减少方差）
3. 降低 learning_rate
4. 检查 policy 是否退化（生成乱码）

**如果 Loss 爆炸**：
1. 降低 learning_rate
2. 增加 `kl_coef`（限制变化）
3. 检查梯度裁剪（`max_grad_norm=1.0`）
4. 检查数据（是否有异常样本）

**如果 KL 过大**：
1. Policy 变化过快，降低 learning_rate
2. 增加 `kl_coef`
3. 减小 `clip_range`
4. 检查是否有 mode collapse

### 6.5 评估方法

**定量评估**：
- Reward 分数（训练集和测试集）
- 与 reference model 的 KL 散度
- 生成长度分布
- 困惑度（PPL）

**定性评估**（更重要）：
- 人工评估生成质量（抽样 100-200 条）
- A/B 测试（对比 SFT 和 RL 后的模型）
- 任务相关指标（如准确率、F1）
- 安全性检查（有害内容过滤）

### 6.6 资源规划

**最小可运行配置**：
- GPU：1x A100 (40GB)
- 模型：7B 参数
- Batch size：16-32
- 训练时间：SFT 2-4小时，RM 1小时，RL 4-8小时

**推荐生产配置**：
- GPU：4-8x A100 (80GB)
- 模型：13B-70B 参数
- 使用 DeepSpeed ZeRO-3 + Offload
- 训练时间：SFT 1天，RM 4小时，RL 2-3天

---

## 7. 参考资源

### 论文

- **GRPO**: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- **DPO**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- **PPO**: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- **InstructGPT**: [Training language models to follow instructions](https://arxiv.org/abs/2203.02155)

### 博客文章

- [OpenAI: Learning from Human Preferences](https://openai.com/blog/deep-reinforcement-learning-from-human-preferences)
- [Hugging Face: RLHF Tutorial](https://huggingface.co/blog/rlhf)
- [Anthropic: Constitutional AI](https://www.anthropic.com/constitutional-ai)

### 代码库

- [Hugging Face TRL](https://github.com/huggingface/trl) - PPO/DPO 实现
- [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF) - 开源 RLHF 框架
- [本项目](../README.md) - 简化版教学实现

---

## 8. 更新日志

- **2026-02-11**: 初始版本，包含 RLHF 核心概念和实践经验
- 欢迎提交 Issue 或 PR 补充内容！

---

**维护者**: Claude Code
**最后更新**: 2026-02-11

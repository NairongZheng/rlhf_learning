# RL训练框架使用指南

## 概述

本RL框架支持GRPO、DPO、PPO等算法，用于语言模型的强化学习训练。

## 快速开始

### 1. GRPO训练示例

GRPO (Group Relative Policy Optimization) 是DeepSeek-V3中使用的RL算法，特点是无需Value Model。

```bash
cd rl/examples
python train_grpo.py
```

### 2. 使用自定义Reward函数

```python
from rl.models.policy_model import PolicyModel
from rl.trainers.grpo_trainer import GRPOTrainer
from rl.config_rl import GRPOConfig

# 定义reward函数
def my_reward_function(prompt: str, completion: str) -> float:
    """自定义reward函数"""
    # 例如：根据长度、关键词、格式等打分
    score = 0.0
    
    # 长度奖励
    length = len(completion.split())
    if 50 <= length <= 100:
        score += 1.0
    
    # 关键词奖励
    if "感谢" in completion:
        score += 0.5
    
    return score

# 配置
config = GRPOConfig(
    num_generations=4,  # 每个prompt生成的样本数
    use_reward_model=False,  # 使用自定义函数
    reward_func=my_reward_function
)

# 创建模型和trainer
policy_model = PolicyModel(vocab_size=151657, hidden_dim=128)
trainer = GRPOTrainer(
    config=config,
    policy_model=policy_model,
    tokenizer=tokenizer,
    reward_func=my_reward_function
)

# 训练
prompts = ["请介绍一下Python", "如何学习机器学习"]
trainer.train(prompts, num_epochs=10)
```

### 3. Reward Model 训练和使用

#### 3.1 什么是 Reward Model？

Reward Model (RM) 是 RLHF 流程的第2阶段，位于 SFT 和 RL 训练之间：

```
阶段1: SFT (Supervised Fine-Tuning)
  ↓ 得到基础模型
阶段2: RM Training ← 当前阶段
  ↓ 得到 Reward Model
阶段3: RL (PPO/GRPO/DPO等)
  ↓ 得到最终对齐模型
```

**RM 的作用**：
- 对模型生成的文本进行自动打分
- 替代昂贵的人工评估
- 在 RL 训练时作为固定的评分器（**不会在 RL 过程中更新**）

#### 3.2 数据准备

RM 训练需要**偏好数据**（Preference Data），每个样本包含：

```python
{
    "prompt": "输入提示",
    "chosen": "更好的回答",      # 高质量
    "rejected": "较差的回答"     # 低质量
}
```

数据来源：
1. **人工标注**（最可靠）：标注员对比两个回答，选择更好的
2. **模型排序**：用强模型（如GPT-4）对多个回答排序
3. **规则+采样**：用规则筛选明显的好/坏样本

数据量建议：**数千到数万对**（取决于任务复杂度）

#### 3.3 训练 Reward Model

**完整训练示例**（见 `rl/examples/train_reward_model.py`）：

```bash
cd rl/examples
python train_reward_model.py
```

**核心训练代码**：

```python
from rl.models.reward_model import RewardModel
import torch

# 1. 准备偏好数据
preference_data = [
    {
        "prompt": "请介绍Python",
        "chosen": "Python是一种高级编程语言...",  # 好
        "rejected": "Python是蛇"  # 差
    },
    # ... 更多数据 ...
]

# 2. 创建 Reward Model
reward_model = RewardModel(
    vocab_size=151657,
    hidden_dim=128,
    num_heads=4,
    num_layers=2  # TextEncoder 层数
)

# 3. 训练循环
optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        chosen_ids = batch['chosen_ids']
        rejected_ids = batch['rejected_ids']

        # 计算 ranking loss
        # 目标：让 reward(chosen) > reward(rejected)
        loss, metrics = reward_model.compute_reward_loss(
            chosen_ids,
            rejected_ids
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")

# 4. 保存模型
torch.save(reward_model.state_dict(), "reward_model.pt")
```

**训练监控指标**：
- **Loss**：应该逐渐下降
- **Accuracy**：RM 正确区分 chosen/rejected 的比例（目标 > 70%）

#### 3.4 在 RL 训练中使用 Reward Model

训练好 RM 后，在 GRPO/PPO 训练中使用（RM 保持冻结，不更新）：

```python
from rl.models.reward_model import RewardModel
from rl.models.policy_model import PolicyModel
from rl.trainers.grpo_trainer import GRPOTrainer
from rl.config_rl import GRPOConfig

# 1. 加载训练好的 Reward Model
reward_model = RewardModel(vocab_size=151657, hidden_dim=128)
reward_model.load_state_dict(torch.load("reward_model.pt"))
reward_model.eval()  # 设为评估模式（冻结）

# 2. 配置 GRPO 使用 Reward Model
config = GRPOConfig(
    use_reward_model=True,  # 使用 RM 打分
    num_generations=4
)

# 3. 创建 trainer（注意传入 reward_model）
policy_model = PolicyModel(vocab_size=151657, hidden_dim=128)
trainer = GRPOTrainer(
    config=config,
    policy_model=policy_model,
    tokenizer=tokenizer,
    reward_model=reward_model  # 传入 RM
)

# 4. RL 训练
prompts = ["请介绍Python", "如何学习机器学习"]
trainer.train(prompts, num_epochs=10)
```

#### 3.5 RM vs 自定义 Reward 函数

**何时使用 Reward Model**：
- 有偏好数据
- 任务复杂，难以用规则定义
- 追求质量，愿意付出训练成本

**何时使用自定义函数**：
- 快速实验
- 有明确的规则（如长度、格式、关键词）
- 多目标优化（如：`reward = 0.5*质量 + 0.3*安全 + 0.2*长度`）

**GRPO 特别优势**：
- 灵活切换两种方式
- 甚至可以组合使用：`final_reward = rm_score + rule_bonus`

## 配置说明

### GRPOConfig主要参数

```python
GRPOConfig(
    # 生成参数
    num_generations=4,      # 每个prompt生成的样本数（G值）
    max_new_tokens=128,     # 最大生成长度
    temperature=1.0,        # 采样温度
    top_p=0.9,             # nucleus sampling
    
    # GRPO核心参数
    clip_range_low=0.1,    # clip下界
    clip_range_high=0.2,   # clip上界
    scale_rewards=True,    # 是否归一化rewards
    reward_baseline="mean", # baseline类型
    
    # Reward设置
    use_reward_model=False, # 是否使用Reward Model
    reward_func=None,      # 自定义reward函数
    
    # 训练参数
    learning_rate=1e-5,    # 学习率
    kl_coef=0.0,          # KL惩罚系数
    max_grad_norm=1.0     # 梯度裁剪
)
```

## 核心组件

### 1. PolicyModel

策略模型，用于生成文本和计算log概率。

```python
from rl.models.policy_model import PolicyModel

model = PolicyModel(
    vocab_size=151657,
    hidden_dim=128,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2
)

# 生成文本
generated = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9
)

# 计算log概率
log_probs = model.get_log_probs(input_ids)
```

### 2. RewardModel

奖励模型，用于对生成结果打分。

```python
from rl.models.reward_model import RewardModel

reward_model = RewardModel(
    vocab_size=151657,
    hidden_dim=128
)

# 计算reward
rewards = reward_model(input_ids)  # [batch]

# 训练Reward Model
loss, metrics = reward_model.compute_reward_loss(
    chosen_ids,
    rejected_ids
)
```

### 3. GRPO Trainer

```python
from rl.trainers.grpo_trainer import GRPOTrainer

trainer = GRPOTrainer(
    config=config,
    policy_model=policy_model,
    tokenizer=tokenizer,
    reward_func=my_reward_func
)

# 训练
metrics = trainer.train(
    train_prompts=["prompt1", "prompt2"],
    num_epochs=10
)

# 生成样本
samples = trainer.generate_samples(
    prompts=["test prompt"],
    num_samples=5
)
```

## 训练监控

训练过程中会输出以下指标：

- **Loss**: 总损失（越小越好）
- **Reward Mean**: 平均reward（应该逐渐上升）
- **KL**: 与reference policy的KL散度（应该保持较小）
- **Mean Ratio**: π_new / π_old的平均值（应该接近1.0）
- **Clip Fraction**: 被clip的比例（0.1-0.3为正常范围）

## 算法对比

### GRPO
- **优势**: 无需Value Model，实现简单，内存效率高
- **适用**: 快速实验，灵活的reward设计
- **推荐场景**: 首选算法

### DPO（待实现）
- **优势**: 无需Reward Model，直接从偏好数据学习
- **适用**: 有高质量偏好数据
- **推荐场景**: 有标注数据时使用

### PPO（可选）
- **优势**: 理论保证强，性能最优
- **劣势**: 需要Value Model，实现复杂
- **推荐场景**: 追求最优性能且有充足资源

## 注意事项

1. **Tokenizer**: 示例中使用的是简化版tokenizer，实际使用时请替换为真实的tokenizer（如Qwen2Tokenizer）

2. **Reward函数设计**: 
   - 应该返回有意义的数值（不要全是0或全是1）
   - 可以结合多个维度（长度、质量、格式等）
   - 建议先用简单reward函数验证流程

3. **超参数调优**:
   - `learning_rate`: 通常使用1e-5到1e-4
   - `clip_range`: GRPO推荐(0.1, 0.2)
   - `num_generations`: 越大variance越小，但计算量越大
   - `temperature`: 生成时建议1.0或略高，增加多样性

4. **训练稳定性**:
   - 监控KL散度，过大说明policy变化太快
   - 监控clip_fraction，应该在0.1-0.3之间
   - 如果训练不稳定，降低learning_rate或增加kl_coef

## 故障排查

### Reward不上升
- 检查reward函数是否有效（是否返回有区分度的值）
- 增加num_generations
- 降低learning_rate

### Loss爆炸
- 降低learning_rate
- 增加kl_coef
- 检查梯度裁剪是否生效

### Clip Fraction过高
- Policy变化过快，降低learning_rate
- 或增加clip_range

## 扩展开发

如果需要添加新的RL算法：

1. 在`rl/config_rl.py`中添加新的Config类
2. 在`rl/losses/`中实现对应的损失函数
3. 在`rl/trainers/`中实现Trainer类
4. 在`rl/examples/`中添加使用示例

参考GRPO的实现即可。

## 相关资源

- [GRPO论文](https://arxiv.org/abs/2412.19437) - DeepSeek-V3
- [DPO论文](https://arxiv.org/abs/2305.18290)
- [PPO论文](https://arxiv.org/abs/1707.06347)

## 贡献

欢迎提交Issue和Pull Request！
# 多模态大模型训练框架

一个用于学习的完整多模态大模型训练框架，包含**监督微调(SFT)**和**强化学习(RL)**两大训练范式。专注于理解原理和快速实验，代码详细注释，模块化设计。

## 📋 项目特点

- ✨ **完整框架**：同时支持SFT和RL训练（GRPO/DPO/PPO）
- 🏗️ **模块化设计**：core共享组件，sft和rl独立模块
- 🔍 **详细调试**：每个步骤都有详细的调试输出
- 📚 **丰富注释**：代码包含大量中文注释，解释每个技术点
- 🚀 **快速运行**：可在CPU上运行，不需要GPU
- 🎯 **教学导向**：专注于理解原理，而非追求SOTA性能

## 🏗️ 项目结构

```
mllm_training_debug/
├── core/                       # 共享核心组件
│   ├── config.py               # 基础配置类
│   ├── modules/                # 共享模型组件
│   │   ├── attention.py        # Attention机制（Multi-Head、Cross、Transformer Block）
│   │   ├── position_encoding.py  # 位置编码（RoPE、2D Position）
│   │   ├── text_encoder.py     # 文本编码器
│   │   └── text_decoder.py     # 文本解码器
│   ├── tokenizers/             # 文本处理工具
│   │   └── qwen_tokenizer.py   # Qwen2-VL Tokenizer封装
│   └── utils/
│       └── debug_utils.py      # 调试工具函数
│
├── sft/                        # 监督微调模块
│   ├── config_sft.py           # SFT配置
│   ├── models/
│   │   ├── vision_encoder.py  # 图像编码器
│   │   ├── fusion_layer.py    # 跨模态融合层
│   │   └── multimodal_model.py  # 完整多模态模型
│   ├── train_sft.py            # SFT训练脚本
│   └── example_sft.py          # SFT推理示例
│
├── rl/                         # 强化学习模块
│   ├── config_rl.py            # RL配置（RLConfig, GRPOConfig, DPOConfig, PPOConfig）
│   ├── models/
│   │   ├── policy_model.py     # 策略模型（带生成功能）
│   │   ├── reward_model.py     # 奖励模型
│   │   └── value_model.py      # 价值模型（PPO）
│   ├── losses/
│   │   ├── grpo_loss.py        # GRPO损失函数
│   │   ├── dpo_loss.py         # DPO损失函数
│   │   └── ppo_loss.py         # PPO损失函数
│   ├── trainers/
│   │   ├── grpo_trainer.py     # GRPO训练器
│   │   ├── dpo_trainer.py      # DPO训练器
│   │   └── ppo_trainer.py      # PPO训练器
│   ├── data/
│   │   └── preference_dataset.py  # 偏好数据集
│   ├── utils/
│   │   └── rl_utils.py         # RL工具函数（GAE、Advantage计算等）
│   ├── examples/
│   │   ├── train_grpo.py       # GRPO训练示例
│   │   ├── train_dpo.py        # DPO训练示例
│   │   └── train_ppo.py        # PPO训练示例
│   └── README_RL.md            # RL模块详细文档
│
├── plan.md                     # 开发笔记和设计文档
├── README.md                   # 项目说明（本文件）
└── .gitignore                  # Git忽略配置
```

## 🚀 快速开始

### 环境要求

```bash
python >= 3.8
torch >= 1.10
numpy
transformers (可选，用于真实tokenizer)
```

### 安装依赖

```bash
pip install torch numpy
# 如果要使用真实tokenizer
pip install transformers
```

### SFT训练

```bash
# 方式1: 从项目根目录运行（推荐）
python sft/train_sft.py              # 完整训练（20步）
python sft/train_sft.py quick        # 快速测试（5步）
python sft/example_sft.py            # 推理示例

# 方式2: 从sft目录运行（也支持）
cd sft
python train_sft.py                  # 完整训练
python train_sft.py quick            # 快速测试
python example_sft.py                # 推理示例
```

### RL训练

```bash
# 方式1: 从项目根目录运行（推荐）
python rl/examples/train_grpo.py     # GRPO训练
python rl/examples/train_dpo.py      # DPO训练
python rl/examples/train_ppo.py      # PPO训练

# 方式2: 从rl/examples目录运行（也支持）
cd rl/examples
python train_grpo.py                 # GRPO训练
python train_dpo.py                  # DPO训练
python train_ppo.py                  # PPO训练
```

详细的RL使用说明请查看 [`rl/README_RL.md`](rl/README_RL.md)

## 📚 两大训练范式

### 1. 监督微调（SFT）

**架构设计：**

```
输入图像 -> Vision Encoder -> 图像特征
                                    |
                                    v
                              Cross Attention Fusion
                                    ^
                                    |
输入文字 -> Text Encoder -> 文字特征

融合特征 -> Text Decoder -> 输出文字
```

**核心组件：**
- **Vision Encoder**: Patch Embedding + 2D Position + Transformer
- **Text Encoder**: Token Embedding + RoPE + Transformer
- **Fusion Layer**: Cross Attention融合跨模态信息
- **Text Decoder**: Transformer Decoder + LM Head

**特点：**
- 多模态输入（图像+文本）
- 使用交叉熵损失训练
- 适合有标注数据的场景

### 2. 强化学习（RL）

支持三种主流RL算法：

#### GRPO (Group Relative Policy Optimization)
- **特点**: 组内相对优化，无需Value Model
- **适用场景**: 快速实验，灵活的reward设计
- **工业应用**: DeepSeek-V3
- **复杂度**: 中等

#### DPO (Direct Preference Optimization)
- **特点**: 直接从偏好数据学习，最简单
- **适用场景**: 有高质量偏好标注数据
- **工业应用**: Llama 3
- **复杂度**: 最低

#### PPO (Proximal Policy Optimization)
- **特点**: 多轮优化，理论保证强
- **适用场景**: 追求最优性能
- **工业应用**: ChatGPT (RLHF)
- **复杂度**: 最高

**算法对比：**

| 特性 | GRPO | DPO | PPO |
|------|------|-----|-----|
| **需要Value Model** | ❌ | ❌ | ✅ |
| **需要在线生成** | ✅ | ❌ | ✅ |
| **数据要求** | Prompts | 偏好对 | Prompts |
| **内存效率** | 高 | 最高 | 低 |
| **理论保证** | 弱 | 中 | 强 |
| **实现难度** | 中 | 低 | 高 |

## 🏛️ 核心技术

### 位置编码
- **RoPE（旋转位置编码）**：用于文本序列，现代LLM标准
- **2D位置编码**：用于图像patch，编码二维空间信息

### Attention机制
- **Multi-Head Attention**：多头自注意力
- **Cross Attention**：跨模态信息融合
- **Causal Mask**：自回归生成的mask机制

### 标准化策略
- **Pre-LayerNorm**：现代Transformer标准
- **Residual Connection**：帮助梯度流动

### RL核心功能
- **Advantage计算**：组内归一化、GAE
- **Log概率计算**：用于policy ratio
- **多种采样策略**：Greedy、Temperature、Top-K、Top-P
- **KL散度惩罚**：防止policy偏离过多

## 📊 模型配置

### SFT默认配置

```python
hidden_dim = 128          # embedding维度
num_heads = 4             # attention头数
num_layers = 1            # 每个组件的层数
vocab_size = 151657       # 词汇表大小（Qwen tokenizer）
image_size = 64           # 输入图像大小
patch_size = 16           # 图像patch大小
max_seq_len = 32          # 最大文本长度
dropout = 0.1             # dropout率
learning_rate = 1e-4      # 学习率
batch_size = 2            # 批次大小
```

### RL默认配置

```python
# GRPO配置
num_samples_per_prompt = 4    # 每个prompt生成的样本数
epsilon_low = 0.2             # asymmetric clipping下限
epsilon_high = 0.2            # asymmetric clipping上限
kl_coeff = 0.01              # KL散度惩罚系数

# DPO配置
beta = 0.1                    # DPO温度参数
loss_type = 'sigmoid'         # 损失类型: sigmoid/hinge/ipo
label_smoothing = 0.0         # 标签平滑

# PPO配置
num_epochs = 4                # 优化轮数
value_loss_coeff = 0.5        # value loss系数
entropy_coeff = 0.01          # entropy bonus系数
gae_lambda = 0.95             # GAE lambda参数
```

## 🔍 测试单个模块

```bash
# 测试Core模块
python -c "from core.modules import MultiHeadAttention, TextEncoder, TextDecoder"
python -c "from core.utils import print_tensor_info, check_gradient_flow"

# 测试SFT模块
python -c "from sft.models.multimodal_model import SimpleMultimodalModel"
cd sft/models && python vision_encoder.py
cd sft/models && python fusion_layer.py

# 测试RL模块
python -c "from rl.models.policy_model import PolicyModel"
python -c "from rl.trainers.grpo_trainer import GRPOTrainer"
cd rl/models && python policy_model.py
cd rl/models && python value_model.py
```

## 📖 学习要点

通过这个项目，你将深入理解：

### SFT相关
1. **多模态融合**：如何通过Cross Attention融合图像和文本
2. **Vision Transformer**：如何将图像转换为序列
3. **位置编码**：RoPE和2D位置编码的实现原理
4. **自回归训练**：如何对齐logits和labels

### RL相关
5. **Policy Gradient方法**：如何用梯度优化策略
6. **Advantage计算**：组内归一化、GAE的实现
7. **偏好学习**：DPO如何直接从偏好学习
8. **在线RL**：GRPO和PPO的完整训练流程
9. **采样策略**：各种decoding策略的实现

### 工程实践
10. **模块化设计**：如何组织大型项目代码
11. **代码复用**：core组件的共享机制
12. **Debug技巧**：如何添加完善的调试功能
13. **配置管理**：继承式配置系统设计

## 🎓 学习路径建议

### 初学者路径
1. **第一步**：阅读 `plan.md`，了解项目设计思路
2. **第二步**：运行SFT示例，理解基础架构
3. **第三步**：阅读core模块代码，理解共享组件
4. **第四步**：运行RL示例，了解三种算法区别
5. **第五步**：修改配置，观察不同设置的影响

### 进阶路径
1. 实现自己的reward函数
2. 在真实数据集上训练
3. 添加新的RL算法
4. 优化训练效率
5. 实现分布式训练

## 🔧 自定义和扩展

### 添加自定义Reward函数（GRPO）

```python
def my_reward_function(prompts, completions):
    """自定义reward函数"""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # 你的reward逻辑
        reward = len(completion)  # 示例：根据长度打分
        rewards.append(reward)
    return torch.tensor(rewards)

# 在GRPOTrainer中使用
trainer = GRPOTrainer(
    policy_model=policy_model,
    ref_policy_model=ref_policy_model,
    reward_fn=my_reward_function,  # 使用自定义函数
    config=grpo_config
)
```

### 使用真实Tokenizer

```python
from transformers import AutoTokenizer

# 替换SimpleTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

# 在训练中使用
encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
```

### 修改模型架构

```python
# 使用更大的模型
config = ModelConfig(
    hidden_dim=256,
    num_heads=8,
    num_layers=2,
    max_seq_len=64
)

# 使用更多fusion层
from sft.models.fusion_layer import MultipleFusionLayers

fusion = MultipleFusionLayers(
    hidden_dim=128,
    num_heads=4,
    num_layers=2  # 多层融合
)
```

## 📈 代码统计

- **总文件数**: 30+
- **总代码量**: ~4000行（含详细注释）
- **核心组件**: 10+个
- **训练算法**: 4种（SFT + GRPO + DPO + PPO）
- **配置类**: 6个
- **示例脚本**: 6个

## ⚠️ 注意事项

1. **教学项目**：专注于理解原理，不追求SOTA性能
2. **简化设计**：真实的工业级模型会更复杂
3. **CPU友好**：可在CPU上运行，但GPU会更快
4. **代码质量**：详细注释，模块化设计，适合学习

## 📖 相关资源

### 论文
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Transformer
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - Vision Transformer
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE
- [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) - 多模态模型
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - DPO
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) - PPO

### 工业实现
- **GRPO**: DeepSeek-V3
- **DPO**: Llama 3
- **PPO**: ChatGPT (RLHF)

## ❓ 常见问题

### Q: SFT和RL有什么区别？
A: SFT是监督学习，需要标注数据；RL是强化学习，通过reward信号优化。SFT用于基础能力训练，RL用于对齐人类偏好。

### Q: 应该选择哪个RL算法？
A:
- 快速实验：GRPO
- 有偏好数据：DPO
- 追求最优：PPO

### Q: 可以在真实数据上训练吗？
A: 可以。需要准备图像-文本对数据（SFT）或prompt数据/偏好数据（RL）。

### Q: 如何保存和加载模型？
```python
# 保存
torch.save(model.state_dict(), 'model.pt')

# 加载
model.load_state_dict(torch.load('model.pt'))
```

### Q: 参数量大概多少？
A:
- SFT模型：~15万参数（默认配置）
- RL Policy Model：~10万参数（默认配置）

### Q: 代码可以直接用于生产吗？
A: 不建议。这是教学代码，需要更多优化才能用于生产环境。

## 🤝 贡献

欢迎提出问题和改进建议！这是一个学习项目，目标是帮助大家理解多模态模型和RL训练。

## 📝 许可

本项目仅用于学习和教育目的。

## 🎉 更新日志

- **2026-02-10**: 完成项目清理和模块化重构，更新README
- **2026-02-09**: 完成GRPO、DPO、PPO三种RL算法实现
- **2026-02-09**: 添加RL训练框架，重组项目结构
- **2026-02-06**: 完成基础SFT多模态模型实现

---

**祝学习愉快！🎉**

**如有问题，请查看 `plan.md` 获取详细的设计文档和开发笔记。**

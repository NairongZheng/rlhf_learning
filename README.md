# 多模态大模型学习代码

一个用于学习的简化版多模态大模型实现，帮助理解多模态模型的基本原理和训练流程。

## 📋 项目特点

- ✨ **极简设计**：最简单的架构，最少的层数，易于理解
- 🔍 **详细调试**：每个步骤都有详细的调试输出，方便观察中间变量
- 📚 **丰富注释**：代码包含大量中文注释，解释每个技术点
- 🚀 **快速运行**：可在CPU上运行，不需要GPU
- 🎯 **教学导向**：专注于理解原理，而非追求性能

## 🏗️ 模型架构

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

### 核心组件

1. **Vision Encoder**（图像编码器）
   - Patch Embedding：将图像切分成小块
   - 2D Position Encoding：添加位置信息
   - 1层Transformer：提取视觉特征

2. **Text Encoder**（文本编码器）
   - Token Embedding：词嵌入
   - RoPE位置编码：旋转位置编码
   - 1层Transformer：提取文本特征

3. **Fusion Layer**（融合层）
   - Cross Attention：融合图像和文本特征
   - Feed-Forward Network：进一步处理

4. **Text Decoder**（文本解码器）
   - 1层Transformer Decoder
   - Language Model Head：输出词汇表logits

## 📦 项目结构

```
mllm_training_debug/
├── config.py                    # 模型配置
├── train.py                     # 训练脚本
├── example.py                   # 推理示例
├── models/
│   ├── attention.py            # Attention机制
│   ├── position_encoding.py   # 位置编码（RoPE、2D）
│   ├── vision_encoder.py      # 图像编码器
│   ├── text_encoder.py        # 文本编码器
│   ├── fusion_layer.py        # 跨模态融合层
│   ├── text_decoder.py        # 文本解码器
│   └── multimodal_model.py    # 整体模型
├── utils/
│   └── debug_utils.py         # 调试工具
└── README.md                   # 项目说明
```

## 🚀 快速开始

### 环境要求

```bash
python >= 3.8
torch >= 1.10
numpy
```

### 安装依赖

```bash
pip install torch numpy
```

### 运行训练

```bash
# 完整训练（20步）
python train.py

# 快速测试（5步，更小的模型）
python train.py quick
```

### 运行推理示例

```bash
python example.py
```

### 测试单个模块

```bash
# 测试Vision Encoder
python models/vision_encoder.py

# 测试Text Encoder
python models/text_encoder.py

# 测试Fusion Layer
python models/fusion_layer.py

# 测试Text Decoder
python models/text_decoder.py

# 测试完整模型
python models/multimodal_model.py
```

## 📊 模型配置

默认超参数（超简单版本）：

```python
hidden_dim = 128          # embedding维度
num_heads = 4             # attention头数
num_layers = 1            # 每个组件的层数
vocab_size = 1000         # 词汇表大小
image_size = 64           # 输入图像大小
patch_size = 16           # 图像patch大小
max_seq_len = 32          # 最大文本长度
dropout = 0.1             # dropout率
learning_rate = 1e-4      # 学习率
batch_size = 2            # 批次大小
```

可以在 `config.py` 中修改这些参数。

## 🔍 调试功能

### 训练过程中的调试信息

- **前向传播**：每层的输出shape和统计信息
- **反向传播**：梯度信息（均值、标准差、范数）
- **梯度流检查**：检测梯度消失/爆炸
- **损失曲线**：可视化训练进度

### 推理过程中的调试信息

- **中间特征**：查看每个模块的输出
- **Attention权重**：可视化跨模态attention
- **输出分析**：预测的token和概率分布

## 📚 学习要点

通过这个代码，你将理解：

### 1. 位置编码
- **RoPE（旋转位置编码）**：现代LLM的标准位置编码
  - 如何通过旋转变换注入位置信息
  - 为什么相对位置编码更好
- **2D位置编码**：图像patch的位置编码
  - 如何编码2D空间信息

### 2. Attention机制
- **Multi-Head Attention**：多头注意力的工作原理
- **Cross Attention**：如何融合不同模态
- **Causal Mask**：自回归生成中的mask机制

### 3. 标准化和残差连接
- **Pre-LayerNorm vs Post-LayerNorm**：两种标准化方式的区别
- **Residual Connection**：残差连接如何帮助梯度流动

### 4. 多模态融合
- 文本如何"查询"图像中的相关信息
- Cross Attention中Q、K、V的作用
- Attention权重的含义

### 5. 训练过程
- 完整的前向传播流程
- 反向传播和梯度计算
- 优化器如何更新参数
- 如何监控训练过程

### 6. Debug技巧
- 如何检查中间变量
- 如何发现训练问题
- 如何可视化Attention
- 如何验证模型是否在学习

## 📈 预期输出

### 训练输出示例

```
模型参数统计
============================================================
总参数量: 156,928
可训练参数量: 156,928
不可训练参数量: 0
============================================================

训练步骤 1
============================================================
Loss: 6.9123

梯度信息汇总
============================================================
总梯度范数: 12.3456
✓ 未检测到梯度消失
✓ 未检测到梯度爆炸

训练步骤 20
============================================================
Loss: 5.1234

训练统计:
  总步数: 20
  最终loss: 5.1234
  最低loss: 5.0123
  平均loss: 6.0234
```

### 推理输出示例

```
Vision Encoder最终输出
============================================================
Shape: (1, 16, 128)
均值: 0.0234
标准差: 0.9876

Text Encoder最终输出
============================================================
Shape: (1, 16, 128)
均值: -0.0123
标准差: 1.0234

融合后的特征
============================================================
Shape: (1, 16, 128)

输出Logits
============================================================
Shape: (1, 16, 1000)
```

## 🔧 自定义和扩展

### 修改模型大小

在 `config.py` 中调整：

```python
# 更小的模型（更快）
config = ModelConfig(
    hidden_dim=64,
    num_heads=2,
    num_layers=1,
    image_size=32,
    patch_size=8
)

# 更大的模型（表达能力更强）
config = ModelConfig(
    hidden_dim=256,
    num_heads=8,
    num_layers=2,
    image_size=128,
    patch_size=16
)
```

### 添加更多层

```python
config = ModelConfig(
    num_layers=2  # 每个组件使用2层Transformer
)
```

### 使用真实数据

修改 `train.py` 中的 `generate_dummy_data()` 函数，加载真实的图像-文本对数据。

## ⚠️ 注意事项

1. **这是教学代码**：专注于理解原理，不追求性能
2. **随机初始化**：没有预训练权重，loss可能不会降低很多
3. **简化设计**：真实的多模态模型会更复杂
4. **随机数据**：使用随机数据训练，仅用于演示流程

## 🎓 学习路径建议

1. **第一步**：阅读 `config.py`，了解超参数
2. **第二步**：运行 `python example.py`，看推理效果
3. **第三步**：阅读各个模块的代码，理解实现
4. **第四步**：运行 `python train.py`，观察训练过程
5. **第五步**：修改配置，观察不同设置的影响
6. **第六步**：尝试添加新功能或改进

## 📖 相关资源

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原论文
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - Vision Transformer (ViT)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE位置编码
- [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) - 多模态大模型

## 🤝 贡献

欢迎提出问题和改进建议！这是一个学习项目，目标是帮助大家理解多模态模型。

## 📝 许可

本项目仅用于学习和教育目的。

## ❓ 常见问题

### Q: 为什么loss不下降？
A: 这是正常的。我们使用随机数据训练，模型可能无法找到有意义的模式。在真实数据上训练会有明显的loss下降。

### Q: 可以在GPU上运行吗？
A: 可以。代码会自动使用可用的GPU。如果要强制使用CPU，设置环境变量 `CUDA_VISIBLE_DEVICES=""`。

### Q: 如何保存和加载模型？
A: 添加以下代码：
```python
# 保存
torch.save(model.state_dict(), 'model.pt')

# 加载
model.load_state_dict(torch.load('model.pt'))
```

### Q: 参数量大概多少？
A: 默认配置约15万个参数，模型大小约0.6MB（float32）。

### Q: 可以用于实际应用吗？
A: 不建议。这是一个简化的教学模型。实际应用需要更大的模型、预训练权重和更复杂的架构。

---

**祝学习愉快！🎉**

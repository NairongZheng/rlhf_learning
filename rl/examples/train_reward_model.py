"""
Reward Model 训练示例

本示例演示如何训练一个 Reward Model (RM)。
RM 是 RLHF 流程的第2阶段（SFT之后，RL之前），用于对模型生成的文本进行打分。

训练流程：
1. 准备偏好数据（chosen vs rejected 对）
2. 创建 Reward Model
3. 使用 ranking loss 训练
4. 评估和保存模型

注意：
- 本示例使用简化版 tokenizer 用于演示
- 实际使用时，请替换为真实的 tokenizer：
  from core.tokenizers import QwenTokenizerWrapper
  tokenizer = QwenTokenizerWrapper()

Author: Claude Code
Date: 2026-02-11
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from rl.models.reward_model import RewardModel


# ============================================================================
# 0. 简化版 Tokenizer（用于演示）
# ============================================================================

class SimpleTokenizer:
    """
    简化版 Tokenizer（仅用于演示）

    实际使用时，应替换为真实的 tokenizer，例如：
    from core.tokenizers import QwenTokenizerWrapper
    tokenizer = QwenTokenizerWrapper()
    """

    def __init__(self):
        """初始化简化版 tokenizer"""
        # 模拟词汇表大小（Qwen2 实际大小约为 151657）
        self.vocab_size = 10000
        self.pad_token_id = 0
        self.eos_token_id = 2

    def encode(self, text: str) -> List[int]:
        """
        将文本编码为 token IDs（简化实现）

        实际实现会使用 BPE/WordPiece 等算法
        这里简化为字符级编码用于演示

        Args:
            text: 输入文本

        Returns:
            token IDs 列表
        """
        # 简化：每个字符映射到一个 ID
        # 实际 tokenizer 使用更复杂的子词编码
        token_ids = []
        for char in text:
            # 使用字符的 Unicode 码点，并对 vocab_size 取模
            token_id = ord(char) % (self.vocab_size - 100) + 100
            token_ids.append(token_id)
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        将 token IDs 解码为文本（简化实现）

        Args:
            token_ids: token IDs 列表

        Returns:
            解码后的文本
        """
        # 这只是一个占位实现，实际解码需要词汇表
        return f"<decoded_{len(token_ids)}_tokens>"


# ============================================================================
# 1. 数据准备
# ============================================================================

class PreferenceDataset(Dataset):
    """
    偏好数据集

    每个样本包含：
    - prompt: 输入提示
    - chosen: 更好的回答
    - rejected: 较差的回答

    RM训练目标：让 reward(prompt+chosen) > reward(prompt+rejected)
    """

    def __init__(self, data: List[Dict[str, str]], tokenizer, max_length: int = 128):
        """
        Args:
            data: 偏好数据列表，每个元素包含 prompt, chosen, rejected
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回一对偏好数据的token IDs
        """
        item = self.data[idx]

        # 拼接prompt和completion
        chosen_text = item['prompt'] + item['chosen']
        rejected_text = item['prompt'] + item['rejected']

        # Tokenize（这里简化处理，实际应该用真实tokenizer）
        chosen_ids = self.tokenizer.encode(chosen_text)
        rejected_ids = self.tokenizer.encode(rejected_text)

        # 截断或padding到固定长度
        chosen_ids = chosen_ids[:self.max_length]
        rejected_ids = rejected_ids[:self.max_length]

        # Padding
        chosen_ids = chosen_ids + [0] * (self.max_length - len(chosen_ids))
        rejected_ids = rejected_ids + [0] * (self.max_length - len(rejected_ids))

        return {
            'chosen_ids': torch.tensor(chosen_ids, dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_ids, dtype=torch.long)
        }


def get_sample_data() -> List[Dict[str, str]]:
    """
    获取示例偏好数据

    实际训练时，这些数据应该来自：
    1. 人工标注（最可靠）
    2. 已有强模型的排序结果
    3. 规则+采样混合

    Returns:
        偏好数据列表
    """
    return [
        {
            "prompt": "请介绍一下Python编程语言。",
            "chosen": "Python是一种高级、解释型、通用的编程语言，由Guido van Rossum于1991年创建。它强调代码可读性和简洁性，使用缩进来定义代码块。Python广泛应用于Web开发、数据科学、人工智能、自动化脚本等领域。",
            "rejected": "Python是一种编程语言。"
        },
        {
            "prompt": "如何学习机器学习？",
            "chosen": "学习机器学习建议按以下步骤进行：1) 掌握Python和NumPy等基础工具；2) 学习线性代数、微积分和概率统计；3) 理解基本算法（线性回归、决策树等）；4) 学习深度学习框架如PyTorch；5) 通过实际项目巩固知识。推荐资源包括Andrew Ng的课程和《机器学习实战》等书籍。",
            "rejected": "看看书，写写代码就行了。"
        },
        {
            "prompt": "什么是神经网络？",
            "chosen": "神经网络是一种受生物神经系统启发的计算模型，由多层互连的节点（神经元）组成。每个连接有权重，通过调整权重来学习输入到输出的映射关系。基本结构包括输入层、隐藏层和输出层。训练过程使用反向传播算法和梯度下降来优化参数。",
            "rejected": "神经网络就是一些节点连在一起。"
        },
        {
            "prompt": "解释一下深度学习中的过拟合。",
            "chosen": "过拟合是指模型在训练集上表现很好，但在测试集上表现较差的现象。发生原因包括模型过于复杂、训练数据不足等。解决方法有：1) 增加训练数据；2) 使用正则化（L1/L2）；3) 应用Dropout；4) 早停法；5) 数据增强。关键是找到模型复杂度和泛化能力的平衡。",
            "rejected": "过拟合就是模型记住了训练数据，可以用正则化解决。"
        },
        {
            "prompt": "GPU在深度学习中的作用是什么？",
            "chosen": "GPU（图形处理单元）在深度学习中扮演关键角色，因为其具有大量并行计算核心，非常适合执行深度学习中的矩阵运算。相比CPU，GPU可以将训练速度提升10-100倍。主流深度学习框架（PyTorch、TensorFlow）都提供GPU加速支持。常用的有NVIDIA的CUDA平台和Tesla/RTX系列显卡。",
            "rejected": "GPU可以加快训练速度。"
        }
    ]


# ============================================================================
# 2. 训练函数
# ============================================================================

def train_reward_model(
    model: RewardModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 5,
    device: str = 'cpu'
) -> List[Dict[str, float]]:
    """
    训练 Reward Model

    Args:
        model: Reward Model
        train_loader: 数据加载器
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 设备（cpu或cuda）

    Returns:
        训练历史（loss和accuracy）
    """
    model.to(device)
    model.train()

    history = []

    print("\n开始训练 Reward Model...")
    print("=" * 70)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        for batch in train_loader:
            # 获取数据
            chosen_ids = batch['chosen_ids'].to(device)
            rejected_ids = batch['rejected_ids'].to(device)

            # 计算 ranking loss
            # 目标：reward(chosen) > reward(rejected)
            loss, metrics = model.compute_reward_loss(chosen_ids, rejected_ids)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 累积指标
            epoch_loss += loss.item()
            epoch_accuracy += metrics['accuracy']
            num_batches += 1

        # 计算平均值
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches

        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': avg_accuracy
        })

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Accuracy: {avg_accuracy:.4f}")

    print("=" * 70)
    print("训练完成！\n")

    return history


def evaluate_reward_model(
    model: RewardModel,
    prompts: List[str],
    completions: List[str],
    tokenizer,
    device: str = 'cpu'
):
    """
    评估 Reward Model

    展示模型对不同文本的打分结果

    Args:
        model: 训练好的 Reward Model
        prompts: 提示列表
        completions: 补全列表
        tokenizer: 分词器
        device: 设备
    """
    model.eval()
    model.to(device)

    print("\n评估 Reward Model...")
    print("=" * 70)

    with torch.no_grad():
        for prompt, completion in zip(prompts, completions):
            # 拼接并tokenize
            text = prompt + completion
            token_ids = tokenizer.encode(text)
            token_ids = token_ids[:128]  # 截断
            token_ids = token_ids + [0] * (128 - len(token_ids))  # padding

            # 转为tensor
            input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)

            # 计算reward
            reward = model(input_ids)

            print(f"\nPrompt: {prompt}")
            print(f"Completion: {completion}")
            print(f"Reward Score: {reward.item():.4f}")

    print("=" * 70)


# ============================================================================
# 3. 主函数
# ============================================================================

def main():
    """
    主训练流程
    """
    print("\n" + "=" * 70)
    print("Reward Model 训练示例")
    print("=" * 70)

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 1. 准备数据
    print("\n1. 准备偏好数据...")
    preference_data = get_sample_data()
    print(f"   - 偏好对数量: {len(preference_data)}")

    # 2. 创建tokenizer和dataset
    print("\n2. 创建数据集...")
    tokenizer = SimpleTokenizer()  # 简化版tokenizer（仅用于演示）
    dataset = PreferenceDataset(preference_data, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"   - Tokenizer: 简化版（演示用）")
    print(f"   - 词汇表大小: {tokenizer.vocab_size}")
    print(f"   - Batch size: 2")
    print(f"   - 序列长度: 128")
    print(f"   - 提示: 实际使用时请替换为 QwenTokenizerWrapper")

    # 3. 创建模型
    print("\n3. 创建 Reward Model...")
    model = RewardModel(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,  # TextEncoder 的层数
        max_seq_len=128
    )
    print(f"   - 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 4. 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 5. 训练
    print("\n4. 开始训练...")
    history = train_reward_model(
        model=model,
        train_loader=dataloader,
        optimizer=optimizer,
        num_epochs=5,
        device=device
    )

    # 6. 评估
    print("\n5. 评估模型...")
    test_cases = [
        ("请介绍一下Python。", "Python是一种高级编程语言，广泛用于Web开发、数据科学和人工智能。"),
        ("请介绍一下Python。", "Python是蛇。"),
        ("什么是机器学习？", "机器学习是人工智能的一个分支，使计算机能够从数据中学习规律。"),
        ("什么是机器学习？", "不知道。")
    ]

    evaluate_reward_model(
        model=model,
        prompts=[case[0] for case in test_cases],
        completions=[case[1] for case in test_cases],
        tokenizer=tokenizer,
        device=device
    )


if __name__ == "__main__":
    main()

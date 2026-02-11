"""
DPO训练示例

演示如何使用DPO从偏好数据训练语言模型

⚠️ 重要：DPO 算法的正确使用方式
========================================

DPO (Direct Preference Optimization) 通常用于 RLHF 的第二阶段：

标准流程：
  阶段1 (SFT): 监督微调
    随机初始化 → SFT训练 → 基础模型（能生成合理文本）
    数据：(prompt, completion) 对

  阶段2 (DPO): 偏好对齐 ← DPO 在这里使用
    基础模型 → ref_model（冻结的基础模型快照）
             → policy（继续学习偏好的模型）
    数据：(prompt, chosen, rejected) 三元组

为什么需要先 SFT？
==================
- ref_model 作为"对比基准"，必须是一个已训练好的合理模型
- 如果 ref_model 是随机初始化的，DPO 训练效果会很差
- DPO 的作用是"微调偏好"，而非"从零学习语言"

当前示例：
=========
⚠️ 本示例使用随机初始化（仅用于演示代码流程）
⚠️ 实际应用中应该：
  1. 先运行 sft/train_sft.py 进行监督微调
  2. 加载 SFT 模型作为起点
  3. 然后使用 DPO 进行偏好对齐

正确用法示例：
============
# Step 1: SFT训练（先执行）
python sft/train_sft.py
# 结果：得到 checkpoints/sft_model.pt

# Step 2: 加载SFT模型
sft_checkpoint = torch.load("checkpoints/sft_model.pt")
policy_model = PolicyModel(...)
policy_model.load_state_dict(sft_checkpoint)

# Step 3: DPO训练
trainer = DPOTrainer(
    policy_model=policy_model,  # ← 已有基础能力
    ref_model=None  # ← ref_model 会是 SFT 模型的快照
)
trainer.train(dpo_data)
"""

import torch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.models.policy_model import PolicyModel
from rl.trainers.dpo_trainer import DPOTrainer
from rl.config_rl import DPOConfig
from rl.data.preference_dataset import PreferenceDataset, create_preference_data_from_rankings
from core.tokenizers import QwenTokenizerWrapper


# =====================================
# 主训练函数
# =====================================

def main():
    print("\n" + "="*60)
    print("DPO训练示例")
    print("="*60)
    
    # =====================================
    # 1. 配置
    # =====================================
    config = DPOConfig(
        # 模型参数
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=151657,  # Qwen2-VL 真实词汇表大小
        max_seq_len=128,
        
        # DPO参数
        beta=0.1,  # 温度参数
        loss_type="sigmoid",  # 损失类型
        label_smoothing=0.0,
        
        # 训练参数
        learning_rate=1e-4,
        batch_size=2,
        max_grad_norm=1.0
    )
    
    print("\n[1] 配置初始化完成")
    print(f"  模型维度: {config.hidden_dim}")
    print(f"  Beta: {config.beta}")
    print(f"  Loss Type: {config.loss_type}")
    print(f"  学习率: {config.learning_rate}")
    
    # =====================================
    # 2. 创建Policy Model
    # =====================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2] 使用设备: {device}")
    
    policy_model = PolicyModel(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_encoder_layers=config.num_layers,
        num_decoder_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in policy_model.parameters())
    print(f"  Policy Model参数量: {total_params:,}")

    # =====================================
    # 3. 创建tokenizer（真实 Qwen2 Tokenizer）
    # =====================================
    tokenizer = QwenTokenizerWrapper()
    print(f"\n[3] Tokenizer初始化完成（真实 Qwen2 Tokenizer）")
    print(f"  实际词汇表大小: {tokenizer.vocab_size}")
    print(f"  PAD token ID: {tokenizer.pad_token_id}")
    print(f"  EOS token ID: {tokenizer.eos_token_id}")
    
    # =====================================
    # 4. 准备偏好数据
    # =====================================
    print("\n[4] 准备偏好数据")
    
    # 方式1: 直接提供偏好对
    preference_data = [
        {
            "prompt": "请介绍一下Python",
            "chosen": "Python是一种高级编程语言，简单易学、功能强大，广泛应用于Web开发、数据分析、人工智能等领域。",
            "rejected": "Python是蛇。"
        },
        {
            "prompt": "如何学习机器学习？",
            "chosen": "建议从数学基础开始，包括线性代数、概率论和统计学。然后学习Python编程，接着深入学习机器学习算法和框架如scikit-learn、TensorFlow等。",
            "rejected": "直接学就行了。"
        },
        {
            "prompt": "什么是深度学习？",
            "chosen": "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的表示。它在图像识别、自然语言处理等任务上取得了突破性进展。",
            "rejected": "就是很深的学习。"
        },
    ]
    
    print(f"  偏好数据数量: {len(preference_data)}")
    for i, item in enumerate(preference_data):
        print(f"    {i+1}. {item['prompt']}")
    
    # 方式2: 从排名数据创建（演示）
    print("\n  也可以从排名数据创建：")
    prompts = ["解释什么是AI"]
    responses = [[
        "AI是人工智能，模拟人类智能的技术。",  # 排名0（最好）
        "AI是计算机技术。",  # 排名1
        "AI是东西。"  # 排名2（最差）
    ]]
    rankings = [[0, 1, 2]]
    
    extra_data = create_preference_data_from_rankings(prompts, responses, rankings)
    print(f"    从1个prompt和3个响应生成了{len(extra_data)}对偏好数据")
    
    # 合并数据
    all_data = preference_data + extra_data
    
    # =====================================
    # 5. 创建Trainer
    # =====================================
    trainer = DPOTrainer(
        config=config,
        policy_model=policy_model,
        tokenizer=tokenizer
    )
    
    print("\n[5] DPO Trainer创建完成")
    
    # =====================================
    # 6. 开始训练
    # =====================================
    print("\n[6] 开始训练")
    
    metrics_history = trainer.train(
        train_data=all_data,
        num_epochs=5,
        log_interval=1
    )
    
    # =====================================
    # 7. 分析训练结果
    # =====================================
    print("\n[7] 训练结果分析")
    print("\n  训练曲线：")
    print("  Epoch | Loss   | Accuracy | Reward Margin")
    print("  " + "-"*60)
    for i, m in enumerate(metrics_history):
        print(f"  {i+1:5d} | {m['loss']:.4f} | {m['accuracy']:.4f}    | {m['reward_margin']:.4f}")
    
    # 计算改进
    initial_accuracy = metrics_history[0]['accuracy']
    final_accuracy = metrics_history[-1]['accuracy']
    improvement = final_accuracy - initial_accuracy
    
    print(f"\n  准确率提升: {initial_accuracy:.4f} -> {final_accuracy:.4f} "
          f"(+{improvement:.4f})")
    
    initial_margin = metrics_history[0]['reward_margin']
    final_margin = metrics_history[-1]['reward_margin']
    print(f"  Reward Margin: {initial_margin:.4f} -> {final_margin:.4f} "
          f"(+{final_margin - initial_margin:.4f})")
    
    # =====================================
    # 8. 评估（使用部分数据作为验证集）
    # =====================================
    print("\n[8] 评估模型")
    eval_data = preference_data[:2]  # 使用前2个样本评估
    eval_metrics = trainer.evaluate(eval_data)
    
    print(f"  验证集Loss: {eval_metrics['loss']:.4f}")
    print(f"  验证集Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"  验证集Reward Margin: {eval_metrics['reward_margin']:.4f}")
    
    # =====================================
    # 9. 总结
    # =====================================
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print("\n关键观察：")
    print(f"  1. Loss是否下降: {metrics_history[0]['loss']:.4f} -> {metrics_history[-1]['loss']:.4f}")
    print(f"  2. Accuracy是否提升: {metrics_history[0]['accuracy']:.4f} -> {metrics_history[-1]['accuracy']:.4f}")
    print(f"  3. Reward Margin是否增大: {metrics_history[0]['reward_margin']:.4f} -> {metrics_history[-1]['reward_margin']:.4f}")
    
    print("\nDPO的优势：")
    print("  ✅ 无需Reward Model")
    print("  ✅ 无需在线生成（离线学习）")
    print("  ✅ 实现简单，内存效率高")
    print("  ✅ 直接从人类偏好学习")
    
    print("\n适用场景：")
    print("  - 有高质量的偏好标注数据")
    print("  - 希望模型遵循人类偏好")
    print("  - 资源有限，无法运行复杂的RL算法")
    
    print("\n提示：")
    print("  - 这是一个简化的演示，实际应用中需要：")
    print("    1. 更多的偏好数据（通常需要数千到数万对）")
    print("    2. 更大的模型")
    print("    3. 调优超参数（beta, learning rate等）")
    print("    4. 使用验证集进行early stopping")


if __name__ == "__main__":
    main()

"""
PPO训练示例（多步版本，T=3）

学习要点：
1. 理解GAE如何通过反向计算利用"未来"的value
2. 观察多步rollout的数据收集和advantage计算
3. 对比T=1 vs T=3的训练行为差异

GAE时序计算原理：
- Rollout阶段：生成完整completion，记录所有rewards和values
- GAE阶段：从后向前计算advantages，可以使用V(s_{t+1})
- 这不是"作弊"，而是离线优化的标准做法

实现了完整的PPO算法，包括：
- 多步数据收集（T=3）
- GAE反向计算
- 模块化训练流程
- 详细的理论注释

运行方式：
python rl/examples/train_ppo.py
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.models.policy_model import PolicyModel
from rl.models.value_model import ValueModel
from rl.trainers.ppo_trainer import PPOTrainer
from rl.config_rl import PPOConfig
from core.tokenizers import QwenTokenizerWrapper


# Reward函数
def length_reward(prompt: str, completion: str) -> float:
    """长度奖励"""
    length = len(completion.split())
    if 50 <= length <= 100:
        return 1.0
    elif 30 <= length <= 150:
        return 0.5
    return 0.0


def main():
    print("\n" + "="*60)
    print("PPO训练示例（多步版本，T=3）")
    print("="*60)

    # 配置
    config = PPOConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=151657,  # Qwen2-VL 真实词汇表大小
        max_seq_len=128,

        # 多步训练参数（核心）
        num_chunks=3,  # T=3，将completion分成3个chunk

        # PPO参数
        clip_range=0.2,
        value_clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=4,

        # 损失权重
        value_loss_coef=0.5,
        entropy_coef=0.01,

        # Reward设置
        use_reward_model=False,

        # 训练参数
        learning_rate=1e-4,
        batch_size=2,
        max_grad_norm=1.0,
        normalize_advantages=True,

        # 生成参数（调整以适配多步训练）
        max_new_tokens=60,  # 增加生成长度，60/3=20 tokens per chunk
        temperature=1.0,
        top_p=0.9
    )

    print("\n[1] 配置初始化完成")
    print(f"  模型维度: {config.hidden_dim}")
    print(f"  PPO Epochs: {config.ppo_epochs}")
    print(f"  Clip Range: {config.clip_range}")
    print(f"\n  多步训练配置:")
    print(f"    时间步数 T = {config.num_chunks}")
    print(f"    折扣因子 γ = {config.gamma}")
    print(f"    GAE λ = {config.gae_lambda}")
    print(f"    生成长度 = {config.max_new_tokens} tokens")
    print(f"    每个chunk约 {config.max_new_tokens // config.num_chunks} tokens")
    print(f"\n  GAE时序计算说明:")
    print(f"    - GAE将从 t={config.num_chunks-1} 反向计算到 t=0")
    print(f"    - 每个时间步会使用'未来'的value来估计advantage")
    print(f"    - 这是离线RL的标准做法，不是'作弊'")

    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[2] 使用设备: {device}")

    policy_model = PolicyModel(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_encoder_layers=config.num_layers,
        num_decoder_layers=config.num_layers,
        max_seq_len=config.max_seq_len
    ).to(device)

    value_model = ValueModel(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len
    ).to(device)

    policy_params = sum(p.numel() for p in policy_model.parameters())
    value_params = sum(p.numel() for p in value_model.parameters())
    print(f"  Policy Model参数量: {policy_params:,}")
    print(f"  Value Model参数量: {value_params:,}")

    # Tokenizer
    tokenizer = QwenTokenizerWrapper()
    print(f"\n[3] Tokenizer初始化完成（真实 Qwen2 Tokenizer）")
    print(f"  实际词汇表大小: {tokenizer.vocab_size}")
    print(f"  PAD token ID: {tokenizer.pad_token_id}")
    print(f"  EOS token ID: {tokenizer.eos_token_id}")

    # 创建Trainer
    trainer = PPOTrainer(
        config=config,
        policy_model=policy_model,
        value_model=value_model,
        tokenizer=tokenizer,
        reward_func=length_reward
    )

    print("\n[4] PPO Trainer创建完成")

    # 训练数据
    train_prompts = [
        "请介绍一下人工智能的发展历史。",
        "如何学习编程？",
        "什么是强化学习？",
        "请写一首关于春天的诗。",
        "未来的科技会是什么样子？"
    ]

    print("\n[5] 训练数据准备完成")
    print(f"  Prompts数量: {len(train_prompts)}")

    # 开始训练
    print("\n[6] 开始训练")

    metrics_history = trainer.train(
        train_prompts=train_prompts,
        num_epochs=5,
        log_interval=1
    )

    # 分析结果
    print("\n[7] 训练结果分析")
    print("\n  训练曲线：")
    print("  Epoch | Reward | R(t=0) | R(t=1) | R(t=2) | Policy Loss | Value Loss")
    print("  " + "-"*80)
    for i, m in enumerate(metrics_history):
        r_t0 = m.get('reward_t0', 0)
        r_t1 = m.get('reward_t1', 0)
        r_t2 = m.get('reward_t2', 0)
        print(f"  {i+1:5d} | {m['reward_mean']:.4f} | "
              f"{r_t0:.4f}  | {r_t1:.4f}  | {r_t2:.4f}  | "
              f"{m['policy_loss']:.4f}      | {m['value_loss']:.4f}")

    # 总结
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)

    print("\n观察要点：")
    print("  ✅ R(t=0), R(t=1), R(t=2)的变化趋势")
    print("     - 通常后期reward更高（completion更完整）")
    print("  ✅ GAE如何利用这些时序信息")
    print("     - A_0受到r_0, r_1, r_2的影响")
    print("     - 通过(γλ)^l进行加权")

    print("\nPPO的特点：")
    print("  ✅ 理论保证强（单调改进）")
    print("  ✅ 多轮优化（提高样本效率）")
    print("  ✅ 使用Value Model（减少variance）")
    print("  ✅ 工业界广泛应用（ChatGPT使用PPO）")

    print("\nPPO vs GRPO vs DPO：")
    print("  - PPO: 最复杂，性能最好，需要Value Model")
    print("  - GRPO: 平衡，无需Value Model，适合快速实验")
    print("  - DPO: 最简单，直接从偏好数据学习")

    print("\n多步训练的意义：")
    print("  ✅ 理解GAE的时序计算过程")
    print("  ✅ 观察'未来信息'如何被利用")
    print("  ✅ 体验离线RL与在线RL的区别")
    print("  ✅ 为实际长文本任务做准备")

    print("\n实现特性：")
    print("  ✅ 完整的GAE（Generalized Advantage Estimation）")
    print("  ✅ 模块化设计（数据收集、GAE计算、策略更新）")
    print("  ✅ 详细的理论注释（代码与论文公式对应）")
    print("  ✅ Clipped Surrogate Objective（防止更新过大）")
    print("  ✅ 梯度裁剪和Advantage归一化")
    print("  ✅ 多轮优化（提高样本效率）")
    print("  ✅ 多步训练（T=3，理解GAE原理）")


if __name__ == "__main__":
    main()

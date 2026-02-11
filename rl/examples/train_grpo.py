"""
GRPO训练示例

演示如何使用GRPO训练语言模型：
1. 定义简单的reward函数（例如：长度奖励、关键词奖励）
2. 创建prompt数据
3. 训练policy模型
4. 观导olicy变化
"""

import torch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl.models.policy_model import PolicyModel
from rl.trainers.grpo_trainer import GRPOTrainer
from rl.config_rl import GRPOConfig
from core.tokenizers import QwenTokenizerWrapper

def length_reward(prompt: str, completion: str) -> float:
    """
    长度奖励：鼓励生成长度适中的回复

    Reward规则：
        - 长度在50-100: +1.0
        - 长度在30-50或100-150: +0.5
        - 其他: 0.0

    Args:
        prompt: 输入的prompt
        completion: 生成的completion

    Returns:
        reward分数
    """
    length = len(completion.split())

    if 50 <= length <= 100:
        return 1.0
    elif 30 <= length <= 150:
        return 0.5
    else:
        return 0.0


def keyword_reward(prompt: str, completion: str, keywords: list = None) -> float:
    """
    关键词奖励：鼓励包含特定关键词的回复

    Args:
        prompt: 输入的prompt
        completion: 生成的completion
        keywords: 关键词列表

    Returns:
        reward分数（包含关键词的比例）
    """
    if keywords is None:
        keywords = ["谢谢", "很高兴", "希望"]

    score = sum(1.0 for kw in keywords if kw in completion)
    return score / len(keywords)


def combined_reward(prompt: str, completion: str) -> float:
    """
    组合奖励：结合长度和关键词奖励

    Args:
        prompt: 输入的prompt
        completion: 生成的completion

    Returns:
        reward分数
    """
    length_r = length_reward(prompt, completion)
    keyword_r = keyword_reward(prompt, completion)

    # 加权平均
    return 0.7 * length_r + 0.3 * keyword_r


# =====================================
# 主训练函数
# =====================================

def main():
    print("\n" + "="*60)
    print("GRPO训练示例")
    print("="*60)

    # =====================================
    # 1. 配置
    # =====================================
    config = GRPOConfig(
        # 模型参数
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        vocab_size=151657,  # Qwen2-VL 真实词汇表大小
        max_seq_len=128,

        # GRPO参数
        num_generations=4,  # 每个prompt生成的样本数
        clip_range_low=0.1,
        clip_range_high=0.2,
        scale_rewards=True,
        reward_baseline="mean",

        # Reward设置
        use_reward_model=False,  # 使用自定义reward函数
        reward_func=length_reward,

        # 训练参数
        learning_rate=1e-4,
        batch_size=2,
        max_grad_norm=1.0,

        # 生成参数
        max_new_tokens=64,
        temperature=1.0,
        top_p=0.9
    )

    print("\n[1] 配置初始化完成")
    print(f"  模型维度: {config.hidden_dim}")
    print(f"  每个prompt生成数: {config.num_generations}")
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
    # 3. 创建 tokenizer（真实 Qwen2 Tokenizer）
    # =====================================
    tokenizer = QwenTokenizerWrapper()
    print(f"\n[3] Tokenizer初始化完成（真实 Qwen2 Tokenizer）")
    print(f"  词汇表大小: {tokenizer.vocab_size}")
    print(f"  PAD token ID: {tokenizer.pad_token_id}")
    print(f"  EOS token ID: {tokenizer.eos_token_id}")

    # =====================================
    # 4. 创建 Trainer
    # =====================================
    trainer = GRPOTrainer(
        config=config,
        policy_model=policy_model,
        tokenizer=tokenizer,
        reward_func=length_reward  # 使用长度奖励
    )

    print("\n[4] GRPO Trainer创建完成")

    # =====================================
    # 5. 准备训练数据
    # =====================================
    train_prompts = [
        "请介绍一下人工智能的发展历史。",
        "如何学习编程？",
        "什么是强化学习？",
        "请写一首关于春天的诗。",
        "未来的科技会是什么样子？"
    ]

    print("\n[5] 训练数据准备完成")
    print(f"  Prompts数量: {len(train_prompts)}")
    for i, prompt in enumerate(train_prompts):
        print(f"    {i+1}. {prompt}")

    # =====================================
    # 6. 训练前生成样本（查看初始状态）
    # =====================================
    print("\n[6] 训练前生成样本")
    before_samples = trainer.generate_samples(
        prompts=train_prompts[:1],
        num_samples=2,
        temperature=0.8
    )

    print(f"  Prompt: {train_prompts[0]}")
    for i, sample in enumerate(before_samples[0]):
        print(f"    样本{i+1}: {sample[:50]}...")

    # =====================================
    # 7. 开始训练
    # =====================================
    print("\n[7] 开始训练")

    metrics_history = trainer.train(
        train_prompts=train_prompts,
        num_epochs=5,
        log_interval=1
    )

    # =====================================
    # 8. 训练后生成样本（查看policy变化）
    # =====================================
    print("\n[8] 训练后生成样本")
    after_samples = trainer.generate_samples(
        prompts=train_prompts[:1],
        num_samples=2,
        temperature=0.8
    )

    print(f"  Prompt: {train_prompts[0]}")
    for i, sample in enumerate(after_samples[0]):
        print(f"    样本{i+1}: {sample[:50]}...")

    # =====================================
    # 9. 分析训练结果
    # =====================================
    print("\n[9] 训练结果分析")
    print("\n  训练曲线：")
    print("  Epoch | Loss   | Reward | KL     | Ratio  | Clip %")
    print("  " + "-"*60)
    for i, m in enumerate(metrics_history):
        print(f"  {i+1:5d} | {m['loss']:.4f} | {m['reward_mean']:.4f} | "
              f"{m['kl']:.4f} | {m['mean_ratio']:.4f} | {m['clip_fraction']:.4f}")

    # 计算reward提升
    initial_reward = metrics_history[0]['reward_mean']
    final_reward = metrics_history[-1]['reward_mean']
    improvement = final_reward - initial_reward

    if abs(initial_reward) > 1e-6:  # 避免除零错误
        improvement_pct = improvement/abs(initial_reward)*100
        print(f"\n  Reward提升: {initial_reward:.4f} -> {final_reward:.4f} "
              f"(+{improvement:.4f}, {improvement_pct:.1f}%)")
    else:
        print(f"\n  Reward提升: {initial_reward:.4f} -> {final_reward:.4f} "
              f"(+{improvement:.4f})")

    # =====================================
    # 10. 总结
    # =====================================
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print("\n关键观察：")
    print(f"  1. Loss是否下降: "
          f"{metrics_history[0]['loss']:.4f} -> {metrics_history[-1]['loss']:.4f}")
    print(f"  2. Reward是否提升: "
          f"{metrics_history[0]['reward_mean']:.4f} -> {metrics_history[-1]['reward_mean']:.4f}")
    print(f"  3. KL散度是否可控: {metrics_history[-1]['kl']:.4f}")
    print(f"  4. Clip Fraction: {metrics_history[-1]['clip_fraction']:.4f} "
          f"(0.1-0.3为正常范围)")

    print("\n提示：")
    print("  - 这是一个简化的演示，实际应用中需要：")
    print("    1. 更多的训练数据")
    print("    2. 更大的模型")
    print("    3. 更复杂的reward函数或训练好的Reward Model")
    print("    4. 调优超参数（learning rate, clip range等）")


if __name__ == "__main__":
    main()
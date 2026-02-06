"""
训练脚本
演示多模态模型的完整训练流程，包括前向传播、反向传播和参数更新
"""
import torch
import torch.optim as optim
from config import ModelConfig
from models.multimodal_model import SimpleMultimodalModel
from models.tokenizer import QwenTokenizerWrapper
from utils.debug_utils import (
    print_gradient_info,
    check_gradient_flow,
    plot_loss_curve,
    count_parameters
)


def generate_dummy_data(batch_size: int, config: ModelConfig, tokenizer):
    """
    生成训练数据（使用真实文本）

    在真实场景中，这里应该是从数据集加载的图像-文本对
    现在使用tokenizer编码真实的中文文本

    Args:
        batch_size: 批次大小
        config: 模型配置
        tokenizer: Qwen tokenizer封装

    Returns:
        image: 图像tensor，shape [batch, channels, height, width]
        text: 文本token IDs，shape [batch, seq_len]
        labels: 目标token IDs，shape [batch, seq_len]
        attention_mask: attention mask，shape [batch, seq_len]
    """
    # 随机图像数据（保持不变）
    image = torch.randn(
        batch_size,
        config.in_channels,
        config.image_size,
        config.image_size
    )

    # 使用tokenizer生成真实文本数据
    seq_len = config.max_seq_len // 2  # 使用较短的序列
    sample_texts = tokenizer.get_sample_texts(batch_size)

    # 编码文本
    encoded = tokenizer.batch_encode(
        sample_texts,
        max_length=seq_len,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    text = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    # 标签：简单起见，使用相同文本作为标签（自监督学习）
    labels = text.clone()

    return image, text, labels, attention_mask


def train_one_step(model: SimpleMultimodalModel,
                  optimizer: torch.optim.Optimizer,
                  image: torch.Tensor,
                  text: torch.Tensor,
                  labels: torch.Tensor,
                  attention_mask: torch.Tensor,
                  step: int,
                  debug: bool = True) -> float:
    """
    训练一步，展示完整的前向和反向传播过程

    Args:
        model: 多模态模型
        optimizer: 优化器
        image: 输入图像
        text: 输入文本
        labels: 目标标签
        attention_mask: attention mask
        step: 当前步数
        debug: 是否打印调试信息

    Returns:
        loss: 损失值
    """
    if debug:
        print("\n" + "╔" + "═"*78 + "╗")
        print(f"║  训练步骤 {step:3d}  " + " "*63 + "║")
        print("╚" + "═"*78 + "╝")

    # ========== 前向传播 ==========
    if debug:
        print("\n[阶段1] 前向传播")
        print("─"*80)

    model.train()  # 设置为训练模式（启用dropout等）

    # 前向传播（传递attention_mask）
    logits, loss = model(image, text, labels, attention_mask=attention_mask, debug=debug)

    if not debug:
        print(f"Step {step}: Loss = {loss.item():.4f}")

    # ========== 反向传播 ==========
    if debug:
        print("\n[阶段2] 反向传播")
        print("─"*80)

    # 清空之前的梯度
    optimizer.zero_grad()

    if debug:
        print("✓ 梯度已清零")

    # 计算梯度
    loss.backward()

    if debug:
        print("✓ 梯度计算完成")

    # ========== 梯度信息 ==========
    if debug:
        print("\n[阶段3] 梯度分析")
        print("─"*80)

        # 打印梯度信息
        total_grad_norm = print_gradient_info(model, detailed=False)

        # 检查梯度消失/爆炸
        check_gradient_flow(model, threshold=1e-7)

    # ========== 参数更新 ==========
    if debug:
        print("\n[阶段4] 参数更新")
        print("─"*80)

    # 梯度裁剪（防止梯度爆炸）
    max_grad_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    if debug:
        print(f"✓ 梯度裁剪完成 (max_norm={max_grad_norm})")

    # 更新参数
    optimizer.step()

    if debug:
        print("✓ 参数已更新")
        print("\n" + "─"*80)
        print(f"第{step}步训练完成！Loss: {loss.item():.4f}")
        print("─"*80)

    return loss.item()


def train(config: ModelConfig, num_steps: int = 20, debug_interval: int = 5):
    """
    完整的训练流程

    Args:
        config: 模型配置
        num_steps: 训练步数
        debug_interval: 每隔多少步打印详细调试信息
    """
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*25 + "开始训练多模态模型" + " "*25 + "║")
    print("╚" + "═"*78 + "╝\n")

    # ========== 1. 创建模型 ==========
    print("─"*80)
    print("1. 初始化模型")
    print("─"*80)

    model = SimpleMultimodalModel(config)
    model.print_model_info()
    count_parameters(model)

    # ========== 2. 初始化Tokenizer ==========
    print("\n" + "─"*80)
    print("2. 初始化Tokenizer")
    print("─"*80)

    tokenizer = QwenTokenizerWrapper()

    # 更新config中的pad_token_id（用于loss计算）
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id

    # ========== 3. 创建优化器 ==========
    print("─"*80)
    print("3. 创建优化器")
    print("─"*80)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    print(f"优化器: AdamW")
    print(f"学习率: {config.learning_rate}")
    print(f"权重衰减: {config.weight_decay}")

    # ========== 4. 训练循环 ==========
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*30 + "开始训练循环" + " "*30 + "║")
    print("╚" + "═"*78 + "╝\n")

    loss_history = []

    for step in range(1, num_steps + 1):
        # 生成训练数据（包含真实文本）
        image, text, labels, attention_mask = generate_dummy_data(
            config.batch_size, config, tokenizer
        )

        # 决定是否打印详细调试信息
        debug = (step % debug_interval == 0) or (step == 1)

        # 训练一步
        loss = train_one_step(
            model=model,
            optimizer=optimizer,
            image=image,
            text=text,
            labels=labels,
            attention_mask=attention_mask,
            step=step,
            debug=debug
        )

        loss_history.append(loss)

    # ========== 4. 训练总结 ==========
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*30 + "训练总结" + " "*36 + "║")
    print("╚" + "═"*78 + "╝")

    # 绘制损失曲线
    plot_loss_curve(loss_history, window_size=3)

    # 训练统计
    print(f"\n训练统计:")
    print(f"  总步数: {num_steps}")
    print(f"  最终loss: {loss_history[-1]:.4f}")
    print(f"  最低loss: {min(loss_history):.4f}")
    print(f"  平均loss: {sum(loss_history)/len(loss_history):.4f}")

    # 检查是否在学习
    if len(loss_history) >= 10:
        early_avg = sum(loss_history[:5]) / 5
        late_avg = sum(loss_history[-5:]) / 5
        improvement = (early_avg - late_avg) / early_avg * 100

        print(f"\n学习进度:")
        print(f"  前5步平均loss: {early_avg:.4f}")
        print(f"  后5步平均loss: {late_avg:.4f}")
        print(f"  改善幅度: {improvement:.1f}%")

        if improvement > 0:
            print(f"  ✓ 模型正在学习!")
        else:
            print(f"  ⚠ 模型可能没有学习，请检查学习率和数据")

    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*30 + "训练完成!" + " "*35 + "║")
    print("╚" + "═"*78 + "╝\n")

    return model, loss_history


def quick_test():
    """
    快速测试（少量步数，用于验证代码）
    """
    print("\n快速测试模式")
    print("="*80)

    config = ModelConfig(
        hidden_dim=64,      # 更小的模型
        num_heads=2,
        num_layers=1,
        vocab_size=151657,  # 使用Qwen tokenizer的词汇表大小
        image_size=32,      # 更小的图像
        patch_size=8,
        max_seq_len=16,     # 更短的序列
        batch_size=2,
        learning_rate=1e-3,
        debug=False
    )

    # 只训练5步
    model, loss_history = train(config, num_steps=5, debug_interval=1)

    print("\n快速测试完成!")


def full_training():
    """
    完整训练（按照计划中的配置）
    """
    print("\n完整训练模式")
    print("="*80)

    config = ModelConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=1,
        vocab_size=151657,  # 使用Qwen tokenizer的词汇表大小
        image_size=64,
        patch_size=16,
        max_seq_len=32,
        batch_size=2,
        learning_rate=1e-4,
        debug=False
    )

    config.print_config()

    # 训练20步
    model, loss_history = train(config, num_steps=20, debug_interval=5)

    print("\n完整训练完成!")

    return model, loss_history


if __name__ == "__main__":
    import sys

    # 根据命令行参数选择模式
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        full_training()

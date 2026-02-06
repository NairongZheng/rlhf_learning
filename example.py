"""
推理示例脚本
演示如何使用多模态模型进行推理
"""
import torch
from config import ModelConfig
from models.multimodal_model import SimpleMultimodalModel
from models.tokenizer import QwenTokenizerWrapper
from utils.debug_utils import print_tensor_info, visualize_attention


def create_sample_data(config: ModelConfig):
    """
    创建示例数据（使用真实文本）

    Args:
        config: 模型配置

    Returns:
        image: 示例输入图像
        text: 示例输入文本token IDs
        tokenizer: tokenizer实例
        original_text: 原始文本字符串
    """
    # 初始化tokenizer
    tokenizer = QwenTokenizerWrapper()

    # 创建一个示例图像（随机数据）
    # 在真实应用中，这里应该是加载真实图像并预处理
    image = torch.randn(1, config.in_channels, config.image_size, config.image_size)

    # 使用tokenizer编码真实文本
    seq_len = 16
    sample_text = tokenizer.get_sample_texts(1)  # 获取一个样本
    print(f"\n输入文本: {sample_text}")

    encoded = tokenizer.encode(
        sample_text,
        max_length=seq_len,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    text = encoded['input_ids']

    return image, text, tokenizer, sample_text


def inference_with_debug(model: SimpleMultimodalModel,
                        image: torch.Tensor,
                        text: torch.Tensor,
                        tokenizer: QwenTokenizerWrapper,
                        original_text: str,
                        config: ModelConfig):
    """
    带详细调试信息的推理

    Args:
        model: 多模态模型
        image: 输入图像
        text: 输入文本
        tokenizer: tokenizer实例
        original_text: 原始文本字符串
        config: 模型配置

    Returns:
        logits: 输出logits
        predictions: 预测的token IDs
    """
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*25 + "多模态模型推理示例" + " "*25 + "║")
    print("╚" + "═"*78 + "╝\n")

    # 设置为评估模式（关闭dropout等）
    model.eval()

    print("─"*80)
    print("模型已设置为评估模式（eval mode）")
    print("─"*80)

    # 输入信息
    print("\n输入数据信息:")
    print_tensor_info("输入图像", image, detailed=True)
    print_tensor_info("输入文本token IDs", text, detailed=False)

    # 关闭梯度计算（推理时不需要梯度）
    with torch.no_grad():
        print("\n" + "╔" + "═"*78 + "╗")
        print("║" + " "*30 + "开始推理" + " "*36 + "║")
        print("╚" + "═"*78 + "╝")

        # 前向传播（带详细调试信息）
        logits, _ = model(image, text, labels=None, debug=True)

    # ========== 分析输出 ==========
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*30 + "输出分析" + " "*36 + "║")
    print("╚" + "═"*78 + "╝")

    # 获取预测的token IDs
    predictions = torch.argmax(logits, dim=-1)

    print(f"\n预测结果:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  预测token IDs shape: {predictions.shape}")

    # 打印前10个预测token
    print(f"\n前10个预测token IDs:")
    print(f"  {predictions[0, :10].tolist()}")

    # Softmax概率
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs[0, 0], k=5)

    print(f"\n第一个位置的Top-5预测:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        print(f"  #{i+1}: token {idx.item()}, 概率 {prob.item():.4f}")

    # 统计信息
    print(f"\nLogits统计:")
    print(f"  最小值: {logits.min().item():.4f}")
    print(f"  最大值: {logits.max().item():.4f}")
    print(f"  均值: {logits.mean().item():.4f}")
    print(f"  标准差: {logits.std().item():.4f}")

    # ========== 文本解码分析 ==========
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*30 + "文本分析" + " "*36 + "║")
    print("╚" + "═"*78 + "╝")

    # 解码输入和输出
    input_decoded = tokenizer.decode(text[0], skip_special_tokens=True)
    output_decoded = tokenizer.decode(predictions[0], skip_special_tokens=True)

    print(f"\n原始输入文本: {original_text}")
    print(f"解码输入文本: {input_decoded}")
    print(f"预测输出文本: {output_decoded}")

    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*30 + "推理完成!" + " "*35 + "║")
    print("╚" + "═"*78 + "╝\n")

    return logits, predictions


def simple_inference(model: SimpleMultimodalModel,
                    image: torch.Tensor,
                    text: torch.Tensor):
    """
    简单的推理示例（不打印详细信息）

    Args:
        model: 多模态模型
        image: 输入图像
        text: 输入文本

    Returns:
        predictions: 预测的token IDs
    """
    model.eval()

    with torch.no_grad():
        logits, _ = model(image, text, labels=None, debug=False)

    predictions = torch.argmax(logits, dim=-1)

    return predictions


def compare_multiple_inputs(model: SimpleMultimodalModel, config: ModelConfig, tokenizer: QwenTokenizerWrapper):
    """
    比较多个不同输入的输出

    演示模型对不同输入的反应

    Args:
        model: 多模态模型
        config: 模型配置
        tokenizer: tokenizer实例
    """
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*25 + "多输入对比实验" + " "*29 + "║")
    print("╚" + "═"*78 + "╝\n")

    model.eval()

    num_samples = 3
    print(f"生成 {num_samples} 个不同的输入样本...\n")

    with torch.no_grad():
        for i in range(num_samples):
            print(f"─"*80)
            print(f"样本 {i+1}")
            print(f"─"*80)

            # 生成随机输入
            image, text, _, original_text = create_sample_data(config)

            # 推理
            predictions = simple_inference(model, image, text)

            # 解码显示
            input_decoded = tokenizer.decode(text[0][:5], skip_special_tokens=True)
            output_decoded = tokenizer.decode(predictions[0][:5], skip_special_tokens=True)

            print(f"原始文本: {original_text}")
            print(f"输入文本 (前5 tokens): {input_decoded}")
            print(f"预测输出 (前5 tokens): {output_decoded}")
            print()


def visualize_model_internals(model: SimpleMultimodalModel,
                             image: torch.Tensor,
                             text: torch.Tensor):
    """
    可视化模型内部状态

    包括attention权重等

    Args:
        model: 多模态模型
        image: 输入图像
        text: 输入文本
    """
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*25 + "模型内部可视化" + " "*29 + "║")
    print("╚" + "═"*78 + "╝\n")

    model.eval()

    with torch.no_grad():
        # 获取融合层的attention权重
        print("提取融合层的attention权重...\n")

        # 先获取中间特征
        image_features = model.vision_encoder(image, debug=False)
        text_features = model.text_encoder(text, debug=False)

        # 获取fusion层的attention权重
        _, attention_weights = model.fusion_layer(
            text_features=text_features,
            image_features=image_features,
            return_attention=True,
            debug=False
        )

        # 可视化attention
        visualize_attention(
            attention_weights,
            name="Fusion Layer Cross-Attention",
            show_top_k=3
        )


def main():
    """
    主函数：运行所有示例
    """
    print("\n" + "="*80)
    print(" "*25 + "多模态模型推理示例程序")
    print("="*80)

    # ========== 1. 创建配置和模型 ==========
    print("\n[1] 初始化模型")
    print("─"*80)

    config = ModelConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=1,
        vocab_size=151657,  # 使用Qwen tokenizer的词汇表大小
        image_size=64,
        patch_size=16,
        max_seq_len=32,
        batch_size=1,
        debug=False
    )

    model = SimpleMultimodalModel(config)
    model.print_model_info()

    print("✓ 模型初始化完成\n")

    # ========== 2. 创建示例数据 ==========
    print("[2] 准备示例数据")
    print("─"*80)

    image, text, tokenizer, original_text = create_sample_data(config)

    print(f"✓ 数据准备完成")
    print(f"  图像shape: {image.shape}")
    print(f"  文本shape: {text.shape}\n")

    # ========== 3. 详细推理示例 ==========
    print("[3] 运行详细推理示例")
    print("─"*80)

    logits, predictions = inference_with_debug(
        model, image, text, tokenizer, original_text, config
    )

    # ========== 4. 简单推理示例 ==========
    print("\n[4] 运行简单推理示例")
    print("─"*80)

    simple_predictions = simple_inference(model, image, text)
    decoded_simple = tokenizer.decode(simple_predictions[0][:10], skip_special_tokens=True)
    print(f"✓ 简单推理完成")
    print(f"  预测结果 (前10个token): {decoded_simple}\n")

    # ========== 5. 多输入对比 ==========
    print("[5] 运行多输入对比实验")
    print("─"*80)

    compare_multiple_inputs(model, config, tokenizer)

    # ========== 6. 可视化内部状态 ==========
    print("[6] 可视化模型内部状态")
    print("─"*80)

    visualize_model_internals(model, image, text)

    # ========== 完成 ==========
    print("\n" + "="*80)
    print(" "*30 + "所有示例运行完成!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

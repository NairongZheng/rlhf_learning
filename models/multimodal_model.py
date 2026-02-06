"""
多模态模型（Multimodal Model）
整合Vision Encoder、Text Encoder、Fusion Layer和Text Decoder
"""
import torch
import torch.nn as nn
from .vision_encoder import VisionEncoder
from .text_encoder import TextEncoder
from .fusion_layer import FusionLayer
from .text_decoder import TextDecoder
from utils.debug_utils import print_tensor_info, count_parameters


class SimpleMultimodalModel(nn.Module):
    """
    简单的多模态大模型

    架构流程：
    输入图像 -> Vision Encoder -> 图像特征
                                        |
                                        v
                                  Cross Attention Fusion
                                        ^
                                        |
    输入文字 -> Text Encoder -> 文字特征

    融合特征 -> Text Decoder -> 输出文字logits

    这是一个教学用的简化版本，帮助理解多模态模型的基本原理
    """

    def __init__(self, config):
        """
        初始化多模态模型

        Args:
            config: ModelConfig对象，包含所有超参数
        """
        super().__init__()
        self.config = config

        # 1. Vision Encoder：图像编码器
        self.vision_encoder = VisionEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
            norm_type=config.norm_type
        )

        # 2. Text Encoder：文本编码器
        self.text_encoder = TextEncoder(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_seq_len=config.max_seq_len,
            rope_base=config.rope_base if config.use_rope else None,
            dropout=config.dropout,
            norm_type=config.norm_type
        )

        # 3. Fusion Layer：跨模态融合层
        self.fusion_layer = FusionLayer(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            norm_type=config.norm_type
        )

        # 4. Text Decoder：文本解码器
        self.text_decoder = TextDecoder(
            hidden_dim=config.hidden_dim,
            vocab_size=config.vocab_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_seq_len=config.max_seq_len,
            rope_base=config.rope_base if config.use_rope else None,
            dropout=config.dropout,
            norm_type=config.norm_type
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        初始化模型权重

        使用标准的初始化策略：
        - Linear层：xavier uniform初始化
        - Embedding层：正态分布初始化
        - LayerNorm：权重为1，偏置为0
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor, text: torch.Tensor,
                labels: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                debug: bool = False) -> tuple:
        """
        前向传播

        Args:
            image: 输入图像，shape [batch, channels, height, width]
            text: 输入文本token IDs，shape [batch, seq_len]
            labels: 目标文本token IDs（用于计算loss），shape [batch, seq_len]
            attention_mask: attention mask，shape [batch, seq_len]
            debug: 是否打印详细调试信息

        Returns:
            logits: 预测的词汇表logits，shape [batch, seq_len, vocab_size]
            loss: 如果提供了labels，返回交叉熵损失
        """
        if debug:
            print("\n" + "="*80)
            print("多模态模型前向传播")
            print("="*80)

        # ========== Step 1: 图像编码 ==========
        if debug:
            print("\n" + "█"*80)
            print("Step 1: Vision Encoder - 图像编码")
            print("█"*80)

        image_features = self.vision_encoder(image, debug=debug)    # image: [2, 3, 64, 64] -> image_features: [2, num_patches, hidden_dim]

        if debug:
            print(f"\n✓ 图像编码完成")
            print(f"  输出shape: {image_features.shape}")
            print(f"  统计: 均值={image_features.mean().item():.4f}, "
                  f"标准差={image_features.std().item():.4f}")

        # ========== Step 2: 文本编码 ==========
        if debug:
            print("\n" + "█"*80)
            print("Step 2: Text Encoder - 文本编码")
            print("█"*80)

        text_features = self.text_encoder(text, debug=debug)

        if debug:
            print(f"\n✓ 文本编码完成")
            print(f"  输出shape: {text_features.shape}")
            print(f"  统计: 均值={text_features.mean().item():.4f}, "
                  f"标准差={text_features.std().item():.4f}")

        # ========== Step 3: 跨模态融合 ==========
        if debug:
            print("\n" + "█"*80)
            print("Step 3: Fusion Layer - 跨模态融合")
            print("█"*80)

        fused_features = self.fusion_layer(
            text_features=text_features,
            image_features=image_features,
            debug=debug
        )

        if debug:
            print(f"\n✓ 跨模态融合完成")
            print(f"  输出shape: {fused_features.shape}")
            print(f"  统计: 均值={fused_features.mean().item():.4f}, "
                  f"标准差={fused_features.std().item():.4f}")

        # ========== Step 4: 文本解码 ==========
        if debug:
            print("\n" + "█"*80)
            print("Step 4: Text Decoder - 文本生成")
            print("█"*80)

        # 生成causal mask（用于自回归生成）
        seq_len = text.shape[1]
        causal_mask = self.text_decoder.generate_causal_mask(
            seq_len, device=text.device
        )

        logits = self.text_decoder(fused_features, causal_mask, debug=debug)

        if debug:
            print(f"\n✓ 文本解码完成")
            print(f"  输出shape: {logits.shape}")
            print(f"  统计: 均值={logits.mean().item():.4f}, "
                  f"标准差={logits.std().item():.4f}")

        # ========== Step 5: 计算损失（如果提供了labels） ==========
        loss = None
        if labels is not None:
            if debug:
                print("\n" + "█"*80)
                print("Step 5: 计算损失")
                print("█"*80)

            # 重塑logits和labels以计算交叉熵
            # logits: [batch, seq_len, vocab_size] -> [batch*seq_len, vocab_size]
            # labels: [batch, seq_len] -> [batch*seq_len]
            logits_flat = logits.view(-1, self.config.vocab_size)
            labels_flat = labels.view(-1)

            # 使用ignore_index忽略padding token
            # 如果tokenizer有pad_token_id，使用它；否则使用-100（CrossEntropyLoss默认ignore值）
            pad_id = getattr(self.config, 'pad_token_id', -100)
            if pad_id is None:
                pad_id = -100  # 默认值

            # 计算交叉熵损失（忽略padding token）
            loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
            loss = loss_fn(logits_flat, labels_flat)

            if debug:
                print(f"\n✓ 损失计算完成")
                print(f"  Loss: {loss.item():.4f}")

                # 计算准确率（用于监控）
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == labels).float().mean().item()
                print(f"  Token准确率: {accuracy*100:.2f}%")

        if debug:
            print("\n" + "="*80)
            print("前向传播完成")
            print("="*80 + "\n")

        return logits, loss

    def print_model_info(self):
        """打印模型结构信息"""
        print("\n" + "="*80)
        print("多模态模型结构信息")
        print("="*80)

        # 统计各组件的参数量
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        text_encoder_params = sum(p.numel() for p in self.text_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_layer.parameters())
        decoder_params = sum(p.numel() for p in self.text_decoder.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        print(f"\n各组件参数量:")
        print(f"  Vision Encoder:  {vision_params:>10,} ({vision_params/total_params*100:.1f}%)")
        print(f"  Text Encoder:    {text_encoder_params:>10,} ({text_encoder_params/total_params*100:.1f}%)")
        print(f"  Fusion Layer:    {fusion_params:>10,} ({fusion_params/total_params*100:.1f}%)")
        print(f"  Text Decoder:    {decoder_params:>10,} ({decoder_params/total_params*100:.1f}%)")
        print(f"  {'─'*50}")
        print(f"  总计:            {total_params:>10,}")

        # 估算模型大小（假设float32）
        model_size_mb = total_params * 4 / (1024 ** 2)
        print(f"\n模型大小 (float32): {model_size_mb:.2f} MB")

        print("="*80 + "\n")


def test_multimodal_model():
    """
    测试完整的多模态模型
    """
    from config import ModelConfig

    print("\n" + "="*80)
    print("测试多模态模型")
    print("="*80)

    # 创建配置
    config = ModelConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=1,
        vocab_size=1000,
        image_size=64,
        patch_size=16,
        max_seq_len=32,
        debug=True
    )

    config.print_config()

    # 创建模型
    model = SimpleMultimodalModel(config)
    model.print_model_info()

    # 创建随机输入数据
    batch_size = 2
    image = torch.randn(batch_size, 3, 64, 64)
    text = torch.randint(0, 1000, (batch_size, 16))
    labels = torch.randint(0, 1000, (batch_size, 16))

    print("\n输入数据信息:")
    print(f"  图像shape: {image.shape}")
    print(f"  文本shape: {text.shape}")
    print(f"  标签shape: {labels.shape}")

    # 前向传播
    print("\n" + "="*80)
    print("开始前向传播...")
    print("="*80)

    logits, loss = model(image, text, labels, debug=True)

    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)
    print(f"输出logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    test_multimodal_model()

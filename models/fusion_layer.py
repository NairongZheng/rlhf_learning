"""
跨模态融合层（Fusion Layer）
使用Cross Attention融合图像和文本特征
"""
import torch
import torch.nn as nn
from .attention import CrossAttention, FeedForward
from utils.debug_utils import print_tensor_info


class FusionLayer(nn.Module):
    """
    跨模态融合层

    使用Cross Attention将文本特征和图像特征融合：
    - Query: 文本特征（"我想从图像中查询什么信息？"）
    - Key & Value: 图像特征（提供视觉信息）

    流程：
    1. Cross Attention: 文本查询图像信息
    2. Feed-Forward: 进一步处理融合后的特征
    3. LayerNorm + Residual Connection: 稳定训练
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 dropout: float = 0.1, norm_type: str = "pre"):
        """
        初始化融合层

        Args:
            hidden_dim: 输入维度
            num_heads: attention头数
            dropout: dropout率
            norm_type: LayerNorm类型（"pre"或"post"）
        """
        super().__init__()
        self.norm_type = norm_type

        # Cross Attention层
        self.cross_attention = CrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Feed-Forward Network
        self.ffn = FeedForward(
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # LayerNorm层
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor,
                mask: torch.Tensor = None, return_attention: bool = False,
                debug: bool = False) -> torch.Tensor:
        """
        前向传播

        Args:
            text_features: 文本特征，shape [batch, seq_len_text, hidden_dim]
            image_features: 图像特征，shape [batch, num_patches, hidden_dim]
            mask: 可选的attention mask
            return_attention: 是否返回attention权重
            debug: 是否打印调试信息

        Returns:
            融合后的特征，shape [batch, seq_len_text, hidden_dim]
            (如果return_attention=True，还会返回attention权重)
        """
        if debug:
            print("\n" + "="*60)
            print("Fusion Layer - 跨模态融合")
            print("="*60)
            print_tensor_info("输入文本特征", text_features, detailed=False)
            print_tensor_info("输入图像特征", image_features, detailed=False)

        # Pre-LayerNorm或Post-LayerNorm
        if self.norm_type == "pre":
            # Pre-LayerNorm: 先norm再计算

            # Step 1: Cross Attention
            if debug:
                print("\n[1] Cross Attention (文本查询图像)")

            residual = text_features
            text_features = self.norm1(text_features)

            if return_attention:
                attn_out, attn_weights = self.cross_attention(
                    query=text_features,
                    context=image_features,
                    mask=mask,
                    return_attention=True
                )
            else:
                attn_out = self.cross_attention(
                    query=text_features,
                    context=image_features,
                    mask=mask,
                    return_attention=False
                )

            text_features = residual + self.dropout(attn_out)

            if debug:
                print(f"  Cross Attention输出shape: {attn_out.shape}")

            # Step 2: Feed-Forward
            if debug:
                print("\n[2] Feed-Forward Network")

            residual = text_features
            text_features = self.norm2(text_features)
            ffn_out = self.ffn(text_features)
            text_features = residual + self.dropout(ffn_out)

        else:
            # Post-LayerNorm: 先计算再norm

            # Step 1: Cross Attention
            if debug:
                print("\n[1] Cross Attention (文本查询图像)")

            residual = text_features

            if return_attention:
                attn_out, attn_weights = self.cross_attention(
                    query=text_features,
                    context=image_features,
                    mask=mask,
                    return_attention=True
                )
            else:
                attn_out = self.cross_attention(
                    query=text_features,
                    context=image_features,
                    mask=mask,
                    return_attention=False
                )

            text_features = self.norm1(residual + self.dropout(attn_out))

            if debug:
                print(f"  Cross Attention输出shape: {attn_out.shape}")

            # Step 2: Feed-Forward
            if debug:
                print("\n[2] Feed-Forward Network")

            residual = text_features
            ffn_out = self.ffn(text_features)
            text_features = self.norm2(residual + self.dropout(ffn_out))

        if debug:
            print_tensor_info("融合后的特征", text_features)

        if return_attention:
            return text_features, attn_weights
        return text_features


class MultipleFusionLayers(nn.Module):
    """
    多层融合

    堆叠多个FusionLayer，可以进行更深层次的跨模态交互
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 num_layers: int = 1, dropout: float = 0.1,
                 norm_type: str = "pre"):
        """
        初始化多层融合

        Args:
            hidden_dim: 输入维度
            num_heads: attention头数
            num_layers: 融合层数
            dropout: dropout率
            norm_type: LayerNorm类型
        """
        super().__init__()

        self.layers = nn.ModuleList([
            FusionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                norm_type=norm_type
            )
            for _ in range(num_layers)
        ])

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor,
                mask: torch.Tensor = None, debug: bool = False) -> torch.Tensor:
        """
        前向传播

        Args:
            text_features: 文本特征
            image_features: 图像特征
            mask: 可选的mask
            debug: 是否打印调试信息

        Returns:
            融合后的特征
        """
        if debug:
            print(f"\n多层融合 (共{len(self.layers)}层)")

        for i, layer in enumerate(self.layers):
            if debug:
                print(f"\n--- 第{i+1}层融合 ---")

            text_features = layer(
                text_features=text_features,
                image_features=image_features,
                mask=mask,
                debug=debug
            )

        return text_features


def test_fusion_layer():
    """
    测试Fusion Layer
    """
    print("\n" + "="*60)
    print("测试Fusion Layer")
    print("="*60)

    # 参数
    batch_size = 2
    seq_len_text = 16
    num_patches = 16
    hidden_dim = 128
    num_heads = 4

    # 创建模型
    fusion = FusionLayer(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=0.1
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in fusion.parameters())
    print(f"\n总参数量: {total_params:,}")

    # 创建随机输入
    text_features = torch.randn(batch_size, seq_len_text, hidden_dim)
    image_features = torch.randn(batch_size, num_patches, hidden_dim)

    # 前向传播
    print("\n执行前向传播...")
    output = fusion(text_features, image_features, debug=True)

    print("\n测试完成!")
    print(f"文本特征shape: {text_features.shape}")
    print(f"图像特征shape: {image_features.shape}")
    print(f"融合后shape: {output.shape}")

    # 测试返回attention权重
    print("\n\n测试返回attention权重...")
    output, attn_weights = fusion(
        text_features, image_features,
        return_attention=True, debug=False
    )
    print(f"Attention权重shape: {attn_weights.shape}")


if __name__ == "__main__":
    test_fusion_layer()

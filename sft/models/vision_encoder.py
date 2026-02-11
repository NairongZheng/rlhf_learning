"""
图像编码器（Vision Encoder）
将图像转换为特征表示
"""
import torch
import torch.nn as nn
from core.modules.position_encoding import Image2DPositionEncoding
from core.modules.attention import TransformerBlock
from core.utils.debug_utils import print_tensor_info


class PatchEmbedding(nn.Module):
    """
    图像Patch嵌入层

    将图像分割成小块(patch)，然后将每个patch映射到embedding空间
    这是Vision Transformer (ViT)的核心思想
    """

    def __init__(self, image_size: int, patch_size: int,
                 in_channels: int, hidden_dim: int):
        """
        初始化Patch Embedding

        Args:
            image_size: 输入图像大小（正方形）
            patch_size: patch大小（正方形）
            in_channels: 输入图像通道数（RGB为3）
            hidden_dim: 输出embedding维度
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # 使用卷积层实现patch切分和线性投影
        # 卷积核大小=patch_size，步长=patch_size，相当于不重叠地切分图像
        # 输入: [batch, in_channels, image_size, image_size]
        # 输出: [batch, hidden_dim, num_patches_h, num_patches_w]
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像，shape [batch, channels, height, width]
            debug: 是否打印调试信息

        Returns:
            patch embeddings，shape [batch, num_patches, hidden_dim]
        """
        if debug:
            print_tensor_info("输入图像", x)

        # 应用卷积投影
        # [batch, channels, H, W] -> [batch, hidden_dim, H/patch_size, W/patch_size]
        x = self.projection(x)  # x: [2, 128, 4, 4] (假设image_size=64, patch_size=16)

        if debug:
            print(f"卷积后shape: {x.shape}")

        # 重塑为序列形式
        # [batch, hidden_dim, H', W'] -> [batch, hidden_dim, num_patches]
        batch_size = x.shape[0]
        x = x.flatten(2)        # x: [2, 128, 16] (num_patches=4*4=16)

        # [batch, hidden_dim, num_patches] -> [batch, num_patches, hidden_dim]
        x = x.transpose(1, 2)   # x: [2, 16, 128]

        if debug:
            print_tensor_info("Patch Embedding输出", x, detailed=False)

        return x


class VisionEncoder(nn.Module):
    """
    视觉编码器

    完整流程：
    1. Patch Embedding: 将图像切分成patches并投影到embedding空间
    2. Position Encoding: 为每个patch添加2D位置信息
    3. Transformer: 提取视觉特征
    4. LayerNorm: 最终的标准化
    """

    def __init__(self, image_size: int = 64, patch_size: int = 16,
                 in_channels: int = 3, hidden_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 1,
                 dropout: float = 0.1, norm_type: str = "pre"):
        """
        初始化视觉编码器

        Args:
            image_size: 输入图像大小
            patch_size: patch大小
            in_channels: 输入通道数
            hidden_dim: embedding维度
            num_heads: attention头数
            num_layers: Transformer层数
            dropout: dropout率
            norm_type: LayerNorm类型
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # 1. Patch Embedding层
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_dim=hidden_dim
        )

        # 2. 2D位置编码
        num_patches_per_side = image_size // patch_size
        self.pos_encoding = Image2DPositionEncoding(
            hidden_dim=hidden_dim,
            height=num_patches_per_side,
            width=num_patches_per_side,
            learnable=True  # 使用可学习的位置编码
        )

        # 3. Transformer块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                norm_type=norm_type
            )
            for _ in range(num_layers)
        ])

        # 4. 最终的LayerNorm
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像，shape [batch, channels, height, width]
            debug: 是否打印调试信息

        Returns:
            图像特征，shape [batch, num_patches, hidden_dim]
        """
        if debug:
            print("\n" + "="*60)
            print("Vision Encoder - 图像编码")
            print("="*60)

        # Step 1: Patch Embedding
        if debug:
            print("\n[1] Patch Embedding")
        x = self.patch_embed(x, debug=debug)

        # Step 2: 添加位置编码
        if debug:
            print("\n[2] 添加2D位置编码")
        x = self.pos_encoding(x)

        if debug:
            print_tensor_info("添加位置编码后", x, detailed=False)

        # Step 3: Transformer处理
        if debug:
            print(f"\n[3] Transformer处理 ({len(self.transformer_blocks)}层)")

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, debug=False)
            if debug:
                print(f"  第{i+1}层输出shape: {x.shape}")

        # Step 4: 最终LayerNorm
        if debug:
            print("\n[4] 最终LayerNorm")
        x = self.norm(x)

        if debug:
            print_tensor_info("Vision Encoder最终输出", x)

        return x


def test_vision_encoder():
    """
    测试Vision Encoder
    """
    print("\n" + "="*60)
    print("测试Vision Encoder")
    print("="*60)

    # 创建模型
    encoder = VisionEncoder(
        image_size=64,
        patch_size=16,
        in_channels=3,
        hidden_dim=128,
        num_heads=4,
        num_layers=1
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n总参数量: {total_params:,}")

    # 创建随机输入
    batch_size = 2
    image = torch.randn(batch_size, 3, 64, 64)

    # 前向传播
    print("\n执行前向传播...")
    output = encoder(image, debug=True)

    print("\n测试完成!")
    print(f"输入shape: {image.shape}")
    print(f"输出shape: {output.shape}")


if __name__ == "__main__":
    test_vision_encoder()

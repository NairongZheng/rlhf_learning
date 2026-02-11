"""
位置编码模块
包含RoPE（旋转位置编码）和2D位置编码
"""
import torch
import torch.nn as nn
import math


class RotaryPositionEncoding(nn.Module):
    """
    旋转位置编码（RoPE - Rotary Position Embedding）

    RoPE是一种相对位置编码方式，通过旋转变换将位置信息注入到query和key中。
    相比传统的绝对位置编码，RoPE能更好地捕捉相对位置关系，适合长序列。

    工作原理：
    1. 将hidden_dim维度分成多对（每对2维）
    2. 对每一对应用旋转矩阵，旋转角度随位置变化
    3. 旋转角度的频率随维度递减，类似Transformer的正弦位置编码
    """

    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        """
        初始化RoPE

        Args:
            dim: 每个attention head的维度
            max_seq_len: 支持的最大序列长度
            base: 频率的base值，控制不同维度的旋转速度
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # 计算每个维度对的旋转频率
        # inv_freq的shape: [dim/2]
        # 频率随维度递减: 1/(base^(2i/dim)), i=0,1,2,...,dim/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # 预计算位置编码（可选，用于加速）
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """
        预计算位置编码的sin和cos值

        Args:
            seq_len: 序列长度
        """
        # 位置索引: [seq_len]
        positions = torch.arange(seq_len).float()

        # 计算每个位置的旋转角度
        # freqs的shape: [seq_len, dim/2]
        # freqs[i,j] = i * inv_freq[j] = 位置i在维度对j的旋转角度
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq)

        # 将频率拼接成完整维度: [seq_len, dim]
        # 每个维度对(2i, 2i+1)使用相同的旋转角度
        emb = torch.cat([freqs, freqs], dim=-1)

        # 预计算sin和cos值，避免重复计算
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入进行旋转变换的辅助函数

        将x的维度分成两半，前半部分取负号并移到后半部分，后半部分移到前半部分
        这是实现旋转矩阵乘法的技巧

        例如: [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]

        Args:
            x: 输入tensor，shape: [batch, seq_len, dim]

        Returns:
            旋转后的tensor
        """
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """
        对query和key应用旋转位置编码

        旋转公式:
        q' = q * cos(θ) + rotate_half(q) * sin(θ)
        k' = k * cos(θ) + rotate_half(k) * sin(θ)

        其中θ是位置相关的旋转角度

        Args:
            q: query tensor，shape: [batch, seq_len, num_heads, head_dim]
            k: key tensor，shape: [batch, seq_len, num_heads, head_dim]

        Returns:
            应用RoPE后的(q, k)
        """
        seq_len = q.shape[1]

        # 如果序列长度超过缓存，重新构建缓存
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        # 获取当前序列长度的sin和cos值
        # shape: [seq_len, dim]
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        # 调整维度以匹配q和k的shape
        # 从 [seq_len, dim] -> [1, seq_len, 1, dim]
        # 这样可以广播到 [batch, seq_len, num_heads, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # 应用旋转变换
        q_rotated = q * cos + self._rotate_half(q) * sin
        k_rotated = k * cos + self._rotate_half(k) * sin

        return q_rotated, k_rotated

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """
        前向传播，应用RoPE

        Args:
            q: query tensor
            k: key tensor

        Returns:
            应用RoPE后的(q, k)
        """
        return self.apply_rope(q, k)


class Image2DPositionEncoding(nn.Module):
    """
    图像2D位置编码

    为图像的每个patch添加2D位置信息（x坐标和y坐标）
    可以使用可学习的embedding或固定的sinusoidal编码
    """

    def __init__(self, hidden_dim: int, height: int, width: int, learnable: bool = True):
        """
        初始化2D位置编码

        Args:
            hidden_dim: embedding维度
            height: 图像patch的行数
            width: 图像patch的列数
            learnable: 是否使用可学习的位置编码
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.height = height
        self.width = width
        self.learnable = learnable

        if learnable:
            # 可学习的位置编码
            # 为每个位置创建一个可学习的embedding
            self.pos_embedding = nn.Parameter(
                torch.randn(1, height * width, hidden_dim) * 0.02
            )
        else:
            # 固定的sinusoidal位置编码
            pos_enc = self._build_sinusoidal_encoding()
            self.register_buffer('pos_embedding', pos_enc)

    def _build_sinusoidal_encoding(self) -> torch.Tensor:
        """
        构建2D sinusoidal位置编码

        原理类似于Transformer的位置编码，但扩展到2D:
        - 前半部分维度编码x坐标
        - 后半部分维度编码y坐标

        Returns:
            位置编码tensor，shape: [1, height*width, hidden_dim]
        """
        # 创建位置网格
        y_pos = torch.arange(self.height).unsqueeze(1).repeat(1, self.width).flatten()
        x_pos = torch.arange(self.width).unsqueeze(0).repeat(self.height, 1).flatten()

        # 归一化到[0, 1]
        y_pos = y_pos.float() / self.height
        x_pos = x_pos.float() / self.width

        # 构建位置编码
        num_pos = self.height * self.width
        half_dim = self.hidden_dim // 2

        # 频率
        div_term = torch.exp(torch.arange(0, half_dim, 2).float() *
                            (-math.log(10000.0) / half_dim))

        pos_enc = torch.zeros(num_pos, self.hidden_dim)

        # 编码x坐标（前半部分维度）
        pos_enc[:, 0:half_dim:2] = torch.sin(x_pos.unsqueeze(1) * div_term)
        pos_enc[:, 1:half_dim:2] = torch.cos(x_pos.unsqueeze(1) * div_term)

        # 编码y坐标（后半部分维度）
        pos_enc[:, half_dim::2] = torch.sin(y_pos.unsqueeze(1) * div_term)
        pos_enc[:, half_dim+1::2] = torch.cos(y_pos.unsqueeze(1) * div_term)

        return pos_enc.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        为输入添加2D位置编码

        Args:
            x: 输入tensor，shape: [batch, num_patches, hidden_dim]

        Returns:
            添加位置编码后的tensor
        """
        # 直接加上位置编码
        # pos_embedding会自动广播到batch维度
        return x + self.pos_embedding

    def visualize_positions(self):
        """
        可视化位置编码（用于调试）
        """
        print(f"\n2D位置编码信息:")
        print(f"  图像尺寸: {self.height} x {self.width}")
        print(f"  总patch数: {self.height * self.width}")
        print(f"  Hidden维度: {self.hidden_dim}")
        print(f"  编码类型: {'可学习' if self.learnable else '固定sinusoidal'}")
        print(f"  位置编码shape: {tuple(self.pos_embedding.shape)}")

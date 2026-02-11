"""
文本编码器（Text Encoder）
将文本序列转换为特征表示
"""
import torch
import torch.nn as nn
from .position_encoding import RotaryPositionEncoding
from .attention import TransformerBlock, MultiHeadAttention, FeedForward
from core.utils.debug_utils import print_tensor_info


class TextEmbedding(nn.Module):
    """
    文本嵌入层

    将token ID转换为embedding向量
    """

    def __init__(self, vocab_size: int, hidden_dim: int):
        """
        初始化文本嵌入

        Args:
            vocab_size: 词汇表大小
            hidden_dim: embedding维度
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # 初始化embedding权重（使用较小的标准差）
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: token IDs，shape [batch, seq_len]

        Returns:
            embeddings，shape [batch, seq_len, hidden_dim]
        """
        return self.embedding(x)


class TransformerBlockWithRoPE(nn.Module):
    """
    带RoPE的Transformer块

    与标准TransformerBlock的区别：
    - 在attention计算前对Q和K应用RoPE
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 max_seq_len: int = 512, rope_base: int = 10000,
                 dropout: float = 0.1, norm_type: str = "pre"):
        """
        初始化

        Args:
            hidden_dim: 输入维度
            num_heads: 注意力头数
            max_seq_len: 最大序列长度
            rope_base: RoPE的base频率
            dropout: dropout率
            norm_type: 标准化类型
        """
        super().__init__()
        self.norm_type = norm_type
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # RoPE位置编码
        self.rope = RotaryPositionEncoding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base
        )

        # Self-Attention的投影层
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Feed-Forward Network
        self.ffn = FeedForward(hidden_dim, dropout=dropout)

        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将hidden_dim分割成多个头

        Args:
            x: shape [batch, seq_len, hidden_dim]

        Returns:
            shape [batch, seq_len, num_heads, head_dim]
        """
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim)

    def _attention_with_rope(self, query: torch.Tensor, key: torch.Tensor,
                            value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        带RoPE的attention计算

        Args:
            query: [batch, seq_len, hidden_dim]
            key: [batch, seq_len, hidden_dim]
            value: [batch, seq_len, hidden_dim]
            mask: 可选的mask

        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # 投影到Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # 分割成多头: [batch, seq_len, num_heads, head_dim]
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # 应用RoPE到Q和K
        Q, K = self.rope(Q, K)

        # 转置以便计算attention: [batch, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 计算attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        output = torch.matmul(attention_weights, V)

        # 重塑: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 输出投影
        output = self.out_proj(output)

        return output

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入，shape [batch, seq_len, hidden_dim]
            mask: 可选的attention mask

        Returns:
            输出，shape [batch, seq_len, hidden_dim]
        """
        if self.norm_type == "pre":
            # Pre-LayerNorm
            residual = x
            x = self.norm1(x)
            attn_out = self._attention_with_rope(x, x, x, mask)
            x = residual + self.dropout(attn_out)

            residual = x
            x = self.norm2(x)
            ffn_out = self.ffn(x)
            x = residual + self.dropout(ffn_out)
        else:
            # Post-LayerNorm
            residual = x
            attn_out = self._attention_with_rope(x, x, x, mask)
            x = self.norm1(residual + self.dropout(attn_out))

            residual = x
            ffn_out = self.ffn(x)
            x = self.norm2(residual + self.dropout(ffn_out))

        return x


class TextEncoder(nn.Module):
    """
    文本编码器

    完整流程：
    1. Text Embedding: 将token IDs转换为embeddings
    2. RoPE: 通过旋转位置编码注入位置信息（在Transformer块内部）
    3. Transformer: 提取文本特征
    4. LayerNorm: 最终的标准化
    """

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 1,
                 max_seq_len: int = 32, rope_base: int = 10000,
                 dropout: float = 0.1, norm_type: str = "pre"):
        """
        初始化文本编码器

        Args:
            vocab_size: 词汇表大小
            hidden_dim: embedding维度
            num_heads: attention头数
            num_layers: Transformer层数
            max_seq_len: 最大序列长度
            rope_base: RoPE的base频率
            dropout: dropout率
            norm_type: LayerNorm类型
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # 1. Token Embedding
        self.token_embed = TextEmbedding(vocab_size, hidden_dim)

        # 2. Transformer块（带RoPE）
        self.transformer_blocks = nn.ModuleList([
            TransformerBlockWithRoPE(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                dropout=dropout,
                norm_type=norm_type
            )
            for _ in range(num_layers)
        ])

        # 3. 最终的LayerNorm
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                debug: bool = False) -> torch.Tensor:
        """
        前向传播

        Args:
            x: token IDs，shape [batch, seq_len]
            mask: 可选的attention mask
            debug: 是否打印调试信息

        Returns:
            文本特征，shape [batch, seq_len, hidden_dim]
        """
        if debug:
            print("\n" + "="*60)
            print("Text Encoder - 文本编码")
            print("="*60)

        # Step 1: Token Embedding
        if debug:
            print("\n[1] Token Embedding")
            print(f"  输入token IDs shape: {x.shape}")

        x = self.token_embed(x)

        if debug:
            print_tensor_info("Token Embeddings", x, detailed=False)

        # Step 2: Transformer处理（包含RoPE）
        if debug:
            print(f"\n[2] Transformer处理 ({len(self.transformer_blocks)}层，带RoPE)")

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, mask)
            if debug:
                print(f"  第{i+1}层输出shape: {x.shape}")

        # Step 3: 最终LayerNorm
        if debug:
            print("\n[3] 最终LayerNorm")
        x = self.norm(x)

        if debug:
            print_tensor_info("Text Encoder最终输出", x)

        return x


def test_text_encoder():
    """
    测试Text Encoder
    """
    print("\n" + "="*60)
    print("测试Text Encoder")
    print("="*60)

    # 创建模型
    encoder = TextEncoder(
        vocab_size=1000,
        hidden_dim=128,
        num_heads=4,
        num_layers=1,
        max_seq_len=32
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n总参数量: {total_params:,}")

    # 创建随机输入
    batch_size = 2
    seq_len = 16
    tokens = torch.randint(0, 1000, (batch_size, seq_len))

    # 前向传播
    print("\n执行前向传播...")
    output = encoder(tokens, debug=True)

    print("\n测试完成!")
    print(f"输入shape: {tokens.shape}")
    print(f"输出shape: {output.shape}")


if __name__ == "__main__":
    test_text_encoder()

"""
文本解码器（Text Decoder）
基于融合特征生成文本输出
"""
import torch
import torch.nn as nn
from .position_encoding import RotaryPositionEncoding
from .attention import FeedForward
from core.utils.debug_utils import print_tensor_info


class DecoderBlockWithRoPE(nn.Module):
    """
    带RoPE的解码器块

    包含：
    1. Self-Attention (带RoPE)
    2. Feed-Forward Network
    3. LayerNorm + Residual Connection
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 max_seq_len: int = 512, rope_base: int = 10000,
                 dropout: float = 0.1, norm_type: str = "pre"):
        """
        初始化解码器块

        Args:
            hidden_dim: 输入维度
            num_heads: attention头数
            max_seq_len: 最大序列长度
            rope_base: RoPE的base频率
            dropout: dropout率
            norm_type: LayerNorm类型
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
        """分割成多头"""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim)

    def _attention_with_rope(self, query: torch.Tensor, key: torch.Tensor,
                            value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        带RoPE的attention计算

        Args:
            query, key, value: [batch, seq_len, hidden_dim]
            mask: 可选的causal mask

        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # 投影
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # 分割成多头
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # 应用RoPE
        Q, K = self.rope(Q, K)

        # 转置: [batch, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 计算attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        output = torch.matmul(attention_weights, V)

        # 重塑
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 输出投影
        output = self.out_proj(output)

        return output

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入，shape [batch, seq_len, hidden_dim]
            mask: 可选的causal mask

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


class TextDecoder(nn.Module):
    """
    文本解码器

    完整流程：
    1. Decoder Transformer: 处理融合后的特征（带RoPE）
    2. LayerNorm: 最终标准化
    3. Language Model Head: 投影到词汇表，生成logits
    """

    def __init__(self, hidden_dim: int = 128, vocab_size: int = 1000,
                 num_heads: int = 4, num_layers: int = 1,
                 max_seq_len: int = 32, rope_base: int = 10000,
                 dropout: float = 0.1, norm_type: str = "pre"):
        """
        初始化文本解码器

        Args:
            hidden_dim: embedding维度
            vocab_size: 词汇表大小
            num_heads: attention头数
            num_layers: Transformer层数
            max_seq_len: 最大序列长度
            rope_base: RoPE的base频率
            dropout: dropout率
            norm_type: LayerNorm类型
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # 1. Decoder Transformer块（带RoPE）
        self.decoder_blocks = nn.ModuleList([
            DecoderBlockWithRoPE(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                dropout=dropout,
                norm_type=norm_type
            )
            for _ in range(num_layers)
        ])

        # 2. 最终LayerNorm
        self.norm = nn.LayerNorm(hidden_dim)

        # 3. Language Model Head：投影到词汇表
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None,
                debug: bool = False) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 融合后的特征，shape [batch, seq_len, hidden_dim]
            causal_mask: 可选的causal mask（用于自回归生成）
            debug: 是否打印调试信息

        Returns:
            logits，shape [batch, seq_len, vocab_size]
        """
        if debug:
            print("\n" + "="*60)
            print("Text Decoder - 文本解码")
            print("="*60)
            print_tensor_info("输入特征", x, detailed=False)

        # Step 1: Decoder Transformer处理
        if debug:
            print(f"\n[1] Decoder Transformer ({len(self.decoder_blocks)}层，带RoPE)")

        for i, block in enumerate(self.decoder_blocks):
            x = block(x, causal_mask)
            if debug:
                print(f"  第{i+1}层输出shape: {x.shape}")

        # Step 2: 最终LayerNorm
        if debug:
            print("\n[2] 最终LayerNorm")
        x = self.norm(x)

        # Step 3: LM Head投影到词汇表
        if debug:
            print("\n[3] Language Model Head (投影到词汇表)")
        logits = self.lm_head(x)

        if debug:
            print_tensor_info("输出Logits", logits)

        return logits

    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        生成causal mask用于自回归生成

        Causal mask确保位置i只能看到位置<=i的信息，不能看到未来的token

        Args:
            seq_len: 序列长度
            device: 设备

        Returns:
            mask，shape [1, 1, seq_len, seq_len]
        """
        # 创建下三角矩阵
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

        # 添加batch和head维度
        mask = mask.unsqueeze(0).unsqueeze(0)

        return mask


def test_text_decoder():
    """
    测试Text Decoder
    """
    print("\n" + "="*60)
    print("测试Text Decoder")
    print("="*60)

    # 参数
    batch_size = 2
    seq_len = 16
    hidden_dim = 128
    vocab_size = 1000
    num_heads = 4

    # 创建模型
    decoder = TextDecoder(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_heads=num_heads,
        num_layers=1,
        max_seq_len=32
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n总参数量: {total_params:,}")

    # 创建随机输入（融合后的特征）
    fused_features = torch.randn(batch_size, seq_len, hidden_dim)

    # 创建causal mask
    causal_mask = decoder.generate_causal_mask(seq_len, fused_features.device)
    print(f"\nCausal mask shape: {causal_mask.shape}")

    # 前向传播
    print("\n执行前向传播...")
    logits = decoder(fused_features, causal_mask, debug=True)

    print("\n测试完成!")
    print(f"输入特征shape: {fused_features.shape}")
    print(f"输出logits shape: {logits.shape}")

    # 生成预测
    predicted_tokens = torch.argmax(logits, dim=-1)
    print(f"预测token shape: {predicted_tokens.shape}")
    print(f"前3个token预测: {predicted_tokens[0, :3].tolist()}")


if __name__ == "__main__":
    test_text_decoder()

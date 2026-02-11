"""
Attention机制模块
包含Multi-Head Attention和Cross Attention
"""
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制（Multi-Head Attention）

    核心思想：
    1. 将输入投影到多个子空间（多头）
    2. 在每个子空间独立计算attention
    3. 将所有头的输出拼接起来

    优势：
    - 多个头可以关注不同的特征模式
    - 增加模型的表达能力
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        """
        初始化多头注意力

        Args:
            hidden_dim: 输入的维度
            num_heads: 注意力头的数量
            dropout: dropout率
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # 每个头的维度
        self.scale = math.sqrt(self.head_dim)    # 缩放因子，防止softmax饱和

        # Query、Key、Value的线性投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None, return_attention: bool = False) -> tuple:
        """
        多头注意力的前向传播

        Attention计算公式：
        Attention(Q,K,V) = softmax(QK^T / √d_k) V

        Args:
            query: shape [batch, seq_len_q, hidden_dim]
            key: shape [batch, seq_len_k, hidden_dim]
            value: shape [batch, seq_len_k, hidden_dim]
            mask: 可选的mask，shape [batch, seq_len_q, seq_len_k]
            return_attention: 是否返回attention权重

        Returns:
            output: shape [batch, seq_len_q, hidden_dim]
            attention_weights: 如果return_attention=True，返回attention权重
        """
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]

        # 1. 线性投影: [batch, seq_len, hidden_dim] -> [batch, seq_len, hidden_dim]
        Q = self.q_proj(query)      # Q: [2, 16, 128]
        K = self.k_proj(key)        # K: [2, 16, 128]
        V = self.v_proj(value)      # V: [2, 16, 128]

        # 2. 重塑为多头形式: [batch, seq_len, hidden_dim]
        #    -> [batch, seq_len, num_heads, head_dim]
        #    -> [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)    # Q: [2, 4, 16, 32]
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)    # K: [2, 4, 16, 32]
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)    # V: [2, 4, 16, 32]

        # 3. 计算attention分数: QK^T / √d_k
        # [batch, num_heads, seq_len_q, head_dim] x [batch, num_heads, head_dim, seq_len_k]
        # -> [batch, num_heads, seq_len_q, seq_len_k]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale    # attention_scores: [2, 4, 16, 16]

        # 4. 应用mask（如果有）
        if mask is not None:
            # mask中为True的位置会被设置为一个很大的负数，softmax后接近0
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # 5. Softmax归一化
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 6. 应用dropout
        attention_weights = self.dropout(attention_weights)

        # 7. 加权求和: attention_weights x V
        # [batch, num_heads, seq_len_q, seq_len_k] x [batch, num_heads, seq_len_k, head_dim]
        # -> [batch, num_heads, seq_len_q, head_dim]
        output = torch.matmul(attention_weights, V)

        # 8. 重塑回原始维度
        # [batch, num_heads, seq_len_q, head_dim]
        # -> [batch, seq_len_q, num_heads, head_dim]
        # -> [batch, seq_len_q, hidden_dim]
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.hidden_dim
        )

        # 9. 输出投影
        output = self.out_proj(output)

        if return_attention:
            return output, attention_weights
        return output


class CrossAttention(nn.Module):
    """
    跨注意力机制（Cross Attention）

    用于融合两种不同模态的信息：
    - Query来自一个模态（如文本）
    - Key和Value来自另一个模态（如图像）

    这样文本可以"查询"图像中相关的视觉信息
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        """
        初始化跨注意力

        Args:
            hidden_dim: 输入的维度
            num_heads: 注意力头的数量
            dropout: dropout率
        """
        super().__init__()

        # 使用MultiHeadAttention作为基础
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor,
                mask: torch.Tensor = None, return_attention: bool = False) -> tuple:
        """
        跨注意力的前向传播

        Args:
            query: query模态的输入，shape [batch, seq_len_q, hidden_dim]
            context: context模态的输入，shape [batch, seq_len_c, hidden_dim]
            mask: 可选的mask
            return_attention: 是否返回attention权重

        Returns:
            output: 融合后的特征，shape [batch, seq_len_q, hidden_dim]
        """
        # Cross attention: query从第一个模态，key和value从第二个模态
        return self.attention(query, context, context, mask, return_attention)


class FeedForward(nn.Module):
    """
    前馈神经网络（Feed-Forward Network）

    Transformer中的FFN通常采用两层线性变换加激活函数：
    FFN(x) = Linear2(GELU(Linear1(x)))

    中间层维度通常是输入维度的4倍
    """

    def __init__(self, hidden_dim: int, ffn_dim: int = None, dropout: float = 0.1):
        """
        初始化FFN

        Args:
            hidden_dim: 输入输出维度
            ffn_dim: 中间层维度，默认为4*hidden_dim
            dropout: dropout率
        """
        super().__init__()

        if ffn_dim is None:
            ffn_dim = hidden_dim * 4

        # 两层线性变换
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)

        # GELU激活函数（比ReLU更平滑）
        self.activation = nn.GELU()

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入tensor，shape [batch, seq_len, hidden_dim]

        Returns:
            输出tensor，shape [batch, seq_len, hidden_dim]
        """
        # Linear1 -> GELU -> Dropout -> Linear2 -> Dropout
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer块

    包含：
    1. Multi-Head Self-Attention
    2. Feed-Forward Network
    3. LayerNorm和Residual Connection

    支持Pre-LayerNorm和Post-LayerNorm两种结构
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 dropout: float = 0.1, norm_type: str = "pre"):
        """
        初始化Transformer块

        Args:
            hidden_dim: 输入维度
            num_heads: 注意力头数
            dropout: dropout率
            norm_type: 标准化类型，"pre"或"post"
        """
        super().__init__()
        self.norm_type = norm_type

        # Self-Attention
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)

        # Feed-Forward Network
        self.ffn = FeedForward(hidden_dim, dropout=dropout)

        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                debug: bool = False) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入tensor，shape [batch, seq_len, hidden_dim]
            mask: 可选的attention mask
            debug: 是否打印调试信息

        Returns:
            输出tensor
        """
        if self.norm_type == "pre":
            # Pre-LayerNorm: 先norm再计算
            # x = x + Attention(LayerNorm(x))
            residual = x
            x = self.norm1(x)
            attn_out = self.attention(x, x, x, mask)
            x = residual + self.dropout(attn_out)

            # x = x + FFN(LayerNorm(x))
            residual = x
            x = self.norm2(x)
            ffn_out = self.ffn(x)
            x = residual + self.dropout(ffn_out)

        else:  # post
            # Post-LayerNorm: 先计算再norm
            # x = LayerNorm(x + Attention(x))
            residual = x
            attn_out = self.attention(x, x, x, mask)
            x = self.norm1(residual + self.dropout(attn_out))

            # x = LayerNorm(x + FFN(x))
            residual = x
            ffn_out = self.ffn(x)
            x = self.norm2(residual + self.dropout(ffn_out))

        return x

"""
模型配置文件
定义所有超参数，方便调整和实验
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """多模态模型配置"""

    # ============ 模型结构参数 ============
    hidden_dim: int = 128          # embedding维度
    num_heads: int = 4             # attention头数
    num_layers: int = 2            # 每个组件的Transformer层数
    vocab_size: int = 151657       # 词汇表大小（匹配Qwen2-VL tokenizer）

    # ============ 图像相关参数 ============
    image_size: int = 64           # 输入图像大小（正方形）
    patch_size: int = 16           # 图像patch大小
    in_channels: int = 3           # 图像通道数（RGB）

    # 计算图像patch数量
    @property
    def num_patches(self) -> int:
        """计算图像被分成多少个patch"""
        return (self.image_size // self.patch_size) ** 2

    # ============ 文本相关参数 ============
    max_seq_len: int = 32          # 最大文本序列长度

    # ============ Tokenizer相关参数 ============
    # 注意：这些值由Qwen tokenizer自动提供，这里仅作记录
    # 实际使用时从tokenizer.pad_token_id等获取
    pad_token_id: int = None       # 由tokenizer提供
    eos_token_id: int = None       # 由tokenizer提供

    # ============ 位置编码参数 ============
    use_rope: bool = True          # 是否使用RoPE（旋转位置编码）
    rope_base: int = 10000         # RoPE的base频率

    # ============ 标准化参数 ============
    norm_type: str = "pre"         # LayerNorm类型："pre" or "post"
    layer_norm_eps: float = 1e-6   # LayerNorm的epsilon值

    # ============ 正则化参数 ============
    dropout: float = 0.1           # dropout率

    # ============ 训练参数 ============
    learning_rate: float = 1e-4    # 学习率
    weight_decay: float = 0.01     # 权重衰减
    batch_size: int = 2            # 批次大小
    num_epochs: int = 10           # 训练轮数

    # ============ Debug参数 ============
    debug: bool = True             # 是否打印详细调试信息
    print_model_info: bool = True  # 是否打印模型结构信息

    def __post_init__(self):
        """初始化后的检查"""
        # 确保hidden_dim能被num_heads整除
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) 必须能被 num_heads ({self.num_heads}) 整除"

        # 确保图像大小能被patch大小整除
        assert self.image_size % self.patch_size == 0, \
            f"image_size ({self.image_size}) 必须能被 patch_size ({self.patch_size}) 整除"

    def print_config(self):
        """打印配置信息"""
        print("=" * 50)
        print("模型配置信息")
        print("=" * 50)
        print(f"Hidden维度: {self.hidden_dim}")
        print(f"Attention头数: {self.num_heads}")
        print(f"每头维度: {self.hidden_dim // self.num_heads}")
        print(f"Transformer层数: {self.num_layers}")
        print(f"词汇表大小: {self.vocab_size}")
        print(f"图像大小: {self.image_size}x{self.image_size}")
        print(f"Patch大小: {self.patch_size}x{self.patch_size}")
        print(f"Patch数量: {self.num_patches}")
        print(f"最大序列长度: {self.max_seq_len}")
        print(f"使用RoPE: {self.use_rope}")
        print(f"LayerNorm类型: {self.norm_type}")
        print(f"Dropout率: {self.dropout}")
        print(f"学习率: {self.learning_rate}")
        print(f"批次大小: {self.batch_size}")
        print("=" * 50)

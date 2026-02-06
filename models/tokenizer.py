"""
Tokenizer模块
封装Qwen2-VL的tokenizer，提供统一接口和样本文本生成
"""

from transformers import AutoTokenizer
from typing import List, Union, Dict
import torch


class QwenTokenizerWrapper:
    """
    Qwen2-VL Tokenizer封装类

    功能：
    1. 加载Qwen2-VL的预训练tokenizer
    2. 提供统一的编码/解码接口
    3. 生成多样化的中文样本文本（用于训练演示）
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        """
        初始化tokenizer

        Args:
            model_name: Hugging Face模型名称
        """
        print(f"正在加载tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print(f"✓ Tokenizer加载完成")
        print(f"  词汇表大小: {len(self.tokenizer)}")
        print(f"  PAD token ID: {self.tokenizer.pad_token_id}")
        print(f"  EOS token ID: {self.tokenizer.eos_token_id}")

        # 样本文本模板
        self.sample_texts = [
            # 图像描述类
            "一只可爱的小猫坐在窗台上晒太阳。",
            "蓝天白云，青山绿水，风景优美。",
            "这是一张城市夜景照片，灯光璀璨。",
            "画面中有三个人在公园里愉快地交谈。",
            "桌子上摆放着一杯咖啡和一本书。",

            # 问答类
            "图片里有什么动物？",
            "这个场景是在哪里拍摄的？",
            "画面中的主要物体是什么颜色？",

            # 简短描述
            "红色的花朵盛开着。",
            "快乐的小狗在草地上奔跑。",
            "美丽的日落景色。",

            # 混合中英文
            "这是一个multimodal model的训练样例。",
            "使用Python进行深度学习开发。",
            "The cat is sleeping on the sofa.",
        ]

    def encode(self, text: str, max_length: int = 32,
               padding: bool = True, truncation: bool = True,
               return_tensors: str = "pt") -> Dict:
        """
        编码文本为token IDs

        Args:
            text: 输入文本
            max_length: 最大序列长度
            padding: 是否padding
            truncation: 是否截断
            return_tensors: 返回格式 ("pt" for PyTorch)

        Returns:
            包含input_ids和attention_mask的字典
        """
        return self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length' if padding else False,
            truncation=truncation,
            return_tensors=return_tensors
        )

    def batch_encode(self, texts: List[str], **kwargs) -> Dict:
        """
        批量编码文本

        Args:
            texts: 文本列表
            **kwargs: 传递给encode的其他参数

        Returns:
            包含input_ids和attention_mask的字典
        """
        # 对于列表输入，tokenizer会自动进行批处理
        return self.tokenizer(
            texts,
            max_length=kwargs.get('max_length', 32),
            padding='max_length' if kwargs.get('padding', True) else False,
            truncation=kwargs.get('truncation', True),
            return_tensors=kwargs.get('return_tensors', 'pt')
        )

    def decode(self, token_ids: Union[List[int], torch.Tensor],
               skip_special_tokens: bool = True) -> str:
        """
        解码token IDs为文本

        Args:
            token_ids: token ID列表或tensor
            skip_special_tokens: 是否跳过特殊token

        Returns:
            解码后的文本字符串
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, batch_token_ids: torch.Tensor,
                     skip_special_tokens: bool = True) -> List[str]:
        """
        批量解码

        Args:
            batch_token_ids: batch的token ID tensor
            skip_special_tokens: 是否跳过特殊token

        Returns:
            解码后的文本列表
        """
        return self.tokenizer.batch_decode(batch_token_ids, skip_special_tokens=skip_special_tokens)

    def get_sample_texts(self, num_samples: int = 1) -> Union[str, List[str]]:
        """
        获取随机样本文本（用于训练演示）

        Args:
            num_samples: 需要的样本数量

        Returns:
            单个样本文本或文本列表
        """
        import random
        samples = random.choices(self.sample_texts, k=num_samples)
        return samples[0] if num_samples == 1 else samples

    @property
    def vocab_size(self) -> int:
        """获取词汇表大小"""
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        """获取padding token ID"""
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        """获取EOS token ID"""
        return self.tokenizer.eos_token_id


def test_tokenizer():
    """测试Qwen tokenizer封装"""
    print("="*80)
    print("测试Qwen Tokenizer封装")
    print("="*80)

    # 初始化
    tokenizer = QwenTokenizerWrapper()
    print(f"\n✓ Tokenizer初始化成功")
    print(f"  词汇表大小: {tokenizer.vocab_size}")

    # 测试编码
    text = "这是一个测试文本。"
    encoded = tokenizer.encode(text, max_length=20, padding=True)
    print(f"\n✓ 编码测试通过")
    print(f"  输入: {text}")
    print(f"  Token IDs: {encoded['input_ids'][0][:10].tolist()}...")

    # 测试解码
    decoded = tokenizer.decode(encoded['input_ids'][0])
    print(f"\n✓ 解码测试通过")
    print(f"  解码结果: {decoded}")

    # 测试批量处理
    texts = tokenizer.get_sample_texts(3)
    batch_encoded = tokenizer.batch_encode(texts, max_length=20, padding=True)
    print(f"\n✓ 批量处理测试通过")
    print(f"  Batch shape: {batch_encoded['input_ids'].shape}")

    # 测试样本生成
    samples = tokenizer.get_sample_texts(5)
    print(f"\n✓ 样本生成测试通过")
    for i, sample in enumerate(samples, 1):
        print(f"  样本{i}: {sample}")

    print("\n" + "="*80)
    print("所有测试通过！")
    print("="*80)


if __name__ == "__main__":
    test_tokenizer()

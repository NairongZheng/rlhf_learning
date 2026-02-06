"""
调试工具函数
提供各种用于debug和可视化的辅助函数
"""
import torch
import numpy as np


def print_tensor_info(name: str, tensor: torch.Tensor, detailed: bool = True):
    """
    打印tensor的详细信息

    Args:
        name: tensor的名称
        tensor: 要检查的tensor
        detailed: 是否打印详细统计信息
    """
    print(f"\n{'='*60}")
    print(f"Tensor: {name}")
    print(f"{'='*60}")
    print(f"Shape: {tuple(tensor.shape)}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Device: {tensor.device}")

    if detailed:
        # 统计信息（转换为float避免整数tensor的问题）
        tensor_float = tensor.float()
        print(f"均值 (Mean): {tensor_float.mean().item():.6f}")
        print(f"标准差 (Std): {tensor_float.std().item():.6f}")
        print(f"最小值 (Min): {tensor_float.min().item():.6f}")
        print(f"最大值 (Max): {tensor_float.max().item():.6f}")

        # 检查异常值
        has_nan = torch.isnan(tensor_float).any()
        has_inf = torch.isinf(tensor_float).any()
        if has_nan:
            print(f"⚠️  警告: 包含 NaN 值!")
        if has_inf:
            print(f"⚠️  警告: 包含 Inf 值!")

    print(f"{'='*60}\n")


def check_nan_inf(tensor: torch.Tensor, name: str = "Tensor") -> bool:
    """
    检查tensor中是否包含NaN或Inf值

    Args:
        tensor: 要检查的tensor
        name: tensor的名称

    Returns:
        如果包含NaN或Inf返回True，否则返回False
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        print(f"\n❌ {name} 包含异常值:")
        if has_nan:
            print(f"   - NaN数量: {torch.isnan(tensor).sum().item()}")
        if has_inf:
            print(f"   - Inf数量: {torch.isinf(tensor).sum().item()}")
        return True

    return False


def print_gradient_info(model: torch.nn.Module, detailed: bool = False):
    """
    打印模型所有参数的梯度信息

    Args:
        model: 要检查的模型
        detailed: 是否打印每个参数的详细信息
    """
    print(f"\n{'='*60}")
    print("梯度信息汇总")
    print(f"{'='*60}")

    total_norm = 0.0
    num_params_with_grad = 0
    num_params_without_grad = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            # 计算梯度的统计信息
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()

            total_norm += grad_norm ** 2
            num_params_with_grad += 1

            if detailed:
                print(f"\n{name}:")
                print(f"  梯度均值: {grad_mean:.6f}")
                print(f"  梯度标准差: {grad_std:.6f}")
                print(f"  梯度范数: {grad_norm:.6f}")

                # 检查异常梯度
                if check_nan_inf(param.grad, f"{name}.grad"):
                    continue
        else:
            num_params_without_grad += 1
            if detailed:
                print(f"\n{name}: 无梯度")

    # 计算总梯度范数
    total_norm = total_norm ** 0.5

    print(f"\n总体统计:")
    print(f"  有梯度的参数数量: {num_params_with_grad}")
    print(f"  无梯度的参数数量: {num_params_without_grad}")
    print(f"  总梯度范数: {total_norm:.6f}")
    print(f"{'='*60}\n")

    return total_norm


def check_gradient_flow(model: torch.nn.Module, threshold: float = 1e-7):
    """
    检查梯度消失或梯度爆炸问题

    Args:
        model: 要检查的模型
        threshold: 梯度消失的阈值
    """
    print(f"\n{'='*60}")
    print("梯度流检查")
    print(f"{'='*60}")

    vanishing_grads = []
    exploding_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()

            # 检查梯度消失
            if grad_norm < threshold:
                vanishing_grads.append((name, grad_norm))

            # 检查梯度爆炸（梯度范数 > 10认为是异常大）
            if grad_norm > 10.0:
                exploding_grads.append((name, grad_norm))

    # 报告梯度消失
    if vanishing_grads:
        print(f"\n⚠️  检测到 {len(vanishing_grads)} 个梯度消失的参数:")
        for name, norm in vanishing_grads[:5]:  # 只显示前5个
            print(f"   {name}: {norm:.2e}")
        if len(vanishing_grads) > 5:
            print(f"   ... 还有 {len(vanishing_grads) - 5} 个")
    else:
        print(f"\n✓ 未检测到梯度消失")

    # 报告梯度爆炸
    if exploding_grads:
        print(f"\n⚠️  检测到 {len(exploding_grads)} 个梯度爆炸的参数:")
        for name, norm in exploding_grads[:5]:
            print(f"   {name}: {norm:.2e}")
        if len(exploding_grads) > 5:
            print(f"   ... 还有 {len(exploding_grads) - 5} 个")
    else:
        print(f"\n✓ 未检测到梯度爆炸")

    print(f"{'='*60}\n")


def visualize_attention(attention_weights: torch.Tensor,
                       name: str = "Attention",
                       show_top_k: int = 5):
    """
    可视化attention权重（文本形式）

    Args:
        attention_weights: attention权重，shape: [batch, num_heads, seq_len, seq_len]
        name: 名称
        show_top_k: 显示每个query的top-k个attention值
    """
    print(f"\n{'='*60}")
    print(f"Attention可视化: {name}")
    print(f"{'='*60}")
    print(f"Shape: {tuple(attention_weights.shape)}")

    # 取第一个batch的第一个head
    attn = attention_weights[0, 0].detach().cpu().numpy()
    seq_len = attn.shape[0]

    print(f"\n第一个样本，第一个head的attention权重:")
    print(f"(每行代表一个query token对所有key tokens的attention)")

    # 显示前几个query的attention分布
    for i in range(min(3, seq_len)):
        # 找到top-k的attention值
        top_indices = np.argsort(attn[i])[-show_top_k:][::-1]
        print(f"\nQuery {i} 的 top-{show_top_k} attention:")
        for j, idx in enumerate(top_indices):
            print(f"  #{j+1}: Key {idx} = {attn[i, idx]:.4f}")

    # 统计信息
    print(f"\n统计信息:")
    print(f"  均值: {attn.mean():.4f}")
    print(f"  标准差: {attn.std():.4f}")
    print(f"  最大值: {attn.max():.4f}")
    print(f"  最小值: {attn.min():.4f}")
    print(f"{'='*60}\n")


def plot_loss_curve(losses: list, window_size: int = 5):
    """
    绘制损失曲线（文本形式）

    Args:
        losses: 损失值列表
        window_size: 移动平均窗口大小
    """
    print(f"\n{'='*60}")
    print("损失曲线")
    print(f"{'='*60}")

    if len(losses) == 0:
        print("没有损失数据")
        return

    # 计算移动平均
    if len(losses) >= window_size:
        smoothed = []
        for i in range(len(losses) - window_size + 1):
            smoothed.append(np.mean(losses[i:i+window_size]))
    else:
        smoothed = losses

    # 打印数值
    print(f"\n最近 {min(10, len(losses))} 个损失值:")
    for i, loss in enumerate(losses[-10:]):
        step = len(losses) - 10 + i
        print(f"  Step {step}: {loss:.4f}")

    # 简单的ASCII图表
    if len(smoothed) > 0:
        print(f"\n损失趋势 (移动平均, 窗口={window_size}):")
        min_loss = min(smoothed)
        max_loss = max(smoothed)
        range_loss = max_loss - min_loss if max_loss > min_loss else 1.0

        # 归一化到0-20的范围用于显示
        normalized = [(loss - min_loss) / range_loss * 20 for loss in smoothed]

        # 每隔一定间隔显示一个点
        step = max(1, len(normalized) // 20)
        for i in range(0, len(normalized), step):
            bar_len = int(normalized[i])
            bar = '█' * bar_len
            print(f"  Step {i:3d}: {bar} {smoothed[i]:.4f}")

    print(f"{'='*60}\n")


def count_parameters(model: torch.nn.Module) -> dict:
    """
    统计模型参数数量

    Args:
        model: 要统计的模型

    Returns:
        包含参数统计信息的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print("模型参数统计")
    print(f"{'='*60}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"不可训练参数量: {total_params - trainable_params:,}")
    print(f"{'='*60}\n")

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

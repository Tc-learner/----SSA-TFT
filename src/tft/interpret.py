"""TFT 可解释性 —— 变量重要性 & 注意力权重分析"""

import numpy as np
import tensorflow as tf
from typing import Optional, List


def get_variable_selection_weights(model: tf.keras.Model,
                                   X_sample: np.ndarray) -> np.ndarray:
    """从 TFT 模型中提取变量选择权重。

    通过构建一个子模型输出 VSN 层的权重来实现。

    Args:
        model: 已训练的 TFT 模型
        X_sample: 单个样本 (1, n_steps, n_features) 用于确定形状

    Returns:
        (n_steps, n_features) 的时间步-变量重要性矩阵
    """
    # 查找 VSN 层
    vsn_layer = None
    for layer in model.layers:
        if 'time_distributed_vsn' in layer.name.lower() or 'vsn' in layer.name.lower():
            vsn_layer = layer
            break

    if vsn_layer is None:
        # 尝试通过子模型提取：找到 VSN 之前的层并构建中间模型
        raise ValueError("未在模型中找到 VariableSelectionNetwork 层。"
                         "请确保使用 build_tft() 构建的完整模型。")

    # 创建中间模型输出 VSN 权重
    vsn_output_model = tf.keras.Model(
        inputs=model.input,
        outputs=vsn_layer.output
    )
    # VSN 输出是 (selected, weights)，取 weights
    _, weights = vsn_output_model(X_sample)
    return weights.numpy().squeeze()  # (n_steps, n_features)


def get_attention_weights(model: tf.keras.Model,
                          X_sample: np.ndarray) -> np.ndarray:
    """从 TFT 模型中提取注意力权重。

    Args:
        model: 已训练的 TFT 模型
        X_sample: 单个样本 (1, n_steps, n_features)

    Returns:
        (num_heads, n_steps, n_steps) 的注意力权重矩阵
    """
    attn_layer = None
    for layer in model.layers:
        if 'causal_temporal_self_attention' in layer.name.lower() or \
           'self_attn' in layer.name.lower() or \
           'attention' in layer.name.lower():
            attn_layer = layer
            break

    if attn_layer is None:
        raise ValueError("未在模型中找到 Attention 层。")

    # 通过 attention 层获取权重
    attn_output = attn_layer(X_sample, return_attention=True)
    if isinstance(attn_output, tuple):
        _, attn_weights = attn_output
        return attn_weights.numpy().squeeze()  # (num_heads, steps, steps)

    raise ValueError("Attention 层未配置 return_attention=True。"
                     "模型需要重建以启用可解释性。")


def compute_variable_importance(vsn_weights: np.ndarray,
                                feature_names: Optional[List[str]] = None
                                ) -> dict:
    """从 VSN 权重计算变量重要性统计。

    Args:
        vsn_weights: (n_steps, n_features) 的 VSN 权重
        feature_names: 可选的变量名列表

    Returns:
        {'mean': (n_features,), 'std': (n_features,), 'names': [...]}
    """
    n_features = vsn_weights.shape[1]
    mean_imp = vsn_weights.mean(axis=0)
    std_imp = vsn_weights.std(axis=0)

    if feature_names is None:
        feature_names = [f'Var_{i + 1}' for i in range(n_features)]

    return {
        'mean': mean_imp,
        'std': std_imp,
        'names': feature_names,
        'sorted_indices': np.argsort(mean_imp)[::-1],  # 降序
    }


def compute_attention_importance(attn_weights: np.ndarray) -> dict:
    """从注意力权重计算时序重要性。

    Args:
        attn_weights: (num_heads, n_steps, n_steps) 注意力矩阵

    Returns:
        {'mean_per_step': (n_steps,), 'std_per_step': (n_steps,),
         'mean_per_head': (num_heads,)}
    """
    # 按时间步汇总（每个时间步被"关注"的总量）
    attention_received = attn_weights.sum(axis=-2)  # (heads, steps)

    mean_per_step = attention_received.mean(axis=0)  # (steps,)
    std_per_step = attention_received.std(axis=0)
    mean_per_head = attention_received.mean(axis=1)  # (heads,)

    return {
        'mean_per_step': mean_per_step,
        'std_per_step': std_per_step,
        'mean_per_head': mean_per_head,
    }

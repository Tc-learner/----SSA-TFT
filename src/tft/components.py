"""TFT 顶层组装 —— build_tft()"""

from typing import Tuple, Optional, Dict, List

import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, LayerNormalization, Add, Input, Layer, Lambda,
    Reshape,
)
from tensorflow.keras.models import Model

from .gated import GatedResidualNetwork
from .vsn import TimeDistributedVSN
from .static_encoder import StaticCovariateEncoder
from .attention import CausalTemporalSelfAttention


def quantile_loss_fn(quantiles: List[float]):
    """构建分位数损失函数。

    Args:
        quantiles: 分位数列表，如 [0.1, 0.5, 0.9]
    Returns:
        loss_fn(y_true, y_pred) -> scalar tensor
    """
    n_quantiles = len(quantiles)

    def loss(y_true, y_pred):
        # y_pred: (batch, n_quantiles * n_features)
        # 重组为 (batch, n_quantiles, n_features)
        n_features = tf.shape(y_true)[-1]
        y_pred_reshaped = tf.reshape(y_pred, [-1, n_quantiles, n_features])

        total_loss = 0.0
        for i, q in enumerate(quantiles):
            pred_q = y_pred_reshaped[:, i, :]
            errors = y_true - pred_q
            total_loss += tf.reduce_mean(
                tf.maximum(q * errors, (q - 1.0) * errors)
            )
        return total_loss / n_quantiles

    return loss


def build_tft(
    input_shape: Tuple[int, int],
    hidden_dim: int = 64,
    num_heads: int = 4,
    dropout_rate: float = 0.1,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    n_static_features: int = 0,
    lstm_layers: int = 1,
    lstm_units: int = 32,
) -> Model:
    """构建完整的 Temporal Fusion Transformer 模型。

    Args:
        input_shape: (n_steps, n_features) 输入序列形状
        hidden_dim: GRN / 全连接层的隐藏维度
        num_heads: 多头注意力头数
        dropout_rate: Dropout 比例
        quantiles: 输出分位数列表
        n_static_features: 静态特征数量（0 表示无静态特征）
        lstm_layers: LSTM 编码器层数
        lstm_units: 每层 LSTM 单元数

    Returns:
        tf.keras.Model, 输入输出如下：
        - 无静态特征时: inputs -> (batch, steps, features), outputs -> (batch, n_quantiles * n_features)
        - 有静态特征时: [dynamic_inputs, static_inputs] -> ...
    """
    n_steps, n_features = input_shape
    n_quantiles = len(quantiles)
    has_static = n_static_features > 0

    # ---- Inputs ----
    dynamic_inputs = Input(shape=input_shape, name='dynamic_inputs')

    if has_static:
        static_inputs = Input(shape=(n_static_features,), name='static_inputs')
    else:
        static_inputs = None

    # ---- Static Covariate Encoder ----
    if has_static:
        static_encoder = StaticCovariateEncoder(hidden_dim, dropout_rate)
        static_contexts = static_encoder(static_inputs)
    else:
        static_contexts = {
            'vsn_context': None, 'lstm_h': None,
            'lstm_c': None, 'grn_context': None,
        }

    # ---- Variable Selection (time-distributed) ----
    vsn = TimeDistributedVSN(hidden_dim, dropout_rate)
    x, vsn_weights = vsn(dynamic_inputs, static_context=static_contexts['vsn_context'])

    # ---- LSTM Encoder ----
    lstm_out = x
    for layer_idx in range(lstm_layers):
        return_seq = True  # 保持序列维度给 attention
        lstm_layer = LSTM(
            lstm_units, return_sequences=return_seq,
            name=f'lstm_encoder_{layer_idx}'
        )
        # 第一层 LSTM 可用静态 context 作为初始状态
        if layer_idx == 0 and has_static and static_contexts['lstm_h'] is not None:
            init_h = Dense(lstm_units, name='lstm_init_h')(static_contexts['lstm_h'])
            init_c = Dense(lstm_units, name='lstm_init_c')(static_contexts['lstm_c'])
            init_h = Lambda(lambda x: tf.expand_dims(x, axis=0),
                           name='lstm_init_h_expand')(init_h)
            init_c = Lambda(lambda x: tf.expand_dims(x, axis=0),
                           name='lstm_init_c_expand')(init_c)
            lstm_out = lstm_layer(lstm_out, initial_state=[init_h, init_c])
        else:
            lstm_out = lstm_layer(lstm_out)

    # Project LSTM output to hidden_dim for skip connection
    if lstm_units != hidden_dim:
        lstm_out = Dense(hidden_dim, name='lstm_proj')(lstm_out)

    # Skip connection + LayerNorm
    x = Add(name='skip_lstm')([lstm_out, x])
    x = LayerNormalization(name='ln_lstm')(x)

    # ---- Temporal Self-Attention (causal) ----
    self_attn = CausalTemporalSelfAttention(hidden_dim, num_heads, dropout_rate)
    attn_out = self_attn(x)
    x = Add(name='skip_attn')([attn_out, x])
    x = LayerNormalization(name='ln_attn')(x)

    # ---- Feed-forward GRN (per time step) ----
    grn = GatedResidualNetwork(hidden_dim, dropout_rate, use_context=has_static)

    # 将序列展平为 (B*T, D) 应用 GRN，再恢复
    x_flat = Reshape((-1, hidden_dim), name='flatten_for_grn')(x)
    grn_ctx_flat = None
    if has_static and static_contexts['grn_context'] is not None:
        # 将静态 context 复制到每个时间步
        grn_ctx_flat = Lambda(
            lambda s: tf.tile(tf.expand_dims(s, axis=1), [1, n_steps, 1]),
            name='tile_static_context'
        )(static_contexts['grn_context'])
        grn_ctx_flat = Reshape((-1, hidden_dim), name='flatten_ctx')(grn_ctx_flat)

    x_flat = grn(x_flat, context=grn_ctx_flat)
    x = Reshape((n_steps, hidden_dim), name='reshape_from_grn')(x_flat)

    # Final LayerNorm
    x = LayerNormalization(name='ln_final')(x)

    # ---- Global pooling over time ----
    x = tf.keras.layers.GlobalAveragePooling1D(name='global_pool')(x)

    # ---- Output: n_quantiles * n_features ----
    outputs = Dense(n_quantiles * n_features, name='quantile_output')(x)

    # ---- Model ----
    if has_static:
        model = Model([dynamic_inputs, static_inputs], outputs, name='TFT')
    else:
        model = Model(dynamic_inputs, outputs, name='TFT')

    # 编译（使用分位数损失）
    model.compile(
        optimizer='adam',
        loss=quantile_loss_fn(quantiles),
    )
    return model


def extract_p50_prediction(y_pred: tf.Tensor, n_quantiles: int,
                           quantile_idx: int = 1) -> tf.Tensor:
    """从分位数输出中提取 P50（中位数）预测。

    Args:
        y_pred: (batch, n_quantiles * n_features)
        n_quantiles: 分位数总数
        quantile_idx: 中位数的索引（默认 1，即 [P10, P50, P90] 中的第二个）
    Returns:
        (batch, n_features) P50 预测值
    """
    n_features = y_pred.shape[-1] // n_quantiles
    y_reshaped = tf.reshape(y_pred, [-1, n_quantiles, n_features])
    return y_reshaped[:, quantile_idx, :]


def extract_prediction_intervals(y_pred: tf.Tensor, n_quantiles: int):
    """提取所有分位数的预测值。

    Returns:
        dict: {'P10': array, 'P50': array, 'P90': array} (取决于 quantiles 配置)
    """
    n_features = y_pred.shape[-1] // n_quantiles
    return tf.reshape(y_pred, [-1, n_quantiles, n_features])

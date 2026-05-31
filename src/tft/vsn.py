"""Variable Selection Network —— 自动加权不同特征的贡献"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

from .gated import GatedResidualNetwork


class VariableSelectionNetwork(Layer):
    """对输入变量做非线性变换并学习每个变量的重要性权重。

    与标准实现不同，这里每个变量独立通过 GRN 再按学习的权重加权求和。
    支持静态变量：当 static_context 不为 None 时，将其融入权重计算。
    """

    def __init__(self, hidden_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        n_features = input_shape[-1]

        # 每个特征一个 GRN 用于非线性变换
        self.feat_grns = [
            GatedResidualNetwork(self.hidden_dim, self.dropout_rate)
            for _ in range(n_features)
        ]
        # 权重组：GRN + Dense(n_features) + Softmax
        self.weight_grn = GatedResidualNetwork(self.hidden_dim, self.dropout_rate)
        self.score_dense = Dense(n_features)

        super().build(input_shape)

    def call(self, x, static_context=None, training=False):
        n_features = x.shape[-1]

        # 逐特征非线性变换
        transformed_list = []
        for j in range(n_features):
            feat_j = x[..., j:j + 1]
            t = self.feat_grns[j](feat_j, training=training)
            transformed_list.append(t)
        transformed = tf.stack(transformed_list, axis=-2)  # (batch, n_features, hidden)

        # 计算变量选择权重
        weights = self.weight_grn(x, context=static_context, training=training)
        weights = self.score_dense(weights)
        weights = tf.nn.softmax(weights, axis=-1)  # (batch, n_features)
        weights = tf.expand_dims(weights, axis=-1)  # (batch, n_features, 1)

        # 加权求和
        selected = tf.reduce_sum(transformed * weights, axis=-2)  # (batch, hidden)
        return selected, weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'hidden_dim': self.hidden_dim, 'dropout_rate': self.dropout_rate})
        return cfg


class TimeDistributedVSN(Layer):
    """逐时间步的 Variable Selection —— 用于序列输入。

    对 (batch, steps, features) 的每个时间步独立做 VSN，
    输出 (batch, steps, hidden) 和 (batch, steps, features, 1) 的权重张量。
    """

    def __init__(self, hidden_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.vsn = VariableSelectionNetwork(self.hidden_dim, self.dropout_rate)
        super().build(input_shape)

    def call(self, x, static_context=None, training=False):
        batch_size = tf.shape(x)[0]
        n_steps = x.shape[1]

        # 将 (B, T, F) 展平为 (B*T, F) 逐时间步处理
        x_flat = tf.reshape(x, [-1, x.shape[-1]])
        selected_flat, weights_flat = self.vsn(
            x_flat, static_context=static_context, training=training
        )
        selected = tf.reshape(selected_flat, [batch_size, n_steps, self.hidden_dim])
        weights = tf.reshape(
            weights_flat, [batch_size, n_steps, x.shape[-1], 1]
        )
        return selected, weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'hidden_dim': self.hidden_dim, 'dropout_rate': self.dropout_rate})
        return cfg

"""Temporal Self-Attention with causal masking (TFT-style)"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class CausalTemporalSelfAttention(Layer):
    """因果时序自注意力 —— 每个时间步只能注意到过去和当前，不能看到未来。

    与 Transformer 标准 MHA 的区别：
    - 使用因果 mask（下三角），确保预测时不会"偷看"未来
    - 输出维度与输入相同，便于残差连接
    """

    def __init__(self, hidden_dim, num_heads=4, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # 确保 hidden_dim 能被 num_heads 整除
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) 必须能被 num_heads ({num_heads}) 整除"
        self.head_dim = hidden_dim // num_heads

    def build(self, input_shape):
        # Q, K, V 投影矩阵
        self.wq = tf.keras.layers.Dense(self.hidden_dim, name='attn_q')
        self.wk = tf.keras.layers.Dense(self.hidden_dim, name='attn_k')
        self.wv = tf.keras.layers.Dense(self.hidden_dim, name='attn_v')
        self.wo = tf.keras.layers.Dense(self.hidden_dim, name='attn_o')
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, mask=None, return_attention=False, training=False):
        """Args:
            x: (batch, steps, hidden_dim)
            mask: 可选的 padding mask (batch, steps)
            return_attention: 是否返回注意力权重（可解释性分析用）
        Returns:
            output: (batch, steps, hidden_dim)
            attention_weights: (batch, num_heads, steps, steps) if return_attention
        """
        batch_size = tf.shape(x)[0]
        n_steps = tf.shape(x)[1]

        # 线性投影并分头
        q = self._split_heads(self.wq(x), batch_size, n_steps)   # (B, H, T, d)
        k = self._split_heads(self.wk(x), batch_size, n_steps)
        v = self._split_heads(self.wv(x), batch_size, n_steps)

        # 缩放点积注意力
        scale = tf.cast(self.head_dim, tf.float32) ** 0.5
        scores = tf.matmul(q, k, transpose_b=True) / scale  # (B, H, T, T)

        # 因果 mask: 上三角为 -inf
        causal_mask = tf.linalg.band_part(tf.ones((n_steps, n_steps)), -1, 0)
        causal_mask = tf.cast(causal_mask, tf.float32)
        causal_mask = (1.0 - causal_mask) * -1e9

        # Padding mask
        if mask is not None:
            pad_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], tf.float32)
            pad_mask = (1.0 - pad_mask) * -1e9
            causal_mask = causal_mask + pad_mask

        scores = scores + causal_mask[tf.newaxis, tf.newaxis, :, :]

        attention_weights = tf.nn.softmax(scores, axis=-1)  # (B, H, T, T)
        attention_weights = self.dropout(attention_weights, training=training)

        # 加权求和
        context = tf.matmul(attention_weights, v)  # (B, H, T, d)
        context = self._merge_heads(context, batch_size, n_steps)  # (B, T, D)
        output = self.wo(context)

        if return_attention:
            return output, attention_weights
        return output

    def _split_heads(self, x, batch_size, n_steps):
        """(B, T, D) -> (B, H, T, d)"""
        x = tf.reshape(x, [batch_size, n_steps, self.num_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def _merge_heads(self, x, batch_size, n_steps):
        """(B, H, T, d) -> (B, T, D)"""
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [batch_size, n_steps, self.hidden_dim])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
        })
        return cfg

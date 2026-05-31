"""Static Covariate Encoder —— 编码时不变特征

TFT 论文中，静态变量通过独立的 GRN 编码后作为 context 注入到
Variable Selection、LSTM 初始状态和 GRN 中。

对于 TE 过程数据，天然缺乏静态特征。但可以通过以下方式构造：
- 手动标注的操作模式标识
- 每个传感器通道的一阶统计量（均值/方差）作为 quasi-static 特征
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer

from .gated import GatedResidualNetwork


class StaticCovariateEncoder(Layer):
    """将静态变量编码为多个 context vector。

    输出四个 context:
    - vsn_context: 注入 Variable Selection Network
    - lstm_h_context: LSTM 初始隐藏状态
    - lstm_c_context: LSTM 初始细胞状态
    - grn_context: 注入后续 GRN 层
    """

    def __init__(self, hidden_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.vsn_grn = GatedResidualNetwork(self.hidden_dim, self.dropout_rate)
        self.lstm_h_grn = GatedResidualNetwork(self.hidden_dim, self.dropout_rate)
        self.lstm_c_grn = GatedResidualNetwork(self.hidden_dim, self.dropout_rate)
        self.grn_context = GatedResidualNetwork(self.hidden_dim, self.dropout_rate)
        super().build(input_shape)

    def call(self, static_vars, training=False):
        """Args:
            static_vars: (batch, n_static_features) 或 None
        Returns:
            dict with keys: vsn_context, lstm_h, lstm_c, grn_context
        """
        if static_vars is None:
            return {
                'vsn_context': None,
                'lstm_h': None,
                'lstm_c': None,
                'grn_context': None,
            }

        vsn_ctx = self.vsn_grn(static_vars, training=training)
        lstm_h = self.lstm_h_grn(static_vars, training=training)
        lstm_c = self.lstm_c_grn(static_vars, training=training)
        grn_ctx = self.grn_context(static_vars, training=training)

        return {
            'vsn_context': vsn_ctx,
            'lstm_h': lstm_h,
            'lstm_c': lstm_c,
            'grn_context': grn_ctx,
        }

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'hidden_dim': self.hidden_dim, 'dropout_rate': self.dropout_rate})
        return cfg

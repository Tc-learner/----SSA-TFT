"""Gated Linear Unit & Gated Residual Network"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Layer


class GLU(Layer):
    """Gated Linear Unit: sigmoid gate * linear value"""

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.gate = Dense(hidden_dim, activation='sigmoid')
        self.value = Dense(hidden_dim)

    def call(self, x):
        return self.gate(x) * self.value(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'hidden_dim': self.hidden_dim})
        return cfg


class GatedResidualNetwork(Layer):
    """GRN: ELU -> Dense -> Dropout -> GLU -> Skip + LayerNorm"""

    def __init__(self, hidden_dim, dropout_rate=0.1, use_context=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_context = use_context

        self.dense1 = Dense(hidden_dim)
        self.dense_context = Dense(hidden_dim) if use_context else None
        self.dense2 = Dense(hidden_dim)
        self.dropout = Dropout(dropout_rate)
        self.glu = GLU(hidden_dim)
        self.layer_norm = LayerNormalization()
        self.projection = Dense(hidden_dim)

    def call(self, x, context=None, training=False):
        skip = x

        eta = self.dense1(x)
        if self.use_context and context is not None:
            eta = eta + self.dense_context(context)
        eta = tf.nn.elu(eta)
        eta = self.dense2(eta)
        eta = self.dropout(eta, training=training)

        if skip.shape[-1] != self.hidden_dim:
            skip = self.projection(skip)

        return self.layer_norm(skip + self.glu(eta))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'hidden_dim': self.hidden_dim,
            'dropout_rate': self.dropout_rate,
            'use_context': self.use_context,
        })
        return cfg

"""模型层 —— LSTM 及其改进变体 + TFT（对比基线保留）"""

from typing import Tuple, List, Optional

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Layer


# ============================================================
# 1. 标准 LSTM
# ============================================================

def build_lstm(input_shape: Tuple[int, int], units: int = 50) -> Sequential:
    """构建标准 LSTM 预测模型"""
    model = Sequential([
        LSTM(units, activation='relu', input_shape=input_shape),
        Dense(input_shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ============================================================
# 2. Attention LSTM（内部注意力机制）
# ============================================================

class AttentionLayer(Layer):
    """Bahdanau 风格的注意力层"""

    def __init__(self, units):
        super().__init__()
        self.units = units
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


class CustomLSTMCellWithAttention(tf.keras.layers.LSTMCell):
    """内嵌注意力机制的 LSTM 单元"""

    def __init__(self, units, **kwargs):
        super().__init__(units=units, **kwargs)
        self.attention = AttentionLayer(units=self.units)

    def call(self, inputs, states, training=None):
        hidden_state = states[0]
        context_vector = self.attention(inputs, hidden_state)
        adjusted_input = inputs * context_vector
        lstm_output, new_states = super().call(adjusted_input, states, training=training)
        return lstm_output, new_states


class CustomLSTMWithAttention(tf.keras.layers.RNN):
    """注意力 LSTM 的 RNN 包装层"""

    def __init__(self, units, return_sequences=True, **kwargs):
        super().__init__(
            cell=CustomLSTMCellWithAttention(units),
            return_sequences=return_sequences,
            **kwargs
        )

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, initial_state=None, training=None):
        return super().call(inputs, initial_state=initial_state, training=training)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.cell.units)
        else:
            return (input_shape[0], self.cell.units)


def build_attention_lstm(input_shape: Tuple[int, int], units: int = 50) -> Sequential:
    """构建带注意力机制的 LSTM 模型"""
    model = Sequential([
        CustomLSTMWithAttention(units=units, input_shape=input_shape, return_sequences=False),
        Dense(input_shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ============================================================
# 3. 自适应遗忘门 LSTM（鲁棒型）
# ============================================================

class LSTMCellWithAdaptiveForgetGate(Layer):
    """
    自适应遗忘门 LSTM 单元
    将输入与期望值（0）的差异融入遗忘门计算，对异常值更鲁棒
    """

    def __init__(self, units):
        super().__init__()
        self.units = units
        self.forget_gate = Dense(units, activation='sigmoid')
        self.input_gate = Dense(units, activation='sigmoid')
        self.cell_gate = Dense(units, activation='tanh')
        self.output_gate = Dense(units, activation='sigmoid')
        self.state_size = [units, units]
        self.output_size = units

    def call(self, inputs, states):
        cell_state, hidden_state = states
        expected_value = 0.0
        input_difference = inputs - expected_value

        adaptive_forget_gate = self.forget_gate(
            tf.concat([inputs, hidden_state, input_difference], axis=-1))
        cell_state = (cell_state * adaptive_forget_gate
                      + self.input_gate(inputs) * self.cell_gate(inputs))
        output_gate_output = self.output_gate(
            tf.concat([inputs, hidden_state], axis=-1))
        hidden_state = output_gate_output * tf.keras.activations.tanh(cell_state)

        return hidden_state, [cell_state, hidden_state]


class CustomLSTMWithAdaptiveForgetGate(Layer):
    """自适应遗忘门 LSTM 的包装层"""

    def __init__(self, units):
        super().__init__()
        self.units = units
        self.lstm_cell = LSTMCellWithAdaptiveForgetGate(units)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        state = [tf.zeros((batch_size, self.units)),
                 tf.zeros((batch_size, self.units))]
        outputs = []
        for t in range(inputs.shape[1]):
            output, state = self.lstm_cell(inputs[:, t, :], state)
            outputs.append(output)
        return tf.stack(outputs, axis=1)


def build_robust_lstm(input_shape: Tuple[int, int], units: int = 8) -> Sequential:
    """构建自适应遗忘门 LSTM 模型"""
    model = Sequential([
        CustomLSTMWithAdaptiveForgetGate(units=units),
        Dense(input_shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ============================================================
# 4. 控制门 LSTM（双重门控）
# ============================================================

class LSTMCellWithAdaptiveForgetAndControlGates(Layer):
    """
    带控制门的自适应遗忘门 LSTM 单元
    在遗忘门基础上增加 control_gate，协同调节细胞状态
    """

    def __init__(self, units):
        super().__init__()
        self.units = units
        self.forget_gate = Dense(units, activation='sigmoid')
        self.input_gate = Dense(units, activation='sigmoid')
        self.cell_gate = Dense(units, activation='tanh')
        self.output_gate = Dense(units, activation='sigmoid')
        self.control_gate = Dense(units, activation='sigmoid')
        self.state_size = [units, units]
        self.output_size = units

    def call(self, inputs, states):
        cell_state, hidden_state = states
        expected_value = 0.0
        input_difference = inputs - expected_value

        adaptive_forget_gate = self.forget_gate(
            tf.concat([inputs, hidden_state, input_difference], axis=-1))
        control_gate = self.control_gate(
            tf.concat([inputs, hidden_state, input_difference], axis=-1))

        cell_state = (cell_state * (adaptive_forget_gate + control_gate)
                      + self.input_gate(inputs) * self.cell_gate(inputs))
        output_gate_output = self.output_gate(
            tf.concat([inputs, hidden_state], axis=-1))
        hidden_state = output_gate_output * tf.keras.activations.tanh(cell_state)

        return hidden_state, [cell_state, hidden_state]


class CustomLSTMWithAdaptiveForgetAndControlGates(Layer):
    """控制门 LSTM 的包装层"""

    def __init__(self, units):
        super().__init__()
        self.units = units
        self.lstm_cell = LSTMCellWithAdaptiveForgetAndControlGates(units)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        state = [tf.zeros((batch_size, self.units)),
                 tf.zeros((batch_size, self.units))]
        outputs = []
        for t in range(inputs.shape[1]):
            output, state = self.lstm_cell(inputs[:, t, :], state)
            outputs.append(output)
        return tf.stack(outputs, axis=1)


def build_control_gate_lstm(input_shape: Tuple[int, int], units: int = 8) -> Sequential:
    """构建控制门 LSTM 模型"""
    model = Sequential([
        CustomLSTMWithAdaptiveForgetAndControlGates(units=units),
        Dense(input_shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ============================================================
# 5. Temporal Fusion Transformer (TFT) —— 从 src.tft 模块加载
# ============================================================

def build_tft(input_shape: Tuple[int, int], units: int = 32) -> Model:
    """构建完整 TFT 模型（分位数输出: P10/P50/P90）

    委托给 src.tft.components.build_tft()，保持向后兼容的接口。
    """
    import config
    from .tft.components import build_tft as _build_tft

    hidden_dim = getattr(config, 'TFT_HIDDEN_DIM', 64)
    num_heads = getattr(config, 'TFT_NUM_HEADS', 4)
    dropout_rate = getattr(config, 'TFT_DROPOUT', 0.1)
    quantiles = getattr(config, 'TFT_QUANTILES', [0.1, 0.5, 0.9])
    n_static = getattr(config, 'TFT_STATIC_FEATURES', 0)
    lstm_layers = getattr(config, 'TFT_LSTM_LAYERS', 1)

    return _build_tft(
        input_shape,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        quantiles=quantiles,
        n_static_features=n_static,
        lstm_layers=lstm_layers,
        lstm_units=units,
    )


# ============================================================
# 6. 模型工厂
# ============================================================

def create_model(model_type: str, input_shape: Tuple[int, int],
                 units: int = 50) -> tf.keras.Model:
    """
    根据类型创建模型

    Args:
        model_type: "lstm" / "attention_lstm" / "robust_lstm" / "control_gate_lstm" / "tft"
        input_shape: (n_steps, n_features)
        units: LSTM 隐藏单元数（TFT 模式下列为 lstm_units）

    Returns:
        编译好的 tf.keras 模型
    """
    builders = {
        "lstm": build_lstm,
        "attention_lstm": build_attention_lstm,
        "robust_lstm": build_robust_lstm,
        "control_gate_lstm": build_control_gate_lstm,
        "tft": build_tft,
    }
    if model_type not in builders:
        raise ValueError(f"未知模型类型: {model_type}，可选: {list(builders.keys())}")
    return builders[model_type](input_shape, units)

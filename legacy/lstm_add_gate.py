import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Layer

class CustomLSTMWithOutlierGate(Layer):
    def __init__(self, units, outlier_threshold=10.0):
        super(CustomLSTMWithOutlierGate, self).__init__()
        self.units = units
        self.outlier_threshold = outlier_threshold
        self.lstm_cell = LSTMCellWithOutlierGate(units)

    def call(self, inputs):
        lstm_output, state = self.lstm_cell(inputs)
        # 添加一个门，判断是否剔除最后一个值
        outlier_gate = self.compute_outlier_gate(inputs[:, -1, :], lstm_output[:, -1, :])
        # 替代原有的
        # lstm_output = tf.where(outlier_gate, lstm_output, 0.0)
        # 使用权重为零的方式
        # lstm_output = lstm_output * tf.cast(tf.math.logical_not(outlier_gate), dtype=tf.float32)
        # 假设异常值在序列中的位置为 outlier_index
        outlier_index = 10  # 仅作为示例，需要根据实际情况调整

        # 使用权重为零的方式，仅对异常值的位置应用
        lstm_output *= tf.concat(
            [tf.ones((outlier_index, 1)), tf.zeros((1, 1)), tf.ones((lstm_output.shape[0] - outlier_index - 1, 1))],
            axis=0)

        return lstm_output

    def compute_outlier_gate(self, last_input, last_output):
        # 计算当前预测值与实际值之间的差异
        prediction_diff = tf.abs(last_output - last_input)
        # 使用门控制策略，如果差异大于阈值则认为是异常
        outlier_gate = prediction_diff > self.outlier_threshold
        return outlier_gate

# 模拟一个LSTMCell的实现
class LSTMCellWithOutlierGate(Layer):
    def __init__(self, units):
        super(LSTMCellWithOutlierGate, self).__init__()
        self.units = units
        self.lstm_cell = LSTMCell(units)

    def call(self, inputs, states):
        lstm_output, new_states = self.lstm_cell(inputs, states)
        return lstm_output, new_states

# 使用示例
lstm_units = 64
model = tf.keras.Sequential([
    CustomLSTMWithOutlierGate(lstm_units),
    Dense(1)
])

# 编译模型和训练
model.compile(optimizer='adam', loss='mse')
model.fit(train_data, train_labels, epochs=num_epochs)
